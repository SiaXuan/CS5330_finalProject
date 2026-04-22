#!/usr/bin/env python3
"""
Train a Combiner MLP on Fashion-IQ training triplets using FashionCLIP.

──────────────────────────────────────────────────────────────────────────────
WHAT IS THE COMBINER?
──────────────────────────────────────────────────────────────────────────────
The fusion baseline (evaluate.py) is *late fusion*:

    text_score  = text_emb  @ gallery.T          # (N, G)
    image_score = image_emb @ gallery.T
    final_score = α * text_score + (1-α) * image_score

The Combiner is *early fusion*: a small MLP that takes the candidate image
embedding and the modification text embedding and outputs a single unified
query embedding, which is then matched against the gallery:

    query_emb = Combiner(image_emb, text_emb)    # (N, 512)
    score     = query_emb @ gallery.T            # (N, G)

The MLP is *learned* from training triplets, so it can do more than a fixed
weighted average — it learns to produce embeddings that land near the target
region of the CLIP embedding space.

──────────────────────────────────────────────────────────────────────────────
ARCHITECTURE  (CLIP4CIR residual MLP, Baldrati et al. 2022)
──────────────────────────────────────────────────────────────────────────────
    input:  image_emb (512)  +  text_emb (512)
                                    │
                          concat → (1024)
                                    │
                           Linear(1024, 1024)
                                    │
                                  GELU
                                    │
                               Dropout(0.1)
                                    │
                           Linear(1024, 512)   ← delta
                                    │
                    image_emb  ─── (+) ←── residual: start from image,
                                    │        apply text-conditioned shift
                               L2-normalize
                                    │
                             query_emb (512)

The residual design is important: at initialisation the delta ≈ 0, so the
combiner starts out as pure image retrieval and gradually learns to shift
toward the target. This makes training stable.

The backbone (FashionCLIP) is *frozen*. Only the 2-layer MLP is trained
(~1.6 M parameters vs ~150 M in the backbone).

──────────────────────────────────────────────────────────────────────────────
TRAINING
──────────────────────────────────────────────────────────────────────────────
Loss: InfoNCE (batch contrastive cross-entropy).

In a batch of B triplets, query_emb[i] should be closest to target_emb[i]
among all B targets.  We build a (B, B) cosine similarity matrix and minimise
cross-entropy with the diagonal as ground truth:

    logits = (query_emb @ target_emb.T) / temperature
    loss   = 0.5 * (CE(logits, I) + CE(logits.T, I))

Temperature 0.07 matches the original CLIP pre-training (Radford et al. 2021).

All image and text embeddings are pre-computed once before training, so the
training loop never touches the FashionCLIP backbone again — it is pure matrix
ops and therefore very fast (~seconds per epoch on CPU).

──────────────────────────────────────────────────────────────────────────────
USAGE
──────────────────────────────────────────────────────────────────────────────
    # Train on all 3 categories (default), evaluate every 2 epochs
    python train_combiner.py

    # Train on a single category
    python train_combiner.py --category dress

    # Custom hyperparameters
    python train_combiner.py --epochs 20 --batch-size 256 --lr 2e-4

    # Only run evaluation with a saved checkpoint
    python train_combiner.py --eval-only

    # Specify a different checkpoint path
    python train_combiner.py --checkpoint results/my_combiner.pt
"""

import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

CATEGORIES     = ["dress", "shirt", "toptee"]
FEATURES_DIR   = "features"
DATA_DIR       = "fashion-iq"
RESULTS_DIR    = "results"
CHECKPOINT_PATH = "results/combiner_fashionclip.pt"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# ─── Model ────────────────────────────────────────────────────────────────────

class Combiner(nn.Module):
    """
    Residual MLP combiner.

    Takes a candidate image embedding and a modification text embedding,
    outputs a query embedding (same dim) that should be close to the target
    image embedding in cosine space.

    Formula:  normalize( image_emb + FC2(GELU(FC1([image_emb ; text_emb]))) )

    The residual (+image_emb) means the network only needs to learn the *delta*
    from the candidate, not the full target embedding from scratch.
    """

    def __init__(self, embed_dim: int = 512, hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.fc1     = nn.Linear(embed_dim * 2, hidden_dim)
        self.fc2     = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_emb: (B, 512) — L2-normalised candidate image embeddings
            text_emb:  (B, 512) — L2-normalised modification text embeddings
        Returns:
            (B, 512) — L2-normalised query embedding for target retrieval
        """
        fused = torch.cat([image_emb, text_emb], dim=-1)        # (B, 1024)
        delta = self.fc2(self.dropout(F.gelu(self.fc1(fused))))  # (B, 512)
        return F.normalize(image_emb + delta, dim=-1)


# ─── Loss ─────────────────────────────────────────────────────────────────────

def infonce_loss(query_emb: torch.Tensor, target_emb: torch.Tensor,
                 temperature: float = 0.07) -> torch.Tensor:
    """
    Symmetric InfoNCE (NT-Xent) loss.

    Builds a (B, B) cosine similarity matrix. The diagonal entries are the
    positive pairs (query[i] ↔ target[i]); all off-diagonal entries are
    in-batch negatives. The symmetric version averages query→target and
    target→query CE losses to use every sample as both anchor and positive.
    """
    logits  = (query_emb @ target_emb.T) / temperature     # (B, B)
    labels  = torch.arange(len(query_emb), device=query_emb.device)
    loss_qt = F.cross_entropy(logits,   labels)             # query → target
    loss_tq = F.cross_entropy(logits.T, labels)             # target → query
    return (loss_qt + loss_tq) / 2


# ─── Data helpers ─────────────────────────────────────────────────────────────

def _asin_to_row(category: str, model: str = "fashionclip") -> dict[str, int]:
    """Map ASIN → row index in the pre-computed embedding matrix."""
    path = os.path.join(FEATURES_DIR, f"{category}_paths_{model}.txt")
    with open(path) as f:
        paths = [line.strip() for line in f]
    return {Path(p).stem.strip(): i for i, p in enumerate(paths)}


def _encode_text_fashionclip(fclip, texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Encode texts using FashionCLIP's text tower (same path as evaluate.py)."""
    processor  = fclip.preprocess
    text_model = fclip.model.text_model
    proj       = fclip.model.text_projection
    dev        = fclip.device

    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch  = texts[i : i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            pooled = text_model(
                input_ids      = inputs["input_ids"].to(dev),
                attention_mask = inputs["attention_mask"].to(dev),
            ).pooler_output
            emb = proj(pooled)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embs.append(emb.cpu().numpy().astype("float32"))
    return np.concatenate(all_embs, axis=0)


# ─── Pre-computation ──────────────────────────────────────────────────────────

def precompute_train(fclip, categories: list[str]):
    """
    Pre-compute everything needed for the training loop.

    Returns:
        cand_embs  (N, 512) — candidate image embeddings
        text_embs  (N, 512) — average of 2 modification captions
        tgt_embs   (N, 512) — target image embeddings
    """
    print("Loading pre-computed image embeddings...")
    asin_to_emb: dict[str, np.ndarray] = {}
    for cat in categories:
        embs = np.load(
            os.path.join(FEATURES_DIR, f"{cat}_embeddings_fashionclip.npy")
        ).astype("float32")
        for asin, idx in _asin_to_row(cat).items():
            asin_to_emb[asin] = embs[idx]
    print(f"  Loaded {len(asin_to_emb)} unique image embeddings")

    print("Loading training triplets...")
    triplets = []
    for cat in categories:
        with open(os.path.join(DATA_DIR, "captions", f"cap.{cat}.train.json")) as f:
            triplets.extend(json.load(f))

    # Only keep triplets where both images are available (some downloads fail)
    triplets = [t for t in triplets
                if t["candidate"] in asin_to_emb and t["target"] in asin_to_emb]
    print(f"  Valid triplets: {len(triplets)}")

    print("Pre-encoding text captions (runs backbone once, then frozen)...")
    cap1_texts = [t["captions"][0] for t in triplets]
    cap2_texts = [t["captions"][1] for t in triplets]
    cap1_embs  = _encode_text_fashionclip(fclip, cap1_texts)
    cap2_embs  = _encode_text_fashionclip(fclip, cap2_texts)

    # Average of 2 captions, re-normalise — same as evaluate.py
    text_embs  = (cap1_embs + cap2_embs) / 2
    text_embs /= np.linalg.norm(text_embs, axis=1, keepdims=True)

    cand_embs = np.stack([asin_to_emb[t["candidate"]] for t in triplets])
    tgt_embs  = np.stack([asin_to_emb[t["target"]]    for t in triplets])
    return cand_embs, text_embs, tgt_embs


def precompute_val(fclip, categories: list[str]) -> dict[str, dict]:
    """
    Pre-compute val gallery embeddings and val query text embeddings so that
    evaluation during training never calls the FashionCLIP backbone again.

    Returns a dict keyed by category with:
        gal_embs    (G, 512) — val gallery image embeddings
        asin_to_idx dict
        cand_embs   (N, 512) — candidate image embeddings (sliced from gallery)
        text_embs   (N, 512) — encoded val text queries
        tgt_idx     (N,)     — target row indices in gal_embs
        n_valid     int
    """
    print("Pre-computing val data for fast in-training evaluation...")
    val_data = {}
    for cat in categories:
        embs = np.load(
            os.path.join(FEATURES_DIR, f"{cat}_embeddings_fashionclip.npy")
        ).astype("float32")
        with open(os.path.join(FEATURES_DIR, f"{cat}_paths_fashionclip.txt")) as f:
            all_paths = [line.strip() for line in f]

        with open(os.path.join(DATA_DIR, "image_splits", f"split.{cat}.val.json")) as f:
            val_asins = set(json.load(f))

        # Build val gallery (same filter as evaluate.py)
        gal_embs, asin_to_idx = [], {}
        for p, emb in zip(all_paths, embs):
            asin = Path(p).stem.strip()
            if asin in val_asins:
                asin_to_idx[asin] = len(gal_embs)
                gal_embs.append(emb)
        gal_embs = np.stack(gal_embs)

        with open(os.path.join(DATA_DIR, "captions", f"cap.{cat}.val.json")) as f:
            entries = json.load(f)
        valid = [e for e in entries
                 if e["candidate"] in asin_to_idx and e["target"] in asin_to_idx]

        # Encode val text
        cap1 = [e["captions"][0] for e in valid]
        cap2 = [e["captions"][1] for e in valid]
        t1   = _encode_text_fashionclip(fclip, cap1)
        t2   = _encode_text_fashionclip(fclip, cap2)
        text_embs  = (t1 + t2) / 2
        text_embs /= np.linalg.norm(text_embs, axis=1, keepdims=True)

        cand_idx  = np.array([asin_to_idx[e["candidate"]] for e in valid])
        tgt_idx   = np.array([asin_to_idx[e["target"]]    for e in valid])

        val_data[cat] = {
            "gal_embs":  gal_embs,
            "cand_embs": gal_embs[cand_idx],
            "text_embs": text_embs,
            "tgt_idx":   tgt_idx,
            "n_valid":   len(valid),
        }
        print(f"  [{cat}] gallery={len(gal_embs)}  val queries={len(valid)}")
    return val_data


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_combiner(combiner: Combiner, val_data: dict,
                      verbose: bool = True) -> float:
    """
    Evaluate the combiner on the pre-computed val data.
    Returns average R@10 across all categories.
    """
    combiner.eval()
    r10_list = []

    for cat, data in val_data.items():
        gal  = torch.from_numpy(data["gal_embs"])   # (G, 512)
        cand = data["cand_embs"]                    # (N, 512) numpy
        text = data["text_embs"]                    # (N, 512) numpy
        tgt  = data["tgt_idx"]                      # (N,)     numpy

        # Run combiner in batches (avoids OOM on large val sets)
        BATCH = 512
        q_parts = []
        for i in range(0, len(cand), BATCH):
            img_t = torch.from_numpy(cand[i:i+BATCH]).to(device)
            txt_t = torch.from_numpy(text[i:i+BATCH]).to(device)
            with torch.no_grad():
                q_parts.append(combiner(img_t, txt_t).cpu())
        query_embs = torch.cat(q_parts, dim=0)      # (N, 512)

        # Cosine scores against full gallery
        scores = (query_embs @ gal.T).numpy()       # (N, G)
        tgt_scores = scores[np.arange(len(tgt)), tgt]
        ranks = (scores > tgt_scores[:, None]).sum(axis=1) + 1  # 1-indexed

        r10 = float(np.mean(ranks <= 10))
        r50 = float(np.mean(ranks <= 50))
        r10_list.append(r10)
        if verbose:
            print(f"  [{cat:7s}] R@10={r10:.4f}  R@50={r50:.4f}  N={data['n_valid']}")

    return float(np.mean(r10_list)) if r10_list else 0.0


# ─── Training loop ────────────────────────────────────────────────────────────

def train(fclip, categories: list[str], epochs: int, batch_size: int,
          lr: float, temperature: float, checkpoint: str):

    cand_embs, text_embs, tgt_embs = precompute_train(fclip, categories)
    val_data = precompute_val(fclip, categories)

    combiner  = Combiner().to(device)
    optimizer = torch.optim.AdamW(combiner.parameters(), lr=lr, weight_decay=1e-4)

    n        = len(cand_embs)
    best_r10 = 0.0

    print(f"\nTraining Combiner for {epochs} epochs  "
          f"(batch={batch_size}, lr={lr}, temp={temperature})")
    print(f"Device: {device}   Trainable params: "
          f"{sum(p.numel() for p in combiner.parameters()):,}")
    print()

    for epoch in range(1, epochs + 1):
        combiner.train()
        idx_perm   = np.random.permutation(n)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            idx = idx_perm[start : start + batch_size]
            img_t = torch.from_numpy(cand_embs[idx]).to(device)
            txt_t = torch.from_numpy(text_embs[idx]).to(device)
            tgt_t = torch.from_numpy(tgt_embs[idx]).to(device)

            query_emb = combiner(img_t, txt_t)
            loss      = infonce_loss(query_emb, tgt_t, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch:>2}/{epochs}  loss={avg_loss:.4f}", end="")

        # Evaluate every 2 epochs and on the final epoch
        if epoch % 2 == 0 or epoch == epochs:
            print()
            r10_avg = evaluate_combiner(combiner, val_data)
            if r10_avg > best_r10:
                best_r10 = r10_avg
                os.makedirs(RESULTS_DIR, exist_ok=True)
                torch.save(combiner.state_dict(), checkpoint)
                print(f"  ✓ New best R@10={r10_avg:.4f} — saved to {checkpoint}")
        else:
            print()

    print(f"\nTraining done. Best avg R@10 (3 categories): {best_r10:.4f}")
    return combiner


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train a Combiner MLP on Fashion-IQ (FashionCLIP backbone, frozen)"
    )
    parser.add_argument(
        "--category", nargs="+", default=CATEGORIES,
        metavar="CAT",
        help="Categories to train on (default: dress shirt toptee)"
    )
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-4,
                        help="AdamW learning rate")
    parser.add_argument("--temp",       type=float, default=0.07,
                        help="InfoNCE temperature (CLIP default: 0.07)")
    parser.add_argument("--checkpoint", type=str,   default=CHECKPOINT_PATH,
                        help="Where to save / load the combiner weights")
    parser.add_argument("--eval-only",  action="store_true",
                        help="Skip training, evaluate an existing checkpoint")
    args = parser.parse_args()

    print(f"Device: {device}")
    print("Loading FashionCLIP (backbone will be frozen during training)...")
    from fashion_clip.fashion_clip import FashionCLIP
    fclip = FashionCLIP("fashion-clip")
    fclip.model.eval()

    if args.eval_only:
        if not os.path.exists(args.checkpoint):
            print(f"Checkpoint not found: {args.checkpoint}")
            return
        combiner = Combiner().to(device)
        combiner.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=True)
        )
        print(f"Loaded checkpoint: {args.checkpoint}\n")
        val_data = precompute_val(fclip, args.category)
        print("\nEvaluation results:")
        avg = evaluate_combiner(combiner, val_data)
        print(f"\nAverage R@10: {avg:.4f}")
    else:
        combiner = train(
            fclip, args.category, args.epochs,
            args.batch_size, args.lr, args.temp, args.checkpoint
        )
        # Load best checkpoint and do a final clean eval
        if os.path.exists(args.checkpoint):
            combiner.load_state_dict(
                torch.load(args.checkpoint, map_location=device, weights_only=True)
            )
        print("\nFinal evaluation (best checkpoint):")
        val_data = precompute_val(fclip, args.category)
        avg = evaluate_combiner(combiner, val_data)
        print(f"\nFinal avg R@10: {avg:.4f}")
        print(f"Checkpoint saved to: {args.checkpoint}")


if __name__ == "__main__":
    main()
