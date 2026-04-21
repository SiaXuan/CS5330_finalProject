"""Train the re-ranker MLP on Fashion-IQ top-K candidates.

Loss: softmax cross-entropy over the K candidates, with the target's
position as the positive class. Equivalent to InfoNCE with K-1 hard
negatives — the re-ranker learns to push the target above the other
first-stage hits (which are, by construction, visually/semantically
confusable).

All heavy lifting is pre-computed; this script only needs numpy arrays.

Pipeline (run once per encoder):
  1. python prepare_embeddings.py --split train --model {clip|fashionclip}
  2. python prepare_embeddings.py --split val   --model {clip|fashionclip}
  3. python -m rerank.precompute  --split train --model {clip|fashionclip}
  4. python -m rerank.precompute  --split val   --model {clip|fashionclip}
  5. python -m rerank.train                     --model {clip|fashionclip}

Usage:
  python -m rerank.train                           # defaults (dress, clip, 10 epochs)
  python -m rerank.train --model fashionclip
  python -m rerank.train --category shirt --epochs 20
"""
from __future__ import annotations

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model   import ReRankerMLP
from .dataset import FashionIQReRankDataset, collate

FEATURES_DIR = "features"
CKPT_DIR     = "checkpoints"
DATA_DIR     = "fashion-iq"


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_split_gallery(category: str, split: str, model_name: str) -> np.ndarray:
    """Load gallery embeddings filtered to the given split's ASINs.

    Must use the SAME filtering as precompute.py so the topk_idx values
    point into the right array.
    """
    emb_path   = os.path.join(FEATURES_DIR, f"{category}_embeddings_{model_name}.npy")
    paths_path = os.path.join(FEATURES_DIR, f"{category}_paths_{model_name}.txt")
    split_file = os.path.join(DATA_DIR, "image_splits", f"split.{category}.{split}.json")

    all_emb = np.load(emb_path).astype("float32")
    with open(paths_path) as f:
        all_paths = [line.strip() for line in f]
    with open(split_file) as f:
        split_asins = set(json.load(f))

    rows = []
    for p, row_i in zip(all_paths, range(len(all_paths))):
        asin = os.path.splitext(os.path.basename(p))[0].strip()
        if asin in split_asins:
            rows.append(row_i)
    return all_emb[np.array(rows, dtype=np.int64)]


def load_split_arrays(category: str, split: str, top_k: int, model_name: str):
    base = os.path.join(FEATURES_DIR, f"{category}_{split}")
    m = model_name
    required = [
        f"{base}_text_{m}.npy",
        f"{base}_top{top_k}_idx_{m}.npy",
        f"{base}_top{top_k}_score_{m}.npy",
        f"{base}_cand_idx_{m}.npy",
        f"{base}_tgt_idx_{m}.npy",
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing precomputed arrays:\n  " + "\n  ".join(missing) +
            "\nRun prepare_embeddings.py and rerank/precompute.py first."
        )
    return {
        "text_emb":   np.load(f"{base}_text_{m}.npy").astype("float32"),
        "topk_idx":   np.load(f"{base}_top{top_k}_idx_{m}.npy"),
        "topk_score": np.load(f"{base}_top{top_k}_score_{m}.npy").astype("float32"),
        "cand_idx":   np.load(f"{base}_cand_idx_{m}.npy"),
        "tgt_idx":    np.load(f"{base}_tgt_idx_{m}.npy"),
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="  train", leave=False):
        cand = batch["cand_emb"].to(device)
        text = batch["text_emb"].to(device)
        retr = batch["topk_emb"].to(device)
        fus  = batch["topk_fusion"].to(device)
        tgt  = batch["tgt_pos"].to(device)

        scores = model(cand, text, retr, fus)
        loss = criterion(scores, tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += float(loss.item()) * len(tgt)
        n     += len(tgt)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader) -> dict:
    """Return R@1/5/10 on the re-ranked top-K, conditional on target ∈ top-K.

    (rerank/evaluate.py reports the absolute — i.e. unconditional — number.)
    """
    model.eval()
    r1 = r5 = r10 = 0
    n  = 0
    for batch in loader:
        cand = batch["cand_emb"].to(device)
        text = batch["text_emb"].to(device)
        retr = batch["topk_emb"].to(device)
        fus  = batch["topk_fusion"].to(device)
        tgt  = batch["tgt_pos"].to(device)

        scores    = model(cand, text, retr, fus)
        tgt_score = scores.gather(1, tgt.unsqueeze(1))
        ranks     = (scores > tgt_score).sum(dim=1) + 1

        r1  += int((ranks <= 1 ).sum().item())
        r5  += int((ranks <= 5 ).sum().item())
        r10 += int((ranks <= 10).sum().item())
        n   += len(tgt)
    n = max(n, 1)
    return {"R@1": r1 / n, "R@5": r5 / n, "R@10": r10 / n, "n": n}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train re-ranker MLP")
    ap.add_argument("--category",     default="dress")
    ap.add_argument("--model",        choices=["clip", "fashionclip"], default="clip")
    ap.add_argument("--top-k",        type=int, default=50)
    ap.add_argument("--epochs",       type=int, default=10)
    ap.add_argument("--batch-size",   type=int, default=64)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-dim",   type=int, default=512)
    ap.add_argument("--n-layers",     type=int, default=3)
    ap.add_argument("--dropout",      type=float, default=0.1)
    ap.add_argument("--no-fusion-score", action="store_true",
                    help="Drop the first-stage fusion score from input features")
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--ckpt-name",    default=None,
                    help=f"Filename under {CKPT_DIR}/ "
                         f"(default: reranker_{{category}}_{{model}}.pt)")
    args = ap.parse_args()

    if args.ckpt_name is None:
        args.ckpt_name = f"reranker_{args.category}_{args.model}.pt"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading gallery + splits  (category={args.category}, model={args.model})...")
    train_gallery = load_split_gallery(args.category, "train", args.model)
    val_gallery   = load_split_gallery(args.category, "val",   args.model)
    train_d = load_split_arrays(args.category, "train", args.top_k, args.model)
    val_d   = load_split_arrays(args.category, "val",   args.top_k, args.model)
    print(f"  train gallery: {train_gallery.shape}   val gallery: {val_gallery.shape}")
    print(f"  train triples: {len(train_d['text_emb'])}")
    print(f"  val triples:   {len(val_d['text_emb'])}")

    train_ds = FashionIQReRankDataset(train_gallery, **train_d)
    val_ds   = FashionIQReRankDataset(val_gallery,   **val_d)
    print(f"  train usable (tgt in top-{args.top_k}): {len(train_ds)}")
    print(f"  val   usable (tgt in top-{args.top_k}): {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate, num_workers=0)

    D = train_gallery.shape[1]
    model = ReRankerMLP(
        emb_dim=D,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        use_fusion_score=not args.no_fusion_score,
    ).to(device)
    print(f"\nModel: ReRankerMLP(emb_dim={D}, hidden={args.hidden_dim}, "
          f"layers={args.n_layers})   "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, args.ckpt_name)

    best_r10 = -1.0
    history  = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        metrics = evaluate(model, val_loader)
        print(f"Epoch {epoch:02d}  loss={loss:.4f}  "
              f"R@1={metrics['R@1']:.2%}  "
              f"R@5={metrics['R@5']:.2%}  "
              f"R@10={metrics['R@10']:.2%}")
        history.append({"epoch": epoch, "loss": loss, **metrics})

        if metrics["R@10"] > best_r10:
            best_r10 = metrics["R@10"]
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "emb_dim":    D,
                    "hidden_dim": args.hidden_dim,
                    "n_layers":   args.n_layers,
                    "dropout":    args.dropout,
                    "use_fusion_score": not args.no_fusion_score,
                    "top_k":      args.top_k,
                    "category":   args.category,
                    "model":      args.model,
                },
                "epoch":   epoch,
                "metrics": metrics,
            }, ckpt_path)
            print(f"  → new best, saved to {ckpt_path}")

    os.makedirs("results", exist_ok=True)
    hist_path = os.path.join(
        "results",
        f"rerank_train_history_{args.category}_{args.model}.json",
    )
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best val R@10 (conditional) = {best_r10:.2%}   ckpt={ckpt_path}")
    print(f"History: {hist_path}")


if __name__ == "__main__":
    main()
