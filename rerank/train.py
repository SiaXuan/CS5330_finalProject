"""Train the re-ranker MLP on Fashion-IQ top-K candidates.

Loss: softmax cross-entropy over the K candidates, with the target's
position as the positive class. Equivalent to InfoNCE with K-1 hard
negatives — the re-ranker learns to push the target above the other
first-stage hits (which are, by construction, visually/semantically
confusable).

All heavy lifting is pre-computed; this script only needs numpy arrays
and triples. The inner loop is fast (<1s/step on CPU).

Pipeline (run once per encoder):
  1. python prepare_embeddings.py --split train
  2. python prepare_embeddings.py --split val
  3. python -m rerank.precompute --split train
  4. python -m rerank.precompute --split val
  5. python -m rerank.train

Usage:
  python -m rerank.train                         # defaults (dress, 10 epochs)
  python -m rerank.train --epochs 20 --lr 5e-4
  python -m rerank.train --top-k 50 --batch-size 64
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


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_split(category: str, split: str, top_k: int):
    base = os.path.join(FEATURES_DIR, f"{category}_{split}")
    required = [
        f"{base}_text.npy",
        f"{base}_top{top_k}_idx.npy",
        f"{base}_top{top_k}_score.npy",
        f"{base}_cand_idx.npy",
        f"{base}_tgt_idx.npy",
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing precomputed arrays:\n  " + "\n  ".join(missing) +
            "\nRun prepare_embeddings.py and rerank/precompute.py first."
        )
    return {
        "text_emb":   np.load(f"{base}_text.npy").astype("float32"),
        "topk_idx":   np.load(f"{base}_top{top_k}_idx.npy"),
        "topk_score": np.load(f"{base}_top{top_k}_score.npy").astype("float32"),
        "cand_idx":   np.load(f"{base}_cand_idx.npy"),
        "tgt_idx":    np.load(f"{base}_tgt_idx.npy"),
    }


def load_gallery(category: str) -> np.ndarray:
    return np.load(
        os.path.join(FEATURES_DIR, f"{category}_embeddings.npy")
    ).astype("float32")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="  train", leave=False):
        cand = batch["cand_emb"].to(device)         # (B, D)
        text = batch["text_emb"].to(device)         # (B, D)
        retr = batch["topk_emb"].to(device)         # (B, K, D)
        fus  = batch["topk_fusion"].to(device)      # (B, K)
        tgt  = batch["tgt_pos"].to(device)          # (B,)

        scores = model(cand, text, retr, fus)       # (B, K)
        loss = criterion(scores, tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += float(loss.item()) * len(tgt)
        n     += len(tgt)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader) -> dict:
    """Return R@1/5/10 on the re-ranked top-K.

    Only examples where the target is in the first-stage top-K contribute
    (FashionIQReRankDataset filters them). So this is R@K *relative to the
    re-rank ceiling*, not absolute. evaluate.py reports the absolute number.
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

        scores = model(cand, text, retr, fus)        # (B, K)
        # rank = 1 + (#positions scored higher than target's score)
        tgt_score = scores.gather(1, tgt.unsqueeze(1))    # (B, 1)
        ranks = (scores > tgt_score).sum(dim=1) + 1       # (B,)

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
    ap.add_argument("--category",   default="dress")
    ap.add_argument("--top-k",      type=int, default=50)
    ap.add_argument("--epochs",     type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--n-layers",   type=int, default=3)
    ap.add_argument("--dropout",    type=float, default=0.1)
    ap.add_argument("--no-fusion-score", action="store_true",
                    help="Drop the first-stage fusion score from input features")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--ckpt-name",  default="reranker.pt",
                    help=f"Filename under {CKPT_DIR}/")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading gallery + splits...")
    gallery = load_gallery(args.category)
    train_d = load_split(args.category, "train", args.top_k)
    val_d   = load_split(args.category, "val",   args.top_k)
    print(f"  gallery: {gallery.shape}")
    print(f"  train triples: {len(train_d['text_emb'])}")
    print(f"  val triples:   {len(val_d['text_emb'])}")

    train_ds = FashionIQReRankDataset(gallery, **train_d)
    val_ds   = FashionIQReRankDataset(gallery, **val_d)
    print(f"  train usable (tgt in top-{args.top_k}): {len(train_ds)}")
    print(f"  val   usable (tgt in top-{args.top_k}): {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate, num_workers=0)

    D = gallery.shape[1]
    model = ReRankerMLP(
        emb_dim=D,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        use_fusion_score=not args.no_fusion_score,
    ).to(device)
    print(f"\nModel: ReRankerMLP(emb_dim={D}, hidden={args.hidden_dim}, "
          f"layers={args.n_layers})   params={sum(p.numel() for p in model.parameters()):,}")
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
                },
                "epoch":   epoch,
                "metrics": metrics,
            }, ckpt_path)
            print(f"  → new best, saved to {ckpt_path}")

    # Training summary
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "rerank_train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best val R@10 = {best_r10:.2%}   ckpt={ckpt_path}")


if __name__ == "__main__":
    main()
