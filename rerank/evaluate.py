"""Absolute R@K evaluation with re-ranking.

Compares three numbers on the Fashion-IQ val split:

  1. Fusion-only     — first-stage alpha-weighted ranking
  2. Re-ranked top-K — fusion scores are replaced inside top-K by the MLP
  3. Re-rank ceiling — fraction of queries whose target is in top-K
                       (upper bound for any re-ranker; outside top-K we can't
                       help)

Both methods rank over the full val-split gallery (Fashion-IQ protocol).
The re-ranker only reorders top-K; everything below top-K keeps its
first-stage fusion rank.

Usage:
  python -m rerank.evaluate                        # uses ckpt for category+model
  python -m rerank.evaluate --model fashionclip
  python -m rerank.evaluate --category shirt
  python -m rerank.evaluate --ckpt checkpoints/my.pt
  python -m rerank.evaluate --save                 # dump results/rerank_eval_*.json
"""
from __future__ import annotations

import os
import json
import argparse

import numpy as np
import torch

from .model import ReRankerMLP
from .train import load_split_gallery   # reuse the exact same filtering

FEATURES_DIR = "features"
CKPT_DIR     = "checkpoints"
RESULTS_DIR  = "results"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def load_split(category: str, split: str, top_k: int, model_name: str):
    base = os.path.join(FEATURES_DIR, f"{category}_{split}")
    m = model_name
    return {
        "text_emb":   np.load(f"{base}_text_{m}.npy").astype("float32"),
        "topk_idx":   np.load(f"{base}_top{top_k}_idx_{m}.npy"),
        "topk_score": np.load(f"{base}_top{top_k}_score_{m}.npy").astype("float32"),
        "cand_idx":   np.load(f"{base}_cand_idx_{m}.npy"),
        "tgt_idx":    np.load(f"{base}_tgt_idx_{m}.npy"),
        "keep_mask":  np.load(f"{base}_keep_mask_{m}.npy"),
    }


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def main():
    ap = argparse.ArgumentParser(description="Evaluate re-ranker R@K")
    ap.add_argument("--category",   default="dress")
    ap.add_argument("--model",      choices=["clip", "fashionclip"], default="clip")
    ap.add_argument("--ckpt",       default=None,
                    help=f"Checkpoint path (default: {CKPT_DIR}/reranker_"
                         f"{{category}}_{{model}}.pt)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--save",       action="store_true")
    args = ap.parse_args()

    if args.ckpt is None:
        args.ckpt = os.path.join(
            CKPT_DIR, f"reranker_{args.category}_{args.model}.pt"
        )

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg  = ckpt["config"]
    top_k = cfg["top_k"]
    print(f"  config: {cfg}")

    # Consistency check: ckpt-model should match CLI-model
    if cfg.get("model") and cfg["model"] != args.model:
        print(f"  WARNING: ckpt was trained with model={cfg['model']} "
              f"but --model={args.model}. Scores will be meaningless.")

    print(f"Loading gallery + val split (top-{top_k}, model={args.model})...")
    gallery = load_split_gallery(args.category, "val", args.model)
    data    = load_split(args.category, "val", top_k, args.model)
    N = len(data["text_emb"])
    print(f"  val gallery: {gallery.shape}   val triples: {N}")

    model = ReRankerMLP(
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        use_fusion_score=cfg["use_fusion_score"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Score every top-K block with the re-ranker
    print("Scoring top-K with re-ranker...")
    rerank_scores = np.full((N, top_k), -np.inf, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            valid = np.where(data["keep_mask"][start:end])[0] + start
            if len(valid) == 0:
                continue
            cand = torch.from_numpy(
                gallery[data["cand_idx"][valid]]
            ).to(device).float()
            text = torch.from_numpy(data["text_emb"][valid]).to(device).float()
            retr = torch.from_numpy(
                gallery[data["topk_idx"][valid]]
            ).to(device).float()
            fus  = torch.from_numpy(data["topk_score"][valid]).to(device).float()

            s = model(cand, text, retr, fus)
            rerank_scores[valid] = s.cpu().numpy()

    # Compute three sets of ranks ----------------------------------------
    valid_rows = np.where(data["keep_mask"])[0]
    tgt_idx    = data["tgt_idx"]

    in_topk = np.array([
        tgt_idx[i] in data["topk_idx"][i] for i in valid_rows
    ])

    fusion_ranks = np.full(len(valid_rows), top_k + 1, dtype=np.int64)
    rerank_ranks = np.full(len(valid_rows), top_k + 1, dtype=np.int64)
    for i, row in enumerate(valid_rows):
        topk = data["topk_idx"][row]
        t = tgt_idx[row]
        pos = np.where(topk == t)[0]
        if len(pos) == 0:
            continue
        p = int(pos[0])
        fusion_ranks[i] = p + 1   # topk_score is sorted descending

        scores_here = rerank_scores[row]
        tgt_score   = scores_here[p]
        rerank_ranks[i] = int((scores_here > tgt_score).sum()) + 1

    def row(name, ranks):
        return {
            "mode":  name,
            "R@1":   recall_at_k(ranks, 1),
            "R@5":   recall_at_k(ranks, 5),
            "R@10":  recall_at_k(ranks, 10),
            "n":     int(len(ranks)),
        }

    results = [
        row(f"fusion-only (alpha-weighted, top-{top_k} pool)", fusion_ranks),
        row("reranker (MLP)",                                   rerank_ranks),
        {
            "mode": f"ceiling (target ∈ top-{top_k})",
            "R@1":  None, "R@5": None,
            "R@10": float(in_topk.mean()),
            "n":    int(len(valid_rows)),
        },
    ]

    print("\n" + "=" * 68)
    print(f"  [{args.category} / {args.model}]")
    print(f"  {'Mode':<42} {'R@1':>7} {'R@5':>7} {'R@10':>7}")
    print("  " + "-" * 64)
    for r in results:
        r1  = f"{r['R@1']:>7.2%}"  if r["R@1"]  is not None else f"{'—':>7}"
        r5  = f"{r['R@5']:>7.2%}"  if r["R@5"]  is not None else f"{'—':>7}"
        r10 = f"{r['R@10']:>7.2%}" if r["R@10"] is not None else f"{'—':>7}"
        print(f"  {r['mode']:<42} {r1} {r5} {r10}")
    print("=" * 68)
    print(f"  N (valid triples) = {len(valid_rows)}")

    if args.save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(
            RESULTS_DIR,
            f"rerank_eval_{args.category}_{args.model}.json",
        )
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
