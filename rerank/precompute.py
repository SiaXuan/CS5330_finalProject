"""Precompute first-stage fusion top-K for each Fashion-IQ triple.

Reads the triples + gallery embeddings + text embeddings (produced by
`prepare_embeddings.py`) and runs the same alpha-weighted fusion as
`fusion_retrieval.py` to produce:

  features/{category}_{split}_top{K}_idx.npy    (N, K) int64
  features/{category}_{split}_top{K}_score.npy  (N, K) float32
  features/{category}_{split}_cand_idx.npy      (N,)   int64
  features/{category}_{split}_tgt_idx.npy       (N,)   int64   (-1 if missing)
  features/{category}_{split}_keep_mask.npy     (N,)   bool

Model-agnostic: only consumes .npy embeddings. Rerunning prepare_embeddings.py
with FashionCLIP and then this script regenerates everything for the new
encoder.

Usage:
  python -m rerank.precompute --category dress --split train --alpha 0.5 --top-k 50
  python -m rerank.precompute --category dress --split val   --alpha 0.5 --top-k 50
"""
from __future__ import annotations

import os
import json
import argparse

import numpy as np
from tqdm import tqdm

FEATURES_DIR = "features"
DATA_DIR     = "fashion-iq"


def load_gallery(category: str):
    emb = np.load(os.path.join(FEATURES_DIR, f"{category}_embeddings.npy")).astype("float32")
    with open(os.path.join(FEATURES_DIR, f"{category}_paths.txt")) as f:
        paths = [line.strip() for line in f]
    asin_to_idx = {
        os.path.splitext(os.path.basename(p))[0].strip(): i
        for i, p in enumerate(paths)
    }
    return emb, paths, asin_to_idx


def load_triples(category: str, split: str):
    cap_file = os.path.join(DATA_DIR, "captions", f"cap.{category}.{split}.json")
    with open(cap_file) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Precompute top-K fusion candidates")
    ap.add_argument("--category", default="dress")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Fusion weight: alpha * text + (1-alpha) * image")
    ap.add_argument("--top-k", type=int, default=50,
                    help="Candidates to save per triple (re-rank depth)")
    args = ap.parse_args()

    print(f"[{args.category}/{args.split}] loading gallery + triples...")
    gallery_emb, _, asin_to_idx = load_gallery(args.category)
    triples = load_triples(args.category, args.split)
    print(f"  gallery: {gallery_emb.shape}   triples: {len(triples)}")

    text_emb_path = os.path.join(
        FEATURES_DIR, f"{args.category}_{args.split}_text.npy"
    )
    if not os.path.exists(text_emb_path):
        raise FileNotFoundError(
            f"Missing {text_emb_path}. Run prepare_embeddings.py first."
        )
    text_emb = np.load(text_emb_path).astype("float32")   # (N, D)
    assert len(text_emb) == len(triples), \
        f"text_emb has {len(text_emb)} rows but triples has {len(triples)}"

    # Build per-triple: cand_idx, tgt_idx, keep_mask (both must be in gallery)
    cand_idx = np.full(len(triples), -1, dtype=np.int64)
    tgt_idx  = np.full(len(triples), -1, dtype=np.int64)
    for i, e in enumerate(triples):
        cand_idx[i] = asin_to_idx.get(e["candidate"], -1)
        tgt_idx[i]  = asin_to_idx.get(e["target"], -1)
    keep_mask = (cand_idx >= 0) & (tgt_idx >= 0)
    print(f"  valid triples (cand+tgt in gallery): {keep_mask.sum()}/{len(triples)}")

    # Fusion scoring — process in chunks to avoid blowing up memory.
    # text_score  = text_emb  @ gallery.T      (N, G)
    # image_score = cand_emb  @ gallery.T      (N, G)   (via gallery_emb[cand_idx])
    # fusion      = alpha * text + (1-alpha) * image
    # top-K via argpartition per row.
    N, G = len(triples), gallery_emb.shape[0]
    K = args.top_k
    topk_idx   = np.full((N, K), -1, dtype=np.int64)
    topk_score = np.zeros((N, K), dtype=np.float32)

    CHUNK = 512
    for start in tqdm(range(0, N, CHUNK), desc="  scoring"):
        end = min(start + CHUNK, N)
        # Only score rows where both cand and tgt are valid; for skipped
        # rows leave the -1 sentinel and score 0 (they'll be filtered by
        # keep_mask downstream).
        rows = np.arange(start, end)
        valid_rows = rows[keep_mask[start:end]]
        if len(valid_rows) == 0:
            continue

        t = text_emb[valid_rows]                             # (b, D)
        c = gallery_emb[cand_idx[valid_rows]]                # (b, D)
        text_scores  = t @ gallery_emb.T                     # (b, G)
        image_scores = c @ gallery_emb.T                     # (b, G)
        fusion = args.alpha * text_scores + (1 - args.alpha) * image_scores

        # top-K with argpartition (unordered) then sort those K.
        part = np.argpartition(-fusion, K - 1, axis=1)[:, :K]
        # Gather partitioned scores and sort descending
        gathered = np.take_along_axis(fusion, part, axis=1)
        order = np.argsort(-gathered, axis=1)
        sorted_idx   = np.take_along_axis(part,     order, axis=1)
        sorted_score = np.take_along_axis(gathered, order, axis=1)

        topk_idx[valid_rows]   = sorted_idx
        topk_score[valid_rows] = sorted_score

    # Report target-in-topK rate (the ceiling for a re-ranker)
    hit = 0
    for i in range(N):
        if keep_mask[i] and tgt_idx[i] in topk_idx[i]:
            hit += 1
    valid_n = int(keep_mask.sum())
    print(f"  target ∈ top-{K}: {hit}/{valid_n} = {hit/max(valid_n,1):.2%}  "
          f"(this is the re-rank ceiling)")

    # Save everything
    base = os.path.join(FEATURES_DIR, f"{args.category}_{args.split}")
    np.save(f"{base}_top{K}_idx.npy",   topk_idx)
    np.save(f"{base}_top{K}_score.npy", topk_score)
    np.save(f"{base}_cand_idx.npy",     cand_idx)
    np.save(f"{base}_tgt_idx.npy",      tgt_idx)
    np.save(f"{base}_keep_mask.npy",    keep_mask)
    print(f"  saved to {base}_*.npy")


if __name__ == "__main__":
    main()
