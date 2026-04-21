"""Precompute first-stage fusion top-K for each Fashion-IQ triple.

Reads triples + gallery embeddings + text embeddings (produced by
`prepare_embeddings.py`) and runs alpha-weighted fusion to produce:

  features/{cat}_{split}_top{K}_idx_{model}.npy    (N, K) int64
  features/{cat}_{split}_top{K}_score_{model}.npy  (N, K) float32
  features/{cat}_{split}_cand_idx_{model}.npy      (N,)   int64
  features/{cat}_{split}_tgt_idx_{model}.npy       (N,)   int64  (-1 if missing)
  features/{cat}_{split}_keep_mask_{model}.npy     (N,)   bool

Gallery is filtered to the split's image_splits ASINs (the Fashion-IQ
protocol — evaluation is against the split's gallery, not the whole dump).

Model-agnostic: only consumes .npy embeddings. Re-run prepare_embeddings.py
with a different --model and this script regenerates for that encoder.

Usage:
  python -m rerank.precompute --category dress --split train --model clip
  python -m rerank.precompute --category dress --split val   --model fashionclip
"""
from __future__ import annotations

import os
import json
import argparse

import numpy as np
from tqdm import tqdm

FEATURES_DIR = "features"
DATA_DIR     = "fashion-iq"


def load_gallery(category: str, split: str, model_name: str):
    """Gallery = (embeddings, paths, asin_to_idx) filtered to split ASINs.

    Matches the protocol used by evaluate.py on feature/fashionclip: the
    full dump of {category}_embeddings_{model}.npy is filtered down to the
    split's image_splits list.
    """
    emb_path   = os.path.join(FEATURES_DIR, f"{category}_embeddings_{model_name}.npy")
    paths_path = os.path.join(FEATURES_DIR, f"{category}_paths_{model_name}.txt")
    split_file = os.path.join(DATA_DIR, "image_splits", f"split.{category}.{split}.json")

    for p in (emb_path, paths_path, split_file):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {p}")

    all_emb = np.load(emb_path).astype("float32")
    with open(paths_path) as f:
        all_paths = [line.strip() for line in f]
    with open(split_file) as f:
        split_asins = set(json.load(f))

    paths, rows = [], []
    for p, row_i in zip(all_paths, range(len(all_paths))):
        asin = os.path.splitext(os.path.basename(p))[0].strip()
        if asin in split_asins:
            paths.append(p)
            rows.append(row_i)
    emb = all_emb[np.array(rows, dtype=np.int64)]
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
    ap.add_argument("--split",    choices=["train", "val"], default="train")
    ap.add_argument("--model",    choices=["clip", "fashionclip"], default="clip")
    ap.add_argument("--alpha",    type=float, default=0.5,
                    help="Fusion weight: alpha * text + (1-alpha) * image")
    ap.add_argument("--top-k",    type=int, default=50,
                    help="Candidates to save per triple (re-rank depth)")
    args = ap.parse_args()

    print(f"[{args.category}/{args.split}/{args.model}] loading gallery + triples...")
    gallery_emb, _, asin_to_idx = load_gallery(args.category, args.split, args.model)
    triples = load_triples(args.category, args.split)
    print(f"  gallery ({args.split} split): {gallery_emb.shape}   triples: {len(triples)}")

    text_emb_path = os.path.join(
        FEATURES_DIR,
        f"{args.category}_{args.split}_text_{args.model}.npy",
    )
    if not os.path.exists(text_emb_path):
        raise FileNotFoundError(
            f"Missing {text_emb_path}. Run:\n"
            f"  python prepare_embeddings.py --category {args.category} "
            f"--split {args.split} --model {args.model}"
        )
    text_emb = np.load(text_emb_path).astype("float32")  # (N, D)
    assert len(text_emb) == len(triples), \
        f"text_emb has {len(text_emb)} rows but triples has {len(triples)}"
    assert text_emb.shape[1] == gallery_emb.shape[1], \
        f"dim mismatch: text {text_emb.shape[1]} vs gallery {gallery_emb.shape[1]}"

    # Build per-triple: cand_idx, tgt_idx, keep_mask (both must be in gallery)
    cand_idx = np.full(len(triples), -1, dtype=np.int64)
    tgt_idx  = np.full(len(triples), -1, dtype=np.int64)
    for i, e in enumerate(triples):
        cand_idx[i] = asin_to_idx.get(e["candidate"], -1)
        tgt_idx[i]  = asin_to_idx.get(e["target"], -1)
    keep_mask = (cand_idx >= 0) & (tgt_idx >= 0)
    print(f"  valid triples (cand+tgt in {args.split} gallery): "
          f"{keep_mask.sum()}/{len(triples)}")

    N, G = len(triples), gallery_emb.shape[0]
    K = min(args.top_k, G)
    if K < args.top_k:
        print(f"  note: top_k reduced from {args.top_k} to {K} "
              f"(gallery only has {G} images)")
    topk_idx   = np.full((N, K), -1, dtype=np.int64)
    topk_score = np.zeros((N, K), dtype=np.float32)

    CHUNK = 512
    for start in tqdm(range(0, N, CHUNK), desc="  scoring"):
        end = min(start + CHUNK, N)
        rows = np.arange(start, end)
        valid_rows = rows[keep_mask[start:end]]
        if len(valid_rows) == 0:
            continue

        t = text_emb[valid_rows]                             # (b, D)
        c = gallery_emb[cand_idx[valid_rows]]                # (b, D)
        text_scores  = t @ gallery_emb.T                     # (b, G)
        image_scores = c @ gallery_emb.T                     # (b, G)
        fusion = args.alpha * text_scores + (1 - args.alpha) * image_scores

        part = np.argpartition(-fusion, K - 1, axis=1)[:, :K]
        gathered = np.take_along_axis(fusion, part, axis=1)
        order = np.argsort(-gathered, axis=1)
        sorted_idx   = np.take_along_axis(part,     order, axis=1)
        sorted_score = np.take_along_axis(gathered, order, axis=1)

        topk_idx[valid_rows]   = sorted_idx
        topk_score[valid_rows] = sorted_score

    # Report ceiling: how many queries have target in top-K
    hit = 0
    for i in range(N):
        if keep_mask[i] and tgt_idx[i] in topk_idx[i]:
            hit += 1
    valid_n = int(keep_mask.sum())
    print(f"  target ∈ top-{K}: {hit}/{valid_n} = {hit/max(valid_n,1):.2%}  "
          f"(this is the re-rank ceiling)")

    base = os.path.join(
        FEATURES_DIR, f"{args.category}_{args.split}"
    )
    m = args.model
    np.save(f"{base}_top{K}_idx_{m}.npy",   topk_idx)
    np.save(f"{base}_top{K}_score_{m}.npy", topk_score)
    np.save(f"{base}_cand_idx_{m}.npy",     cand_idx)
    np.save(f"{base}_tgt_idx_{m}.npy",      tgt_idx)
    np.save(f"{base}_keep_mask_{m}.npy",    keep_mask)
    print(f"  saved to {base}_*_{m}.npy")


if __name__ == "__main__":
    main()
