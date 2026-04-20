#!/usr/bin/env python3
"""
Recall@K evaluation for text-only, image-only, and fusion retrieval.

Dataset:  Fashion-IQ validation split
Metrics:  Recall@1, Recall@5, Recall@10
Query:    (candidate_image, modification_text) → find target_image in gallery

Text query:  average CLIP embedding of both modification captions
Image query: pre-computed gallery embedding of the candidate image (no extra I/O)
Fusion:      alpha * text_score + (1-alpha) * image_score

All score matrices are built with one matrix multiply — fast even on CPU.

Usage:
  python evaluate.py                             # all modes, alphas 0.3 0.5 0.7
  python evaluate.py --alphas 0.5               # single fusion weight
  python evaluate.py --category shirt           # different category
  python evaluate.py --save                     # write results/eval_results.json

  # Skip the gallery shortcut and re-encode every candidate image from
  # disk. Mathematically equivalent but slower — useful as a sanity check
  # and to confirm the pipeline works without pre-computed vectors.
  python evaluate.py --no-gallery-lookup
"""
import os
import json
import argparse

import clip
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

CATEGORY = "dress"
FEATURES_DIR = "features"
DATA_DIR = "fashion-iq"
RESULTS_DIR = "results"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gallery(category: str, model_name: str = "clip"):
    embeddings = np.load(
        os.path.join(FEATURES_DIR, f"{category}_embeddings_{model_name}.npy")
    ).astype("float32")
    with open(os.path.join(FEATURES_DIR, f"{category}_paths_{model_name}.txt")) as f:
        paths = [line.strip() for line in f]
    asin_to_idx = {
        os.path.splitext(os.path.basename(p))[0].strip(): i
        for i, p in enumerate(paths)
    }
    return embeddings, paths, asin_to_idx


def load_val_queries(category: str):
    cap_file = os.path.join(DATA_DIR, "captions", f"cap.{category}.val.json")
    with open(cap_file) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Batch text encoding
# ---------------------------------------------------------------------------

def encode_text_batch(model, queries: list[str], batch_size: int = 64,
                      model_name: str = "clip") -> np.ndarray:
    embeddings = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        if model_name == "clip":
            with torch.no_grad():
                tokens = clip.tokenize(batch, truncate=True).to(device)
                emb = model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy().astype("float32"))
        else:  # fashionclip — use text_model directly (encode_text() has broken API)
            processor = model.preprocess
            text_model = model.model.text_model
            dev = model.device
            inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs['input_ids'].to(dev)
            attention_mask = inputs['attention_mask'].to(dev)
            with torch.no_grad():
                out = text_model(input_ids=input_ids, attention_mask=attention_mask)
                emb = out.pooler_output
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy().astype("float32"))
    return np.concatenate(embeddings, axis=0)


# ---------------------------------------------------------------------------
# Batch image encoding (used when --no-gallery-lookup is set)
# ---------------------------------------------------------------------------

def encode_image_batch(model, preprocess, paths: list, batch_size: int = 32) -> np.ndarray:
    """Encode a list of image paths to normalized CLIP embeddings.

    Used as the slower-but-general alternative to gallery-slicing: each
    candidate image is re-encoded from disk. Results are mathematically
    equivalent to the gallery lookup (same CLIP weights, same preprocess),
    so Recall@K numbers should match.
    """
    embeddings = []
    for i in tqdm(range(0, len(paths), batch_size), desc="  encode_image_batch"):
        batch_paths = paths[i : i + batch_size]
        tensors = torch.stack(
            [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
        ).to(device)
        with torch.no_grad():
            emb = model.encode_image(tensors)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu().numpy().astype("float32"))
    return np.concatenate(embeddings, axis=0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def evaluate_all(entries, gallery_emb, gallery_paths, asin_to_idx, model, alphas,
                 preprocess=None, no_gallery_lookup: bool = False, model_name: str = "clip"):
    """
    Fast batch evaluation.
    - Text embeddings: average of 2 captions, batch-encoded upfront.
    - Image embeddings: either looked up from the pre-computed gallery
      (default, zero disk I/O) or re-encoded from disk per candidate when
      `no_gallery_lookup=True` (slower, but proves the pipeline works
      without a pre-computed index and supports arbitrary candidates).
    - All similarity matrices built with one matmul per modality.
    """
    # Filter entries where both target and candidate are in the gallery
    valid = [
        e for e in entries
        if e["target"] in asin_to_idx and e["candidate"] in asin_to_idx
    ]
    n_skipped = len(entries) - len(valid)
    if n_skipped:
        print(f"  Skipped {n_skipped} queries (images not in gallery)")
    print(f"  Valid queries: {len(valid)}")

    # Batch-encode all captions (2 captions per query → average)
    print("  Encoding text queries...")
    cap1_list = [e["captions"][0] for e in valid]
    cap2_list = [e["captions"][1] for e in valid]
    emb1 = encode_text_batch(model, cap1_list, model_name=model_name)   # (N, D)
    emb2 = encode_text_batch(model, cap2_list, model_name=model_name)   # (N, D)
    text_embs = (emb1 + emb2) / 2
    text_embs /= np.linalg.norm(text_embs, axis=1, keepdims=True)

    # Candidate image embeddings: gallery lookup (fast) or re-encode (general)
    if no_gallery_lookup:
        if preprocess is None:
            raise ValueError("preprocess must be provided when no_gallery_lookup=True")
        print("  Re-encoding candidate images from disk (--no-gallery-lookup)...")
        cand_paths = [gallery_paths[asin_to_idx[e["candidate"]]] for e in valid]
        image_embs = encode_image_batch(model, preprocess, cand_paths)  # (N, D)
    else:
        print("  Looking up candidate image embeddings...")
        cand_idx = np.array([asin_to_idx[e["candidate"]] for e in valid])
        image_embs = gallery_emb[cand_idx]           # (N, D)

    # Target indices in gallery
    target_idx = np.array([asin_to_idx[e["target"]] for e in valid])

    # Score matrices: (N, G)
    print("  Computing similarity matrices...")
    text_scores  = text_embs  @ gallery_emb.T    # (N, G)
    image_scores = image_embs @ gallery_emb.T    # (N, G)

    # Helper: compute ranks from score matrix
    def ranks_from_scores(score_matrix: np.ndarray) -> np.ndarray:
        target_scores = score_matrix[np.arange(len(valid)), target_idx]
        # rank = number of gallery images scored strictly higher + 1
        return (score_matrix > target_scores[:, None]).sum(axis=1) + 1

    results = []

    # Text-only
    r = ranks_from_scores(text_scores)
    results.append({
        "mode": "text-only",
        "R@1":  recall_at_k(r, 1),
        "R@5":  recall_at_k(r, 5),
        "R@10": recall_at_k(r, 10),
        "n":    len(valid),
        "ranks": r.tolist(),
    })

    # Image-only
    r = ranks_from_scores(image_scores)
    results.append({
        "mode": "image-only",
        "R@1":  recall_at_k(r, 1),
        "R@5":  recall_at_k(r, 5),
        "R@10": recall_at_k(r, 10),
        "n":    len(valid),
        "ranks": r.tolist(),
    })

    # Fusion at each alpha
    for alpha in alphas:
        fusion_scores = alpha * text_scores + (1 - alpha) * image_scores
        r = ranks_from_scores(fusion_scores)
        results.append({
            "mode":  f"fusion α={alpha}",
            "R@1":   recall_at_k(r, 1),
            "R@5":   recall_at_k(r, 5),
            "R@10":  recall_at_k(r, 10),
            "n":     len(valid),
            "ranks": r.tolist(),
        })

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(results):
    print("\n" + "=" * 62)
    print(f"  {'Mode':<22} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'N':>7}")
    print("  " + "-" * 58)
    for row in results:
        print(
            f"  {row['mode']:<22}"
            f" {row['R@1']:>7.2%}"
            f" {row['R@5']:>7.2%}"
            f" {row['R@10']:>7.2%}"
            f" {row['n']:>7}"
        )
    print("=" * 62)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Recall@K evaluation")
    parser.add_argument("--category", type=str, default=CATEGORY)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7],
        help="Fusion weights to test (alpha for text, 1-alpha for image)"
    )
    parser.add_argument("--save", action="store_true",
                        help="Save results to results/eval_results.json")
    parser.add_argument("--model", choices=["clip", "fashionclip"], default="clip")
    parser.add_argument("--no-gallery-lookup", action="store_true",
                        help="Re-encode candidate images from disk instead of "
                             "slicing the pre-computed gallery (slower but "
                             "supports open-set candidates)")
    args = parser.parse_args()

    if args.model == "clip":
        print("Loading CLIP model (ViT-B/32)...")
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
    else:
        print("Loading FashionCLIP...")
        from fashion_clip.fashion_clip import FashionCLIP
        model = FashionCLIP('fashion-clip')
        preprocess = None

    print("Loading gallery...")
    gallery_emb, gallery_paths, asin_to_idx = load_gallery(args.category, args.model)
    print(f"Gallery: {len(gallery_paths)} images  dim={gallery_emb.shape[1]}")

    print("Loading val queries...")
    entries = load_val_queries(args.category)
    print(f"Val queries: {len(entries)}")

    print("\nRunning evaluation...")
    results = evaluate_all(
        entries, gallery_emb, gallery_paths, asin_to_idx, model, args.alphas,
        preprocess=preprocess, no_gallery_lookup=args.no_gallery_lookup,
        model_name=args.model,
    )

    print_table(results)

    if args.save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        tag = f"{args.category}_{args.model}"
        save_data = [{k: v for k, v in r.items() if k != "ranks"} for r in results]
        with open(os.path.join(RESULTS_DIR, f"eval_results_{tag}.json"), "w") as f:
            json.dump(save_data, f, indent=2)
        ranks_data = {r["mode"]: r["ranks"] for r in results}
        with open(os.path.join(RESULTS_DIR, f"eval_ranks_{tag}.json"), "w") as f:
            json.dump(ranks_data, f)
        print(f"\nSaved to {RESULTS_DIR}/eval_results_{tag}.json and eval_ranks_{tag}.json")


if __name__ == "__main__":
    main()
