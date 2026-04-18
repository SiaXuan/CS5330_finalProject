#!/usr/bin/env python3
"""
Text-to-image retrieval using CLIP text encoder.

Supports two modes:
  freeform  : arbitrary text queries ("black dress", "white floral shirt")
  fashioniq : uses Fashion-IQ val captions (candidate + modification → target)

Usage:
  python text_retrieval.py                         # freeform demo + Fashion-IQ demo
  python text_retrieval.py --query "red dress"     # single freeform query
  python text_retrieval.py --mode fashioniq        # Fashion-IQ captions only
  python text_retrieval.py --mode freeform         # freeform demo only
  python text_retrieval.py --category shirt        # change category (dress/shirt/toptee)
"""
import os
import json
import argparse

import clip
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

CATEGORY = "dress"
FEATURES_DIR = "features"
DATA_DIR = "fashion-iq"
RESULTS_DIR = "results/text"
TOP_K = 5

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

FREEFORM_QUERIES = [
    # simple: color + category
    "black dress",
    "red dress",
    "white dress",
    # attribute-rich
    "black sleeveless dress",
    "white long floral dress",
    "short striped sundress",
    # style / mood
    "casual summer dress",
    "elegant evening gown",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gallery(category: str):
    emb_path = os.path.join(FEATURES_DIR, f"{category}_embeddings.npy")
    paths_path = os.path.join(FEATURES_DIR, f"{category}_paths.txt")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"Gallery embeddings not found at {emb_path}. "
            "Run extract_features.py first."
        )
    embeddings = np.load(emb_path).astype("float32")
    with open(paths_path) as f:
        paths = [line.strip() for line in f]
    # .strip() guards against a pre-existing bug in download_images.py that
    # saved some files as "B00XYZ .jpg" (trailing space before the extension).
    asin_to_idx = {
        os.path.splitext(os.path.basename(p))[0].strip(): i
        for i, p in enumerate(paths)
    }
    return embeddings, paths, asin_to_idx


def load_fashioniq_val(category: str):
    cap_file = os.path.join(DATA_DIR, "captions", f"cap.{category}.val.json")
    if not os.path.exists(cap_file):
        return None
    with open(cap_file) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_text(model, query: str) -> np.ndarray:
    with torch.no_grad():
        tokens = clip.tokenize([query], truncate=True).to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")[0]


def retrieve(query_emb: np.ndarray, gallery_emb: np.ndarray, top_k: int):
    scores = gallery_emb @ query_emb
    top_idx = np.argsort(scores)[::-1][:top_k]
    return top_idx, scores[top_idx]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _try_open(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), (180, 180, 180))


def save_result(query_text: str, top_idx, scores, gallery_paths, *,
                target_asin: str = None, save_path: str = None, n_show: int = TOP_K):
    n_show = min(n_show, len(top_idx))
    fig, axes = plt.subplots(1, n_show + 1, figsize=(3 * (n_show + 1), 4))

    # Left panel: query text
    axes[0].text(0.5, 0.5, f'Query:\n"{query_text}"',
                 ha="center", va="center", fontsize=9,
                 wrap=True, transform=axes[0].transAxes)
    axes[0].set_facecolor("#e8e8e8")
    axes[0].set_title("Text Query", fontsize=9)
    axes[0].axis("off")

    for i in range(n_show):
        idx = top_idx[i]
        axes[i + 1].imshow(_try_open(gallery_paths[idx]))
        asin = os.path.splitext(os.path.basename(gallery_paths[idx]))[0]
        is_target = target_asin is not None and asin == target_asin
        title = f"#{i+1}  {scores[i]:.3f}" + (" ✓" if is_target else "")
        axes[i + 1].set_title(title, fontsize=8,
                              color="green" if is_target else "black")
        if is_target:
            for spine in axes[i + 1].spines.values():
                spine.set_edgecolor("green")
                spine.set_linewidth(3)
        axes[i + 1].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Demo runners
# ---------------------------------------------------------------------------

def run_freeform(model, gallery_emb, gallery_paths, top_k: int):
    print("\n=== Freeform Text Query Demo ===")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for query in FREEFORM_QUERIES:
        emb = encode_text(model, query)
        top_idx, scores = retrieve(emb, gallery_emb, top_k)
        slug = query.replace(" ", "_")
        save_result(query, top_idx, scores, gallery_paths,
                    save_path=os.path.join(RESULTS_DIR, f"freeform_{slug}.png"))
        print(f"  '{query}'  top-1 score: {scores[0]:.4f}")


def run_fashioniq(model, gallery_emb, gallery_paths, asin_to_idx,
                  entries, top_k: int, n_samples: int = 8):
    print(f"\n=== Fashion-IQ Caption Queries (first {n_samples} valid) ===")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    saved = 0
    for entry in entries:
        if saved >= n_samples:
            break
        target_asin = entry["target"]
        candidate_asin = entry["candidate"]
        if target_asin not in asin_to_idx or candidate_asin not in asin_to_idx:
            continue

        cap1, cap2 = entry["captions"][0], entry["captions"][1]
        emb1 = encode_text(model, cap1)
        emb2 = encode_text(model, cap2)
        query_emb = (emb1 + emb2) / 2
        query_emb = query_emb / np.linalg.norm(query_emb)

        top_idx, scores = retrieve(query_emb, gallery_emb, top_k)
        retrieved_asins = [
            os.path.splitext(os.path.basename(gallery_paths[i]))[0] for i in top_idx
        ]
        hit = target_asin in retrieved_asins
        rank = retrieved_asins.index(target_asin) + 1 if hit else f">{top_k}"

        display_cap = f"{cap1[:45]}... / {cap2[:45]}..."
        print(f"  [{saved:02d}] '{display_cap}'")
        print(f"        target: {target_asin}  rank: {rank}")

        save_result(
            display_cap, top_idx, scores, gallery_paths,
            target_asin=target_asin,
            save_path=os.path.join(RESULTS_DIR, f"fashioniq_{saved:03d}.png"),
        )
        saved += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Text-to-image retrieval with CLIP")
    parser.add_argument("--query", type=str, help="Single freeform text query")
    parser.add_argument("--mode", choices=["freeform", "fashioniq", "both"],
                        default="both")
    parser.add_argument("--category", type=str, default=CATEGORY)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--n_fashioniq", type=int, default=8,
                        help="Number of Fashion-IQ examples to visualize")
    args = parser.parse_args()

    print("Loading CLIP model (ViT-B/32)...")
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()

    print("Loading gallery embeddings...")
    gallery_emb, gallery_paths, asin_to_idx = load_gallery(args.category)
    print(f"Gallery: {len(gallery_paths)} images  dim={gallery_emb.shape[1]}")

    if args.query:
        print(f"\nQuery: '{args.query}'")
        emb = encode_text(model, args.query)
        top_idx, scores = retrieve(emb, gallery_emb, args.top_k)
        for i, idx in enumerate(top_idx):
            print(f"  #{i+1}: {os.path.basename(gallery_paths[idx])}  score={scores[i]:.4f}")
        slug = args.query.replace(" ", "_").replace("/", "-")
        save_result(args.query, top_idx, scores, gallery_paths,
                    save_path=os.path.join(RESULTS_DIR, f"query_{slug}.png"))
        return

    if args.mode in ("freeform", "both"):
        run_freeform(model, gallery_emb, gallery_paths, args.top_k)

    if args.mode in ("fashioniq", "both"):
        entries = load_fashioniq_val(args.category)
        if entries is None:
            print("Fashion-IQ captions not found — skipping.")
        else:
            run_fashioniq(model, gallery_emb, gallery_paths, asin_to_idx,
                          entries, args.top_k, args.n_fashioniq)


if __name__ == "__main__":
    main()
