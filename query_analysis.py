#!/usr/bin/env python3
"""
Analyze text query quality across 4 query types.
Produces a grid image (results/text/query_analysis_grid.png) for the report.

Query types:
  simple     : "black dress", "red dress", "white dress"
  attribute  : "black sleeveless dress", "white long floral dress", "short striped dress"
  style      : "casual summer dress", "elegant evening gown", "vintage boho dress"
  vague      : "nice outfit", "pretty clothes", "fashion"

Usage:
  python query_analysis.py
  python query_analysis.py --category shirt
  python query_analysis.py --top_k 5
"""
import os
import argparse

import clip
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

CATEGORY = "dress"
FEATURES_DIR = "features"
RESULTS_DIR = "results/text"
TOP_K = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

QUERY_GROUPS = {
    "Simple (color+category)": [
        "black dress",
        "red dress",
        "white dress",
    ],
    "Attribute-rich": [
        "black sleeveless dress",
        "white long floral dress",
        "short striped dress",
    ],
    "Style / mood": [
        "casual summer dress",
        "elegant evening gown",
        "vintage boho dress",
    ],
    "Vague / hard": [
        "nice outfit",
        "pretty clothes",
        "fashion",
    ],
}


def load_gallery(category: str):
    embeddings = np.load(
        os.path.join(FEATURES_DIR, f"{category}_embeddings.npy")
    ).astype("float32")
    with open(os.path.join(FEATURES_DIR, f"{category}_paths.txt")) as f:
        paths = [line.strip() for line in f]
    return embeddings, paths


def encode_text(model, query: str) -> np.ndarray:
    with torch.no_grad():
        tokens = clip.tokenize([query], truncate=True).to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")[0]


def _try_open(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), (180, 180, 180))


def make_analysis_grid(model, gallery_emb, gallery_paths, top_k: int, save_path: str):
    all_queries = [
        (group, q)
        for group, queries in QUERY_GROUPS.items()
        for q in queries
    ]
    n_rows = len(all_queries)
    n_cols = top_k + 1

    row_colors = {
        "Simple (color+category)": "#DDEEFF",
        "Attribute-rich":          "#DDFFD9",
        "Style / mood":            "#FFF3DD",
        "Vague / hard":            "#FFE0E0",
    }

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 3.0 * n_rows))
    fig.suptitle(
        f"Text Query Analysis — {CATEGORY.capitalize()} Retrieval  (CLIP ViT-B/32)",
        fontsize=13, fontweight="bold", y=1.01
    )

    print(f"\n{'Query':<38} {'Group':<26} {'Top-1 Score'}")
    print("-" * 72)

    for row, (group, query) in enumerate(all_queries):
        query_emb = encode_text(model, query)
        scores = gallery_emb @ query_emb
        top_idx = np.argsort(scores)[::-1][:top_k]

        # Query label column
        bg = row_colors.get(group, "#FFFFFF")
        axes[row, 0].set_facecolor(bg)
        axes[row, 0].text(
            0.5, 0.5,
            f"[{group}]\n\"{query}\"",
            ha="center", va="center",
            fontsize=8, transform=axes[row, 0].transAxes,
        )
        axes[row, 0].axis("off")

        for col, idx in enumerate(top_idx):
            axes[row, col + 1].imshow(_try_open(gallery_paths[idx]))
            axes[row, col + 1].set_title(f"{scores[idx]:.3f}", fontsize=7)
            axes[row, col + 1].axis("off")

        print(f"{query:<38} {group:<26} {scores[top_idx[0]]:.4f}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze text query types")
    parser.add_argument("--category", type=str, default=CATEGORY)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    args = parser.parse_args()

    print("Loading CLIP model...")
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()

    print("Loading gallery...")
    gallery_emb, gallery_paths = load_gallery(args.category)
    print(f"Gallery: {len(gallery_paths)} images")

    save_path = os.path.join(RESULTS_DIR, "query_analysis_grid.png")
    make_analysis_grid(model, gallery_emb, gallery_paths, args.top_k, save_path)


if __name__ == "__main__":
    main()
