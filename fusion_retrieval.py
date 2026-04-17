#!/usr/bin/env python3
"""
fusion_retrieval.py — Member C 的融合检索 demo

公式：final_score = alpha * text_score + (1-alpha) * image_score
     两个分数都是归一化后的 cosine 相似度，值域相同，可以直接加权平均。

用途：生成报告里的 demo 图（不是跑完整评估用的，完整评估用 evaluate.py）

--compare 参数：生成 3 行图（text-only / image-only / fusion），
               直观展示 fusion 是否改善了 rank。

如何选 alpha：先跑 evaluate.py --alphas 0.3 0.5 0.7，
             找 R@10 最高的 alpha，再用那个值跑这个脚本。

详见 notes/05_fusion_and_tuning.md — Fusion 策略与调参
详见 notes/02_data.md             — Fashion-IQ query 格式

Usage:
  python fusion_retrieval.py                     # demo with alpha=0.5
  python fusion_retrieval.py --alpha 0.3         # custom weight
  python fusion_retrieval.py --n 10              # show 10 examples
  python fusion_retrieval.py --compare           # show text/image/fusion per example
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
RESULTS_DIR = "results/fusion"
TOP_K = 5

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gallery(category: str):
    embeddings = np.load(
        os.path.join(FEATURES_DIR, f"{category}_embeddings.npy")
    ).astype("float32")
    with open(os.path.join(FEATURES_DIR, f"{category}_paths.txt")) as f:
        paths = [line.strip() for line in f]
    asin_to_idx = {
        os.path.splitext(os.path.basename(p))[0]: i for i, p in enumerate(paths)
    }
    return embeddings, paths, asin_to_idx


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


def get_query_embeddings(model, entry, gallery_emb, asin_to_idx):
    """一条 Fashion-IQ query → (text_emb, image_emb)"""
    cap1, cap2 = entry["captions"][0], entry["captions"][1]
    emb1 = encode_text(model, cap1)
    emb2 = encode_text(model, cap2)
    text_emb = (emb1 + emb2) / 2     # 两句 caption 向量平均
    text_emb /= np.linalg.norm(text_emb)  # 重新归一化

    # candidate 在 gallery 里，直接切片，不读硬盘
    # 详见 notes/02_data.md — "为什么 candidate 和 target 都在 gallery 里"
    cand_idx = asin_to_idx[entry["candidate"]]
    image_emb = gallery_emb[cand_idx]
    return text_emb, image_emb


def compute_ranks(text_scores, image_scores, target_idx, alpha):
    fusion_scores = alpha * text_scores + (1 - alpha) * image_scores
    text_rank   = int((text_scores   > text_scores[target_idx]).sum())   + 1
    image_rank  = int((image_scores  > image_scores[target_idx]).sum())  + 1
    fusion_rank = int((fusion_scores > fusion_scores[target_idx]).sum()) + 1
    return text_rank, image_rank, fusion_rank, fusion_scores


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def save_simple(entry, top_idx, scores, gallery_paths, target_asin, label, save_path):
    n = len(top_idx)
    fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 4))

    axes[0].text(0.5, 0.5, label, ha="center", va="center",
                 fontsize=8, transform=axes[0].transAxes)
    axes[0].set_facecolor("#e8f5e9")
    axes[0].set_title("Query", fontsize=8)
    axes[0].axis("off")

    for i, idx in enumerate(top_idx):
        axes[i + 1].imshow(_try_open(gallery_paths[idx]))
        asin = os.path.splitext(os.path.basename(gallery_paths[idx]))[0]
        is_tgt = asin == target_asin
        axes[i + 1].set_title(
            f"#{i+1} {scores[i]:.3f}" + (" ✓" if is_tgt else ""),
            fontsize=8, color="green" if is_tgt else "black"
        )
        if is_tgt:
            for spine in axes[i + 1].spines.values():
                spine.set_edgecolor("green"); spine.set_linewidth(3)
        axes[i + 1].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def save_comparison(entry, gallery_emb, gallery_paths, asin_to_idx,
                    model, alpha, save_path, n_show=5):
    """3-row figure: text-only / image-only / fusion."""
    target_asin = entry["target"]
    candidate_asin = entry["candidate"]
    captions = entry["captions"]

    text_emb, image_emb = get_query_embeddings(model, entry, gallery_emb, asin_to_idx)
    target_idx = asin_to_idx[target_asin]

    text_scores  = gallery_emb @ text_emb
    image_scores = gallery_emb @ image_emb
    text_rank, image_rank, fusion_rank, fusion_scores = compute_ranks(
        text_scores, image_scores, target_idx, alpha
    )

    text_top   = np.argsort(text_scores)[::-1][:n_show]
    image_top  = np.argsort(image_scores)[::-1][:n_show]
    fusion_top = np.argsort(fusion_scores)[::-1][:n_show]

    n_cols = n_show + 2  # label | target | top-k
    fig, axes = plt.subplots(3, n_cols, figsize=(2.8 * n_cols, 8))

    rows = [
        ("Text-only",       f"Rank: {text_rank}",         "#FFF3E0", text_top,   text_scores),
        ("Image-only",      f"Rank: {image_rank}",         "#E3F2FD", image_top,  image_scores),
        (f"Fusion α={alpha}", f"Rank: {fusion_rank}",      "#E8F5E9", fusion_top, fusion_scores),
    ]

    for row_i, (mode_label, rank_label, bg, top_arr, score_arr) in enumerate(rows):
        # Col 0: mode label (image-only shows candidate image)
        if row_i == 1:
            cand_path = gallery_paths[asin_to_idx[candidate_asin]]
            axes[row_i, 0].imshow(_try_open(cand_path))
            axes[row_i, 0].set_title(f"{mode_label}\n{rank_label}", fontsize=7)
        else:
            label_txt = f"{mode_label}\n{rank_label}"
            if row_i == 0:
                cap_short = f"\"{captions[0][:35]}...\"\n\"{captions[1][:35]}...\""
                label_txt += f"\n{cap_short}"
            axes[row_i, 0].text(0.5, 0.5, label_txt,
                                ha="center", va="center", fontsize=7,
                                transform=axes[row_i, 0].transAxes)
            axes[row_i, 0].set_facecolor(bg)
        axes[row_i, 0].axis("off")

        # Col 1: target image
        tgt_path = gallery_paths[target_idx]
        axes[row_i, 1].imshow(_try_open(tgt_path))
        axes[row_i, 1].set_title("★ Target", fontsize=7, color="red")
        axes[row_i, 1].axis("off")
        for spine in axes[row_i, 1].spines.values():
            spine.set_edgecolor("red"); spine.set_linewidth(2)

        # Cols 2+: retrieved results
        for j, idx in enumerate(top_arr):
            axes[row_i, j + 2].imshow(_try_open(gallery_paths[idx]))
            asin = os.path.splitext(os.path.basename(gallery_paths[idx]))[0]
            is_tgt = asin == target_asin
            axes[row_i, j + 2].set_title(
                f"#{j+1} {score_arr[idx]:.3f}" + (" ✓" if is_tgt else ""),
                fontsize=7, color="green" if is_tgt else "black"
            )
            if is_tgt:
                for spine in axes[row_i, j + 2].spines.values():
                    spine.set_edgecolor("green"); spine.set_linewidth(2)
            axes[row_i, j + 2].axis("off")

    plt.suptitle(
        f"Candidate → Target: {candidate_asin[:8]} → {target_asin[:8]}",
        fontsize=9
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    return text_rank, image_rank, fusion_rank


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multimodal fusion retrieval demo")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for text score (1-alpha for image)")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of examples to visualize")
    parser.add_argument("--compare", action="store_true",
                        help="Generate 3-row comparison figures (text/image/fusion)")
    parser.add_argument("--category", type=str, default=CATEGORY)
    args = parser.parse_args()

    print("Loading CLIP model...")
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()

    print("Loading gallery...")
    gallery_emb, gallery_paths, asin_to_idx = load_gallery(args.category)
    print(f"Gallery: {len(gallery_paths)} images")

    cap_file = os.path.join(DATA_DIR, "captions", f"cap.{args.category}.val.json")
    with open(cap_file) as f:
        entries = json.load(f)
    print(f"Val queries: {len(entries)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    saved = 0
    for entry in entries:
        if saved >= args.n:
            break
        if entry["target"] not in asin_to_idx or entry["candidate"] not in asin_to_idx:
            continue

        if args.compare:
            tr, ir, fr = save_comparison(
                entry, gallery_emb, gallery_paths, asin_to_idx,
                model, args.alpha,
                save_path=os.path.join(RESULTS_DIR, f"compare_{saved:03d}.png"),
            )
            print(f"  [{saved:02d}] text_rank={tr}  image_rank={ir}  fusion_rank={fr}")
        else:
            text_emb, image_emb = get_query_embeddings(
                model, entry, gallery_emb, asin_to_idx
            )
            text_scores  = gallery_emb @ text_emb
            image_scores = gallery_emb @ image_emb
            _, _, fusion_rank, fusion_scores = compute_ranks(
                text_scores, image_scores, asin_to_idx[entry["target"]], args.alpha
            )
            top_idx = np.argsort(fusion_scores)[::-1][:TOP_K]
            label = (
                f"Text: \"{entry['captions'][0][:40]}...\"\n"
                f"α={args.alpha}  fusion_rank={fusion_rank}"
            )
            save_simple(
                entry, top_idx, fusion_scores[top_idx], gallery_paths,
                entry["target"], label,
                save_path=os.path.join(RESULTS_DIR, f"fusion_{saved:03d}.png"),
            )
            print(f"  [{saved:02d}] fusion rank: {fusion_rank}")

        saved += 1

    print(f"\nDone. {saved} figures saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
