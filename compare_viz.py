#!/usr/bin/env python3
"""
compare_viz.py — Member C 的报告可视化脚本

生成三类图片用于报告：
  comparison_NNN.png  : 每条 query 的 3 行对比（text-only / image-only / fusion）
  success_cases.png   : fusion 明显优于两个单模态的案例（"Fusion wins"）
  failure_cases.png   : 三种模式都失败的案例（用于 failure analysis）
  baseline_wins.png   : 单模态好但 fusion 反而更差的案例

scan_entries 函数：批量计算前 N 条 query 的 rank，自动筛选三类案例，
                  速度快（用矩阵乘法，同 evaluate.py 的方式）。

先跑 evaluate.py 确定最好的 alpha，再用 --alpha <best> 跑这个脚本。

详见 notes/05_fusion_and_tuning.md — 三类案例的含义和报告写法
详见 notes/04_evaluation.md       — rank 的计算方式

Usage:
  python compare_viz.py                    # 5 comparison + success + failure grids
  python compare_viz.py --n 10            # 10 per-query comparisons
  python compare_viz.py --alpha 0.3       # different fusion weight
  python compare_viz.py --scan 300        # scan first N queries to find cases
  python compare_viz.py --no_compare      # skip per-query, only grids
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
RESULTS_DIR = "results/compare"
TOP_K = 5

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# ---------------------------------------------------------------------------
# Data loading
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


def encode_text_batch(model, queries: list[str], batch_size=64) -> np.ndarray:
    out = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        with torch.no_grad():
            tokens = clip.tokenize(batch, truncate=True).to(device)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        out.append(emb.cpu().numpy().astype("float32"))
    return np.concatenate(out, axis=0)


def _try_open(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), (180, 180, 180))


# ---------------------------------------------------------------------------
# Per-query comparison figure
# ---------------------------------------------------------------------------

def make_comparison_fig(entry, gallery_emb, gallery_paths, asin_to_idx,
                        model, alpha, n_show=TOP_K):
    target_asin    = entry["target"]
    candidate_asin = entry["candidate"]
    captions       = entry["captions"]

    # Text embedding: average of 2 captions
    emb1 = encode_text(model, captions[0])
    emb2 = encode_text(model, captions[1])
    text_emb = (emb1 + emb2) / 2
    text_emb /= np.linalg.norm(text_emb)

    # Image embedding: candidate looked up from gallery
    cand_idx   = asin_to_idx[candidate_asin]
    image_emb  = gallery_emb[cand_idx]
    target_idx = asin_to_idx[target_asin]

    text_scores   = gallery_emb @ text_emb
    image_scores  = gallery_emb @ image_emb
    fusion_scores = alpha * text_scores + (1 - alpha) * image_scores

    def rank_of(scores):
        return int((scores > scores[target_idx]).sum()) + 1

    text_rank   = rank_of(text_scores)
    image_rank  = rank_of(image_scores)
    fusion_rank = rank_of(fusion_scores)

    text_top   = np.argsort(text_scores)[::-1][:n_show]
    image_top  = np.argsort(image_scores)[::-1][:n_show]
    fusion_top = np.argsort(fusion_scores)[::-1][:n_show]

    n_cols = n_show + 2   # label | target | top-k
    fig, axes = plt.subplots(3, n_cols, figsize=(2.8 * n_cols, 8))

    row_config = [
        ("Text-only",       f"Rank: {text_rank}",         "#FFF3E0",
         text_top,   text_scores,   None),
        ("Image-only",      f"Rank: {image_rank}",         "#E3F2FD",
         image_top,  image_scores,  gallery_paths[cand_idx]),
        (f"Fusion α={alpha}", f"Rank: {fusion_rank}",      "#E8F5E9",
         fusion_top, fusion_scores, None),
    ]

    for row_i, (mode, rank_lbl, bg, top_arr, scores, cand_path) in enumerate(row_config):
        # Col 0: label
        if cand_path:
            axes[row_i, 0].imshow(_try_open(cand_path))
            axes[row_i, 0].set_title(f"{mode}\n{rank_lbl}", fontsize=7)
        else:
            txt = f"{mode}\n{rank_lbl}"
            if row_i == 0:
                txt += f"\n\"{captions[0][:30]}...\"\n\"{captions[1][:30]}...\""
            axes[row_i, 0].text(0.5, 0.5, txt, ha="center", va="center",
                                fontsize=7, transform=axes[row_i, 0].transAxes)
            axes[row_i, 0].set_facecolor(bg)
        axes[row_i, 0].axis("off")

        # Col 1: target
        axes[row_i, 1].imshow(_try_open(gallery_paths[target_idx]))
        axes[row_i, 1].set_title("★ Target", fontsize=7, color="red")
        axes[row_i, 1].axis("off")
        for sp in axes[row_i, 1].spines.values():
            sp.set_edgecolor("red"); sp.set_linewidth(2)

        # Cols 2+: results
        for j, idx in enumerate(top_arr):
            axes[row_i, j + 2].imshow(_try_open(gallery_paths[idx]))
            asin = os.path.splitext(os.path.basename(gallery_paths[idx]))[0]
            is_tgt = asin == target_asin
            axes[row_i, j + 2].set_title(
                f"#{j+1} {scores[idx]:.3f}" + (" ✓" if is_tgt else ""),
                fontsize=7, color="green" if is_tgt else "black"
            )
            if is_tgt:
                for sp in axes[row_i, j + 2].spines.values():
                    sp.set_edgecolor("green"); sp.set_linewidth(2)
            axes[row_i, j + 2].axis("off")

    plt.suptitle(
        f"Candidate → Target: {candidate_asin[:8]} → {target_asin[:8]}\n"
        f"Text rank={text_rank}  Image rank={image_rank}  Fusion rank={fusion_rank}",
        fontsize=8
    )
    plt.tight_layout()
    return fig, text_rank, image_rank, fusion_rank


# ---------------------------------------------------------------------------
# Summary grids (success / failure)
# ---------------------------------------------------------------------------

def make_summary_grid(cases, title, gallery_paths, save_path, n_show=5):
    """
    cases: list of (entry, fusion_scores, text_rank, image_rank, fusion_rank)
    Rows = cases; cols = [candidate | target | top-k fusion results]
    """
    n = len(cases)
    if n == 0:
        print(f"  No cases for: {title}")
        return

    n_cols = n_show + 2
    fig, axes = plt.subplots(n, n_cols, figsize=(2.6 * n_cols, 3.0 * n))
    if n == 1:
        axes = axes[None, :]  # ensure 2D

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    for row, (entry, fusion_scores, t_rank, i_rank, f_rank) in enumerate(cases):
        target_asin    = entry["target"]
        candidate_asin = entry["candidate"]

        # Col 0: candidate image
        # (we don't have asin_to_idx here, resolve from fusion_scores arg)
        # Use the gallery path at the candidate index stored as metadata
        cand_path, tgt_path = entry["_cand_path"], entry["_tgt_path"]

        axes[row, 0].imshow(_try_open(cand_path))
        axes[row, 0].set_title(
            f"Candidate\nT:{t_rank} I:{i_rank} F:{f_rank}", fontsize=6
        )
        axes[row, 0].axis("off")

        # Col 1: target
        axes[row, 1].imshow(_try_open(tgt_path))
        axes[row, 1].set_title("★ Target", fontsize=7, color="red")
        axes[row, 1].axis("off")
        for sp in axes[row, 1].spines.values():
            sp.set_edgecolor("red"); sp.set_linewidth(2)

        # Cols 2+: top fusion results
        top_idx = np.argsort(fusion_scores)[::-1][:n_show]
        for j, idx in enumerate(top_idx):
            axes[row, j + 2].imshow(_try_open(gallery_paths[idx]))
            asin = os.path.splitext(os.path.basename(gallery_paths[idx]))[0]
            is_tgt = asin == target_asin
            axes[row, j + 2].set_title(
                f"#{j+1}" + (" ✓" if is_tgt else ""),
                fontsize=7, color="green" if is_tgt else "black"
            )
            if is_tgt:
                for sp in axes[row, j + 2].spines.values():
                    sp.set_edgecolor("green"); sp.set_linewidth(2)
            axes[row, j + 2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Batch scanning to find interesting cases
# ---------------------------------------------------------------------------

def scan_entries(entries, gallery_emb, gallery_paths, asin_to_idx,
                 model, alpha, n_scan, n_per_case=5):
    """
    Scan up to n_scan queries and classify into:
      - fusion_wins:  fusion_rank < min(text_rank, image_rank) and fusion_rank <= 10
      - all_fail:     all three ranks > 20
      - baseline_wins: text or image rank <= 5 but fusion worse (fusion degraded)
    """
    fusion_wins  = []
    all_fail     = []
    baseline_wins = []

    valid_count = 0
    print(f"  Scanning {n_scan} entries...")

    # Batch encode all captions upfront for speed
    valid_entries = [
        e for e in entries[:n_scan]
        if e["target"] in asin_to_idx and e["candidate"] in asin_to_idx
    ]
    cap1_list = [e["captions"][0] for e in valid_entries]
    cap2_list = [e["captions"][1] for e in valid_entries]
    emb1 = encode_text_batch(model, cap1_list)
    emb2 = encode_text_batch(model, cap2_list)
    text_embs = (emb1 + emb2) / 2
    text_embs /= np.linalg.norm(text_embs, axis=1, keepdims=True)

    cand_indices  = np.array([asin_to_idx[e["candidate"]] for e in valid_entries])
    target_indices = np.array([asin_to_idx[e["target"]] for e in valid_entries])
    image_embs   = gallery_emb[cand_indices]

    text_scores_all  = text_embs  @ gallery_emb.T   # (N, G)
    image_scores_all = image_embs @ gallery_emb.T
    fusion_scores_all = alpha * text_scores_all + (1 - alpha) * image_scores_all

    def rank_of(score_row, tgt_idx):
        return int((score_row > score_row[tgt_idx]).sum()) + 1

    for i, entry in enumerate(valid_entries):
        tgt_idx = target_indices[i]
        t_rank = rank_of(text_scores_all[i],   tgt_idx)
        i_rank = rank_of(image_scores_all[i],  tgt_idx)
        f_rank = rank_of(fusion_scores_all[i], tgt_idx)

        # Annotate with image paths for grid rendering
        entry_copy = dict(entry)
        entry_copy["_cand_path"] = gallery_paths[cand_indices[i]]
        entry_copy["_tgt_path"]  = gallery_paths[tgt_idx]

        case = (entry_copy, fusion_scores_all[i], t_rank, i_rank, f_rank)

        if f_rank <= 10 and f_rank < min(t_rank, i_rank) - 2:
            if len(fusion_wins) < n_per_case:
                fusion_wins.append(case)
        if t_rank > 20 and i_rank > 20 and f_rank > 20:
            if len(all_fail) < n_per_case:
                all_fail.append(case)
        if min(t_rank, i_rank) <= 5 and f_rank > min(t_rank, i_rank) + 5:
            if len(baseline_wins) < n_per_case:
                baseline_wins.append(case)

        if (len(fusion_wins) >= n_per_case and
                len(all_fail) >= n_per_case and
                len(baseline_wins) >= n_per_case):
            break

    return fusion_wins, all_fail, baseline_wins


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Comparison figures for report")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--n", type=int, default=5,
                        help="Number of per-query comparison figures")
    parser.add_argument("--scan", type=int, default=300,
                        help="How many queries to scan for success/failure cases")
    parser.add_argument("--n_cases", type=int, default=5,
                        help="Examples per success/failure grid")
    parser.add_argument("--no_compare", action="store_true",
                        help="Skip per-query comparison figures")
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

    # Per-query comparison figures
    if not args.no_compare:
        print(f"\nGenerating {args.n} per-query comparison figures...")
        saved = 0
        for entry in entries:
            if saved >= args.n:
                break
            if entry["target"] not in asin_to_idx or entry["candidate"] not in asin_to_idx:
                continue
            fig, tr, ir, fr = make_comparison_fig(
                entry, gallery_emb, gallery_paths, asin_to_idx,
                model, args.alpha
            )
            out = os.path.join(RESULTS_DIR, f"comparison_{saved:03d}.png")
            fig.savefig(out, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"  [{saved:02d}] text={tr} image={ir} fusion={fr}  → {out}")
            saved += 1

    # Scan and build case grids
    print(f"\nScanning {args.scan} entries for interesting cases...")
    fusion_wins, all_fail, baseline_wins = scan_entries(
        entries, gallery_emb, gallery_paths, asin_to_idx,
        model, args.alpha, args.scan, args.n_cases
    )

    print(f"  fusion_wins: {len(fusion_wins)}  |  all_fail: {len(all_fail)}  |  baseline_wins: {len(baseline_wins)}")

    if fusion_wins:
        make_summary_grid(
            fusion_wins,
            f"Fusion Wins (α={args.alpha}) — fusion improves over both text-only and image-only",
            gallery_paths,
            os.path.join(RESULTS_DIR, "success_cases.png"),
        )

    if all_fail:
        make_summary_grid(
            all_fail,
            f"Failure Cases — all three modes fail (ranks > 20)",
            gallery_paths,
            os.path.join(RESULTS_DIR, "failure_cases.png"),
        )

    if baseline_wins:
        make_summary_grid(
            baseline_wins,
            f"Fusion Degradation — baseline retrieves well but fusion hurts rank",
            gallery_paths,
            os.path.join(RESULTS_DIR, "baseline_wins.png"),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
