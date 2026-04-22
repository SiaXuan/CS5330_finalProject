#!/usr/bin/env python3
"""
Visualise Combiner vs Fusion (α=0.7) retrieval results on Fashion-IQ.

Generates report-quality figures to results/combiner_viz/:

  comparison_NNN.png   — per-query 2-row grid:
                           row 0: Fusion α=0.7  top-5
                           row 1: Combiner      top-5
                           each row shows [candidate | ★ target | #1 … #5]

  combiner_wins.png    — cases where combiner rank << fusion rank (combiner helps)
  fusion_wins.png      — cases where fusion rank << combiner rank (combiner hurts)
  both_fail.png        — cases where both methods fail (rank > 20)

Usage:
  python viz_combiner.py                           # 5 comparisons, scan 500 for grids
  python viz_combiner.py --category toptee         # different category
  python viz_combiner.py --n 10 --scan 1000        # more figures
  python viz_combiner.py --no-compare              # skip per-query, only grids
  python viz_combiner.py --checkpoint results/my_combiner.pt
"""

import os
import json
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Reuse model/data helpers from train_combiner
from train_combiner import (
    Combiner,
    _encode_text_fashionclip,
    _asin_to_row,
    device,
)

CATEGORY        = "toptee"
FEATURES_DIR    = "features"
DATA_DIR        = "fashion-iq"
RESULTS_DIR     = "results/combiner_viz"
CHECKPOINT_PATH = "results/combiner_fashionclip.pt"
TOP_K           = 5


# ─── Image loading ────────────────────────────────────────────────────────────

def _try_open(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), (180, 180, 180))


# ─── Gallery / query loading ──────────────────────────────────────────────────

def load_val_gallery(category: str):
    """Load val-split gallery embeddings (same filter as evaluate.py)."""
    embs = np.load(
        os.path.join(FEATURES_DIR, f"{category}_embeddings_fashionclip.npy")
    ).astype("float32")
    with open(os.path.join(FEATURES_DIR, f"{category}_paths_fashionclip.txt")) as f:
        all_paths = [line.strip() for line in f]

    with open(os.path.join(DATA_DIR, "image_splits", f"split.{category}.val.json")) as f:
        val_asins = set(json.load(f))

    gal_embs, gal_paths, asin_to_idx = [], [], {}
    for p, emb in zip(all_paths, embs):
        asin = Path(p).stem.strip()
        if asin in val_asins:
            asin_to_idx[asin] = len(gal_embs)
            gal_embs.append(emb)
            gal_paths.append(p)

    return np.stack(gal_embs), gal_paths, asin_to_idx


# ─── Batch score computation ──────────────────────────────────────────────────

def compute_all_scores(entries, gallery_emb, asin_to_idx, fclip, combiner):
    """
    Returns parallel arrays for all valid entries:
      fusion_scores   (N, G)
      combiner_scores (N, G)
      cand_idx        (N,)
      tgt_idx         (N,)
      valid_entries   list[dict]
    """
    valid = [e for e in entries
             if e["candidate"] in asin_to_idx and e["target"] in asin_to_idx]

    cap1 = [e["captions"][0] for e in valid]
    cap2 = [e["captions"][1] for e in valid]
    print(f"  Encoding {len(valid)} text queries...")
    t1 = _encode_text_fashionclip(fclip, cap1)
    t2 = _encode_text_fashionclip(fclip, cap2)
    text_embs  = (t1 + t2) / 2
    text_embs /= np.linalg.norm(text_embs, axis=1, keepdims=True)

    cand_idx = np.array([asin_to_idx[e["candidate"]] for e in valid])
    tgt_idx  = np.array([asin_to_idx[e["target"]]    for e in valid])
    cand_embs = gallery_emb[cand_idx]   # (N, 512)

    # Fusion scores
    gal_t = torch.from_numpy(gallery_emb)
    text_t = torch.from_numpy(text_embs)
    cand_t = torch.from_numpy(cand_embs)
    text_scores  = (text_t  @ gal_t.T).numpy()   # (N, G)
    image_scores = (cand_t  @ gal_t.T).numpy()
    fusion_scores = 0.7 * text_scores + 0.3 * image_scores

    # Combiner scores
    print("  Running combiner...")
    combiner.eval()
    BATCH = 512
    q_parts = []
    for i in range(0, len(valid), BATCH):
        img_t = torch.from_numpy(cand_embs[i:i+BATCH]).to(device)
        txt_t = torch.from_numpy(text_embs[i:i+BATCH]).to(device)
        with torch.no_grad():
            q_parts.append(combiner(img_t, txt_t).cpu())
    query_embs = torch.cat(q_parts, dim=0)
    combiner_scores = (query_embs @ gal_t.T).numpy()   # (N, G)

    return fusion_scores, combiner_scores, cand_idx, tgt_idx, valid


# ─── Per-query comparison figure ──────────────────────────────────────────────

def make_comparison_fig(entry, gallery_paths, asin_to_idx,
                        fusion_scores_row, combiner_scores_row,
                        cand_idx, tgt_idx, n_show=TOP_K):
    """
    2-row figure: [Fusion α=0.7] and [Combiner].
    Each row: [candidate | ★ target | #1 #2 #3 #4 #5]
    """
    def rank_of(scores):
        return int((scores > scores[tgt_idx]).sum()) + 1

    f_rank = rank_of(fusion_scores_row)
    c_rank = rank_of(combiner_scores_row)

    fusion_top   = np.argsort(fusion_scores_row)[::-1][:n_show]
    combiner_top = np.argsort(combiner_scores_row)[::-1][:n_show]

    captions = entry["captions"]
    cap_text = f'"{captions[0][:40]}"\n"{captions[1][:40]}"'

    n_cols = n_show + 2   # [candidate/label] [★ target] [top-1 … top-5]
    fig, axes = plt.subplots(2, n_cols, figsize=(2.6 * n_cols, 5.5))

    row_cfg = [
        ("Fusion α=0.7", f_rank, "#FFF9C4", fusion_scores_row,   fusion_top),
        ("Combiner",     c_rank, "#E8F5E9", combiner_scores_row, combiner_top),
    ]

    for row_i, (label, rank, bg, scores, top_arr) in enumerate(row_cfg):
        ax0 = axes[row_i, 0]
        if row_i == 0:
            # Show candidate image on first row
            ax0.imshow(_try_open(gallery_paths[cand_idx]))
            ax0.set_title(f"Candidate\n{cap_text}", fontsize=6)
        else:
            # Repeat candidate image on second row too for easy comparison
            ax0.imshow(_try_open(gallery_paths[cand_idx]))
            ax0.set_title("(same candidate)", fontsize=6)
        # Colour border to distinguish rows
        for sp in ax0.spines.values():
            sp.set_edgecolor(bg); sp.set_linewidth(3)
        ax0.set_facecolor(bg)
        ax0.axis("off")

        # Row label inset
        ax0.text(0.03, 0.97, f"{label}\nRank: {rank}",
                 transform=ax0.transAxes, fontsize=7,
                 va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.2", fc=bg, alpha=0.85))

        # Col 1: target
        ax1 = axes[row_i, 1]
        ax1.imshow(_try_open(gallery_paths[tgt_idx]))
        ax1.set_title("★ Target", fontsize=7, color="red")
        ax1.axis("off")
        for sp in ax1.spines.values():
            sp.set_edgecolor("red"); sp.set_linewidth(2)

        # Cols 2+: top-k retrieved
        for j, idx in enumerate(top_arr):
            ax = axes[row_i, j + 2]
            ax.imshow(_try_open(gallery_paths[idx]))
            asin = Path(gallery_paths[idx]).stem
            is_tgt = asin == entry["target"]
            ax.set_title(
                f"#{j+1}" + (" ✓" if is_tgt else ""),
                fontsize=7, color="green" if is_tgt else "black"
            )
            if is_tgt:
                for sp in ax.spines.values():
                    sp.set_edgecolor("green"); sp.set_linewidth(2)
            ax.axis("off")

    delta = c_rank - f_rank   # negative = combiner is better
    direction = f"Combiner {'better' if delta < 0 else 'worse'} by {abs(delta)} ranks"
    plt.suptitle(
        f"{entry['candidate'][:8]} → {entry['target'][:8]}   "
        f"Fusion rank={f_rank}  Combiner rank={c_rank}   ({direction})",
        fontsize=8
    )
    plt.tight_layout()
    return fig


# ─── Summary grids ────────────────────────────────────────────────────────────

def make_grid(cases, title, gallery_paths, save_path, n_show=TOP_K):
    """
    cases: list of dicts with keys:
        entry, fusion_scores, combiner_scores, cand_idx, tgt_idx
    Shows 2 rows per case (fusion + combiner), labelled.
    """
    n = len(cases)
    if n == 0:
        print(f"  No cases found for: {title}")
        return

    n_cols = n_show + 2
    # 2 rows per case
    fig, axes = plt.subplots(n * 2, n_cols, figsize=(2.5 * n_cols, 3.8 * n))
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.005)

    row_colors = ["#FFF9C4", "#E8F5E9"]   # fusion=yellow, combiner=green
    row_labels  = ["Fusion α=0.7", "Combiner"]

    for case_i, c in enumerate(cases):
        entry  = c["entry"]
        tgt_idx = c["tgt_idx"]
        cand_idx = c["cand_idx"]

        for sub, (scores, color, lbl, rank) in enumerate([
            (c["fusion_scores"],   row_colors[0], row_labels[0], c["f_rank"]),
            (c["combiner_scores"], row_colors[1], row_labels[1], c["c_rank"]),
        ]):
            row = case_i * 2 + sub
            top_idx = np.argsort(scores)[::-1][:n_show]

            ax0 = axes[row, 0]
            ax0.imshow(_try_open(gallery_paths[cand_idx]))
            ax0.set_title(f"{lbl}\nRank: {rank}", fontsize=6)
            ax0.set_facecolor(color)
            for sp in ax0.spines.values():
                sp.set_edgecolor(color); sp.set_linewidth(3)
            ax0.axis("off")

            ax1 = axes[row, 1]
            ax1.imshow(_try_open(gallery_paths[tgt_idx]))
            ax1.set_title("★ Target", fontsize=6, color="red")
            ax1.axis("off")
            for sp in ax1.spines.values():
                sp.set_edgecolor("red"); sp.set_linewidth(2)

            for j, idx in enumerate(top_idx):
                ax = axes[row, j + 2]
                ax.imshow(_try_open(gallery_paths[idx]))
                is_tgt = Path(gallery_paths[idx]).stem == entry["target"]
                ax.set_title(f"#{j+1}" + (" ✓" if is_tgt else ""),
                             fontsize=6, color="green" if is_tgt else "black")
                if is_tgt:
                    for sp in ax.spines.values():
                        sp.set_edgecolor("green"); sp.set_linewidth(2)
                ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─── Case scanning ────────────────────────────────────────────────────────────

def scan_cases(valid_entries, fusion_scores, combiner_scores, cand_idx, tgt_idx,
               gallery_paths, n_scan: int, n_per_case: int = 5):
    """Find up to n_per_case examples of each case type."""
    combiner_wins = []   # combiner rank << fusion rank
    fusion_wins   = []   # fusion rank << combiner rank
    both_fail     = []   # both rank > 20

    for i, entry in enumerate(valid_entries[:n_scan]):
        t_idx = tgt_idx[i]
        f_row = fusion_scores[i]
        c_row = combiner_scores[i]

        f_rank = int((f_row > f_row[t_idx]).sum()) + 1
        c_rank = int((c_row > c_row[t_idx]).sum()) + 1

        meta = dict(
            entry=entry,
            fusion_scores=f_row,
            combiner_scores=c_row,
            cand_idx=int(cand_idx[i]),
            tgt_idx=int(t_idx),
            f_rank=f_rank,
            c_rank=c_rank,
        )

        # Combiner wins: clearly better rank and actually in top-10
        if c_rank <= 10 and c_rank < f_rank - 3 and len(combiner_wins) < n_per_case:
            combiner_wins.append(meta)
        # Fusion wins: combiner clearly hurts
        elif f_rank <= 10 and f_rank < c_rank - 3 and len(fusion_wins) < n_per_case:
            fusion_wins.append(meta)
        # Both struggle
        elif f_rank > 20 and c_rank > 20 and len(both_fail) < n_per_case:
            both_fail.append(meta)

        if (len(combiner_wins) >= n_per_case and
                len(fusion_wins) >= n_per_case and
                len(both_fail) >= n_per_case):
            break

    return combiner_wins, fusion_wins, both_fail


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Combiner vs Fusion visualisations")
    parser.add_argument("--category",   type=str,   default=CATEGORY)
    parser.add_argument("--checkpoint", type=str,   default=CHECKPOINT_PATH)
    parser.add_argument("--n",          type=int,   default=5,
                        help="Number of per-query comparison figures")
    parser.add_argument("--scan",       type=int,   default=500,
                        help="Queries to scan when finding case examples")
    parser.add_argument("--n-cases",    type=int,   default=5,
                        help="Examples per case-type grid")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip per-query figures, only generate case grids")
    args = parser.parse_args()

    print(f"Device: {device}")

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading FashionCLIP...")
    from fashion_clip.fashion_clip import FashionCLIP
    fclip = FashionCLIP("fashion-clip")
    fclip.model.eval()

    print(f"Loading Combiner from {args.checkpoint}...")
    combiner = Combiner().to(device)
    combiner.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    combiner.eval()

    # ── Load gallery + val queries ────────────────────────────────────────────
    print(f"Loading {args.category} gallery...")
    gallery_emb, gallery_paths, asin_to_idx = load_val_gallery(args.category)
    print(f"  Gallery: {len(gallery_paths)} images")

    with open(os.path.join(DATA_DIR, "captions", f"cap.{args.category}.val.json")) as f:
        entries = json.load(f)
    print(f"  Val queries: {len(entries)}")

    # ── Compute all scores upfront (one pass) ────────────────────────────────
    print("\nComputing fusion + combiner scores for all val queries...")
    fusion_sc, combiner_sc, cand_idx, tgt_idx, valid = compute_all_scores(
        entries, gallery_emb, asin_to_idx, fclip, combiner
    )
    print(f"  Valid queries: {len(valid)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Per-query comparison figures ─────────────────────────────────────────
    if not args.no_compare:
        print(f"\nGenerating {args.n} per-query comparison figures...")
        saved = 0
        for i in range(min(args.n * 3, len(valid))):   # oversample in case some entries are boring
            if saved >= args.n:
                break
            entry = valid[i]
            fig = make_comparison_fig(
                entry, gallery_paths, asin_to_idx,
                fusion_sc[i], combiner_sc[i],
                cand_idx[i], tgt_idx[i],
            )
            out = os.path.join(RESULTS_DIR, f"comparison_{saved:03d}.png")
            fig.savefig(out, dpi=100, bbox_inches="tight")
            plt.close(fig)
            f_rank = int((fusion_sc[i] > fusion_sc[i][tgt_idx[i]]).sum()) + 1
            c_rank = int((combiner_sc[i] > combiner_sc[i][tgt_idx[i]]).sum()) + 1
            print(f"  [{saved:02d}] fusion_rank={f_rank}  combiner_rank={c_rank}  → {out}")
            saved += 1

    # ── Case grids ────────────────────────────────────────────────────────────
    print(f"\nScanning {args.scan} queries for case examples...")
    combiner_wins, fusion_wins, both_fail = scan_cases(
        valid, fusion_sc, combiner_sc, cand_idx, tgt_idx,
        gallery_paths, args.scan, args.n_cases
    )
    print(f"  combiner_wins={len(combiner_wins)}  "
          f"fusion_wins={len(fusion_wins)}  "
          f"both_fail={len(both_fail)}")

    if combiner_wins:
        make_grid(
            combiner_wins,
            f"Combiner Wins — combiner rank << fusion rank  [{args.category}]",
            gallery_paths,
            os.path.join(RESULTS_DIR, "combiner_wins.png"),
        )
    if fusion_wins:
        make_grid(
            fusion_wins,
            f"Fusion Still Wins — cases where combiner hurts  [{args.category}]",
            gallery_paths,
            os.path.join(RESULTS_DIR, "fusion_wins.png"),
        )
    if both_fail:
        make_grid(
            both_fail,
            f"Hard Cases — both methods fail (rank > 20)  [{args.category}]",
            gallery_paths,
            os.path.join(RESULTS_DIR, "both_fail.png"),
        )

    print(f"\nAll figures saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
