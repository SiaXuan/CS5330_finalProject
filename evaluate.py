#!/usr/bin/env python3
"""
evaluate.py — Member C 的核心评估脚本

作用：在 Fashion-IQ val 集上跑 Recall@K，对比三种检索模式：
  text-only   : 只用 caption 文字
  image-only  : 只用 candidate 图片
  fusion      : alpha * text_score + (1-alpha) * image_score

关键设计：用矩阵乘法一次计算所有 query 对所有 gallery 图片的分数，
          不需要 for 循环，速度极快（见 evaluate_all 函数）。

image-only 的 trick：candidate 本身也在 gallery 里，所以 image embedding
          可以直接从已有的 gallery_emb 里查，不需要重新读取图片文件。

详见 notes/04_evaluation.md — Recall@K 是什么，以及快速计算原理
详见 notes/05_fusion_and_tuning.md — 如何选最好的 alpha

Usage:
  python evaluate.py                             # all modes, alphas 0.3 0.5 0.7
  python evaluate.py --alphas 0.5               # single fusion weight
  python evaluate.py --category shirt           # different category
  python evaluate.py --save                     # write results/eval_results.json
"""
import os
import json
import argparse

import clip
import torch
import numpy as np
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


def load_val_queries(category: str):
    cap_file = os.path.join(DATA_DIR, "captions", f"cap.{category}.val.json")
    with open(cap_file) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Batch text encoding
# ---------------------------------------------------------------------------

def encode_text_batch(model, queries: list[str], batch_size: int = 64) -> np.ndarray:
    embeddings = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        with torch.no_grad():
            tokens = clip.tokenize(batch, truncate=True).to(device)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu().numpy().astype("float32"))
    return np.concatenate(embeddings, axis=0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def evaluate_all(entries, gallery_emb, asin_to_idx, model, alphas):
    """
    Fast batch evaluation.
    - Text embeddings: average of 2 captions, batch-encoded upfront.
    - Image embeddings: looked up from pre-computed gallery (zero disk I/O).
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

    # --- 文本向量：批量编码所有 caption，取两句平均 ---
    # 每条 query 有两句 caption，分别编码再平均，捕捉两者的共同语义
    # encode_text_batch 按 batch_size=64 分批，避免 GPU/内存溢出
    # 详见 notes/03_retrieval.md — "Fashion-IQ Caption 的特殊处理"
    print("  Encoding text queries...")
    cap1_list = [e["captions"][0] for e in valid]
    cap2_list = [e["captions"][1] for e in valid]
    emb1 = encode_text_batch(model, cap1_list)   # (N, D)
    emb2 = encode_text_batch(model, cap2_list)   # (N, D)
    text_embs = (emb1 + emb2) / 2
    text_embs /= np.linalg.norm(text_embs, axis=1, keepdims=True)  # 重新归一化

    # --- 图像向量：直接从 gallery 里查，不读硬盘 ---
    # candidate 本身就在 gallery 里，embedding 已经算好了
    # 详见 notes/02_data.md — "为什么 candidate 和 target 都在 gallery 里"
    print("  Looking up candidate image embeddings...")
    cand_idx = np.array([asin_to_idx[e["candidate"]] for e in valid])
    image_embs = gallery_emb[cand_idx]           # (N, D)  ← 直接切片，极快

    target_idx = np.array([asin_to_idx[e["target"]] for e in valid])

    # --- 核心：一次矩阵乘法算出所有分数 ---
    # text_scores[i, j]  = 第 i 条 query 的文本向量 和 第 j 张 gallery 图 的 cosine 相似度
    # 详见 notes/04_evaluation.md — "为什么这么快"
    print("  Computing similarity matrices...")
    text_scores  = text_embs  @ gallery_emb.T    # (N, G)
    image_scores = image_embs @ gallery_emb.T    # (N, G)

    # --- rank 计算：有几张图比 target 分数高 + 1 ---
    # 详见 notes/04_evaluation.md — "evaluate.py 里的快速计算"
    def ranks_from_scores(score_matrix: np.ndarray) -> np.ndarray:
        target_scores = score_matrix[np.arange(len(valid)), target_idx]  # (N,) 每条 query 中 target 的分数
        # target_scores[:, None] 广播成 (N, 1)，和 (N, G) 比较得到布尔矩阵
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
    args = parser.parse_args()

    print("Loading CLIP model (ViT-B/32)...")
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()

    print("Loading gallery...")
    gallery_emb, gallery_paths, asin_to_idx = load_gallery(args.category)
    print(f"Gallery: {len(gallery_paths)} images  dim={gallery_emb.shape[1]}")

    print("Loading val queries...")
    entries = load_val_queries(args.category)
    print(f"Val queries: {len(entries)}")

    print("\nRunning evaluation...")
    results = evaluate_all(entries, gallery_emb, asin_to_idx, model, args.alphas)

    print_table(results)

    if args.save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_data = [{k: v for k, v in r.items() if k != "ranks"} for r in results]
        with open(os.path.join(RESULTS_DIR, "eval_results.json"), "w") as f:
            json.dump(save_data, f, indent=2)
        ranks_data = {r["mode"]: r["ranks"] for r in results}
        with open(os.path.join(RESULTS_DIR, "eval_ranks.json"), "w") as f:
            json.dump(ranks_data, f)
        print(f"\nSaved to {RESULTS_DIR}/eval_results.json and eval_ranks.json")


if __name__ == "__main__":
    main()
