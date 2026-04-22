# FashionCLIP Integration — What We Did

## Overview

Extended the baseline CLIP retrieval pipeline to support **FashionCLIP** (`patrickjohncyh/fashion-clip`), a fashion-domain fine-tuned CLIP model, and ran a full Recall@K evaluation across all three Fashion-IQ categories (dress, shirt, toptee).

---

## Changes Made

### 1. `extract_features.py`
Added `encode_images_fashionclip()` to extract image embeddings using FashionCLIP:
- Bypassed the broken `FashionCLIP.encode_images()` API
- Used `model.model.vision_model` directly to get `pooler_output` (768-dim)
- Applied `model.model.visual_projection` to project into the shared 512-dim CLIP space
- Forced CPU device (FashionCLIP hangs on Apple Silicon MPS)

### 2. `evaluate.py`
- Added FashionCLIP text encoding path using `text_model` + `text_projection` (same shared 512-dim space)
- Fixed gallery filtering: load only val-split ASINs (~3600–6200 images), not all downloaded images (~18000+). This was a critical bug — using the full download as gallery inflated the denominator and suppressed R@K by ~3×
- Added **R@50** metric to all evaluation modes and the results table
- Save paths include model tag: `eval_results_{category}_{model}.json`

### 3. Downstream fixes (FashionCLIP library patch)
- `use_auth_token` → `token` (removed in newer `transformers`)
- Hardcoded device to CPU to avoid MPS hang

---

## Data & Features

Downloaded images and extracted pre-computed embeddings for all three categories:

| Category | Gallery size | CLIP .npy | FashionCLIP .npy |
|----------|-------------|-----------|-----------------|
| dress | ~3,653 | 18 MB | 35 MB |
| shirt | ~6,182 | 30 MB | 60 MB |
| toptee | ~5,261 | 26 MB | 51 MB |

All `.npy` files committed to `features/` so teammates can run `evaluate.py` without re-extracting.

---

## Results (fusion α=0.7)

| Method | Dress R@10 | Dress R@50 | Shirt R@10 | Shirt R@50 | Toptee R@10 | Toptee R@50 |
|--------|-----------|-----------|-----------|-----------|------------|------------|
| CLIP text-only | 12.14% | 29.34% | 16.59% | 30.19% | 18.43% | 35.62% |
| CLIP image-only | 4.03% | 11.23% | 6.96% | 14.68% | 6.70% | 14.09% |
| CLIP fusion α=0.7 | 16.39% | 34.93% | 16.85% | 31.32% | 21.05% | 37.71% |
| FashionCLIP text-only | 21.44% | 40.30% | 21.84% | 37.87% | 28.17% | 47.88% |
| FashionCLIP image-only | 5.70% | 14.40% | 9.22% | 18.70% | 7.61% | 17.78% |
| FashionCLIP fusion α=0.7 | **26.97%** | **46.64%** | **27.92%** | **45.29%** | **33.00%** | **54.37%** |

FashionCLIP consistently outperforms CLIP across all categories. The gap is largest on toptee (+12 pp R@10). Fusion with α=0.7 (text-heavy) beats text-only in all cases, confirming that candidate image context adds signal even when weighted lightly.

---

## How to Reproduce

```bash
# Feature extraction (skip if using committed .npy files)
python extract_features.py --model clip --category dress
python extract_features.py --model fashionclip --category dress

# Evaluation
python evaluate.py --model clip --category dress --alphas 0.7 --save
python evaluate.py --model fashionclip --category dress --alphas 0.7 --save
```
