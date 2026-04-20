# CS5330 Final Project — Multimodal Fashion Image Retrieval

CLIP-based image retrieval on the Fashion-IQ dataset.  
Given a reference image + a text description of changes, retrieve the target image from a gallery.

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Activate the venv (`source venv/bin/activate`) every time you open a new terminal.  
> GPU (CUDA or Apple MPS) is detected automatically — no config needed.

---

## Data

You need two things:

**1. Fashion-IQ dataset** — download from the [official repo](https://github.com/XiaoxiaoGuo/fashion-iq) and place it as:

```
fashion-iq/
├── captions/
│   ├── cap.dress.train.json
│   ├── cap.dress.val.json
│   └── cap.dress.test.json
└── image_splits/
    ├── split.dress.train.json
    ├── split.dress.val.json
    └── split.dress.test.json
```

Same structure for `shirt` and `toptee` categories.

**2. Fashion-IQ metadata** — URL list for downloading images:

```
fashion-iq-metadata/
└── image_url/
    ├── asin2url.dress.txt
    ├── asin2url.shirt.txt
    └── asin2url.toptee.txt
```

**Download images** (run once, takes a while):

```bash
python download_images.py
```

Images are saved to `images/dress/`, `images/shirt/`, `images/toptee/`.  
Some images may fail to download (server issues) — this is expected, those queries are skipped at eval time.

---

## Pipeline

Run these steps **in order**. Steps 1–2 are offline (run once); steps 3+ are online.

### Step 1 — Extract features

Encode all gallery images with CLIP → save as `.npy`.

```bash
python extract_features.py --model clip         # default
python extract_features.py --model fashionclip  # requires: pip install fashion-clip
```

Output: `features/dress_embeddings_clip.npy` or `features/dress_embeddings_fashionclip.npy`

### Step 2 — Build FAISS index

Wrap the embeddings in a FAISS index for fast retrieval.

```bash
python build_index.py
```

Output: `features/dress_index.faiss`

### Step 3 — Evaluate (Recall@K)

Run the full evaluation on the val set. Tests multiple fusion weights at once.

```bash
python evaluate.py --alphas 0.3 0.5 0.7                        # CLIP (default)
python evaluate.py --model fashionclip --alphas 0.3 0.5 0.7    # FashionCLIP
```

Prints a table like:

```
  Mode                    R@1      R@10     R@50       N
  text-only             6.84%   28.91%   52.3%     1823
  image-only            5.21%   24.33%   47.1%     1823
  fusion α=0.3          7.53%   32.45%   56.2%     1823
  fusion α=0.5          8.12%   34.87%   58.9%     1823   ← best
  fusion α=0.7          7.91%   33.20%   57.4%     1823
```

Pick the alpha with the highest R@10, then use it below.

### Step 4 — Generate comparison figures

```bash
# Side-by-side retrieval results: text-only / image-only / fusion
python fusion_retrieval.py --compare --alpha 0.5 --n 5

# Success / failure / baseline-wins case grids (for report)
python compare_viz.py --alpha 0.5 --scan 300
```

Output images saved to `results/`.

---

## Open-Set Demo

Retrieve from the gallery using **any image** (not from the dataset):

```bash
python fusion_retrieval.py \
  --query-image path/to/my_photo.jpg \
  --query-text "shorter and more floral" \
  --alpha 0.5
```

The query image does not need to be in the gallery.

---

## Text-Only Retrieval

```bash
python text_retrieval.py
```

Baseline: text query only, no reference image.

---

## Results (val split)

Run `python print_results.py` to reprint this table from the saved JSON files.

| Method | Dress R@10 | Dress R@50 | Shirt R@10 | Shirt R@50 | Toptee R@10 | Toptee R@50 |
|--------|-----------|-----------|-----------|-----------|------------|------------|
| CLIP text-only | 12.14% | 29.34% | 16.59% | 30.19% | 18.43% | 35.62% |
| CLIP image-only | 4.03% | 11.23% | 6.96% | 14.68% | 6.70% | 14.09% |
| CLIP fusion α=0.7 | 16.39% | 34.93% | 16.85% | 31.32% | 21.05% | 37.71% |
| FashionCLIP text-only | 21.44% | 40.30% | 21.84% | 37.87% | 28.17% | 47.88% |
| FashionCLIP image-only | 5.70% | 14.40% | 9.22% | 18.70% | 7.61% | 17.78% |
| FashionCLIP fusion α=0.7 | 26.97% | 46.64% | 27.92% | 45.29% | 33.00% | 54.37% |
| **Combiner (FashionCLIP)** | **33.85%** | **58.57%** | **30.29%** | **53.53%** | **36.80%** | **62.88%** |

> All results on val split. Test split labels are not publicly available.  
> α=0.3 and α=0.5 both perform *worse* than text-only; see `print_results.py` for the full α comparison and explanation.

---

## Combiner (learned early fusion)

The fusion baseline scores text and image separately and combines the scores.
The Combiner instead learns a small MLP that maps `(candidate_image_emb, text_emb) → query_emb`,
then does a single retrieval pass. Trained end-to-end on Fashion-IQ training triplets with
InfoNCE loss; FashionCLIP backbone is frozen.

```bash
# Train on all 3 categories (saves best checkpoint to results/combiner_fashionclip.pt)
python train_combiner.py

# Evaluate a saved checkpoint without re-training
python train_combiner.py --eval-only

# Custom options
python train_combiner.py --epochs 20 --batch-size 256 --lr 2e-4 --category dress shirt
```

See `train_combiner.py` module docstring for architecture details.

---

## File Overview

| File | What it does |
|------|-------------|
| `download_images.py` | Download gallery images from Amazon URLs |
| `extract_features.py` | Encode all images → `features/*.npy` |
| `build_index.py` | Build FAISS index → `features/*.faiss` |
| `evaluate.py` | Compute Recall@K for text / image / fusion |
| `print_results.py` | Print full results table + α=0.7 rationale (run directly) |
| `train_combiner.py` | Train & evaluate learned MLP combiner (FashionCLIP backbone) |
| `fusion_retrieval.py` | Demo: fusion retrieval + comparison figures + open-set |
| `text_retrieval.py` | Baseline: text-only retrieval |
| `retrieve.py` | Minimal FAISS demo (single query) |
| `compare_viz.py` | Generate success/failure case grids for report |
| `data_summary.py` | Print dataset statistics |
| `query_analysis.py` | Analyse caption distributions |
| `visualize.py` | Misc visualisations |
