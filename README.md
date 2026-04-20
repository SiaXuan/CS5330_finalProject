# CS5330 Final Project — Multimodal Fashion Image Retrieval

CLIP-based image retrieval on the Fashion-IQ dataset.  
Given a reference image + a text description of changes, retrieve the target image from a gallery.

---

## Setup

```bash
pip install torch torchvision clip-by-openai faiss-cpu Pillow tqdm matplotlib numpy requests
```

> GPU (CUDA or Apple MPS) is optional but speeds up feature extraction significantly.

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

## Expected Results (val split, dress category)

| Method | Dress R@10 | Dress R@50 | Shirt R@10 | Shirt R@50 | Toptee R@10 | Toptee R@50 |
|--------|-----------|-----------|-----------|-----------|------------|------------|
| CLIP text-only | ? | ? | ? | ? | ? | ? |
| CLIP image-only | ? | ? | ? | ? | ? | ? |
| CLIP fusion α=0.7 | ? | ? | ? | ? | ? | ? |
| FashionCLIP text-only | ? | ? | ? | ? | ? | ? |
| FashionCLIP image-only | ? | ? | ? | ? | ? | ? |
| FashionCLIP fusion α=0.7 | ? | ? | ? | ? | ? | ? |

> All results reported on val split. Test split labels are not publicly available.

---

## File Overview

| File | What it does |
|------|-------------|
| `download_images.py` | Download gallery images from Amazon URLs |
| `extract_features.py` | Encode all images → `features/*.npy` |
| `build_index.py` | Build FAISS index → `features/*.faiss` |
| `evaluate.py` | Compute Recall@K for text / image / fusion |
| `fusion_retrieval.py` | Demo: fusion retrieval + comparison figures + open-set |
| `text_retrieval.py` | Baseline: text-only retrieval |
| `retrieve.py` | Minimal FAISS demo (single query) |
| `compare_viz.py` | Generate success/failure case grids for report |
| `data_summary.py` | Print dataset statistics |
| `query_analysis.py` | Analyse caption distributions |
| `visualize.py` | Misc visualisations |
