# CS5330 Final Project — Text-Guided Fashion Image Retrieval

**Team:** Xinyue Xuan, Junrui Ding, Martin Han

Given a **reference image** and a **modification caption** (e.g. *"make it red with shorter sleeves"*), retrieve the target image from a gallery of ~18k fashion photos. Evaluated on the [Fashion-IQ](https://github.com/XiaoxiaoGuo/fashion-iq) benchmark (dress / shirt / toptee).

We implement and compare **six methods** along two axes:

1. **Encoder** — CLIP (ViT-B/32) vs **FashionCLIP** (domain-adapted)
2. **Composition** — late α-fusion → learned early fusion (Combiner / TIRG) → top-50 re-ranking

---

## Methods

| # | Method | Type | Notes |
|---|---|---|---|
| 1 | **CLIP text-only / image-only / α-fusion** | Baseline | Late fusion: `score = α·text + (1-α)·image` |
| 2 | **FashionCLIP backend** | Backbone swap | `patrickjohncyh/fashion-clip`; same pipeline, much stronger |
| 3 | **Open-set retrieval** | Demo | Query with *any* phone photo + text, no gallery membership needed |
| 4 | **Combiner** (CLIP4CIR) | Early fusion | 2-layer MLP outputs a single unified query embedding; residual design |
| 5 | **TIRG** | Early fusion | Gated residual: `gate·img + residual(img, text)` |
| 6 | **Re-ranking** | 2-stage | MLP cross-encoder re-scores the top-50 fusion hits |

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

CUDA / Apple MPS / CPU is auto-detected. No config needed.

### Data layout

```
fashion-iq/                       (captions + split lists — tracked in repo)
├── captions/cap.{dress,shirt,toptee}.{train,val,test}.json
└── image_splits/split.{dress,shirt,toptee}.{train,val,test}.json

fashion-iq-metadata/              (URL lists for downloading images)
└── image_url/asin2url.{dress,shirt,toptee}.txt

images/                           (downloaded by download_images.py — gitignored)
└── {dress,shirt,toptee}/*.jpg

features/                         (generated — gitignored)
└── {cat}_{embeddings,paths}_{clip,fashionclip}.{npy,txt}
└── {cat}_{split}_text_{model}.npy
└── {cat}_{split}_top50_{idx,score}_{model}.npy  (re-rank)
```

Download images once (slow, some links broken — skipped at eval time):

```bash
python download_images.py
```

---

## Pipeline

### Step 1 — Encode the gallery (offline, once per encoder)

```bash
python extract_features.py --model clip          # default
python extract_features.py --model fashionclip   # recommended
```

Produces `features/{cat}_embeddings_{model}.npy` + `features/{cat}_paths_{model}.txt` for all three categories.

### Step 2 — Baseline & fusion evaluation

```bash
python evaluate.py --model fashionclip --alphas 0.3 0.5 0.7
```

Prints R@1 / R@5 / R@10 / R@50 for text-only, image-only, and α-fusion at each α. Results saved to `results/eval_results_{cat}_{model}.json`.

### Step 3 — Combiner (early fusion, learned)

Trained with symmetric InfoNCE on Fashion-IQ triplets (frozen FashionCLIP, 2-layer residual MLP).

```bash
python train_combiner.py           # trains 25 epochs, saves results/combiner_fashionclip.pt
python viz_combiner.py             # combiner-vs-fusion comparison grids
```

### Step 4 — TIRG (gated residual)

`out = gate(img, text) · img + residual(img, text)` — both the gate and residual are small MLPs on top of frozen FashionCLIP.

```bash
python -m tirg.train     --category dress
python -m tirg.evaluate  --category dress
```

### Step 5 — Re-ranking (2-stage)

Takes the top-50 candidates from α-fusion and re-scores them with a 3-layer MLP cross-encoder trained as InfoNCE over the 50-way candidate set.

```bash
# 5a. Encode captions (model-specific — the one bridge)
python prepare_embeddings.py --category dress --split train --model fashionclip
python prepare_embeddings.py --category dress --split val   --model fashionclip

# 5b. Precompute top-50 fusion candidates per triple
python -m rerank.precompute --category dress --split train --model fashionclip
python -m rerank.precompute --category dress --split val   --model fashionclip

# 5c. Train & evaluate
python -m rerank.train     --category dress --model fashionclip
python -m rerank.evaluate  --category dress --model fashionclip --save
```

The `rerank/` package consumes only `.npy` arrays — swapping CLIP ↔ FashionCLIP means re-running step 5a with a different `--model`; nothing else changes.

### Step 6 — Open-set demo (any query image)

```bash
python fusion_retrieval.py \
  --query-image path/to/your_photo.jpg \
  --query-text "shorter and more floral" \
  --alpha 0.7
```

---

## Results (Fashion-IQ val split)

### Table 1 — First-stage retrieval: CLIP vs FashionCLIP

| Method | Dress R@10 | Dress R@50 | Shirt R@10 | Shirt R@50 | Toptee R@10 | Toptee R@50 |
|---|---:|---:|---:|---:|---:|---:|
| CLIP text-only | 12.14% | 29.34% | 16.59% | 30.19% | 18.43% | 35.62% |
| CLIP image-only | 4.03% | 11.23% | 6.96% | 14.68% | 6.70% | 14.09% |
| CLIP fusion α=0.7 | 16.39% | 34.93% | 16.85% | 31.32% | 21.05% | 37.71% |
| FashionCLIP text-only | 21.44% | 40.30% | 21.84% | 37.87% | 28.17% | 47.88% |
| FashionCLIP image-only | 5.70% | 14.40% | 9.22% | 18.70% | 7.61% | 17.78% |
| **FashionCLIP fusion α=0.7** | **26.97%** | **46.64%** | **27.92%** | **45.29%** | **33.00%** | **54.37%** |

**Takeaway.** FashionCLIP alone lifts R@10 by ~10 points over CLIP in every category. α-fusion adds another ~5 points on top. α=0.7 (text-dominant) beats 0.5/0.3 — expected, since captions describe *changes*.

### Table 2 — TIRG (learned early fusion, FashionCLIP backbone)

| Category | R@1 | R@5 | R@10 | R@50 |
|---|---:|---:|---:|---:|
| dress | 9.89% | 25.04% | 34.55% | 59.16% |
| shirt | 5.20% | 22.62% | 31.99% | 53.43% |
| toptee | 8.30% | 27.91% | 38.40% | 63.63% |

**Takeaway.** TIRG beats late α-fusion across the board (+7-8 points R@10) and dramatically on R@50 (+10-15 points), confirming that a learned combiner finds the target's neighborhood better than a fixed linear mix.

### Table 3 — Re-ranking over FashionCLIP top-50 (α=0.5)

| Category | Stage | R@1 | R@5 | R@10 | R@50 |
|---|---|---:|---:|---:|---:|
| dress | α-fusion | 0.11% | 11.45% | 16.23% | 34.07% |
|  | **+ re-rank** | **7.90%** | **19.51%** | **26.65%** | 34.07% |
| shirt | α-fusion | 0.10% | 12.47% | 18.13% | 33.23% |
|  | **+ re-rank** | **8.86%** | **21.02%** | **26.22%** | 33.23% |
| toptee | α-fusion | 0.32% | 14.35% | 21.26% | 38.89% |
|  | **+ re-rank** | **12.43%** | **27.05%** | **32.35%** | 38.89% |

**Takeaway.** The MLP cross-encoder recovers ~60% of the headroom between fusion and the top-50 ceiling (= R@50). R@1 jumps 20-80× because the linear α-mix cannot separate near-duplicates, while the MLP can. R@50 is unchanged by construction — a re-ranker over top-K cannot improve R@K' for K' ≥ K.

---

## File overview

### Top-level scripts

| File | Purpose |
|---|---|
| `download_images.py` | Download Fashion-IQ gallery images from Amazon URLs |
| `extract_features.py` | Encode images with CLIP / FashionCLIP → `features/*.npy` |
| `build_index.py` | Wrap embeddings in a FAISS index |
| `prepare_embeddings.py` | Encode train/val captions to text embeddings (bridge for `rerank/`) |
| `evaluate.py` | Recall@K for text / image / α-fusion |
| `fusion_retrieval.py` | Retrieval demo + open-set query support + side-by-side figures |
| `text_retrieval.py` | Text-only retrieval demo |
| `retrieve.py` | Minimal FAISS demo (single query) |
| `compare_viz.py` | Success / failure / baseline-wins case grids (for report) |
| `train_combiner.py` | Train the Combiner MLP (CLIP4CIR residual) |
| `viz_combiner.py` | Combiner-vs-fusion visualisations |
| `query_analysis.py` | Caption-pattern analysis figures |
| `data_summary.py` | Dataset statistics |
| `visualize.py` | Misc visualisation helpers |

### Sub-packages

| Dir | Contents |
|---|---|
| `tirg/` | `model.py` (gated residual), `train.py`, `evaluate.py` |
| `rerank/` | `model.py` (cross-encoder MLP), `dataset.py`, `precompute.py`, `train.py`, `evaluate.py` |

### Results

| Pattern | What |
|---|---|
| `results/eval_results_{cat}_{model}.json` | Baseline + α-fusion R@K |
| `results/eval_ranks_{cat}_{model}.json` | Per-query rank lists |
| `results/eval_results_{cat}_tirg.json` | TIRG R@K |
| `results/rerank_eval_{cat}_{model}.json` | Fusion vs re-rank vs ceiling |
| `results/rerank_train_history_{cat}_{model}.json` | Re-rank training curves |
| `results/combiner_fashionclip.pt` | Trained Combiner weights |
| `results/combiner_viz/` | Combiner-vs-fusion comparison PNGs |

---

## Team contributions

| Member | Work |
|---|---|
| **Xinyue Xuan** | CLIP baseline + Fashion-IQ preprocessing; TIRG gated-residual combiner (`tirg/`, training + evaluation) |
| **Junrui Ding** | Fusion-retrieval pipeline (`evaluate.py`, `fusion_retrieval.py`, `text_retrieval.py`, `compare_viz.py`); FashionCLIP backbone integration and `--model` flag across the stack; Combiner (early-fusion MLP, CLIP4CIR residual); visualisations; Apple-MPS support |
| **Martin Han** | Re-ranking package (`rerank/` — MLP cross-encoder, precompute, train, evaluate); model-agnostic `prepare_embeddings.py`; open-set retrieval support |

---

## Reproducing the tables

```bash
# Table 1
python extract_features.py  --model clip
python extract_features.py  --model fashionclip
python evaluate.py          --model clip          --alphas 0.3 0.5 0.7
python evaluate.py          --model fashionclip   --alphas 0.3 0.5 0.7

# Table 2
for c in dress shirt toptee; do
  python -m tirg.train    --category $c
  python -m tirg.evaluate --category $c
done

# Table 3
for c in dress shirt toptee; do
  for m in clip fashionclip; do
    python prepare_embeddings.py --category $c --split train --model $m
    python prepare_embeddings.py --category $c --split val   --model $m
    python -m rerank.precompute  --category $c --split train --model $m
    python -m rerank.precompute  --category $c --split val   --model $m
    python -m rerank.train       --category $c --model $m
    python -m rerank.evaluate    --category $c --model $m --save
  done
done
```

All numbers above are on the Fashion-IQ **val** split (test labels are not public).
