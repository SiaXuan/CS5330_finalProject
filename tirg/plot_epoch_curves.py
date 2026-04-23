"""
Evaluate every tirg_all_epochNN.pt checkpoint on the val set and plot
R@10 / R@50 curves for all three categories.

Run from the project root:
    python tirg/plot_epoch_curves.py

Output:
    results/tirg_epoch_metrics.json   — raw numbers
    results/tirg_epoch_curves.png     — the plot
"""
import os, sys, json, glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tirg.model import TIRGCombiner

CATEGORIES   = ["dress", "shirt", "toptee"]
FEATURES_DIR = "features"
DATA_DIR     = "fashion-iq"
RESULTS_DIR  = "results"
CKPT_DIR     = "tirg/checkpoints"
BEST_EPOCH   = 17
BATCH_SIZE   = 128

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def load_gallery(category):
    all_embs = np.load(f"{FEATURES_DIR}/{category}_embeddings_fashionclip.npy").astype("float32")
    with open(f"{FEATURES_DIR}/{category}_paths_fashionclip.txt") as f:
        all_paths = [l.strip() for l in f]
    with open(f"{DATA_DIR}/image_splits/split.{category}.val.json") as f:
        val_asins = set(json.load(f))

    gallery_embs, gallery_paths = [], []
    for p, emb in zip(all_paths, all_embs):
        asin = os.path.splitext(os.path.basename(p))[0].strip()
        if asin in val_asins:
            gallery_paths.append(p)
            gallery_embs.append(emb)

    gallery_embs = np.stack(gallery_embs)
    all_idx      = {os.path.splitext(os.path.basename(p))[0].strip(): i for i, p in enumerate(all_paths)}
    gal_idx      = {os.path.splitext(os.path.basename(p))[0].strip(): i for i, p in enumerate(gallery_paths)}
    return all_embs, all_idx, gallery_embs, gal_idx


def encode_text_batch(fc_model, captions):
    processor  = fc_model.preprocess
    text_model = fc_model.model.text_model
    dev        = fc_model.device
    inputs = processor(text=captions, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        pooled = text_model(
            input_ids=inputs["input_ids"].to(dev),
            attention_mask=inputs["attention_mask"].to(dev),
        ).pooler_output
        emb = fc_model.model.text_projection(pooled)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.to(device)


def evaluate_checkpoint(combiner, fc_model, category_data):
    """Returns {category: {"R@10": float, "R@50": float}} for one checkpoint."""
    results = {}
    for cat, (all_embs, all_idx, gal_embs, gal_idx, valid) in category_data.items():
        query_vecs = []
        for i in range(0, len(valid), BATCH_SIZE):
            batch     = valid[i : i + BATCH_SIZE]
            cand_embs = torch.tensor(
                np.stack([all_embs[all_idx[e["candidate"]]] for e in batch])
            ).to(device)
            t1 = encode_text_batch(fc_model, [e["captions"][0] for e in batch])
            t2 = encode_text_batch(fc_model, [e["captions"][1] for e in batch])
            text_embs = (t1 + t2) / 2
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                q = combiner(cand_embs, text_embs)
            query_vecs.append(q.cpu().numpy())
        query_vecs = np.concatenate(query_vecs, axis=0)

        scores     = query_vecs @ gal_embs.T
        tgt_idx    = np.array([gal_idx[e["target"]] for e in valid])
        tgt_scores = scores[np.arange(len(valid)), tgt_idx]
        ranks      = (scores > tgt_scores[:, None]).sum(axis=1) + 1

        results[cat] = {
            "R@10": float(np.mean(ranks <= 10)),
            "R@50": float(np.mean(ranks <= 50)),
        }
    return results


def main():
    # ── Load FashionCLIP once ────────────────────────────────────────
    print("Loading FashionCLIP...")
    from fashion_clip.fashion_clip import FashionCLIP
    fc_model = FashionCLIP("fashion-clip")

    # ── Load galleries & queries once per category ───────────────────
    print("Loading galleries...")
    category_data = {}
    for cat in CATEGORIES:
        all_embs, all_idx, gal_embs, gal_idx = load_gallery(cat)
        with open(f"{DATA_DIR}/captions/cap.{cat}.val.json") as f:
            entries = json.load(f)
        valid = [e for e in entries if e["candidate"] in all_idx and e["target"] in gal_idx]
        category_data[cat] = (all_embs, all_idx, gal_embs, gal_idx, valid)
        print(f"  {cat}: {len(gal_idx)} gallery, {len(valid)} queries")

    # ── Find & sort checkpoints ──────────────────────────────────────
    ckpts = sorted(glob.glob(f"{CKPT_DIR}/tirg_all_epoch*.pt"))
    if not ckpts:
        print(f"No checkpoints found in {CKPT_DIR}/")
        return
    print(f"\nFound {len(ckpts)} checkpoints. Starting evaluation...\n")

    combiner = TIRGCombiner(feature_dim=512).to(device)
    combiner.eval()

    epoch_results = {}   # epoch_int -> {cat: {R@10, R@50}}

    for ckpt_path in ckpts:
        epoch = int(os.path.basename(ckpt_path).replace("tirg_all_epoch", "").replace(".pt", ""))
        combiner.load_state_dict(torch.load(ckpt_path, map_location=device))
        res = evaluate_checkpoint(combiner, fc_model, category_data)
        epoch_results[epoch] = res
        r10_avg = sum(res[c]["R@10"] for c in CATEGORIES) / len(CATEGORIES)
        print(f"  epoch {epoch:02d}  avg R@10 = {r10_avg:.2%}  "
              + "  ".join(f"{c} {res[c]['R@10']:.2%}" for c in CATEGORIES))

    # ── Save raw numbers ─────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_json = f"{RESULTS_DIR}/tirg_epoch_metrics.json"
    with open(out_json, "w") as f:
        json.dump({str(k): v for k, v in epoch_results.items()}, f, indent=2)
    print(f"\nSaved metrics to {out_json}")

    # ── Plot ─────────────────────────────────────────────────────────
    epochs = sorted(epoch_results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("TIRG — Validation Recall vs. Epoch (FashionCLIP backbone, joint training)",
                 fontsize=13, fontweight="bold")

    colors = {"dress": "#4C72B0", "shirt": "#DD8452", "toptee": "#55A868"}
    metric_labels = {"R@10": "Recall@10", "R@50": "Recall@50"}

    for ax, metric in zip(axes, ["R@10", "R@50"]):
        for cat in CATEGORIES:
            ys = [epoch_results[e][cat][metric] * 100 for e in epochs]
            ax.plot(epochs, ys, marker="o", markersize=4, linewidth=1.8,
                    color=colors[cat], label=cat.capitalize())

        # Mark best epoch
        best_y_max = max(epoch_results[BEST_EPOCH][cat][metric] for cat in CATEGORIES)
        ax.axvline(BEST_EPOCH, color="#CC0000", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(BEST_EPOCH + 0.3, ax.get_ylim()[0] + 1,
                f"epoch {BEST_EPOCH}\n(selected)", color="#CC0000", fontsize=8)

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(f"{metric_labels[metric]} (%)", fontsize=11)
        ax.set_title(metric_labels[metric], fontsize=12)
        ax.legend(fontsize=10)
        ax.set_xticks(epochs[::2])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = f"{RESULTS_DIR}/tirg_epoch_curves.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_png}")


if __name__ == "__main__":
    main()
