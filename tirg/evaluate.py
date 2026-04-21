"""
Evaluate a trained TIRG combiner on the Fashion-IQ validation set.
Outputs Recall@1/5/10/50 in the same format as evaluate.py for easy comparison.

Usage:
    python tirg/evaluate.py --category dress --checkpoint tirg/checkpoints/tirg_dress_epoch10.pt
    python tirg/evaluate.py --category dress --checkpoint tirg/checkpoints/tirg_dress_epoch10.pt --save
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tirg.model import TIRGCombiner

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

FEATURES_DIR = "features"
DATA_DIR     = "fashion-iq"
RESULTS_DIR  = "results"


def load_gallery(category: str, model_name: str = "fashionclip"):
    all_embs = np.load(
        os.path.join(FEATURES_DIR, f"{category}_embeddings_{model_name}.npy")
    ).astype("float32")
    with open(os.path.join(FEATURES_DIR, f"{category}_paths_{model_name}.txt")) as f:
        all_paths = [line.strip() for line in f]

    with open(os.path.join(DATA_DIR, "image_splits", f"split.{category}.val.json")) as f:
        val_asins = set(json.load(f))

    gallery_embs, gallery_paths = [], []
    for p, emb in zip(all_paths, all_embs):
        asin = os.path.splitext(os.path.basename(p))[0].strip()
        if asin in val_asins:
            gallery_paths.append(p)
            gallery_embs.append(emb)

    gallery_embs = np.stack(gallery_embs)
    all_asin_to_idx = {
        os.path.splitext(os.path.basename(p))[0].strip(): i
        for i, p in enumerate(all_paths)
    }
    gallery_asin_to_idx = {
        os.path.splitext(os.path.basename(p))[0].strip(): i
        for i, p in enumerate(gallery_paths)
    }
    return all_embs, all_asin_to_idx, gallery_embs, gallery_asin_to_idx


def encode_text_batch(fc_model, captions: list) -> torch.Tensor:
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


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category",   default="dress")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save",       action="store_true")
    args = parser.parse_args()

    print("Loading FashionCLIP...")
    from fashion_clip.fashion_clip import FashionCLIP
    fc_model = FashionCLIP("fashion-clip")

    print("Loading gallery...")
    all_embs, all_asin_to_idx, gallery_embs, gallery_asin_to_idx = load_gallery(args.category)
    print(f"Gallery: {len(gallery_asin_to_idx)} images")

    print(f"Loading TIRG combiner from {args.checkpoint}...")
    combiner = TIRGCombiner(feature_dim=512).to(device)
    combiner.load_state_dict(torch.load(args.checkpoint, map_location=device))
    combiner.eval()

    with open(os.path.join(DATA_DIR, "captions", f"cap.{args.category}.val.json")) as f:
        entries = json.load(f)
    valid = [
        e for e in entries
        if e["candidate"] in all_asin_to_idx and e["target"] in gallery_asin_to_idx
    ]
    print(f"Valid queries: {len(valid)}")

    # Compute TIRG query vectors
    batch_size  = 64
    query_vecs  = []
    for i in range(0, len(valid), batch_size):
        batch     = valid[i : i + batch_size]
        cand_embs = torch.tensor(
            np.stack([all_embs[all_asin_to_idx[e["candidate"]]] for e in batch])
        ).to(device)
        t1 = encode_text_batch(fc_model, [e["captions"][0] for e in batch])
        t2 = encode_text_batch(fc_model, [e["captions"][1] for e in batch])
        text_embs = (t1 + t2) / 2
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            q = combiner(cand_embs, text_embs)
        query_vecs.append(q.cpu().numpy())
    query_vecs = np.concatenate(query_vecs, axis=0)   # (N, 512)

    # Retrieve and rank
    scores     = query_vecs @ gallery_embs.T           # (N, G)
    target_idx = np.array([gallery_asin_to_idx[e["target"]] for e in valid])
    tgt_scores = scores[np.arange(len(valid)), target_idx]
    ranks      = (scores > tgt_scores[:, None]).sum(axis=1) + 1

    # Print results
    print(f"\nTIRG — {args.category} | {os.path.basename(args.checkpoint)}")
    print("=" * 50)
    result = {}
    for k in [1, 5, 10, 50]:
        r = recall_at_k(ranks, k)
        result[f"R@{k}"] = r
        print(f"  R@{k:<3}: {r:.2%}")
    print("=" * 50)

    if args.save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        import json as _json
        tag      = f"{args.category}_tirg"
        out_path = os.path.join(RESULTS_DIR, f"eval_results_{tag}.json")
        with open(out_path, "w") as f:
            _json.dump([{"mode": "tirg", "n": len(valid), **result}], f, indent=2)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
