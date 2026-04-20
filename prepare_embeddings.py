#!/usr/bin/env python3
"""Encode Fashion-IQ captions to embeddings — the ONE model-specific file.

When the teammate swaps CLIP → FashionCLIP, this is the file that changes.
Everything downstream (rerank/*) consumes only .npy arrays.

Each triple has two captions; we average their normalized embeddings, which
matches what fusion_retrieval.py / evaluate.py already do for queries.

Output:
  features/{category}_{split}_text.npy   (N, D)  float32, L2-normalized

Currently supported backends:
  --backend clip         OpenAI CLIP ViT-B/32  (default; matches the gallery)
  --backend fashionclip  patrickjohncyh/fashion-clip  (requires transformers)

If you add a third backend, keep the output contract identical: shape
(N, D), float32, unit-normalized, rows aligned 1:1 with the triples in
cap.{category}.{split}.json.

Usage:
  python prepare_embeddings.py --category dress --split train
  python prepare_embeddings.py --category dress --split val
  python prepare_embeddings.py --category dress --split train --backend fashionclip
"""
from __future__ import annotations

import os
import json
import argparse

import numpy as np
import torch
from tqdm import tqdm

FEATURES_DIR = "features"
DATA_DIR     = "fashion-iq"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# ---------------------------------------------------------------------------
# Backends — each must expose encode_text(list[str]) -> (N, D) float32 L2-norm
# ---------------------------------------------------------------------------

class CLIPBackend:
    name = "clip"

    def __init__(self):
        import clip                                       # noqa: import inside
        self.clip = clip
        self.model, _ = clip.load("ViT-B/32", device=device)
        self.model.eval()

    def encode_text(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in tqdm(range(0, len(texts), batch_size), desc="  encode_text"):
            batch = texts[i : i + batch_size]
            with torch.no_grad():
                toks = self.clip.tokenize(batch, truncate=True).to(device)
                emb  = self.model.encode_text(toks)
                emb  = emb / emb.norm(dim=-1, keepdim=True)
            out.append(emb.cpu().numpy().astype("float32"))
        return np.concatenate(out, axis=0)


class FashionCLIPBackend:
    """Teammate's path once they finish migrating.

    Uses HuggingFace transformers + patrickjohncyh/fashion-clip. Same contract
    as CLIPBackend so nothing downstream changes.
    """
    name = "fashionclip"

    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor  # noqa: import inside
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
        self.model.eval()

    def encode_text(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in tqdm(range(0, len(texts), batch_size), desc="  encode_text"):
            batch = texts[i : i + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt",
                                    padding=True, truncation=True).to(device)
            with torch.no_grad():
                emb = self.model.get_text_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            out.append(emb.cpu().numpy().astype("float32"))
        return np.concatenate(out, axis=0)


BACKENDS = {"clip": CLIPBackend, "fashionclip": FashionCLIPBackend}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Encode Fashion-IQ captions")
    ap.add_argument("--category", default="dress")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--backend", choices=list(BACKENDS), default="clip")
    args = ap.parse_args()

    cap_file = os.path.join(DATA_DIR, "captions",
                            f"cap.{args.category}.{args.split}.json")
    with open(cap_file) as f:
        triples = json.load(f)
    print(f"[{args.category}/{args.split}] {len(triples)} triples")

    # Collect all captions; average per-triple after encoding.
    cap1 = [e["captions"][0] for e in triples]
    cap2 = [e["captions"][1] for e in triples]

    print(f"Loading backend: {args.backend} on {device}...")
    backend = BACKENDS[args.backend]()

    emb1 = backend.encode_text(cap1)        # (N, D)
    emb2 = backend.encode_text(cap2)        # (N, D)
    text_emb = (emb1 + emb2) / 2
    text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
    text_emb = text_emb.astype("float32")

    out = os.path.join(FEATURES_DIR,
                       f"{args.category}_{args.split}_text.npy")
    os.makedirs(FEATURES_DIR, exist_ok=True)
    np.save(out, text_emb)
    print(f"Saved {text_emb.shape} → {out}")


if __name__ == "__main__":
    main()
