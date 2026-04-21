#!/usr/bin/env python3
"""Encode Fashion-IQ captions to embeddings — the ONE model-specific file.

The rerank/ package consumes only .npy arrays and is model-agnostic; this
script is the single bridge that the chosen encoder (CLIP or FashionCLIP)
must cross. Add a new backend here and nothing else changes downstream.

Both captions in a triple are encoded and averaged (matching the query
construction in evaluate.py / fusion_retrieval.py).

Output:
  features/{category}_{split}_text_{model}.npy   (N, D)  float32, L2-normalized

The filename is model-tagged so CLIP and FashionCLIP text embeddings can
coexist and be swapped by changing one flag.

Usage:
  python prepare_embeddings.py --category dress --split train --model clip
  python prepare_embeddings.py --category dress --split val   --model fashionclip
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
# Backends — each must expose encode_text(list[str]) → (N, D) float32 L2-norm
# ---------------------------------------------------------------------------

class CLIPBackend:
    """OpenAI CLIP ViT-B/32 — matches the CLIP gallery embeddings."""
    name = "clip"

    def __init__(self):
        import clip
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
    """FashionCLIP (patrickjohncyh/fashion-clip).

    Uses the exact same encoding path as evaluate.py on feature/fashionclip:
    `text_model.pooler_output` → `text_projection` → 512-dim shared CLIP space.

    Note: we load the HuggingFace CLIPModel directly rather than going through
    the `fashion_clip` package, because `fashion_clip 0.2.2` calls
    `from_pretrained(..., use_auth_token=...)` which was removed in
    transformers 5.x. The weights and architecture are identical — `fashion_clip`
    is just a thin wrapper around the same HF checkpoint.
    """
    name = "fashionclip"

    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        self.model     = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
        self.model.eval()

    def encode_text(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        text_model = self.model.text_model
        proj       = self.model.text_projection

        out = []
        for i in tqdm(range(0, len(texts), batch_size), desc="  encode_text"):
            batch = texts[i : i + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt",
                                    padding=True, truncation=True)
            input_ids      = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            with torch.no_grad():
                pooled = text_model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).pooler_output
                emb = proj(pooled)
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
    ap.add_argument("--model", choices=list(BACKENDS), default="clip")
    args = ap.parse_args()

    cap_file = os.path.join(DATA_DIR, "captions",
                            f"cap.{args.category}.{args.split}.json")
    with open(cap_file) as f:
        triples = json.load(f)
    print(f"[{args.category}/{args.split}] {len(triples)} triples")

    cap1 = [e["captions"][0] for e in triples]
    cap2 = [e["captions"][1] for e in triples]

    print(f"Loading backend: {args.model} on {device}...")
    backend = BACKENDS[args.model]()

    emb1 = backend.encode_text(cap1)
    emb2 = backend.encode_text(cap2)
    text_emb = (emb1 + emb2) / 2
    text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
    text_emb = text_emb.astype("float32")

    os.makedirs(FEATURES_DIR, exist_ok=True)
    out = os.path.join(
        FEATURES_DIR,
        f"{args.category}_{args.split}_text_{args.model}.npy",
    )
    np.save(out, text_emb)
    print(f"Saved {text_emb.shape} → {out}")


if __name__ == "__main__":
    main()
