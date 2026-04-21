"""
Train the TIRG combiner on Fashion-IQ triplets with a frozen FashionCLIP backbone.

Usage:
    # Joint training on all 3 categories (default)
    python tirg/train.py --epochs 15

    # Single category
    python tirg/train.py --categories dress --epochs 15

    # Resume from checkpoint
    python tirg/train.py --epochs 5 --resume tirg/checkpoints/tirg_all_epoch10.pt --start_epoch 11
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tirg.model import TIRGCombiner

CATEGORIES = ["dress", "shirt", "toptee"]

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class FashionIQDataset(Dataset):
    def __init__(self, category: str, split: str, img_embs: np.ndarray, asin_to_idx: dict):
        with open(f"fashion-iq/captions/cap.{category}.{split}.json") as f:
            entries = json.load(f)
        self.data = [
            e for e in entries
            if e["candidate"] in asin_to_idx and e["target"] in asin_to_idx
        ]
        self.img_embs   = img_embs
        self.asin_to_idx = asin_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e    = self.data[idx]
        cand = torch.tensor(self.img_embs[self.asin_to_idx[e["candidate"]]])
        tgt  = torch.tensor(self.img_embs[self.asin_to_idx[e["target"]]])
        return cand, e["captions"][0], e["captions"][1], tgt


def load_img_embeddings(category: str, model_name: str = "fashionclip"):
    embs = np.load(f"features/{category}_embeddings_{model_name}.npy").astype("float32")
    with open(f"features/{category}_paths_{model_name}.txt") as f:
        paths = [line.strip() for line in f]
    asin_to_idx = {
        os.path.splitext(os.path.basename(p))[0].strip(): i
        for i, p in enumerate(paths)
    }
    return embs, asin_to_idx


def encode_text(fc_model, captions: list) -> torch.Tensor:
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


def symmetric_infonce(query_embs: torch.Tensor, target_embs: torch.Tensor,
                      temperature: float = 0.07) -> torch.Tensor:
    logits = query_embs @ target_embs.T / temperature   # (B, B)
    labels = torch.arange(len(query_embs), device=query_embs.device)
    loss_qt = F.cross_entropy(logits,   labels)   # query → target
    loss_tq = F.cross_entropy(logits.T, labels)   # target → query
    return (loss_qt + loss_tq) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories",  nargs="+", default=CATEGORIES)
    parser.add_argument("--epochs",      type=int,   default=15)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--resume",      default=None)
    parser.add_argument("--start_epoch", type=int,   default=1)
    args = parser.parse_args()

    tag = "_".join(args.categories) if len(args.categories) < 3 else "all"

    print("Loading FashionCLIP (frozen backbone)...")
    from fashion_clip.fashion_clip import FashionCLIP
    fc_model = FashionCLIP("fashion-clip")

    print(f"Loading image embeddings for: {args.categories}")
    datasets = []
    for cat in args.categories:
        img_embs, asin_to_idx = load_img_embeddings(cat)
        datasets.append(FashionIQDataset(cat, "train", img_embs, asin_to_idx))

    dataset = ConcatDataset(datasets)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"Total training samples: {len(dataset)}  |  device: {device}")

    combiner  = TIRGCombiner(feature_dim=512).to(device)
    optimizer = torch.optim.Adam(combiner.parameters(), lr=args.lr)

    if args.resume:
        combiner.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")

    os.makedirs("tirg/checkpoints", exist_ok=True)
    end_epoch = args.start_epoch - 1 + args.epochs

    for epoch in range(args.start_epoch - 1, end_epoch):
        combiner.train()
        total_loss = 0.0

        for cand_embs, cap1_list, cap2_list, tgt_embs in tqdm(loader, desc=f"Epoch {epoch+1}/{end_epoch}"):
            cand_embs = cand_embs.to(device)
            tgt_embs  = tgt_embs.to(device)

            t1 = encode_text(fc_model, list(cap1_list))
            t2 = encode_text(fc_model, list(cap2_list))
            text_embs = (t1 + t2) / 2
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

            query_embs = combiner(cand_embs, text_embs)
            loss = symmetric_infonce(query_embs, tgt_embs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"  loss = {avg_loss:.4f}")

        ckpt = f"tirg/checkpoints/tirg_{tag}_epoch{epoch+1:02d}.pt"
        torch.save(combiner.state_dict(), ckpt)

    print(f"\nDone. Checkpoint saved to {ckpt}")


if __name__ == "__main__":
    main()
