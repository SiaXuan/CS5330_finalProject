"""Dataset for training the re-ranker.

Consumes ONLY numpy embedding arrays + precomputed candidate indices.
Zero model dependencies — swap CLIP → FashionCLIP by regenerating the
.npy inputs, not by touching this file.

Inputs required (produced by prepare_embeddings.py + rerank/precompute.py):
  - gallery_emb : (G, D)     all gallery images, normalized
  - text_emb    : (N, D)     averaged caption embedding per triple
  - topk_idx    : (N, K)     top-K gallery indices from first-stage fusion
  - topk_score  : (N, K)     first-stage fusion scores for those K
  - cand_idx    : (N,)       gallery index of the query image
  - tgt_idx     : (N,)       gallery index of the target (may be -1 if skipped)

Each __getitem__ returns one training example with:
  - cand_emb    (D,)
  - text_emb    (D,)
  - topk_emb    (K, D)
  - topk_fusion (K,)
  - tgt_pos     int        position of target within topk_idx, or -1 if absent
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FashionIQReRankDataset(Dataset):
    def __init__(
        self,
        gallery_emb: np.ndarray,     # (G, D)
        text_emb: np.ndarray,         # (N, D)
        topk_idx: np.ndarray,         # (N, K) int64
        topk_score: np.ndarray,       # (N, K) float32
        cand_idx: np.ndarray,         # (N,) int64
        tgt_idx: np.ndarray,          # (N,) int64  (-1 if target missing)
        only_with_target: bool = True,
    ):
        assert gallery_emb.ndim == 2
        assert text_emb.shape[1] == gallery_emb.shape[1]
        assert len(text_emb) == len(topk_idx) == len(topk_score) \
               == len(cand_idx) == len(tgt_idx)

        self.gallery_emb = gallery_emb
        self.text_emb    = text_emb
        self.topk_idx    = topk_idx
        self.topk_score  = topk_score
        self.cand_idx    = cand_idx
        self.tgt_idx     = tgt_idx

        if only_with_target:
            # Drop entries where the target isn't in top-K; they give no
            # learning signal for a re-ranker that only sees the top-K.
            # (We could also force-inject the target, but that distorts
            # the distribution — keep the ones where first-stage was
            # already "close enough" to help.)
            keep = []
            for i in range(len(tgt_idx)):
                if tgt_idx[i] < 0:
                    continue
                if tgt_idx[i] in topk_idx[i]:
                    keep.append(i)
            self.indices = np.array(keep, dtype=np.int64)
        else:
            self.indices = np.arange(len(text_emb), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        idx = self.indices[i]
        topk = self.topk_idx[idx]
        target = self.tgt_idx[idx]

        tgt_pos = int(np.where(topk == target)[0][0]) if target in topk else -1

        return {
            "cand_emb":    torch.from_numpy(self.gallery_emb[self.cand_idx[idx]]).float(),
            "text_emb":    torch.from_numpy(self.text_emb[idx]).float(),
            "topk_emb":    torch.from_numpy(self.gallery_emb[topk]).float(),
            "topk_fusion": torch.from_numpy(self.topk_score[idx]).float(),
            "tgt_pos":     torch.tensor(tgt_pos, dtype=torch.long),
        }


def collate(batch: list[dict]) -> dict:
    """Stacks a list of __getitem__ outputs into batched tensors."""
    return {
        "cand_emb":    torch.stack([b["cand_emb"]    for b in batch]),  # (B, D)
        "text_emb":    torch.stack([b["text_emb"]    for b in batch]),  # (B, D)
        "topk_emb":    torch.stack([b["topk_emb"]    for b in batch]),  # (B, K, D)
        "topk_fusion": torch.stack([b["topk_fusion"] for b in batch]),  # (B, K)
        "tgt_pos":     torch.stack([b["tgt_pos"]     for b in batch]),  # (B,)
    }
