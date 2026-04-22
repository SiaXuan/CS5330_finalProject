"""Cross-encoder re-ranker: scores a (candidate_img, text, retrieved_img) triple.

Pure PyTorch — no CLIP/FashionCLIP imports. The model only knows embedding
dimension. Whatever encoder produced the embeddings, this MLP will score
them as long as `emb_dim` is set correctly at construction.

Input features per triple:
  - candidate_emb  (D,)      the query image embedding
  - text_emb       (D,)      the modification text embedding (avg of 2 captions)
  - retrieved_emb  (D,)      one gallery candidate being re-ranked
  - cand*retr      (D,)      element-wise product (captures visual similarity)
  - text*retr      (D,)      element-wise product (captures text-image match)
  - fusion_score   (1,)      the alpha-weighted sum that the first stage used

Total input dim = 5*D + 1. For D=512 that's 2561.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ReRankerMLP(nn.Module):
    """MLP that scores one (candidate, text, retrieved) triple.

    Designed to be called in batches of (B, K, *) where K is the re-rank
    depth (e.g. top-50). `forward` flattens the first two dims internally.
    """

    def __init__(self, emb_dim: int = 512, hidden_dim: int = 512,
                 n_layers: int = 3, dropout: float = 0.1,
                 use_fusion_score: bool = True):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_fusion_score = use_fusion_score

        # 3 raw embeddings + 2 hadamard products + optional fusion score
        in_dim = 5 * emb_dim + (1 if use_fusion_score else 0)

        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(prev, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = hidden_dim
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def featurize(self, cand: torch.Tensor, text: torch.Tensor,
                  retr: torch.Tensor, fusion: torch.Tensor | None = None
                  ) -> torch.Tensor:
        """Build the input feature vector from the three embeddings.

        Shapes: cand/text are either (B, D) or (B, K, D); retr is (B, K, D)
        (or (B, D) for a single item). Everything is broadcast to (..., K, D)
        and concatenated on the last dim.
        """
        # Broadcast cand/text up to the retrieved shape
        if retr.dim() == 3 and cand.dim() == 2:
            cand = cand.unsqueeze(1).expand_as(retr)
            text = text.unsqueeze(1).expand_as(retr)

        feats = [cand, text, retr, cand * retr, text * retr]
        out = torch.cat(feats, dim=-1)

        if self.use_fusion_score:
            if fusion is None:
                raise ValueError("fusion score required when use_fusion_score=True")
            # fusion is (B, K) or (B,) — add a trailing feature dim
            out = torch.cat([out, fusion.unsqueeze(-1)], dim=-1)
        return out

    def forward(self, cand: torch.Tensor, text: torch.Tensor,
                retr: torch.Tensor, fusion: torch.Tensor | None = None
                ) -> torch.Tensor:
        """Return a scalar score per triple.

        Shapes in:  cand/text (B, D), retr (B, K, D), fusion (B, K) or None.
        Shape out:  (B, K) scores.
        """
        feats = self.featurize(cand, text, retr, fusion)
        # feats: (B, K, in_dim) → (B*K, in_dim) → MLP → (B*K, 1)
        *batch_dims, in_dim = feats.shape
        flat = feats.reshape(-1, in_dim)
        scores = self.mlp(flat).squeeze(-1)
        return scores.reshape(*batch_dims)
