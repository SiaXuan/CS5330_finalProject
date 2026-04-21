import torch
import torch.nn as nn


class TIRGCombiner(nn.Module):
    """
    TIRG: gate * img_feat + residual(img_feat, text_feat)
    Both inputs and output are L2-normalized 512-dim CLIP vectors.
    """
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid(),
        )
        self.residual = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, img_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([img_feat, text_feat], dim=-1)
        out = self.gate(x) * img_feat + self.residual(x)
        return out / out.norm(dim=-1, keepdim=True)
