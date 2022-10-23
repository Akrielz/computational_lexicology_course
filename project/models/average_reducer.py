from typing import Optional

import torch
from einops import reduce, repeat
from torch import nn


class AverageReducer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        if mask is None:
            return reduce(x, "b n d -> b d", "mean")

        with torch.no_grad():
            lengths = mask.sum(1).view(-1, 1)

        expanded_mask = repeat(mask, "b n -> b n d", d=x.shape[-1])

        masked_features = x.masked_fill(~expanded_mask, 0)
        return masked_features.sum(1).div(lengths)
