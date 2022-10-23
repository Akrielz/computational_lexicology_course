from typing import Optional

import torch
from einops import reduce, repeat
from torch import nn


class AverageReducer(nn.Module):
    """
    AverageReducer is a module that takes a sequence tensor and a mask tensor
    and reduces the sequence len axis by taking the mean of the sequence tensor
    considering the masked elements only.

    If the mask tensor is None, the mean operation is applied to the
    entire sequence tensor.
    """

    def __init__(self):
        super().__init__()

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Sequence tensor with shape (batch_size, length, embedding_dim).
            mask: Mask tensor with shape (batch_size, length).

        Returns:
            Tensor with shape (batch_size, embedding_dim)
        """

        if mask is None:
            return reduce(x, "b n d -> b d", "mean")

        with torch.no_grad():
            lengths = mask.sum(1).view(-1, 1)

        expanded_mask = repeat(mask, "b n -> b n d", d=x.shape[-1])

        masked_features = x.masked_fill(~expanded_mask, 0)
        return masked_features.sum(1).div(lengths)
