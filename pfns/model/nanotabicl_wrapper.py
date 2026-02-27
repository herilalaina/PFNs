"""Wrapper that adapts NanoTabICLv2 to the PFNs TableTransformer interface."""

from __future__ import annotations

import warnings

import torch
from torch import nn

from .nanotabicl_v2 import NanoTabICLv2


class NanoTabICLWrapper(nn.Module):
    """Wraps :class:`NanoTabICLv2` so it can be used as a drop-in replacement
    for :class:`TableTransformer` inside the PFNs training loop.
    """

    cache_trainset_representation: bool = False

    def __init__(self, inner_model: NanoTabICLv2) -> None:
        super().__init__()
        self.inner_model = inner_model
        # criterion will be assigned externally by the config's create_model()
        self.criterion: nn.Module | None = None

    # ------------------------------------------------------------------
    # forward – mirrors TableTransformer.forward()
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        test_x: torch.Tensor | None = None,
        style: torch.Tensor | None = None,
        y_style: torch.Tensor | None = None,
        only_return_standard_out: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        Parameters match :meth:`TableTransformer.forward` exactly.

        * **x**: ``(B, total_seq, num_features)`` when *test_x* is ``None``,
          or ``(B, train_len, num_features)`` when *test_x* is provided.
        * **y**: ``(B, train_len)`` or ``(B, train_len, 1)``.
        * **test_x**: ``(B, test_len, num_features)`` optional.
        """
        if style is not None:
            warnings.warn(
                "NanoTabICLv2 does not support `style`; the argument is ignored.",
                stacklevel=2,
            )
        if y_style is not None:
            warnings.warn(
                "NanoTabICLv2 does not support `y_style`; the argument is ignored.",
                stacklevel=2,
            )

        # --- Determine train_len from y --------------------------------
        train_len = y.shape[1]

        # --- Build x_full (train + test) --------------------------------
        if test_x is not None:
            x_full = torch.cat([x[:, :train_len], test_x], dim=1)
        else:
            x_full = x

        # --- Squeeze y to 2-D if needed ---------------------------------
        if y.ndim == 3:
            y = y.squeeze(-1)  # (B, train_len, 1) -> (B, train_len)

        # --- Inner model forward ----------------------------------------
        output = self.inner_model(x_full, y)  # (B, test_len, out_dim)

        if only_return_standard_out:
            return output

        return {
            "standard": output,
            "train_embeddings": torch.empty(0),
            "test_embeddings": torch.empty(0),
        }

    # ------------------------------------------------------------------
    # Compatibility stubs expected by the PFNs framework
    # ------------------------------------------------------------------
    def empty_trainset_representation_cache(self) -> None:
        """No-op — NanoTabICLv2 has no KV cache."""

    def reset_save_peak_mem_factor(self, factor: int | None = None) -> None:
        """No-op — not applicable to NanoTabICLv2."""
