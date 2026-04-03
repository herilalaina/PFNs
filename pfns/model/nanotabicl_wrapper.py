"""Wrapper that adapts NanoTabICLv2 to the PFNs TableTransformer interface.

Replicates the input-processing pipeline of
:class:`~pfns.model.transformer.TableTransformer` (feature grouping, x/y
encoding, label-leakage prevention, test_x handling) so that the *only*
difference with the standard PFN model is the attention architecture.
"""

from __future__ import annotations

import warnings

import einops
import torch
from torch import nn

from pfns.model.encoders import (
    SequentialEncoder,
    get_linear_x_encoder,
    get_linear_y_encoder,
)
from pfns.model.nanotabicl_v2 import NanoTabICLv2


class NanoTabICLWrapper(nn.Module):
    """Drop-in replacement for :class:`TableTransformer` that uses
    NanoTabICLv2 attention blocks internally.

    The wrapper owns:
    - ``encoder`` / ``y_encoder`` — identical to the ones built by
      :class:`TransformerConfig`.
    - ``decoder_dict`` — maps embeddings to output logits (same structure
      as in :class:`TableTransformer`).
    - ``criterion`` — :class:`BarDistribution`, :class:`GMMDistribution`,
      :class:`GaussianDistribution`, or :class:`CrossEntropyLoss`.
    """

    cache_trainset_representation: bool = False

    def __init__(
        self,
        inner_model: NanoTabICLv2,
        *,
        encoder: nn.Module,
        y_encoder: nn.Module,
        decoder_dict: nn.ModuleDict,
        features_per_group: int = 1,
    ) -> None:
        super().__init__()
        self.inner_model = inner_model
        self.encoder = encoder
        self.y_encoder = y_encoder
        self.decoder_dict = decoder_dict
        self.features_per_group = features_per_group
        self.criterion: nn.Module | None = None  # assigned by config

    # ------------------------------------------------------------------
    # forward — mirrors TableTransformer.forward()
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

        * **x**: ``(B, S_total, num_features)`` when *test_x* is ``None``,
          or ``(B, S_train, num_features)`` when *test_x* is provided.
        * **y**: ``(B, S_train)`` or ``(B, S_train, 1)``.
        * **test_x**: ``(B, S_test, num_features)`` optional.
        """
        if style is not None:
            warnings.warn(
                "NanoTabICLv2 does not support `style`; ignored.",
                stacklevel=2,
            )
        if y_style is not None:
            warnings.warn(
                "NanoTabICLv2 does not support `y_style`; ignored.",
                stacklevel=2,
            )

        # ---- Clone to avoid mutating caller's tensors --------------------
        x_bf = x.clone()
        y_bf = y.clone() if y is not None else None

        # ---- Determine single_eval_pos (train length) --------------------
        single_eval_pos = y_bf.shape[1] if y_bf is not None else 0
        current_context_len = single_eval_pos

        # ---- Combine x + test_x ------------------------------------------
        if test_x is not None:
            x_bf = torch.cat((x_bf[:, :single_eval_pos], test_x), dim=1)

        # Wrap x in dict
        if not isinstance(x_bf, dict):
            x_bf = {"main": x_bf}

        _batch_size, _seq_len, _num_features = x_bf["main"].shape

        # ---- Prepare y dict -----------------------------------------------
        if isinstance(y_bf, torch.Tensor):
            y_bf = {"main": y_bf}

        for k in y_bf:
            if y_bf[k].ndim == 2:
                y_bf[k] = y_bf[k].unsqueeze(-1)  # (B, S) -> (B, S, 1)

            # Pad y to full sequence length with NaN (test positions)
            if y_bf[k].shape[1] < _seq_len:
                y_bf[k] = torch.cat(
                    (
                        y_bf[k],
                        torch.full(
                            (y_bf[k].shape[0], _seq_len - y_bf[k].shape[1], y_bf[k].shape[2]),
                            float("nan"),
                            device=y_bf[k].device,
                            dtype=y_bf[k].dtype,
                        ),
                    ),
                    dim=1,
                )

        # Prevent label leakage
        if "main" in y_bf and y_bf["main"].shape[1] > current_context_len:
            y_bf["main"] = y_bf["main"].clone()
            y_bf["main"][:, current_context_len:] = torch.nan

        # ---- Y encoding (sequence-first for encoder) ----------------------
        y_for_enc = {k: v.transpose(0, 1) for k, v in y_bf.items()}  # B S T -> S B T
        embedded_y = self.y_encoder(
            y_for_enc,
            single_eval_pos=current_context_len,
            cache_trainset_representation=False,
        ).transpose(0, 1)  # S B E -> B S E

        if torch.isnan(embedded_y).any():
            raise ValueError(
                "NaN in embedded_y after y_encoder. "
                "Make sure the y_encoder includes NaN handling."
            )

        del y_bf, y_for_enc

        # ---- Pad x features to multiple of features_per_group -------------
        for k in x_bf:
            num_f = x_bf[k].shape[2]
            missing = (
                self.features_per_group - (num_f % self.features_per_group)
            ) % self.features_per_group
            if missing > 0:
                x_bf[k] = torch.cat(
                    (
                        x_bf[k],
                        torch.zeros(
                            x_bf[k].shape[0],
                            x_bf[k].shape[1],
                            missing,
                            device=x_bf[k].device,
                            dtype=x_bf[k].dtype,
                        ),
                    ),
                    dim=-1,
                )

        # ---- Group features -----------------------------------------------
        for k in x_bf:
            x_bf[k] = einops.rearrange(
                x_bf[k], "b s (f n) -> b s f n", n=self.features_per_group,
            )

        # ---- X encoding (sequence-first for encoder) ----------------------
        for k in x_bf:
            x_bf[k] = einops.rearrange(x_bf[k], "b s f n -> s (b f) n")

        embedded_x = einops.rearrange(
            self.encoder(
                x_bf,
                single_eval_pos=current_context_len,
                cache_trainset_representation=False,
            ),
            "s (b f) e -> b s f e",
            b=_batch_size,
        )  # (B, S, num_groups, emsize)

        del x_bf

        # ---- TabICL attention blocks --------------------------------------
        output_emb = self.inner_model(
            embedded_x, embedded_y, n_train=current_context_len,
        )  # (B, test_len, icl_dim)

        del embedded_x, embedded_y

        # ---- Decoder (same pattern as TableTransformer) -------------------
        output_decoded = {k: v(output_emb) for k, v in self.decoder_dict.items()}
        output_decoded["train_embeddings"] = torch.empty(0)
        output_decoded["test_embeddings"] = output_emb

        if only_return_standard_out:
            return output_decoded["standard"]

        return output_decoded

    # ------------------------------------------------------------------
    # Compatibility stubs expected by the PFNs framework
    # ------------------------------------------------------------------
    def empty_trainset_representation_cache(self) -> None:
        """No-op — NanoTabICLv2 has no KV cache."""

    def reset_save_peak_mem_factor(self, factor: int | None = None) -> None:
        """No-op — not applicable to NanoTabICLv2."""
