"""Config that plugs NanoTabICLv2 into the PFNs training framework.

Usage::

    from pfns.model.nanotabicl_config import NanoTabICLConfig

    model_config = NanoTabICLConfig(
        criterion=BarDistributionConfig(...),
        encoder=EncoderConfig(...),
        y_encoder=EncoderConfig(...),
        emsize=128,
        nhid=512,
        features_per_group=1,
        # TabICL-specific
        embed_dim=128,
        col_num_blocks=3,
        row_num_blocks=3,
        icl_num_blocks=12,
    )

Set ``MainConfig(model=model_config)`` to train with NanoTabICLv2.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

from torch import nn

from pfns import base_config
from pfns.model import encoders
from pfns.model.criterions import (
    BarDistributionConfig,
    CrossEntropyConfig,
    GaussianDistributionConfig,
    GMMDistributionConfig,
)
from pfns.model.nanotabicl_v2 import NanoTabICLv2
from pfns.model.nanotabicl_wrapper import NanoTabICLWrapper


@dataclass(frozen=True)
class NanoTabICLConfig(base_config.BaseConfig):
    """Drop-in replacement for :class:`TransformerConfig` that builds a
    :class:`NanoTabICLWrapper` instead of a :class:`TableTransformer`.

    Encoder / y_encoder / decoder / criterion follow the exact same
    conventions as :class:`TransformerConfig` so that the *only*
    difference is the attention architecture.
    """

    criterion: (
        CrossEntropyConfig
        | BarDistributionConfig
        | GaussianDistributionConfig
        | GMMDistributionConfig
    )

    # Encoder configs — same as TransformerConfig
    encoder: tp.Optional[encoders.EncoderConfig] = None
    y_encoder: tp.Optional[encoders.EncoderConfig] = None

    # Shared dimensions (must match encoder output / decoder input)
    emsize: int = 128
    nhid: int = 512
    features_per_group: int = 1

    # Required by train.py assertion (c.model.features_per_group)
    attention_between_features: bool = True

    # NanoTabICLv2 architecture hyperparameters
    embed_dim: int = 128
    col_num_blocks: int = 3
    row_num_blocks: int = 3
    icl_num_blocks: int = 12
    col_nhead: int = 8
    row_nhead: int = 8
    icl_nhead: int = 8
    n_cls_cols: int = 4
    n_cls_rows: int = 128

    def create_model(self) -> NanoTabICLWrapper:
        # ---- Criterion ----------------------------------------------------
        criterion = self.criterion.get_criterion()

        if hasattr(criterion, "num_bars"):
            n_out = criterion.num_bars
        elif isinstance(criterion, nn.CrossEntropyLoss):
            n_out = criterion.weight.shape[0]
        else:
            raise ValueError(f"Criterion {criterion} not supported")

        # ---- Encoders (identical to TransformerConfig) --------------------
        if self.encoder is not None:
            x_encoder = self.encoder.create_encoder(
                features=self.features_per_group, emsize=self.emsize,
            )
        else:
            from pfns.model.encoders import get_linear_x_encoder
            x_encoder = get_linear_x_encoder(self.emsize, self.features_per_group)

        if self.y_encoder is not None:
            y_encoder = self.y_encoder.create_encoder(
                features=1, emsize=self.emsize,
            )
        else:
            from pfns.model.encoders import get_linear_y_encoder
            y_encoder = get_linear_y_encoder(self.emsize)

        # ---- Inner TabICL model -------------------------------------------
        inner = NanoTabICLv2(
            emsize=self.emsize,
            embed_dim=self.embed_dim,
            col_num_blocks=self.col_num_blocks,
            row_num_blocks=self.row_num_blocks,
            icl_num_blocks=self.icl_num_blocks,
            col_nhead=self.col_nhead,
            row_nhead=self.row_nhead,
            icl_nhead=self.icl_nhead,
            n_cls_cols=self.n_cls_cols,
            n_cls_rows=self.n_cls_rows,
        )

        # ---- Decoder (same pattern as TableTransformer) -------------------
        # Input dimension = icl_dim from TabICL (embed_dim * n_cls_cols)
        icl_dim = inner.icl_dim
        decoder_dict = nn.ModuleDict({
            "standard": nn.Sequential(
                nn.Linear(icl_dim, self.nhid),
                nn.GELU(),
                nn.Linear(self.nhid, n_out),
            ),
        })

        # ---- Assemble wrapper ---------------------------------------------
        wrapper = NanoTabICLWrapper(
            inner,
            encoder=x_encoder,
            y_encoder=y_encoder,
            decoder_dict=decoder_dict,
            features_per_group=self.features_per_group,
        )
        wrapper.criterion = criterion
        return wrapper
