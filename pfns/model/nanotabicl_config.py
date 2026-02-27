"""Config that plugs NanoTabICLv2 into the PFNs training framework."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from pfns import base_config
from pfns.model.bar_distribution import BarDistribution
from pfns.model.criterions import BarDistributionConfig, CrossEntropyConfig
from pfns.model.nanotabicl_v2 import NanoTabICLv2
from pfns.model.nanotabicl_wrapper import NanoTabICLWrapper


@dataclass(frozen=True)
class NanoTabICLConfig(base_config.BaseConfig):
    """Drop-in replacement for :class:`TransformerConfig` that builds a
    :class:`NanoTabICLWrapper` instead of a :class:`TableTransformer`.

    Set ``MainConfig(model=NanoTabICLConfig(...))`` to use NanoTabICLv2.
    """

    criterion: CrossEntropyConfig | BarDistributionConfig

    # NanoTabICLv2 hyperparameters
    embed_dim: int = 128
    col_num_blocks: int = 3
    row_num_blocks: int = 3
    icl_num_blocks: int = 12
    col_nhead: int = 8
    row_nhead: int = 8
    icl_nhead: int = 8
    feature_group_size: int = 3
    n_cls_cols: int = 4
    n_cls_rows: int = 128

    # Required by train.py assertions
    features_per_group: int = 1
    attention_between_features: bool = True

    def create_model(self) -> NanoTabICLWrapper:
        criterion = self.criterion.get_criterion()

        if isinstance(criterion, BarDistribution):
            max_classes = 0
            n_out = criterion.num_bars
        elif isinstance(criterion, nn.CrossEntropyLoss):
            max_classes = criterion.weight.shape[0]
            n_out = max_classes
        else:
            raise ValueError(f"Unsupported criterion type: {type(criterion)}")

        inner = NanoTabICLv2(
            max_classes=max_classes,
            out_dim=n_out,
            embed_dim=self.embed_dim,
            col_num_blocks=self.col_num_blocks,
            row_num_blocks=self.row_num_blocks,
            icl_num_blocks=self.icl_num_blocks,
            col_nhead=self.col_nhead,
            row_nhead=self.row_nhead,
            icl_nhead=self.icl_nhead,
            feature_group_size=self.feature_group_size,
            n_cls_cols=self.n_cls_cols,
            n_cls_rows=self.n_cls_rows,
        )

        wrapper = NanoTabICLWrapper(inner)
        wrapper.criterion = criterion
        return wrapper
