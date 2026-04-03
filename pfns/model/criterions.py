from dataclasses import dataclass

import torch
import torch.nn as nn
from pfns import base_config

from .bar_distribution import BarDistributionConfig
from .gaussian_distribution import GaussianDistributionConfig
from .gmm_distribution import GMMDistributionConfig


@dataclass(frozen=True)
class CrossEntropyConfig(base_config.BaseConfig):
    num_classes: int
    reduction: str = "none"

    def get_criterion(self):
        return nn.CrossEntropyLoss(
            reduction=self.reduction, weight=torch.ones(self.num_classes)
        )


__all__ = ["BarDistributionConfig", "CrossEntropyConfig", "GaussianDistributionConfig", "GMMDistributionConfig"]
