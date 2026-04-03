"""
Single Gaussian output head for PFNs.

Simpler alternative to Bar/GMM: directly parameterizes mean and std (n_out=2).
Trained with Gaussian NLL. Implements the same interface as Bar/GMM distributions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
from torch import nn
import torch.nn.functional as F

from pfns import base_config

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


@dataclass(frozen=True)
class GaussianDistributionConfig(base_config.BaseConfig):
    min_std: float = 1e-4
    max_std: float = 3.0
    ignore_nan_targets: bool = True

    def get_criterion(self) -> "GaussianDistribution":
        return GaussianDistribution(
            min_std=self.min_std,
            max_std=self.max_std,
            ignore_nan_targets=self.ignore_nan_targets,
        )


class GaussianDistribution(nn.Module):
    """Single Gaussian output distribution for PFNs.

    The model outputs 2 logits per prediction:
    - logits[..., 0] = mean (unconstrained)
    - logits[..., 1] = pre-std (transformed via softplus + clamp to get std)
    """

    def __init__(
        self,
        min_std: float = 1e-4,
        max_std: float = 3.0,
        ignore_nan_targets: bool = True,
    ):
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std
        self.ignore_nan_targets = ignore_nan_targets

    @property
    def num_bars(self) -> int:
        """Compatibility with BarDistribution interface for n_out computation."""
        return 2

    def _parse_logits(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split logits into mean and std.

        Args:
            logits: (..., 2) tensor

        Returns:
            mean: (...,)
            std: (...,) in [min_std, max_std]
        """
        mean = logits[..., 0]
        std = F.softplus(logits[..., 1]).clamp(self.min_std, self.max_std)
        return mean, std

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood loss.

        NLL = log(sigma) + 0.5 * ((y - mu) / sigma)^2 + 0.5*log(2*pi)

        Args:
            logits: (..., 2) model output
            y: (...,) or (..., 1) targets

        Returns:
            NLL loss of shape (...)
        """
        y = y.clone().view(*logits.shape[:-1])
        ignore_mask = torch.isnan(y)
        if ignore_mask.any() and not self.ignore_nan_targets:
            raise ValueError(f"Found NaN in target {y}")
        y[ignore_mask] = 0.0

        mean, std = self._parse_logits(logits)

        nll = (
            torch.log(std)
            + 0.5 * ((y - mean) / std) ** 2
            + 0.5 * math.log(2 * math.pi)
        )

        nll[ignore_mask] = 0.0
        return nll

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """Predicted mean."""
        mean, _std = self._parse_logits(logits)
        return mean

    def variance(self, logits: torch.Tensor) -> torch.Tensor:
        """Predicted variance."""
        _mean, std = self._parse_logits(logits)
        return std ** 2

    def _ei_parts(
        self,
        logits: torch.Tensor,
        best_f: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared computation for ei / soft_ei.

        Returns (u, std, normal) where u = (mu - best_f) / sigma.
        """
        mean, std = self._parse_logits(logits)

        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(mean.shape, best_f, device=logits.device)

        u = (mean - best_f) / std
        normal = torch.distributions.Normal(
            torch.zeros_like(u), torch.ones_like(u)
        )
        return u, std, normal

    def ei(
        self,
        logits: torch.Tensor,
        best_f: float | torch.Tensor,
        *,
        maximize: bool = True,
    ) -> torch.Tensor:
        """Expected Improvement: sigma * [u * Phi(u) + phi(u)]
        where u = (mu - best_f) / sigma.
        """
        assert maximize
        u, std, normal = self._ei_parts(logits, best_f)
        return std * (u * normal.cdf(u) + torch.exp(normal.log_prob(u)))

    @staticmethod
    def _log_ei_helper(u: torch.Tensor) -> torch.Tensor:
        """Numerically stable log(u * Phi(u) + phi(u)).

        Three-regime implementation following BoTorch (Ament et al. 2023):

        1. u > UPPER_BOUND (-1): Direct log of the naive EI formula.
           phi(u) + u*Phi(u) is well-behaved and positive here.

        2. NEG_INV_SQRT_EPS < u <= UPPER_BOUND: Rearrange as
           log(phi(u)) + log(1 - exp(w)), where
           w = log(|u| * Phi(u) / phi(u)) = log(erfcx(-u/sqrt(2)) * |u|) + c.
           Uses log1mexp for numerical stability.

        3. u <= NEG_INV_SQRT_EPS: Asymptotic regime where
           log(EI) ≈ log(phi(u)) - 2*log(|u|).
           The log1mexp correction vanishes in machine precision.

        Critical: uses masked_fill to prevent NaN gradient leakage through
        torch.where branches (PyTorch computes gradients for both branches).
        """
        INV_SQRT_2 = math.sqrt(0.5)
        LOG_SQRT_PI_OVER_2 = 0.5 * math.log(math.pi / 2.0)
        LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

        UPPER_BOUND = -1.0
        NEG_INV_SQRT_EPS = -1e6 if u.dtype == torch.float64 else -1e3

        # --- Upper branch (u > UPPER_BOUND): log(phi(u) + u * Phi(u)) ---
        # Mask u to avoid NaN grads from the lower branches
        u_upper = u.masked_fill(u < UPPER_BOUND, UPPER_BOUND)
        # phi(u) + u * Phi(u) via standard normal
        Phi_upper = 0.5 * torch.erfc(-INV_SQRT_2 * u_upper)
        phi_upper = torch.exp(-0.5 * u_upper**2 - LOG_SQRT_2PI)
        log_ei_upper = (phi_upper + u_upper * Phi_upper).log()

        # --- Lower branch (u <= UPPER_BOUND): more careful computation ---
        # Mask u to avoid NaN grads from the upper branch
        u_lower = u.masked_fill(u > UPPER_BOUND, UPPER_BOUND)
        # Further mask for the asymptotic regime
        u_safe = u_lower.masked_fill(u < NEG_INV_SQRT_EPS, NEG_INV_SQRT_EPS)

        # w = log(|u| * Phi(u) / phi(u)) = log(erfcx(-u/sqrt(2)) * |u|) + c
        # For u < 0: -u/sqrt(2) > 0, so erfcx is safe and doesn't overflow
        w = torch.log(torch.special.erfcx(-INV_SQRT_2 * u_safe) * u_safe.abs()) + LOG_SQRT_PI_OVER_2

        # log(phi(u)) = -u^2/2 - log(sqrt(2*pi))
        log_phi_u = -0.5 * u**2 - LOG_SQRT_2PI

        # log1mexp: numerically stable log(1 - exp(w)) for w < 0
        # For w close to 0: use log(-expm1(w)); for w << 0: use log1p(-exp(w))
        LOG_2 = math.log(2.0)
        log1mexp_w = torch.where(
            w > -LOG_2,
            torch.log(-torch.expm1(w.masked_fill(w <= -LOG_2, -1.0))),
            torch.log1p(-torch.exp(w.masked_fill(w > -LOG_2, -1.0))),
        )

        # Asymptotic regime: log(1-exp(w)) ≈ -2*log(|u|) when |u| → ∞
        lower_correction = torch.where(
            u > NEG_INV_SQRT_EPS,
            log1mexp_w,
            -2.0 * u_lower.abs().log(),
        )
        log_ei_lower = log_phi_u + lower_correction

        return torch.where(u > UPPER_BOUND, log_ei_upper, log_ei_lower)

    def log_ei(
        self,
        logits: torch.Tensor,
        best_f: float | torch.Tensor,
        *,
        maximize: bool = True,
    ) -> torch.Tensor:
        """Log Expected Improvement: log(sigma) + log(u * Phi(u) + phi(u)).

        Numerically stable via erfcx trick (Ament et al. 2023).
        """
        assert maximize
        u, std, _normal = self._ei_parts(logits, best_f)
        return torch.log(std) + self._log_ei_helper(u)

    def soft_ei(
        self,
        logits: torch.Tensor,
        best_f: float | torch.Tensor,
        *,
        maximize: bool = True,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Strictly positive EI for EULBO loss: log(soft_ei) is always finite.

        Same analytical formula as ei() but clamped to a small positive floor
        so that log(soft_ei) never produces -inf.

        Args:
            logits: (..., 2) model output (WITH gradients)
            best_f: best observed value
            maximize: whether to maximize
            beta: unused (kept for API compatibility with BarDistribution)

        Returns:
            soft_ei values of shape (...), strictly > 0
        """
        assert maximize
        u, std, normal = self._ei_parts(logits, best_f)
        ei_val = std * (u * normal.cdf(u) + torch.exp(normal.log_prob(u)))
        # Floor at small positive value so log(soft_ei) is finite
        return ei_val.clamp(min=1e-10)

    @staticmethod
    def _gauss_hermite_nodes(n: int, dtype: torch.dtype, device: torch.device):
        """Gauss-Hermite quadrature nodes and weights (cached per call site)."""
        import numpy as np
        nodes_np, weights_np = np.polynomial.hermite.hermgauss(n)
        return (
            torch.as_tensor(nodes_np, dtype=dtype, device=device),
            torch.as_tensor(weights_np, dtype=dtype, device=device),
        )

    def expected_log_utility(
        self,
        logits: torch.Tensor,
        best_f: float | torch.Tensor,
        *,
        maximize: bool = True,
        beta: float = 1.0,
        n_quad: int = 20,
    ) -> torch.Tensor:
        """E[log(softplus(f - best_f))] under Gaussian predictive (EULBO).

        Uses Gauss-Hermite quadrature to compute the expectation, as in
        Moss et al. (2024). The log is INSIDE the expectation.

        The numpy call computes constant quadrature nodes/weights (not part
        of the autograd graph). All differentiable computation uses torch.

        Args:
            logits: (..., 2) model output (WITH gradients)
            best_f: best observed value
            maximize: whether to maximize
            beta: softplus sharpness (default 1.0)
            n_quad: number of Gauss-Hermite quadrature points

        Returns:
            Expected log-utility of shape (...)
        """
        assert maximize
        mean, std = self._parse_logits(logits)

        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(mean.shape, best_f, device=logits.device)

        # Constant quadrature nodes/weights (not differentiable, just constants)
        gh_nodes, gh_weights = self._gauss_hermite_nodes(
            n_quad, logits.dtype, logits.device
        )

        # Transform: f = mean + std * sqrt(2) * node
        # E_N(mu,s^2)[h(f)] = (1/sqrt(pi)) * sum_i w_i * h(sqrt(2)*s*x_i + mu)
        f_samples = mean.unsqueeze(-1) + std.unsqueeze(-1) * math.sqrt(2) * gh_nodes  # (..., n_quad)
        improvement = f_samples - best_f.unsqueeze(-1)  # (..., n_quad)
        log_utility = torch.log(torch.nn.functional.softplus(improvement, beta=beta))

        # Weighted sum with GH weights, normalized by 1/sqrt(pi)
        return (gh_weights * log_utility).sum(-1) / math.sqrt(math.pi)

    def pi(
        self,
        logits: torch.Tensor,
        best_f: float | torch.Tensor,
        *,
        maximize: bool = True,
    ) -> torch.Tensor:
        """Probability of Improvement: Phi((mu - best_f) / sigma)."""
        assert maximize
        mean, std = self._parse_logits(logits)

        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(mean.shape, best_f, device=logits.device)

        u = (mean - best_f) / std
        normal = torch.distributions.Normal(
            torch.zeros_like(u), torch.ones_like(u)
        )
        return normal.cdf(u)

    def ucb(
        self,
        logits: torch.Tensor,
        best_f: float,
        rest_prob: float = (1 - 0.682) / 2,
        *,
        maximize: bool = True,
    ) -> torch.Tensor:
        """UCB: mean + z * std where z = Phi^{-1}(1 - rest_prob)."""
        mean, std = self._parse_logits(logits)
        if maximize:
            z = torch.erfinv(torch.tensor(2.0 * (1 - rest_prob) - 1.0)) * math.sqrt(2.0)
        else:
            z = -torch.erfinv(torch.tensor(2.0 * (1 - rest_prob) - 1.0)) * math.sqrt(2.0)
        return mean + z * std

    def cdf(self, logits: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """CDF: Phi((y - mu) / sigma).

        Args:
            logits: (..., 2)
            ys: (..., n_points) or (n_points,)
        """
        mean, std = self._parse_logits(logits)

        if len(ys.shape) < len(logits.shape) and len(ys.shape) == 1:
            ys = ys.repeat(logits.shape[:-1] + (1,))

        # mean, std: (...), ys: (..., n_points)
        # Need to broadcast: expand mean/std to (..., 1)
        z = (ys - mean.unsqueeze(-1)) / std.unsqueeze(-1)
        return 0.5 * torch.erfc(-z * math.sqrt(0.5))

    def icdf(self, logits: torch.Tensor, left_prob: float) -> torch.Tensor:
        """Inverse CDF: mu + sigma * Phi^{-1}(left_prob)."""
        mean, std = self._parse_logits(logits)
        inv_normal = torch.erfinv(torch.tensor(2.0 * left_prob - 1.0)) * math.sqrt(2.0)
        return mean + std * inv_normal

    def sample(self, logits: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        """Sample: mu + t * sigma * eps, eps ~ N(0,1)."""
        mean, std = self._parse_logits(logits)
        return mean + t * std * torch.randn_like(mean)

    def mode(self, logits: torch.Tensor) -> torch.Tensor:
        """Mode of a Gaussian = mean."""
        mean, _std = self._parse_logits(logits)
        return mean

    def median(self, logits: torch.Tensor) -> torch.Tensor:
        """Median of a Gaussian = mean."""
        mean, _std = self._parse_logits(logits)
        return mean

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Entropy: 0.5 * log(2*pi*e*sigma^2)."""
        _mean, std = self._parse_logits(logits)
        return 0.5 * torch.log(2 * math.pi * math.e * std ** 2)

    def quantile(
        self,
        logits: torch.Tensor,
        center_prob: float = 0.682,
    ) -> torch.Tensor:
        """Quantile interval."""
        side_probs = (1.0 - center_prob) / 2
        return torch.stack(
            (
                self.icdf(logits, side_probs),
                self.icdf(logits, 1.0 - side_probs),
            ),
            -1,
        )

    def plot(
        self,
        logits: torch.Tensor,
        ax: plt.Axes | None = None,
        zoom_to_quantile: float | None = None,
        n_points: int = 200,
        **kwargs: Any,
    ) -> plt.Axes:
        """Plot the Gaussian PDF."""
        import matplotlib.pyplot as plt

        logits = logits.squeeze()
        assert logits.dim() == 1 and logits.shape[0] == 2
        if ax is None:
            ax = plt.gca()

        mean, std = self._parse_logits(logits.unsqueeze(0))
        mean_val = mean.item()
        std_val = std.item()

        if zoom_to_quantile is not None:
            lower, upper = self.quantile(
                logits.unsqueeze(0), zoom_to_quantile
            ).squeeze(0).tolist()
        else:
            lower = mean_val - 4 * std_val
            upper = mean_val + 4 * std_val

        xs = torch.linspace(lower, upper, n_points)
        pdf = torch.exp(
            -0.5 * ((xs - mean_val) / std_val) ** 2
        ) / (std_val * math.sqrt(2 * math.pi))

        ax.plot(xs.numpy(), pdf.detach().numpy(), **kwargs)
        if zoom_to_quantile is not None:
            ax.set_xlim(lower, upper)
        return ax
