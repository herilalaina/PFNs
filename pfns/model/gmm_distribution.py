"""
Gaussian Mixture Model output head for PFNs.

Alternative to BarDistribution that models the predictive distribution
as a mixture of K Gaussians. Implements the same interface as BarDistribution
(forward, mean, variance, ei, pi, ucb, entropy, cdf, icdf, sample, mode, plot).
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
class GMMDistributionConfig(base_config.BaseConfig):
    num_components: int = 16
    min_std: float = 1e-4
    max_std: float = 10.0
    ignore_nan_targets: bool = True

    def get_criterion(self) -> "GMMDistribution":
        return GMMDistribution(
            num_components=self.num_components,
            min_std=self.min_std,
            max_std=self.max_std,
            ignore_nan_targets=self.ignore_nan_targets,
        )


class GMMDistribution(nn.Module):
    """Mixture of Gaussians output distribution for PFNs.

    For K components, the model outputs 3*K logits:
    - K means
    - K log-stds (transformed via softplus + clamp)
    - K logit-weights (transformed via log_softmax)
    """

    def __init__(
        self,
        num_components: int = 16,
        min_std: float = 1e-4,
        max_std: float = 10.0,
        ignore_nan_targets: bool = True,
    ):
        super().__init__()
        self.num_components = num_components
        self.min_std = min_std
        self.max_std = max_std
        self.ignore_nan_targets = ignore_nan_targets

    @property
    def num_bars(self) -> int:
        """Compatibility with BarDistribution interface for n_out computation."""
        return 3 * self.num_components

    def _parse_logits(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split logits into component parameters.

        Args:
            logits: (..., 3*K) tensor

        Returns:
            means: (..., K)
            stds: (..., K) in [min_std, max_std]
            log_weights: (..., K) log-probabilities summing to 0 in log-space
        """
        K = self.num_components
        means = logits[..., :K]
        stds = F.softplus(logits[..., K : 2 * K]).clamp(self.min_std, self.max_std)
        log_weights = F.log_softmax(logits[..., 2 * K : 3 * K], dim=-1)
        return means, stds, log_weights

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood loss.

        Args:
            logits: (..., 3*K) model output
            y: (...,) or (..., 1) targets

        Returns:
            NLL loss of shape (...)
        """
        y = y.clone().view(*logits.shape[:-1])
        ignore_mask = torch.isnan(y)
        if ignore_mask.any() and not self.ignore_nan_targets:
            raise ValueError(f"Found NaN in target {y}")
        y[ignore_mask] = 0.0  # placeholder, will be masked

        means, stds, log_weights = self._parse_logits(logits)

        # log N(y; mu_k, sigma_k) = -0.5*log(2*pi) - log(sigma_k) - 0.5*((y - mu_k)/sigma_k)^2
        y_expanded = y.unsqueeze(-1)  # (..., 1)
        log_normal = (
            -0.5 * math.log(2 * math.pi)
            - torch.log(stds)
            - 0.5 * ((y_expanded - means) / stds) ** 2
        )
        # NLL = -logsumexp_k(log_w_k + log N(y; mu_k, sigma_k))
        nll = -torch.logsumexp(log_weights + log_normal, dim=-1)

        nll[ignore_mask] = 0.0
        return nll

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """Mixture mean: sum_k w_k * mu_k."""
        means, _stds, log_weights = self._parse_logits(logits)
        weights = torch.exp(log_weights)
        return (weights * means).sum(-1)

    def variance(self, logits: torch.Tensor) -> torch.Tensor:
        """Mixture variance: sum_k w_k*(sigma_k^2 + mu_k^2) - (E[X])^2."""
        means, stds, log_weights = self._parse_logits(logits)
        weights = torch.exp(log_weights)
        mean = (weights * means).sum(-1)
        mean_of_square = (weights * (stds**2 + means**2)).sum(-1)
        return mean_of_square - mean**2

    def ei(
        self,
        logits: torch.Tensor,
        best_f: float | torch.Tensor,
        *,
        maximize: bool = True,
    ) -> torch.Tensor:
        """Expected Improvement acquisition function.

        EI = sum_k w_k * sigma_k * [u_k * Phi(u_k) + phi(u_k)]
        where u_k = (mu_k - best_f) / sigma_k
        """
        assert maximize
        means, stds, log_weights = self._parse_logits(logits)
        weights = torch.exp(log_weights)

        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(
                logits[..., 0].shape, best_f, device=logits.device
            )

        best_f = best_f.unsqueeze(-1)  # (..., 1)
        u = (means - best_f) / stds

        normal = torch.distributions.Normal(
            torch.zeros_like(u), torch.ones_like(u)
        )
        ei_per_component = stds * (
            u * normal.cdf(u) + torch.exp(normal.log_prob(u))
        )
        return (weights * ei_per_component).sum(-1)

    def soft_ei(
        self,
        logits: torch.Tensor,
        best_f: float | torch.Tensor,
        *,
        maximize: bool = True,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Softplus-based EI — always positive, differentiable, no log(0) issues.

        For each component k, computes E[softplus(f - best_f)] where f ~ N(mu_k, sigma_k^2).
        Uses the closed-form: E[softplus(X)] for X ~ N(mu, sigma^2)
            = sigma * [u * Phi(u) + phi(u)] + (1/beta) * log(1 + exp(-beta * mu))
        where u = mu / sigma (approximation valid for beta=1).

        For beta=1 the exact formula is:
            E[softplus(X)] = mu * Phi(mu/sigma) + sigma * phi(mu/sigma)
                           + sigma * [z * Phi(z) + phi(z)]  (correction term)
        But simpler: use E[softplus(X)] = E[max(0,X)] + E[softplus(X) - max(0,X)]
        Since softplus(x) >= max(0,x) and softplus(x) ~ max(0,x) + 0.5*exp(-|x|),
        we use: soft_ei_k = ei_k + sigma_k * C  where C is a small positive correction.

        Simplified approach: E_N(mu,s^2)[softplus(x)] = s*[u*Phi(u) + phi(u)] + s*log(2)*Phi(-u)
        where u = mu/s. This equals EI + s*log(2)*Phi(-u) which is always > 0.

        Args:
            logits: (..., 3*K) model output (WITH gradients)
            best_f: best observed value
            maximize: whether to maximize
            beta: softplus sharpness (unused, kept for interface compat)

        Returns:
            soft_ei values of shape (...), strictly > 0
        """
        assert maximize
        means, stds, log_weights = self._parse_logits(logits)
        weights = torch.exp(log_weights)

        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(
                logits[..., 0].shape, best_f, device=logits.device
            )

        best_f = best_f.unsqueeze(-1)  # (..., 1)
        mu = means - best_f  # shifted mean: (..., K)
        u = mu / stds

        normal = torch.distributions.Normal(
            torch.zeros_like(u), torch.ones_like(u)
        )
        phi_u = torch.exp(normal.log_prob(u))
        Phi_u = normal.cdf(u)

        # E[softplus(X)] for X ~ N(mu, s^2):
        # = s * [u * Phi(u) + phi(u)] + s * log(2) * Phi(-u)
        # The first term is the standard EI, the second ensures strict positivity
        ei_per_component = stds * (u * Phi_u + phi_u)
        correction = stds * math.log(2) * normal.cdf(-u)
        soft_ei_per_component = ei_per_component + correction

        return (weights * soft_ei_per_component).sum(-1)

    def expected_log_utility(
        self,
        logits: torch.Tensor,
        best_f: float | torch.Tensor,
        *,
        maximize: bool = True,
        beta: float = 1.0,
        n_quad: int = 20,
    ) -> torch.Tensor:
        """E[log(softplus(f - best_f))] under GMM predictive (EULBO).

        Uses Gauss-Hermite quadrature per component, then mixture-weighted sum.
        The log is INSIDE the expectation, as in Moss et al. (2024).

        Args:
            logits: (..., 3*K) model output (WITH gradients)
            best_f: best observed value
            maximize: whether to maximize
            beta: softplus sharpness (default 1.0)
            n_quad: number of Gauss-Hermite quadrature points

        Returns:
            Expected log-utility of shape (...)
        """
        assert maximize
        means, stds, log_weights = self._parse_logits(logits)
        weights = torch.exp(log_weights)  # (..., K)

        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(
                logits[..., 0].shape, best_f, device=logits.device
            )

        # Constant quadrature nodes/weights (not differentiable, just constants)
        import numpy as np
        gh_nodes_np, gh_weights_np = np.polynomial.hermite.hermgauss(n_quad)
        gh_nodes = torch.as_tensor(gh_nodes_np, dtype=logits.dtype, device=logits.device)
        gh_weights = torch.as_tensor(gh_weights_np, dtype=logits.dtype, device=logits.device)

        # For each component k: E_{N(mu_k, s_k^2)}[log(softplus(f - best_f))]
        # f = mu_k + s_k * sqrt(2) * node
        # means: (..., K), stds: (..., K), gh_nodes: (Q,)
        f_samples = (
            means.unsqueeze(-1)
            + stds.unsqueeze(-1) * math.sqrt(2) * gh_nodes
        )  # (..., K, Q)
        improvement = f_samples - best_f[..., None, None]  # (..., K, Q)
        log_utility = torch.log(
            torch.nn.functional.softplus(improvement, beta=beta)
        )  # (..., K, Q)

        # GH quadrature per component: (1/sqrt(pi)) * sum_q w_q * log_utility
        elu_per_component = (gh_weights * log_utility).sum(-1) / math.sqrt(math.pi)  # (..., K)

        # Mixture-weighted sum
        return (weights * elu_per_component).sum(-1)

    def pi(
        self,
        logits: torch.Tensor,
        best_f: float | torch.Tensor,
        *,
        maximize: bool = True,
    ) -> torch.Tensor:
        """Probability of Improvement: sum_k w_k * Phi((mu_k - best_f) / sigma_k)."""
        assert maximize
        means, stds, log_weights = self._parse_logits(logits)
        weights = torch.exp(log_weights)

        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(
                logits[..., 0].shape, best_f, device=logits.device
            )

        best_f = best_f.unsqueeze(-1)
        u = (means - best_f) / stds
        normal = torch.distributions.Normal(
            torch.zeros_like(u), torch.ones_like(u)
        )
        return (weights * normal.cdf(u)).sum(-1)

    def cdf(self, logits: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """CDF: sum_k w_k * Phi((y - mu_k) / sigma_k).

        Args:
            logits: (..., 3*K)
            ys: (..., n_points) or (n_points,)
        """
        means, stds, log_weights = self._parse_logits(logits)
        weights = torch.exp(log_weights)

        if len(ys.shape) < len(logits.shape) and len(ys.shape) == 1:
            ys = ys.repeat(logits.shape[:-1] + (1,))

        # ys: (..., n_points), means: (..., K) -> need broadcasting
        # ys_expanded: (..., n_points, 1), means_expanded: (..., 1, K)
        ys_expanded = ys.unsqueeze(-1)  # (..., n_points, 1)
        means_expanded = means.unsqueeze(-2)  # (..., 1, K)
        stds_expanded = stds.unsqueeze(-2)  # (..., 1, K)
        weights_expanded = weights.unsqueeze(-2)  # (..., 1, K)

        z = (ys_expanded - means_expanded) / stds_expanded
        normal = torch.distributions.Normal(
            torch.zeros_like(z), torch.ones_like(z)
        )
        return (weights_expanded * normal.cdf(z)).sum(-1)

    def _cdf_at(
        self,
        y: torch.Tensor,
        means: torch.Tensor,
        stds: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate mixture CDF at y. y: (...), means/stds/weights: (..., K)."""
        z = (y.unsqueeze(-1) - means) / stds
        # Use erfc for better numerical precision in tails
        cdf_per_component = 0.5 * torch.erfc(-z * math.sqrt(0.5))
        return (weights * cdf_per_component).sum(-1)

    def icdf(self, logits: torch.Tensor, left_prob: float) -> torch.Tensor:
        """Inverse CDF via bisection.

        Uses per-component quantile bounds for a tight initial bracket,
        then 64 bisection iterations for high precision.

        Args:
            logits: (..., 3*K)
            left_prob: probability mass to the left

        Returns:
            Quantile values of shape (...)
        """
        means, stds, log_weights = self._parse_logits(logits)
        weights = torch.exp(log_weights)

        # Tight initial bracket: use per-component quantile bounds
        # For each component, the left_prob quantile is mu_k + sigma_k * Phi^{-1}(left_prob)
        # The mixture quantile must lie between the min and max of these
        inv_normal = torch.erfinv(torch.tensor(2.0 * left_prob - 1.0)) * math.sqrt(2.0)
        component_quantiles = means + stds * inv_normal  # (..., K)
        lo = component_quantiles.min(dim=-1).values - 0.1 * stds.max(dim=-1).values
        hi = component_quantiles.max(dim=-1).values + 0.1 * stds.max(dim=-1).values

        # Verify bracket contains the target; widen if needed
        cdf_lo = self._cdf_at(lo, means, stds, weights)
        cdf_hi = self._cdf_at(hi, means, stds, weights)
        needs_wider = (cdf_lo > left_prob) | (cdf_hi < left_prob)
        if needs_wider.any():
            weighted_mean = (weights * means).sum(-1)
            weighted_std = torch.sqrt(
                (weights * (stds**2 + means**2)).sum(-1) - weighted_mean**2
            ).clamp(min=1e-6)
            safe_lo = weighted_mean - 10 * weighted_std
            safe_hi = weighted_mean + 10 * weighted_std
            lo = torch.where(needs_wider, safe_lo, lo)
            hi = torch.where(needs_wider, safe_hi, hi)

        for _ in range(64):
            mid = (lo + hi) / 2
            cdf_val = self._cdf_at(mid, means, stds, weights)
            lo = torch.where(cdf_val < left_prob, mid, lo)
            hi = torch.where(cdf_val < left_prob, hi, mid)

        return (lo + hi) / 2

    def ucb(
        self,
        logits: torch.Tensor,
        best_f: float,
        rest_prob: float = (1 - 0.682) / 2,
        *,
        maximize: bool = True,
    ) -> torch.Tensor:
        """UCB via quantile function.

        Args:
            logits: model output
            best_f: unused (kept for interface compatibility)
            rest_prob: tail probability
            maximize: whether to maximize
        """
        if maximize:
            rest_prob = 1 - rest_prob
        return self.icdf(logits, rest_prob)

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Upper-bound entropy: sum_k w_k * [H_k - log(w_k)].

        Where H_k = 0.5 * log(2*pi*e*sigma_k^2) is the Gaussian entropy.
        """
        means, stds, log_weights = self._parse_logits(logits)
        weights = torch.exp(log_weights)
        component_entropy = 0.5 * torch.log(2 * math.pi * math.e * stds**2)
        return (weights * (component_entropy - log_weights)).sum(-1)

    def sample(self, logits: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        """Sample from the mixture.

        Args:
            logits: (..., 3*K)
            t: temperature (applied to log-weights)

        Returns:
            Samples of shape (...)
        """
        means, stds, log_weights = self._parse_logits(logits)
        # Apply temperature to weights
        tempered_weights = F.softmax(log_weights / t, dim=-1)
        # Sample component indices
        indices = torch.multinomial(
            tempered_weights.reshape(-1, self.num_components), 1
        ).reshape(means.shape[:-1])
        # Gather selected component parameters
        selected_means = means.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
        selected_stds = stds.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
        return selected_means + selected_stds * torch.randn_like(selected_means)

    def mode(self, logits: torch.Tensor) -> torch.Tensor:
        """Mode: mean of component with highest w_k / sigma_k (peak density)."""
        means, stds, log_weights = self._parse_logits(logits)
        weights = torch.exp(log_weights)
        # Peak density of component k is w_k / (sigma_k * sqrt(2*pi))
        # Argmax over k is same as argmax of w_k / sigma_k
        density_at_peak = weights / stds
        best_component = density_at_peak.argmax(-1)
        return means.gather(-1, best_component.unsqueeze(-1)).squeeze(-1)

    def median(self, logits: torch.Tensor) -> torch.Tensor:
        """Median of the mixture."""
        return self.icdf(logits, 0.5)

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
        """Plot the mixture PDF."""
        import matplotlib.pyplot as plt

        logits = logits.squeeze()
        assert logits.dim() == 1, "logits should be 1d, at least after squeezing."
        if ax is None:
            ax = plt.gca()

        means, stds, log_weights = self._parse_logits(logits.unsqueeze(0))
        weights = torch.exp(log_weights)

        if zoom_to_quantile is not None:
            lower, upper = (
                self.quantile(logits.unsqueeze(0), zoom_to_quantile)
                .squeeze(0)
                .tolist()
            )
        else:
            # Use range covering all components
            lower = (means - 4 * stds).min().item()
            upper = (means + 4 * stds).max().item()

        xs = torch.linspace(lower, upper, n_points)
        # Evaluate PDF: sum_k w_k * N(x; mu_k, sigma_k)
        z = (xs.unsqueeze(-1) - means.squeeze(0)) / stds.squeeze(0)
        log_pdf = torch.logsumexp(
            log_weights.squeeze(0)
            - 0.5 * math.log(2 * math.pi)
            - torch.log(stds.squeeze(0))
            - 0.5 * z**2,
            dim=-1,
        )
        pdf = torch.exp(log_pdf)

        ax.plot(xs.numpy(), pdf.detach().numpy(), **kwargs)
        if zoom_to_quantile is not None:
            ax.set_xlim(lower, upper)
        return ax
