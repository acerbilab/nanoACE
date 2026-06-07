"""Beta-specific ACEP helpers for bounded continuous latent prior tokens.

Model-side PRIOR token semantics live in `ace.py`: tokens have a mode, a fixed
two-feature prior payload, and an embedder path. This module is narrower: it
implements the Beta-family math used by the current ACEP examples. Finite spread
encodes a Beta prior over a bounded continuous latent; zero spread encodes an
exact known latent value. The Gaussian, SIR, and BO examples use these helpers
to sample prior hyperparameters, draw latent truths, build information tokens,
and score prior densities on diagnostic grids.

Three coordinate spaces appear here:

- **unit** `[0, 1]`: where the Beta math lives. A prior is parameterized by mean
  `mu_unit` and concentration `nu = alpha + beta`; `Beta(1, 1)` (`mu=0.5,
  nu=2`) is the uninformative/uniform case.
- **internal** `[-1, 1]`: the ACE token convention for bounded latents
  (`mean_internal = 2*mu_unit - 1`).
- **native** `[lo, hi]`: the latent's semantic/transformed range, used for
  drawing true values and for grid priors.
"""

from __future__ import annotations

import math

import torch

from ace import PRIOR_FEATURES

assert PRIOR_FEATURES == 2, "ace_prior_beta builds two-feature (mean, spread) tokens"


def sample_prior_params(shape: tuple[int, ...], *, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample Beta prior mean/concentration hyperparameters on unit coordinates."""

    eps = 1e-4
    choice = torch.rand(shape, device=device)
    mu_unit = torch.empty(shape, device=device).uniform_(eps, 1.0 - eps)
    nu = torch.empty(shape, device=device)

    uniform = choice < (1.0 / 3.0)
    broad = (choice >= (1.0 / 3.0)) & (choice < 0.5)
    concentrated = ~(uniform | broad)

    mu_unit = torch.where(uniform, torch.full_like(mu_unit, 0.5), mu_unit)
    nu = torch.where(uniform, torch.full_like(nu, 2.0), nu)
    nu = torch.where(broad, torch.empty(shape, device=device).uniform_(0.1, 2.0), nu)
    log_nu = torch.empty(shape, device=device).uniform_(math.log(2.0), math.log(1000.0))
    nu = torch.where(concentrated, log_nu.exp(), nu)
    return mu_unit.clamp(eps, 1.0 - eps), nu


def beta_alpha_beta(mu_unit: torch.Tensor, nu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert Beta mean/concentration to shape parameters."""

    return mu_unit * nu, (1.0 - mu_unit) * nu


def prior_features(mu_unit: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Encode a finite-spread Beta prior as ACE information-token features."""

    mean_internal = 2.0 * mu_unit - 1.0
    spread_internal = ((1.0 - mean_internal.pow(2)).clamp_min(0.0) / (nu + 1.0)).sqrt()
    return torch.stack([mean_internal, spread_internal], dim=-1)


def known_latent_features(value_internal: torch.Tensor) -> torch.Tensor:
    """Encode an exact known bounded latent as a zero-spread information token."""

    return torch.stack([value_internal, torch.zeros_like(value_internal)], dim=-1)


def draw_from_beta(mu_unit: torch.Tensor, nu: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Draw a native-coordinate value from a rescaled Beta prior."""

    alpha, beta = beta_alpha_beta(mu_unit, nu)
    unit = torch.distributions.Beta(alpha, beta).sample()
    return lo + unit * (hi - lo)


def beta_logprior_on_grid(
    grid_native: torch.Tensor,
    mu_unit: torch.Tensor,
    nu: torch.Tensor,
    lo: float,
    hi: float,
) -> torch.Tensor:
    """Native-coordinate Beta prior log density on a grid."""

    width = hi - lo
    unit = ((grid_native - lo) / width).clamp(1e-6, 1.0 - 1e-6)
    alpha, beta = beta_alpha_beta(mu_unit, nu)
    return torch.distributions.Beta(alpha, beta).log_prob(unit) - math.log(width)


def sample_contaminated(
    mu_unit: torch.Tensor,
    nu: torch.Tensor,
    lo: float,
    hi: float,
    eps: float,
) -> torch.Tensor:
    """Draw a native value from an epsilon-contaminated Beta prior.

    The effective prior is `(1 - eps) * Beta(mu_unit, nu) + eps * Uniform[lo, hi]`,
    a classic robust-Bayes / epsilon-contamination prior. The *token* still encodes
    only the raw Beta (see the prior helpers above); contamination lives only here,
    in the truth draw, so the trained model learns a posterior whose uniform floor
    keeps it from ever fully committing to a confident-but-wrong location. This is
    the one thing `sample_prior_params` cannot do: it always draws truth from the
    same Beta the token encodes, so the token never lies.
    """

    beta_draw = draw_from_beta(mu_unit, nu, lo, hi)
    uniform_draw = torch.empty_like(beta_draw).uniform_(lo, hi)
    contaminate = torch.rand_like(beta_draw) < eps
    return torch.where(contaminate, uniform_draw, beta_draw)
