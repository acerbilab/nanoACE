"""Executable 1D Bayesian optimization example for nanoACE.

Problem: infer the location `x_opt` and value `y_opt` of a black-box function's
global minimum from a few noisy observations. These latents are properties of
the specific sampled function, not class-level GP hyperparameters. Every batch
includes Beta information tokens for the optimum latents; truth is drawn from an
epsilon-contaminated prior so a concentrated but wrong runtime prior still has a
uniform probability floor.

Data-generating process: sample nuisance GP kernel and hyperparameters, draw
`x_opt` and `y_opt`, draw a natural optimum depth `d`, sample a GP draw
conditioned on `g(x_opt) = d` by Matheron's rule, then plant the optimum with a
fold and convex envelope:

    f(x) = |g_c(x) - d| + ENVELOPE * (x - x_opt)^2 + y_opt

Both added terms are >= 0 and vanish together only at x = x_opt, and the
envelope is strictly positive off x_opt, so x_opt is the exact unique global
minimum with value y_opt. The `|.|` fold gives the kinked, multi-basin geometry
and destroys Gaussianity, so this example deliberately has no grid oracle.
Diagnostics instead check token scale and compare ACE under uniform, correct,
and wrong runtime priors on the same fixed observation.

GP sampling and optimum planting use CPU float64. ACE training and prediction
run on the selected device.

File layout:
1. constants and small dataclasses;
2. `variables()` schema, token constructor, and scaling/prior helpers;
3. planted-optimum GP data-generating process and ACE batch construction;
4. no-oracle diagnostics: scale check plus uniform/correct/wrong prior contrast;
5. model construction, training, checkpoint helpers;
6. fixed evaluation, printed metrics, and plot;
7. CLI entry point.

Task-specific generation and diagnostics live here; reusable ACE machinery
stays in `ace.py`, `ace_prior_beta.py`, and `diagnostics.py`.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch

import data
import train
from ace import ACE, Batch, PRIOR, PRIOR_FEATURES, QUERY, VALUE, Tokens, Variable, encode_value, sample_reveal_mask
from ace_prior_beta import (
    beta_logprior_on_grid,
    known_latent_features,
    prior_features,
    sample_contaminated,
    sample_prior_params,
)
from diagnostics import conditional_log_density, normalized_moments, query_log_density


KERNELS = ("RBF", "Matern12", "Matern32", "Matern52")
KERNEL_WEIGHTS = (0.35, 0.1, 0.2, 0.35)
ELL_MEAN = 1.0 / 3.0
ELL_STD = 0.75
ELL_RANGE = (0.05, 2.0)

# x_opt is a bounded latent over [-1, 1] (encode_value is then identity).
X_OPT_RANGE = (-1.0, 1.0)
# y_opt and data `y` are the *same* physical quantity (function values), so they
# share one affine: Y_RANGE is the y_opt latent bounds AND the data-`y` scaling
# (frozen for checkpoint compatibility -- not a CLI arg). Y_OPT_RANGE is the
# sub-interval the optimum *value* prior lives on; the function bump sits on top,
# so Y_RANGE upper = Y_OPT_RANGE upper + the bump budget.
Y_OPT_RANGE = (-1.0, 0.0)
Y_RANGE = (-1.0, 2.0)
ENVELOPE = 0.2  # paper's 1/5 convex envelope constant
D_CAP = 2.0  # cap |natural optimum depth| so it cannot inflate the function height
N_TOTAL = 64  # data observation points per instance (context + targets); the pool's [n, N_TOTAL] data width
# Frozen DGP constants for an offline pool (also the matching CLI defaults, so online and
# the pool share one source of truth for the data-generating process).
GEN_SIGMA_OBS = 0.02
GEN_SIGMA_F_MAX = 0.5
GEN_PRIOR_UNIFORM_MIX = 0.05  # epsilon for the contaminated prior
GEN_JITTER = 1e-5

# Fixed diagnostic case (seeded). The context points are sparse and not at the
# optimum, so the location prior visibly matters.
EVAL_X_OPT = 0.40
EVAL_Y_OPT = -0.60
EVAL_KERNEL = 0  # RBF
EVAL_ELL = 0.30
EVAL_SIGMA_F = 0.40
EVAL_DEPTH = -0.70
EVAL_CONTEXT_X = (-0.80, -0.50, 0.10, 0.70)
EVAL_SEED = 20260607
# Observation noise for the fixed eval case (kept separate so the seeded
# function/noise are stable regardless of the CLI `--sigma-obs`).
EVAL_SIGMA_OBS = 0.02

# x_opt prior settings for the three diagnostic columns, as (mu_unit, nu) over
# X_OPT_RANGE. "correct" concentrates near the true x_opt; "wrong" on the far
# side -- the epsilon floor should let the data recover it anyway.
PRIOR_KINDS = ("uniform", "correct", "wrong")
EVAL_X_OPT_PRIORS = {
    "uniform": (0.5, 2.0),
    "correct": (0.70, 25.0),
    "wrong": (0.18, 25.0),
}
# y_opt prior is uninformative (Beta(1, 1)) in all eval columns.
EVAL_Y_OPT_PRIOR = (0.5, 2.0)


@dataclass
class BOBatch:
    """A BO ACE batch plus the sampled latents and prior hyperparameters.

    `x`/`y` are kept in native coordinates here; the token tensors inside `batch`
    hold the scaled/encoded versions.
    """

    batch: Batch
    x_context: torch.Tensor
    y_context: torch.Tensor
    x_target: torch.Tensor
    y_target: torch.Tensor
    x_opt: torch.Tensor
    y_opt: torch.Tensor
    x_opt_prior_unit: torch.Tensor
    x_opt_prior_nu: torch.Tensor
    y_opt_prior_unit: torch.Tensor
    y_opt_prior_nu: torch.Tensor


@dataclass
class PriorResult:
    """ACE diagnostics for one x_opt prior setting (no oracle)."""

    toy: BOBatch
    x_grid: torch.Tensor
    x_logp: torch.Tensor
    y_grid: torch.Tensor
    y_logp: torch.Tensor
    x_given_y_grid: torch.Tensor
    x_given_y_logp: torch.Tensor
    pred_x: torch.Tensor
    pred_mean: torch.Tensor
    pred_std: torch.Tensor
    true_x: torch.Tensor
    true_f: torch.Tensor
    metrics: dict[str, float]


@dataclass
class Diagnostic:
    """The three x_opt prior settings, keyed by name in `PRIOR_KINDS`."""

    results: dict[str, PriorResult]


def variables() -> list[Variable]:
    """Schema for the function observations and the two optimum latents."""

    return [
        Variable("y", "data", "continuous"),
        Variable("x_opt", "latent", "continuous", bounds=X_OPT_RANGE),
        Variable("y_opt", "latent", "continuous", bounds=Y_RANGE),
    ]


def make_tokens(
    *,
    var_id: torch.Tensor,
    value: torch.Tensor,
    mode: torch.Tensor,
    mask: torch.Tensor,
    x: torch.Tensor | None = None,
    value_index: torch.Tensor | None = None,
    prior: torch.Tensor | None = None,
) -> Tokens:
    """Construct BO tokens, keeping the data `x` covariate explicit."""

    b, t = var_id.shape
    device = value.device
    if x is None:
        x = torch.zeros(b, t, 1, device=device, dtype=value.dtype)
    if value_index is None:
        value_index = torch.zeros(b, t, device=device, dtype=torch.long)
    if prior is None:
        prior = torch.zeros(b, t, PRIOR_FEATURES, device=device, dtype=value.dtype)
    return Tokens(
        var_id=var_id.long(),
        x=x,
        value=value,
        value_index=value_index.long(),
        prior=prior,
        mode=mode.long(),
        mask=mask.bool(),
    )


def scale_y(y: torch.Tensor) -> torch.Tensor:
    """Map a native function value to its token coordinate over `Y_RANGE`.

    This is exactly `encode_value` for the `y_opt` latent (whose bounds are
    `Y_RANGE`), so data `y` and `y_opt` live on one ruler and `y_opt <= all y`
    is legible to the model.
    """

    lo, hi = Y_RANGE
    return 2.0 * (y - lo) / (hi - lo) - 1.0


def unscale_y(v: torch.Tensor) -> torch.Tensor:
    """Invert `scale_y`."""

    lo, hi = Y_RANGE
    return lo + 0.5 * (v + 1.0) * (hi - lo)


def y_opt_prior_features(mu_unit: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Encode a Beta prior over `Y_OPT_RANGE` as a `y_opt` information token.

    The `y_opt` latent's bounds are the wider `Y_RANGE`, but its prior lives on
    the `Y_OPT_RANGE` sub-interval, so the token's `(mean, spread)` are computed
    in `Y_RANGE` internal coordinates directly rather than via `prior_features`
    (which assumes the Beta spans the full bounds).
    """

    a, b = Y_OPT_RANGE
    lo, hi = Y_RANGE
    mean_native = a + mu_unit * (b - a)
    var_unit = mu_unit * (1.0 - mu_unit) / (nu + 1.0)
    std_native = var_unit.sqrt() * (b - a)
    mean_internal = 2.0 * (mean_native - lo) / (hi - lo) - 1.0
    spread_internal = std_native * 2.0 / (hi - lo)
    return torch.stack([mean_internal, spread_internal], dim=-1)


def mixture_logprior_on_grid(
    grid_native: torch.Tensor,
    mu_unit: torch.Tensor,
    nu: torch.Tensor,
    lo: float,
    hi: float,
    eps: float,
) -> torch.Tensor:
    """Native log density of `(1 - eps) Beta + eps Uniform[lo, hi]` for plotting.

    Single consumer (the diagnostic overlay), so it lives here rather than in
    `ace_prior_beta.py`. The grid must lie within `[lo, hi]`.
    """

    beta_logp = beta_logprior_on_grid(grid_native, mu_unit, nu, lo, hi)
    uniform_logp = torch.full_like(beta_logp, -math.log(hi - lo))
    weighted = torch.stack(
        [
            beta_logp + math.log(1.0 - eps),
            uniform_logp + math.log(eps),
        ],
        dim=0,
    )
    return torch.logsumexp(weighted, dim=0)


def _sample_lengthscale(batch_size: int) -> torch.Tensor:
    """Truncated-normal lengthscale `N(ELL_MEAN, ELL_STD)` on `ELL_RANGE` (CPU)."""

    ell = torch.empty(batch_size, dtype=torch.float64).normal_(ELL_MEAN, ELL_STD)
    for _ in range(8):
        bad = (ell < ELL_RANGE[0]) | (ell > ELL_RANGE[1])
        if not bool(bad.any()):
            break
        ell[bad] = torch.empty(int(bad.sum()), dtype=torch.float64).normal_(ELL_MEAN, ELL_STD)
    return ell.clamp(*ELL_RANGE)


def _sample_depth(sigma_f: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
    """Sample the natural optimum depth `d` from the min-value distribution (CPU).

    `d` is the minimum of `N = ceil(2 / ell)` draws from `N(0, sigma_f^2)` (the
    approximate number of uncorrelated GP samples across the width-2 domain),
    sampled exactly via the min CDF and the inverse normal CDF. With probability
    0.1 an extra `Exp(1)` deepens it (the paper's "unexpectedly low optimum").
    Clamped to `[-D_CAP, 0]` so the sign is a genuine dip and `|d|` cannot
    inflate the function height.
    """

    n = torch.ceil(2.0 / ell).clamp_min(1.0)
    u = torch.rand_like(sigma_f)
    z = torch.special.ndtri(1.0 - (1.0 - u).pow(1.0 / n))
    d = sigma_f * z
    kick = torch.rand_like(sigma_f) < 0.1
    exp_draw = -torch.log1p(-torch.rand_like(sigma_f))
    d = torch.where(kick, d - exp_draw, d)
    return d.clamp(-D_CAP, 0.0)


def _kernel_covariance(
    x_left: torch.Tensor,
    x_right: torch.Tensor,
    kernel: torch.Tensor,
    ell: torch.Tensor,
    sigma_f: torch.Tensor,
) -> torch.Tensor:
    """Batch of GP cross-covariance matrices on CPU float64 tensors."""

    r = (x_left[:, :, None] - x_right[:, None, :]).abs()
    ee = ell[:, None, None].clamp_min(1e-6)
    amp2 = sigma_f.pow(2)[:, None, None]
    out = torch.empty_like(r)
    for idx, name in enumerate(KERNELS):
        sel = kernel == idx
        if not bool(sel.any()):
            continue
        rr = r[sel]
        e = ee[sel]
        if name == "RBF":
            base = torch.exp(-0.5 * (rr / e).pow(2))
        elif name == "Matern12":
            base = torch.exp(-rr / e)
        elif name == "Matern32":
            z = math.sqrt(3.0) * rr / e
            base = (1.0 + z) * torch.exp(-z)
        elif name == "Matern52":
            z = math.sqrt(5.0) * rr / e
            base = (1.0 + z + z.pow(2) / 3.0) * torch.exp(-z)
        else:
            raise ValueError(f"unknown kernel {name}")
        out[sel] = amp2[sel] * base
    return out


def _planted_function(
    x_opt: torch.Tensor,
    x_eval: torch.Tensor,
    kernel: torch.Tensor,
    ell: torch.Tensor,
    sigma_f: torch.Tensor,
    depth: torch.Tensor,
    y_opt: torch.Tensor,
    *,
    jitter: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Evaluate the planted-optimum function `f` at `x_eval`, batched on CPU f64.

    Draws a joint GP sample over `{x_opt} u x_eval`, conditions it to pass through
    `depth` at `x_opt` (Matheron's rule -- a true posterior sample), folds and
    adds the convex envelope and the `y_opt` level. The returned values are the
    *clean* function; observation noise is added by the caller.
    """

    x_all = torch.cat([x_opt[:, None], x_eval], dim=1)  # x_opt is index 0
    k = _kernel_covariance(x_all, x_all, kernel, ell, sigma_f)
    n = x_all.shape[1]
    k = k + jitter * torch.eye(n, dtype=k.dtype)
    chol = torch.linalg.cholesky(k)
    eps = torch.randn(x_all.shape[0], n, 1, dtype=k.dtype, generator=generator)
    g = torch.bmm(chol, eps).squeeze(-1)
    # Matheron pin: g_c(x) = g(x) - k(x, x_opt)/k(x_opt, x_opt) * (g(x_opt) - depth).
    k_xo = k[:, :, 0]
    weight = k_xo / k[:, 0, 0:1]
    g_c = g - weight * (g[:, 0:1] - depth[:, None])
    g_eval = g_c[:, 1:]
    x_eval_rel = x_eval - x_opt[:, None]
    return g_eval.sub(depth[:, None]).abs() + ENVELOPE * x_eval_rel.pow(2) + y_opt[:, None]


def draw_instances(
    n_instances: int,
    *,
    n_points: int,
    prior_uniform_mix: float,
    sigma_obs: float,
    sigma_f_max: float,
    jitter: float,
) -> dict[str, torch.Tensor]:
    """Draw the expensive BO physics for `n_instances` tasks (CPU float64, no tokens).

    Struct-of-arrays, `n_points` observation points each. RNG draw order is
    `prior params x/y -> contaminated x_opt/y_opt -> kernel -> ell -> sigma_f -> depth ->
    x_data -> planted f -> noise`; keep it stable so the online path (post per-step reseed)
    matches the old monolithic sampler. Nuisance kernel/ell/sigma_f/depth are consumed into
    `y` and not returned (so a `data.py` pool stores no categorical field). The split /
    reveal / tokenization is `assemble`'s job.
    """

    x_unit, x_nu = sample_prior_params((n_instances,), device="cpu")
    y_unit, y_nu = sample_prior_params((n_instances,), device="cpu")
    x_opt = sample_contaminated(x_unit, x_nu, *X_OPT_RANGE, prior_uniform_mix).double()
    y_opt = sample_contaminated(y_unit, y_nu, *Y_OPT_RANGE, prior_uniform_mix).double()

    kernel = torch.multinomial(torch.tensor(KERNEL_WEIGHTS), n_instances, replacement=True)
    ell = _sample_lengthscale(n_instances)
    sigma_f = torch.empty(n_instances, dtype=torch.float64).uniform_(0.1, sigma_f_max)
    depth = _sample_depth(sigma_f, ell)

    x = 2.0 * torch.rand(n_instances, n_points, dtype=torch.float64) - 1.0
    f = _planted_function(x_opt, x, kernel, ell, sigma_f, depth, y_opt, jitter=jitter)
    y = f + sigma_obs * torch.randn(n_instances, n_points, dtype=torch.float64)
    return {
        "x": x,
        "y": y,
        "x_opt": x_opt,
        "y_opt": y_opt,
        "x_unit": x_unit,
        "x_nu": x_nu,
        "y_unit": y_unit,
        "y_nu": y_nu,
    }


def assemble(
    inst: dict[str, torch.Tensor],
    *,
    variables: list[Variable],
    n_context: torch.Tensor,
    reveal_mask: torch.Tensor,
    max_context: int,
    device: torch.device | str,
) -> Batch:
    """Tokenize drawn BO instances into an ACE `Batch` (RNG-free).

    `n_context` (`[B]`) and `reveal_mask` (`[B, 2]`) are decided by the caller -- the online
    path from the global RNG, the offline `data.py` reader from a stateless index hash -- so
    this is deterministic given its inputs and shared by both. The first `max_context` points
    are context candidates (first `n_context` active); **targets are all non-context points**
    (`n_target = n_points - n_context`), the target block masked `>= n_context`. The two
    optimum latents always carry a Beta PRIOR token (zero-spread known token when revealed).
    CPU-native `inst` physics is moved to `device` as float32 here.
    """

    device = torch.device(device)
    b = int(inst["x"].shape[0])
    n_points = int(inst["x"].shape[1])
    if not 1 <= max_context < n_points:
        raise ValueError(f"need 1 <= max_context ({max_context}) < n_points ({n_points}) for >=1 target")
    x_data_d = inst["x"].float().to(device)
    y_native_d = inst["y"].float().to(device)
    x_opt_d = inst["x_opt"].float().to(device)
    y_opt_d = inst["y_opt"].float().to(device)
    x_opt_internal = encode_value(variables[1], x_opt_d)
    y_opt_internal = encode_value(variables[2], y_opt_d)
    x_unit_d, x_nu_d = inst["x_unit"].to(device), inst["x_nu"].to(device)
    y_unit_d, y_nu_d = inst["y_unit"].to(device), inst["y_nu"].to(device)

    reveal_x, reveal_y = reveal_mask[:, 0], reveal_mask[:, 1]

    # Context: first max_context points (first n_context active) + the 2 optimum PRIOR tokens.
    ctx_t = max_context + 2
    x_pos, y_pos = max_context, max_context + 1
    ctx_var = torch.zeros(b, ctx_t, device=device, dtype=torch.long)
    ctx_var[:, x_pos] = 1
    ctx_var[:, y_pos] = 2
    ctx_x = torch.zeros(b, ctx_t, 1, device=device)
    ctx_x[:, :max_context, 0] = x_data_d[:, :max_context]
    ctx_value = torch.zeros(b, ctx_t, device=device)
    ctx_value[:, :max_context] = scale_y(y_native_d[:, :max_context])
    ctx_value[:, x_pos] = x_opt_internal
    ctx_value[:, y_pos] = y_opt_internal
    ctx_prior = torch.zeros(b, ctx_t, PRIOR_FEATURES, device=device)
    ctx_prior[:, x_pos] = prior_features(x_unit_d, x_nu_d)
    ctx_prior[:, y_pos] = y_opt_prior_features(y_unit_d, y_nu_d)
    ctx_prior[:, x_pos] = torch.where(reveal_x[:, None], known_latent_features(x_opt_internal), ctx_prior[:, x_pos])
    ctx_prior[:, y_pos] = torch.where(reveal_y[:, None], known_latent_features(y_opt_internal), ctx_prior[:, y_pos])
    ctx_mode = torch.full((b, ctx_t), VALUE, device=device)
    ctx_mode[:, x_pos] = PRIOR
    ctx_mode[:, y_pos] = PRIOR
    ctx_mask = torch.zeros(b, ctx_t, device=device, dtype=torch.bool)
    ctx_ar = torch.arange(max_context, device=device)[None, :]
    ctx_mask[:, :max_context] = ctx_ar < n_context[:, None]
    ctx_mask[:, x_pos] = True
    ctx_mask[:, y_pos] = True
    context = make_tokens(var_id=ctx_var, x=ctx_x, value=ctx_value, mode=ctx_mode, mask=ctx_mask, prior=ctx_prior)

    # Target: all non-context points (mask >= n_context) plus the 2 optimum queries.
    tgt_t = n_points + 2
    tgt_var = torch.zeros(b, tgt_t, device=device, dtype=torch.long)
    tgt_var[:, 0] = 1
    tgt_var[:, 1] = 2
    tgt_x = torch.zeros(b, tgt_t, 1, device=device)
    tgt_x[:, 2:, 0] = x_data_d
    tgt_value = torch.zeros(b, tgt_t, device=device)
    tgt_value[:, 0] = x_opt_internal
    tgt_value[:, 1] = y_opt_internal
    tgt_value[:, 2:] = scale_y(y_native_d)
    tgt_mask = torch.ones(b, tgt_t, device=device, dtype=torch.bool)
    tgt_mask[:, 0] = ~reveal_x
    tgt_mask[:, 1] = ~reveal_y
    tgt_ar = torch.arange(n_points, device=device)[None, :]
    tgt_mask[:, 2:] = tgt_ar >= n_context[:, None]
    target = make_tokens(
        var_id=tgt_var,
        x=tgt_x,
        value=tgt_value,
        mode=torch.full((b, tgt_t), QUERY, device=device),
        mask=tgt_mask,
    )
    return Batch(variables, context, target)


def online_batch(model: ACE, args: argparse.Namespace, device: torch.device | str) -> Batch:
    """Draw + assemble one online BO training batch (global RNG; see `assemble`)."""

    inst = draw_instances(
        args.batch_size,
        n_points=N_TOTAL,
        prior_uniform_mix=args.prior_uniform_mix,
        sigma_obs=args.sigma_obs,
        sigma_f_max=args.sigma_f_max,
        jitter=args.jitter,
    )
    n_context = torch.randint(args.min_context, args.max_context + 1, (args.batch_size,), device=device)
    reveal_mask = sample_reveal_mask(2, args.batch_size, q=1.0 - args.latent_context_prob, device=device)
    return assemble(
        inst,
        variables=model.variables,
        n_context=n_context,
        reveal_mask=reveal_mask,
        max_context=args.max_context,
        device=device,
    )


def gen_config() -> dict:
    """Frozen DGP constants that define an offline pool's identity (hashed; drift => regenerate)."""

    return {
        "kernels": list(KERNELS),
        "kernel_weights": list(KERNEL_WEIGHTS),
        "ell_mean": ELL_MEAN,
        "ell_std": ELL_STD,
        "ell_range": list(ELL_RANGE),
        "x_opt_range": list(X_OPT_RANGE),
        "y_opt_range": list(Y_OPT_RANGE),
        "y_range": list(Y_RANGE),
        "envelope": ENVELOPE,
        "d_cap": D_CAP,
        "sigma_f_max": GEN_SIGMA_F_MAX,
        "sigma_obs": GEN_SIGMA_OBS,
        "eps": GEN_PRIOR_UNIFORM_MIX,
        "N_TOTAL": N_TOTAL,
        "jitter": GEN_JITTER,
    }


def draw_pool(n_instances: int) -> dict[str, torch.Tensor]:
    """`draw_instances` bound to the frozen pool DGP config (used by `data.write_pool`)."""

    return draw_instances(
        n_instances,
        n_points=N_TOTAL,
        prior_uniform_mix=GEN_PRIOR_UNIFORM_MIX,
        sigma_obs=GEN_SIGMA_OBS,
        sigma_f_max=GEN_SIGMA_F_MAX,
        jitter=GEN_JITTER,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> ACE:
    """Load a BO checkpoint (2-arg wrapper over `train.load_checkpoint`).

    Kept at this 2-arg signature because `playground/export_weights.py` and
    `playground/parity.py` call `bo1d.load_checkpoint(path, device)` directly.
    """

    return train.load_checkpoint(path, device, variables())


def fixed_eval_batch(vars_: list[Variable], *, device: torch.device | str, points: int, prior_kind: str, jitter: float) -> BOBatch:
    """Build the fixed BO diagnostic case under one x_opt prior setting.

    The observation (context locations, true latents, hyperparameters, seeded GP
    draw and noise) is identical across prior kinds; only the x_opt Beta token
    differs, so the three posteriors are directly comparable. The target is a
    dense x grid for a smooth posterior-predictive curve.
    """

    device = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(EVAL_SEED)
    x_opt = torch.tensor([EVAL_X_OPT], dtype=torch.float64)
    y_opt = torch.tensor([EVAL_Y_OPT], dtype=torch.float64)
    kernel = torch.tensor([EVAL_KERNEL], dtype=torch.long)
    ell = torch.tensor([EVAL_ELL], dtype=torch.float64)
    sigma_f = torch.tensor([EVAL_SIGMA_F], dtype=torch.float64)
    depth = torch.tensor([EVAL_DEPTH], dtype=torch.float64)

    x_ctx = torch.tensor([list(EVAL_CONTEXT_X)], dtype=torch.float64)
    x_tgt = torch.linspace(-1.0, 1.0, points, dtype=torch.float64)[None, :]
    x_all = torch.cat([x_ctx, x_tgt], dim=1)
    f_all = _planted_function(x_opt, x_all, kernel, ell, sigma_f, depth, y_opt, jitter=jitter, generator=gen)
    n = x_ctx.shape[1]
    f_ctx = f_all[:, :n]
    f_tgt = f_all[:, n:]
    noise = torch.randn(f_ctx.shape, generator=gen, dtype=torch.float64)
    y_ctx = f_ctx + noise * EVAL_SIGMA_OBS

    x_prior = EVAL_X_OPT_PRIORS[prior_kind]
    y_prior = EVAL_Y_OPT_PRIOR

    x_ctx_d = x_ctx.float().to(device)
    y_ctx_d = y_ctx.float().to(device)
    x_tgt_d = x_tgt.float().to(device)
    f_tgt_d = f_tgt.float().to(device)
    x_opt_d = x_opt.float().to(device)
    y_opt_d = y_opt.float().to(device)
    x_opt_internal = encode_value(vars_[1], x_opt_d)
    y_opt_internal = encode_value(vars_[2], y_opt_d)
    x_unit = torch.tensor([x_prior[0]], device=device)
    x_nu = torch.tensor([x_prior[1]], device=device)
    y_unit = torch.tensor([y_prior[0]], device=device)
    y_nu = torch.tensor([y_prior[1]], device=device)

    ctx_t = n + 2
    x_pos, y_pos = n, n + 1
    ctx_var = torch.zeros(1, ctx_t, device=device, dtype=torch.long)
    ctx_var[:, x_pos] = 1
    ctx_var[:, y_pos] = 2
    ctx_x = torch.zeros(1, ctx_t, 1, device=device)
    ctx_x[:, :n, 0] = x_ctx_d
    ctx_value = torch.zeros(1, ctx_t, device=device)
    ctx_value[:, :n] = scale_y(y_ctx_d)
    ctx_value[:, x_pos] = x_opt_internal
    ctx_value[:, y_pos] = y_opt_internal
    ctx_prior = torch.zeros(1, ctx_t, PRIOR_FEATURES, device=device)
    ctx_prior[:, x_pos] = prior_features(x_unit, x_nu)
    ctx_prior[:, y_pos] = y_opt_prior_features(y_unit, y_nu)
    ctx_mode = torch.full((1, ctx_t), VALUE, device=device)
    ctx_mode[:, x_pos] = PRIOR
    ctx_mode[:, y_pos] = PRIOR
    context = make_tokens(
        var_id=ctx_var,
        x=ctx_x,
        value=ctx_value,
        mode=ctx_mode,
        mask=torch.ones(1, ctx_t, device=device, dtype=torch.bool),
        prior=ctx_prior,
    )
    target = make_tokens(
        var_id=torch.zeros(1, points, device=device, dtype=torch.long),
        x=x_tgt_d[..., None],
        value=scale_y(f_tgt_d),
        mode=torch.full((1, points), QUERY, device=device),
        mask=torch.ones(1, points, device=device, dtype=torch.bool),
    )
    return BOBatch(
        Batch(vars_, context, target),
        x_ctx_d,
        y_ctx_d,
        x_tgt_d,
        f_tgt_d,
        x_opt_d,
        y_opt_d,
        x_unit,
        x_nu,
        y_unit,
        y_nu,
    )


def _evaluate_one(model: ACE, args: argparse.Namespace, prior_kind: str) -> PriorResult:
    """Run the fixed diagnostic under one x_opt prior setting (no oracle)."""

    device = next(model.parameters()).device
    toy = fixed_eval_batch(model.variables, device=device, points=args.eval_points, prior_kind=prior_kind, jitter=args.jitter)

    pred = model(toy.batch)
    pred_mean = unscale_y(pred.mean(toy.batch.target)[0])
    pred_std = pred.continuous_var()[0].clamp_min(1e-8).sqrt() * (0.5 * (Y_RANGE[1] - Y_RANGE[0]))

    x_grid = torch.linspace(X_OPT_RANGE[0], X_OPT_RANGE[1], args.bins, device=device)
    y_grid = torch.linspace(Y_OPT_RANGE[0], Y_OPT_RANGE[1], args.bins, device=device)
    x_logp = query_log_density(model, toy.batch, 1, encode_value(model.variables[1], x_grid))
    y_logp = query_log_density(model, toy.batch, 2, encode_value(model.variables[2], y_grid))

    # Conditional p(x_opt | y_opt = true, D), to gesture at Thompson sampling.
    y_cond = torch.tensor([float(toy.y_opt[0])], device=device)
    x_given_y_logp = conditional_log_density(
        model,
        toy.batch,
        known_var=2,
        known_values=encode_value(model.variables[2], y_cond),
        query_var=1,
        query_values=encode_value(model.variables[1], x_grid),
    )[0]

    x_mean, x_std = normalized_moments(x_grid, x_logp)
    y_mean, y_std = normalized_moments(y_grid, y_logp)
    metrics = {
        "x_opt_mean": float(x_mean),
        "x_opt_std": float(x_std),
        "y_opt_mean": float(y_mean),
        "y_opt_std": float(y_std),
        "x_opt_true": float(toy.x_opt[0]),
        "y_opt_true": float(toy.y_opt[0]),
        "pred_rmse": float((pred_mean - toy.y_target[0]).pow(2).mean().sqrt()),
    }

    print(f"\nBO diagnostic [{prior_kind} prior on x_opt]")
    print(f"truth x_opt     {float(toy.x_opt[0]): .3f}")
    print(f"ACE x_opt       mean {float(x_mean): .3f}  std {float(x_std): .3f}")
    print(f"truth y_opt     {float(toy.y_opt[0]): .3f}")
    print(f"ACE y_opt       mean {float(y_mean): .3f}  std {float(y_std): .3f}")
    print(f"predictive      rmse {metrics['pred_rmse']: .3f}")
    return PriorResult(
        toy,
        x_grid,
        x_logp,
        y_grid,
        y_logp,
        x_grid,
        x_given_y_logp,
        toy.x_target[0],
        pred_mean,
        pred_std,
        toy.x_target[0],
        toy.y_target[0],
        metrics,
    )


@torch.no_grad()
def evaluate(model: ACE, args: argparse.Namespace) -> Diagnostic:
    """Run the fixed BO diagnostic under each x_opt prior setting and print metrics."""

    return Diagnostic({kind: _evaluate_one(model, args, kind) for kind in PRIOR_KINDS})


def plot_diagnostic(diag: Diagnostic, path: str | Path, *, eps: float) -> None:
    """Save the three-column prior-contrast figure (no oracle)."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_kinds = len(PRIOR_KINDS)
    fig = plt.figure(figsize=(4.2 * n_kinds, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, n_kinds, height_ratios=[1.2, 1.0, 1.0, 1.0])

    for col, kind in enumerate(PRIOR_KINDS):
        res = diag.results[kind]
        toy = res.toy
        x = toy.x_target[0].detach().cpu()
        f_true = res.true_f.detach().cpu()
        x_ctx = toy.x_context[0].detach().cpu()
        y_ctx = toy.y_context[0].detach().cpu()
        pred_mean = res.pred_mean.detach().cpu()
        pred_std = res.pred_std.detach().cpu()
        true_x_opt = float(toy.x_opt[0])
        true_y_opt = float(toy.y_opt[0])

        # Row 0: function + predictive band.
        ax_f = fig.add_subplot(gs[0, col])
        ax_f.plot(x, f_true, color="0.25", linewidth=1.3, label="true f")
        ax_f.plot(x, pred_mean, color="tab:blue", linewidth=1.4, label="ACE mean")
        ax_f.fill_between(x, pred_mean - 2 * pred_std, pred_mean + 2 * pred_std, color="tab:blue", alpha=0.15)
        ax_f.scatter(x_ctx, y_ctx, color="black", s=24, zorder=3, label="context")
        ax_f.scatter([true_x_opt], [true_y_opt], color="tab:red", marker="*", s=120, zorder=4, label="optimum")
        ax_f.set_title(f"{kind} prior")
        if col == 0:
            ax_f.set_ylabel("y")
        ax_f.legend(fontsize=7)

        # Row 1: p(x_opt | D) with the effective (contaminated) prior overlay.
        x_grid = res.x_grid.detach().cpu()
        x_p = (res.x_logp - torch.logsumexp(res.x_logp, dim=0)).exp().detach().cpu()
        prior_logp = mixture_logprior_on_grid(
            x_grid, toy.x_opt_prior_unit.detach().cpu()[0], toy.x_opt_prior_nu.detach().cpu()[0], *X_OPT_RANGE, eps
        )
        prior_p = (prior_logp - torch.logsumexp(prior_logp, dim=0)).exp()
        ax_x = fig.add_subplot(gs[1, col])
        ax_x.plot(x_grid, prior_p, color="0.4", linestyle=":", label="prior (eff.)")
        ax_x.plot(x_grid, x_p, color="tab:blue", label="ACE")
        ax_x.axvline(true_x_opt, color="tab:red", alpha=0.5)
        if col == 0:
            ax_x.set_ylabel("p(x_opt | D)")
        ax_x.legend(fontsize=7)

        # Row 2: p(y_opt | D).
        y_grid = res.y_grid.detach().cpu()
        y_p = (res.y_logp - torch.logsumexp(res.y_logp, dim=0)).exp().detach().cpu()
        ax_y = fig.add_subplot(gs[2, col])
        ax_y.plot(y_grid, y_p, color="tab:green", label="ACE")
        ax_y.axvline(true_y_opt, color="tab:red", alpha=0.5)
        if col == 0:
            ax_y.set_ylabel("p(y_opt | D)")
        ax_y.legend(fontsize=7)

        # Row 3: conditional p(x_opt | y_opt = true, D) (Thompson-style query).
        xy_p = (res.x_given_y_logp - torch.logsumexp(res.x_given_y_logp, dim=0)).exp().detach().cpu()
        ax_xy = fig.add_subplot(gs[3, col])
        ax_xy.plot(x_grid, xy_p, color="tab:purple", label="ACE")
        ax_xy.axvline(true_x_opt, color="tab:red", alpha=0.5)
        ax_xy.set_xlabel("x_opt")
        if col == 0:
            ax_xy.set_ylabel("p(x_opt | y_opt, D)")
        ax_xy.legend(fontsize=7)

    u = diag.results["uniform"].metrics
    c = diag.results["correct"].metrics
    w = diag.results["wrong"].metrics
    fig.suptitle(
        f"x_opt std: uniform {u['x_opt_std']:.3f} -> correct {c['x_opt_std']:.3f}   "
        f"x_opt mean: wrong {w['x_opt_mean']:.3f} (truth {u['x_opt_true']:.2f})"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved diagnostic plot: {path}")


def scale_check(args: argparse.Namespace) -> None:
    """Sample a batch and report token-value spread + contamination marginal.

    The first gate (no oracle): confirm data token values sit ~[-1, 1] and the
    drawn x_opt marginal for a fixed informative token shows the epsilon floor.
    """

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    inst = draw_instances(
        4096,
        n_points=N_TOTAL,
        prior_uniform_mix=args.prior_uniform_mix,
        sigma_obs=args.sigma_obs,
        sigma_f_max=args.sigma_f_max,
        jitter=args.jitter,
    )
    y_tok = scale_y(inst["y"].float().to(device))
    q = torch.tensor([0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0], device=y_tok.device)
    quant = torch.quantile(y_tok.flatten(), q)
    frac_out = float((y_tok.abs() > 1.0).float().mean())
    print("scale check (data y token values)")
    print("  quantiles [min .1% 1% 50% 99% 99.9% max]:")
    print("  " + "  ".join(f"{float(v): .3f}" for v in quant))
    print(f"  fraction |token| > 1: {frac_out:.4f}")
    print(f"  native y_opt range used: {Y_OPT_RANGE}, data Y_RANGE: {Y_RANGE}")
    print(f"  x_opt in [-1,1]: {float((inst['x_opt'].abs() <= 1).float().mean()):.4f}")
    print(f"  y_opt in Y_OPT_RANGE: {float(((inst['y_opt'] >= Y_OPT_RANGE[0]) & (inst['y_opt'] <= Y_OPT_RANGE[1])).float().mean()):.4f}")

    # Contamination marginal: fixed concentrated token, many truth draws.
    mu = torch.full((200000,), 0.75)
    nu = torch.full((200000,), 25.0)
    draws = sample_contaminated(mu, nu, *X_OPT_RANGE, args.prior_uniform_mix)
    tail = float(((draws > 0.6) & (draws < 1.0)).float().mean())  # near the Beta mode
    spread = float(((draws > -1.0) & (draws < 0.0)).float().mean())  # the uniform floor region
    print("contamination marginal (token Beta mu_unit=0.75, nu=25):")
    print(f"  mass near Beta mode (0.6,1.0): {tail:.3f}   mass in wrong half (-1,0): {spread:.3f}")
    print(f"  expected uniform floor in wrong half ~= eps/2 = {args.prior_uniform_mix/2:.3f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the BO example."""

    p = argparse.ArgumentParser(parents=[train.common_parser()], description="Train/evaluate the nanoACE 1D BO example.")
    # Targets are all non-context points (complement-targets); N_TOTAL is the point budget.
    # `--data-targets` is inherited from common_parser but unused by BO (no-op).
    p.set_defaults(
        batch_size=64,
        max_context=20,
        min_context=1,
        d_model=192,
        heads=16,
        layers=6,
        hidden=384,
        components=12,
        plot_path="artifacts/bo1d.png",
    )
    p.add_argument("--eval-points", type=int, default=160)
    p.add_argument("--bins", type=int, default=80, help="diagnostic grid bins")
    p.add_argument("--prior-uniform-mix", type=float, default=GEN_PRIOR_UNIFORM_MIX, help="epsilon for the contaminated prior")
    p.add_argument("--sigma-obs", type=float, default=GEN_SIGMA_OBS)
    p.add_argument("--sigma-f-max", type=float, default=GEN_SIGMA_F_MAX)
    p.add_argument("--jitter", type=float, default=GEN_JITTER)
    p.add_argument("--scale-check", action="store_true", help="sample a batch and report token scale, then exit")
    p.add_argument("--pool", default="", help="train from an offline data.py pool directory instead of online")
    p.add_argument("--pool-force", action="store_true", help="reuse a pool despite a DGP config-hash mismatch")
    return train.apply_config_file(p)


def main() -> None:
    """Run BO training/evaluation from the command line."""

    args = parse_args()
    if args.scale_check:
        scale_check(args)
        return

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    if args.eval_only and not args.load_checkpoint:
        raise SystemExit("--eval-only requires --load-checkpoint")

    if args.load_checkpoint:
        model = load_checkpoint(args.load_checkpoint, device)
    elif args.resume:
        model = load_checkpoint(args.resume, device)
    else:
        model = train.build_model(args, variables(), device)

    if not args.eval_only:
        resume_state = (
            torch.load(args.resume, map_location=device, weights_only=False) if args.resume else None
        )
        if args.pool:
            _frozen = {"jitter": GEN_JITTER, "sigma_obs": GEN_SIGMA_OBS,
                       "sigma_f_max": GEN_SIGMA_F_MAX, "prior_uniform_mix": GEN_PRIOR_UNIFORM_MIX}
            _bad = {k: getattr(args, k) for k, v in _frozen.items() if getattr(args, k) != v}
            if _bad:
                raise SystemExit(
                    f"--pool freezes the DGP; these flags differ from the pool's gen_config and would only "
                    f"affect diagnostics, not the cached data: {_bad}. Regenerate the pool or drop --pool."
                )
            source = data.PoolReader(
                args.pool,
                assemble=assemble,
                variables=model.variables,
                gen_config=gen_config(),
                batch_size=args.batch_size,
                seed=args.seed,
                max_context=args.max_context,
                min_context=args.min_context,
                latent_context_prob=args.latent_context_prob,
                device=device,
                force=args.pool_force,
            )
        else:
            source = lambda step: online_batch(model, args, device)
        model = train.fit(
            model,
            source,
            train.TrainConfig.from_args(args),
            resume_state=resume_state,
            seed=args.seed,
            checkpoint_path=args.save_checkpoint or None,
            ckpt_every=args.ckpt_every,
        )

    diag = evaluate(model, args)
    if args.save_checkpoint:
        train.save_checkpoint(args.save_checkpoint, model, seed=args.seed, config=vars(args))
    if not args.no_plot and args.plot_path:
        plot_diagnostic(diag, args.plot_path, eps=args.prior_uniform_mix)


if __name__ == "__main__":
    main()
