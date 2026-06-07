"""Executable SIR simulation-based-inference example for nanoACE.

This is the simulation-based inference (SBI) example: recover the contact rate
`beta` and recovery rate `gamma` of an epidemic from a noisily observed infected
fraction over time. It is a fusion of the two existing examples:

- runtime **Beta prior injection (ACEP)** like `gaussian_toy.py` -- one PRIOR
  token is always emitted per continuous latent (`Beta(1, 1)` = uniform when
  uninformative);
- **online time-series simulation, 1D-indexed (time) data tokens, and a grid
  oracle** like `gp1d.py`.

The simulator is the deterministic SIR ODE with Gaussian observation noise.
Because the trajectory is deterministic given `(beta, gamma)`, the marginal
likelihood is a product of Gaussian observation densities, so the `(beta,
gamma)` grid posterior is tractable -- the same recipe as `gp1d.py::gp_oracle`.
ACE itself only ever sees simulator draws, never the likelihood. The diagnostic
contrasts a uniform prior against an informative prior on the same observation
to show runtime prior conditioning at work.

ODE integration uses CPU float64; ACE itself runs on the selected device.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ace import ACE, ACEConfig, Batch, PRIOR, PRIOR_FEATURES, QUERY, VALUE, Tokens, Variable, encode_value
from ace_prior import beta_logprior_on_grid, draw_from_beta, known_latent_features, prior_features, sample_prior_params
from diagnostics import normalized_moments, query_log_density


BETA_RANGE = (0.1, 0.8)
GAMMA_RANGE = (0.04, 0.4)
I0 = 0.01
T_MAX = 40.0
T_OBS = 25
FINE_STEPS = 400
SIGMA_OBS = 0.02
DATA_LOC = 0.2
DATA_SCALE = 0.2

PRIOR_KINDS = ("uniform", "informative")
UNIFORM_PRIOR = (0.5, 2.0)

EVAL_BETA = 0.55
EVAL_GAMMA = 0.18
EVAL_BETA_PRIOR = (0.60, 12.0)
EVAL_GAMMA_PRIOR = (0.45, 10.0)
# Sparse, rise-phase observations: early epidemic data identifies the growth
# rate but leaves a broad beta/gamma ridge, so the runtime prior visibly helps.
EVAL_CONTEXT_TIMES = (3.0, 6.0, 9.0, 12.0)
EVAL_SEED = 20260607


@dataclass
class SIRBatch:
    """An SIR ACE batch plus the sampled latents and prior hyperparameters.

    Times and infected-fraction values are kept in native coordinates here; the
    token tensors inside `batch` hold the scaled/encoded versions.
    """

    batch: Batch
    t_context: torch.Tensor
    y_context: torch.Tensor
    t_target: torch.Tensor
    y_target: torch.Tensor
    beta: torch.Tensor
    gamma: torch.Tensor
    beta_prior_unit: torch.Tensor
    beta_prior_nu: torch.Tensor
    gamma_prior_unit: torch.Tensor
    gamma_prior_nu: torch.Tensor


@dataclass
class SIROracle:
    """Grid posterior and posterior predictive for one fixed SIR diagnostic."""

    beta_grid: torch.Tensor
    beta_probs: torch.Tensor
    gamma_grid: torch.Tensor
    gamma_probs: torch.Tensor
    corr: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor


@dataclass
class PriorResult:
    """ACE-vs-oracle comparison for one prior setting (uniform or informative)."""

    toy: SIRBatch
    oracle: SIROracle
    beta_grid: torch.Tensor
    beta_logp: torch.Tensor
    gamma_grid: torch.Tensor
    gamma_logp: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor
    metrics: dict[str, float]


@dataclass
class Diagnostic:
    """The two prior settings, keyed by name in `PRIOR_KINDS`."""

    results: dict[str, PriorResult]


def variables() -> list[Variable]:
    """Schema for the infected-fraction observations and the two rate latents."""

    return [
        Variable("y", "data", "continuous"),
        Variable("beta", "latent", "continuous", bounds=BETA_RANGE),
        Variable("gamma", "latent", "continuous", bounds=GAMMA_RANGE),
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
    """Construct SIR tokens, keeping the data `x` (time) covariate explicit."""

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


def scale_time(t: torch.Tensor) -> torch.Tensor:
    """Map native time in `[0, T_MAX]` to the `[-1, 1]` token covariate."""

    return 2.0 * t / T_MAX - 1.0


def scale_value(i: torch.Tensor) -> torch.Tensor:
    """Map a native infected fraction to a roughly `[-1, 1]` token value."""

    return (i - DATA_LOC) / DATA_SCALE


def unscale_value(v: torch.Tensor) -> torch.Tensor:
    """Invert `scale_value`."""

    return v * DATA_SCALE + DATA_LOC


def _sir_deriv(s: torch.Tensor, i: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """SIR fraction-coordinate derivatives (r = 1 - s - i is implicit)."""

    infection = beta * s * i
    return -infection, infection - gamma * i


def _integrate_fine(beta: torch.Tensor, gamma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched RK4 integration of the SIR ODE over `[0, T_MAX]` on CPU float64.

    Returns the fine time grid and the infected-fraction trajectory of shape
    `(B, FINE_STEPS + 1)`. The ODE is non-stiff for the bounded rate ranges, so
    a fixed-step RK4 is comfortably stable.
    """

    beta = beta.reshape(-1).double()
    gamma = gamma.reshape(-1).double()
    b = beta.shape[0]
    dt = T_MAX / FINE_STEPS
    fine_t = torch.linspace(0.0, T_MAX, FINE_STEPS + 1, dtype=torch.float64)
    s = torch.full((b,), 1.0 - I0, dtype=torch.float64)
    i = torch.full((b,), I0, dtype=torch.float64)
    traj = torch.empty(b, FINE_STEPS + 1, dtype=torch.float64)
    traj[:, 0] = i
    for step in range(FINE_STEPS):
        ds1, di1 = _sir_deriv(s, i, beta, gamma)
        ds2, di2 = _sir_deriv(s + 0.5 * dt * ds1, i + 0.5 * dt * di1, beta, gamma)
        ds3, di3 = _sir_deriv(s + 0.5 * dt * ds2, i + 0.5 * dt * di2, beta, gamma)
        ds4, di4 = _sir_deriv(s + dt * ds3, i + dt * di3, beta, gamma)
        s = (s + (dt / 6.0) * (ds1 + 2.0 * ds2 + 2.0 * ds3 + ds4)).clamp(0.0, 1.0)
        i = (i + (dt / 6.0) * (di1 + 2.0 * di2 + 2.0 * di3 + di4)).clamp_min(0.0)
        traj[:, step + 1] = i
    return fine_t, traj


def _interp(fine_t: torch.Tensor, i_fine: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of `i_fine` (B, N) at native `times` (T,)."""

    times = times.double().clamp(float(fine_t[0]), float(fine_t[-1]))
    idx = torch.searchsorted(fine_t, times).clamp(1, fine_t.numel() - 1)
    t0 = fine_t[idx - 1]
    t1 = fine_t[idx]
    w = ((times - t0) / (t1 - t0)).clamp(0.0, 1.0)
    i0 = i_fine[:, idx - 1]
    i1 = i_fine[:, idx]
    return i0 + (i1 - i0) * w[None, :]


def integrate_sir(beta: torch.Tensor, gamma: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """Infected fraction at native `times` for each `(beta, gamma)`, shape (B, T)."""

    fine_t, i_fine = _integrate_fine(beta, gamma)
    return _interp(fine_t, i_fine, times)


def _prior_pair(kind: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the `(beta_prior, gamma_prior)` `(mu_unit, nu)` pairs for a kind."""

    if kind == "informative":
        return EVAL_BETA_PRIOR, EVAL_GAMMA_PRIOR
    if kind == "uniform":
        return UNIFORM_PRIOR, UNIFORM_PRIOR
    raise ValueError(f"unknown prior kind {kind!r}")


def sample_sir_batch(
    vars_: list[Variable],
    *,
    batch_size: int,
    max_context: int,
    min_context: int,
    data_targets: int,
    device: torch.device | str,
    latent_context_prob: float,
    sigma_obs: float,
) -> SIRBatch:
    """Sample one online SIR training batch.

    Each element observes the epidemic at the fixed `T_OBS` evenly spaced times,
    then a random permutation splits those points into context and data targets
    (so the split is varied, like the random GP locations in `gp1d.py`). Both
    rate latents always appear in context as Beta PRIOR tokens; with probability
    `latent_context_prob` one is revealed as a zero-spread (known) token.
    """

    if max_context + data_targets > T_OBS:
        raise ValueError(f"max_context + data_targets must be <= T_OBS ({T_OBS})")
    device = torch.device(device)

    beta_unit, beta_nu = sample_prior_params((batch_size,), device=device)
    gamma_unit, gamma_nu = sample_prior_params((batch_size,), device=device)
    beta = draw_from_beta(beta_unit, beta_nu, *BETA_RANGE)
    gamma = draw_from_beta(gamma_unit, gamma_nu, *GAMMA_RANGE)
    beta_internal = encode_value(vars_[1], beta)
    gamma_internal = encode_value(vars_[2], gamma)

    times = torch.linspace(0.0, T_MAX, T_OBS, device=device)
    i_traj = integrate_sir(beta.detach().cpu(), gamma.detach().cpu(), times.detach().cpu())
    i_traj = i_traj.float().to(device)
    y_native = i_traj + sigma_obs * torch.randn(batch_size, T_OBS, device=device)

    perm = torch.argsort(torch.rand(batch_size, T_OBS, device=device), dim=1)
    t_perm = times[perm]
    y_perm = torch.gather(y_native, 1, perm)
    x_tok = scale_time(t_perm)
    v_tok = scale_value(y_perm)

    ctx_data_slice = slice(0, max_context)
    tgt_data_slice = slice(max_context, max_context + data_targets)
    t_ctx = t_perm[:, ctx_data_slice]
    y_ctx = y_perm[:, ctx_data_slice]
    t_tgt = t_perm[:, tgt_data_slice]
    y_tgt = y_perm[:, tgt_data_slice]

    n_ctx = torch.randint(min_context, max_context + 1, (batch_size,), device=device)
    ar = torch.arange(max_context, device=device)[None, :]
    reveal = torch.rand(batch_size, device=device) < latent_context_prob
    reveal_beta = reveal & (torch.rand(batch_size, device=device) < 0.5)
    reveal_gamma = reveal & ~reveal_beta

    ctx_t = max_context + 2
    beta_pos, gamma_pos = max_context, max_context + 1
    ctx_var = torch.zeros(batch_size, ctx_t, device=device, dtype=torch.long)
    ctx_var[:, beta_pos] = 1
    ctx_var[:, gamma_pos] = 2
    ctx_x = torch.zeros(batch_size, ctx_t, 1, device=device)
    ctx_x[:, ctx_data_slice, 0] = x_tok[:, ctx_data_slice]
    ctx_value = torch.zeros(batch_size, ctx_t, device=device)
    ctx_value[:, ctx_data_slice] = v_tok[:, ctx_data_slice]
    ctx_value[:, beta_pos] = beta_internal
    ctx_value[:, gamma_pos] = gamma_internal
    ctx_prior = torch.zeros(batch_size, ctx_t, PRIOR_FEATURES, device=device)
    ctx_prior[:, beta_pos] = prior_features(beta_unit, beta_nu)
    ctx_prior[:, gamma_pos] = prior_features(gamma_unit, gamma_nu)
    ctx_prior[:, beta_pos] = torch.where(reveal_beta[:, None], known_latent_features(beta_internal), ctx_prior[:, beta_pos])
    ctx_prior[:, gamma_pos] = torch.where(reveal_gamma[:, None], known_latent_features(gamma_internal), ctx_prior[:, gamma_pos])
    ctx_mode = torch.full((batch_size, ctx_t), VALUE, device=device)
    ctx_mode[:, beta_pos] = PRIOR
    ctx_mode[:, gamma_pos] = PRIOR
    ctx_mask = torch.zeros(batch_size, ctx_t, device=device, dtype=torch.bool)
    ctx_mask[:, ctx_data_slice] = ar < n_ctx[:, None]
    ctx_mask[:, beta_pos] = True
    ctx_mask[:, gamma_pos] = True
    context = make_tokens(var_id=ctx_var, x=ctx_x, value=ctx_value, mode=ctx_mode, mask=ctx_mask, prior=ctx_prior)

    tgt_t = 2 + data_targets
    tgt_var = torch.zeros(batch_size, tgt_t, device=device, dtype=torch.long)
    tgt_var[:, 0] = 1
    tgt_var[:, 1] = 2
    tgt_x = torch.zeros(batch_size, tgt_t, 1, device=device)
    tgt_x[:, 2:, 0] = x_tok[:, tgt_data_slice]
    tgt_value = torch.zeros(batch_size, tgt_t, device=device)
    tgt_value[:, 0] = beta_internal
    tgt_value[:, 1] = gamma_internal
    tgt_value[:, 2:] = v_tok[:, tgt_data_slice]
    tgt_mask = torch.ones(batch_size, tgt_t, device=device, dtype=torch.bool)
    tgt_mask[:, 0] = ~reveal_beta
    tgt_mask[:, 1] = ~reveal_gamma
    target = make_tokens(
        var_id=tgt_var,
        x=tgt_x,
        value=tgt_value,
        mode=torch.full((batch_size, tgt_t), QUERY, device=device),
        mask=tgt_mask,
    )
    return SIRBatch(
        Batch(vars_, context, target),
        t_ctx,
        y_ctx,
        t_tgt,
        y_tgt,
        beta,
        gamma,
        beta_unit,
        beta_nu,
        gamma_unit,
        gamma_nu,
    )


def build_model(args, device: torch.device) -> ACE:
    """Construct the SIR ACE model from CLI hyperparameters."""

    cfg = ACEConfig(
        x_dim=1,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        mlp_hidden=args.hidden,
        head_hidden=args.hidden,
        mdn_components=args.components,
    )
    return ACE(variables(), cfg).to(device)


def train(args: argparse.Namespace, model: ACE | None = None) -> ACE:
    """Train ACE online on freshly simulated SIR batches."""

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = build_model(args, device) if model is None else model
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        toy = sample_sir_batch(
            model.variables,
            batch_size=args.batch_size,
            max_context=args.max_context,
            min_context=args.min_context,
            data_targets=args.data_targets,
            device=device,
            latent_context_prob=args.latent_context_prob,
            sigma_obs=args.sigma_obs,
        )
        loss = model.loss(toy.batch, latent_weight=args.latent_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0:
            print(f"step {step:5d}/{args.steps}  loss {loss.item():.4f}")
    return model


def save_checkpoint(model: ACE, path: str | Path, args: argparse.Namespace) -> None:
    """Save a lightweight SIR checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": asdict(model.cfg), "seed": args.seed, "state_dict": model.state_dict()}, path)
    print(f"saved checkpoint: {path}")


def load_checkpoint(path: str | Path, device: torch.device) -> ACE:
    """Load an SIR checkpoint saved by `save_checkpoint`."""

    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = ACEConfig(**payload["cfg"])
    model = ACE(variables(), cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    return model


def fixed_eval_batch(vars_: list[Variable], *, device: torch.device | str, points: int, prior_kind: str) -> SIRBatch:
    """Build the fixed SIR diagnostic case under one prior setting.

    The observation (context times, true latents, seeded noise) is identical
    across prior kinds; only the Beta PRIOR tokens differ, so the two posteriors
    are directly comparable. The target is a dense time grid for a smooth
    posterior-predictive curve.
    """

    gen = torch.Generator(device="cpu").manual_seed(EVAL_SEED)
    beta = torch.tensor([EVAL_BETA], dtype=torch.float64)
    gamma = torch.tensor([EVAL_GAMMA], dtype=torch.float64)
    t_ctx = torch.tensor(EVAL_CONTEXT_TIMES, dtype=torch.float64)
    t_tgt = torch.linspace(0.0, T_MAX, points, dtype=torch.float64)

    fine_t, i_fine = _integrate_fine(beta, gamma)
    i_ctx = _interp(fine_t, i_fine, t_ctx)
    i_tgt = _interp(fine_t, i_fine, t_tgt)
    noise = torch.randn(i_ctx.shape, generator=gen, dtype=torch.float64)
    y_ctx = i_ctx + SIGMA_OBS * noise

    (beta_prior, gamma_prior) = _prior_pair(prior_kind)

    device = torch.device(device)
    beta_d = beta.float().to(device)
    gamma_d = gamma.float().to(device)
    beta_internal = encode_value(vars_[1], beta_d)
    gamma_internal = encode_value(vars_[2], gamma_d)
    beta_prior_unit = torch.tensor([beta_prior[0]], device=device)
    beta_prior_nu = torch.tensor([beta_prior[1]], device=device)
    gamma_prior_unit = torch.tensor([gamma_prior[0]], device=device)
    gamma_prior_nu = torch.tensor([gamma_prior[1]], device=device)

    t_ctx_d = t_ctx.float().to(device)[None, :]
    y_ctx_d = y_ctx.float().to(device)
    t_tgt_d = t_tgt.float().to(device)[None, :]
    i_tgt_d = i_tgt.float().to(device)

    n = t_ctx.numel()
    ctx_t = n + 2
    beta_pos, gamma_pos = n, n + 1
    ctx_var = torch.zeros(1, ctx_t, device=device, dtype=torch.long)
    ctx_var[:, beta_pos] = 1
    ctx_var[:, gamma_pos] = 2
    ctx_x = torch.zeros(1, ctx_t, 1, device=device)
    ctx_x[:, :n, 0] = scale_time(t_ctx_d)
    ctx_value = torch.zeros(1, ctx_t, device=device)
    ctx_value[:, :n] = scale_value(y_ctx_d)
    ctx_value[:, beta_pos] = beta_internal
    ctx_value[:, gamma_pos] = gamma_internal
    ctx_prior = torch.zeros(1, ctx_t, PRIOR_FEATURES, device=device)
    ctx_prior[:, beta_pos] = prior_features(beta_prior_unit, beta_prior_nu)
    ctx_prior[:, gamma_pos] = prior_features(gamma_prior_unit, gamma_prior_nu)
    ctx_mode = torch.full((1, ctx_t), VALUE, device=device)
    ctx_mode[:, beta_pos] = PRIOR
    ctx_mode[:, gamma_pos] = PRIOR
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
        x=scale_time(t_tgt_d)[..., None],
        value=scale_value(i_tgt_d),
        mode=torch.full((1, points), QUERY, device=device),
        mask=torch.ones(1, points, device=device, dtype=torch.bool),
    )
    return SIRBatch(
        Batch(vars_, context, target),
        t_ctx_d,
        y_ctx_d,
        t_tgt_d,
        i_tgt_d,
        beta_d,
        gamma_d,
        beta_prior_unit,
        beta_prior_nu,
        gamma_prior_unit,
        gamma_prior_nu,
    )


def _moments_from_probs(grid: torch.Tensor, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean/std for normalized probability mass on a grid."""

    mean = (probs * grid).sum()
    std = (probs * (grid - mean).pow(2)).sum().sqrt()
    return mean, std


def sir_oracle(toy: SIRBatch, *, bins: int, sigma_obs: float) -> SIROracle:
    """Numerically integrate the SIR posterior over `(beta, gamma)` on a grid."""

    if bins < 2:
        raise ValueError("SIR oracle needs at least two grid bins")

    t_ctx = toy.t_context[0].detach().cpu().double()
    y_ctx = toy.y_context[0].detach().cpu().double()
    t_tgt = toy.t_target[0].detach().cpu().double()

    beta_grid = torch.linspace(BETA_RANGE[0], BETA_RANGE[1], bins, dtype=torch.float64)
    gamma_grid = torch.linspace(GAMMA_RANGE[0], GAMMA_RANGE[1], bins, dtype=torch.float64)
    bb, gg = torch.meshgrid(beta_grid, gamma_grid, indexing="ij")
    flat_beta = bb.reshape(-1)
    flat_gamma = gg.reshape(-1)

    fine_t, i_fine = _integrate_fine(flat_beta, flat_gamma)
    i_ctx = _interp(fine_t, i_fine, t_ctx)
    loglike = (
        -0.5 * ((y_ctx[None, :] - i_ctx) / sigma_obs).pow(2)
        - math.log(sigma_obs)
        - 0.5 * math.log(2.0 * math.pi)
    ).sum(dim=1)

    logprior_beta = beta_logprior_on_grid(beta_grid, toy.beta_prior_unit.detach().cpu()[0], toy.beta_prior_nu.detach().cpu()[0], *BETA_RANGE)
    logprior_gamma = beta_logprior_on_grid(gamma_grid, toy.gamma_prior_unit.detach().cpu()[0], toy.gamma_prior_nu.detach().cpu()[0], *GAMMA_RANGE)
    logprior = (logprior_beta[:, None] + logprior_gamma[None, :]).reshape(-1)

    logpost = loglike + logprior
    post = (logpost - torch.logsumexp(logpost, dim=0)).exp().reshape(bins, bins)
    beta_probs = post.sum(dim=1)
    gamma_probs = post.sum(dim=0)
    beta_mean = (beta_probs * beta_grid).sum()
    gamma_mean = (gamma_probs * gamma_grid).sum()
    beta_std = (beta_probs * (beta_grid - beta_mean).pow(2)).sum().sqrt()
    gamma_std = (gamma_probs * (gamma_grid - gamma_mean).pow(2)).sum().sqrt()
    cov = (post * (beta_grid[:, None] - beta_mean) * (gamma_grid[None, :] - gamma_mean)).sum()
    corr = cov / (beta_std * gamma_std).clamp_min(1e-8)

    i_tgt = _interp(fine_t, i_fine, t_tgt)
    w = post.reshape(-1)[:, None]
    mean = (w * i_tgt).sum(dim=0)
    second = (w * (sigma_obs ** 2 + i_tgt.pow(2))).sum(dim=0)
    std = (second - mean.pow(2)).clamp_min(1e-12).sqrt()
    return SIROracle(beta_grid, beta_probs, gamma_grid, gamma_probs, corr, mean, std)


def _evaluate_one(model: ACE, args: argparse.Namespace, prior_kind: str) -> PriorResult:
    """Run the fixed diagnostic under one prior setting."""

    device = next(model.parameters()).device
    toy = fixed_eval_batch(model.variables, device=device, points=args.eval_points, prior_kind=prior_kind)
    oracle = sir_oracle(toy, bins=args.bins, sigma_obs=args.sigma_obs)

    pred = model(toy.batch)
    y_mean = unscale_value(pred.mean(toy.batch.target)[0])
    y_std = pred.continuous_var()[0].clamp_min(1e-8).sqrt() * DATA_SCALE

    beta_grid = torch.linspace(BETA_RANGE[0], BETA_RANGE[1], args.bins, device=device)
    gamma_grid = torch.linspace(GAMMA_RANGE[0], GAMMA_RANGE[1], args.bins, device=device)
    beta_logp = query_log_density(model, toy.batch, 1, encode_value(model.variables[1], beta_grid))
    gamma_logp = query_log_density(model, toy.batch, 2, encode_value(model.variables[2], gamma_grid))
    beta_mean, beta_std = normalized_moments(beta_grid, beta_logp)
    gamma_mean, gamma_std = normalized_moments(gamma_grid, gamma_logp)
    oracle_beta_mean, oracle_beta_std = _moments_from_probs(oracle.beta_grid, oracle.beta_probs)
    oracle_gamma_mean, oracle_gamma_std = _moments_from_probs(oracle.gamma_grid, oracle.gamma_probs)

    y_true = toy.y_target[0]
    rmse = (y_mean - y_true).pow(2).mean().sqrt()
    oracle_rmse = (oracle.y_mean.to(device) - y_true).pow(2).mean().sqrt()
    metrics = {
        "beta_mean_abs_err": float((beta_mean - oracle_beta_mean.to(device)).abs()),
        "beta_std_abs_err": float((beta_std - oracle_beta_std.to(device)).abs()),
        "gamma_mean_abs_err": float((gamma_mean - oracle_gamma_mean.to(device)).abs()),
        "gamma_std_abs_err": float((gamma_std - oracle_gamma_std.to(device)).abs()),
        "y_rmse": float(rmse),
        "oracle_y_rmse": float(oracle_rmse),
        "oracle_corr": float(oracle.corr),
        "beta_mean": float(beta_mean),
        "beta_std": float(beta_std),
        "gamma_mean": float(gamma_mean),
        "gamma_std": float(gamma_std),
        "oracle_beta_mean": float(oracle_beta_mean),
        "oracle_beta_std": float(oracle_beta_std),
        "oracle_gamma_mean": float(oracle_gamma_mean),
        "oracle_gamma_std": float(oracle_gamma_std),
    }

    print(f"\nSIR diagnostic [{prior_kind} prior]")
    print(f"truth beta      {float(toy.beta[0]): .3f}")
    print(f"oracle beta     mean {float(oracle_beta_mean): .3f}  std {float(oracle_beta_std): .3f}")
    print(f"ACE beta        mean {float(beta_mean): .3f}  std {float(beta_std): .3f}")
    print(f"truth gamma     {float(toy.gamma[0]): .3f}")
    print(f"oracle gamma    mean {float(oracle_gamma_mean): .3f}  std {float(oracle_gamma_std): .3f}")
    print(f"ACE gamma       mean {float(gamma_mean): .3f}  std {float(gamma_std): .3f}")
    print(f"oracle corr     {float(oracle.corr): .3f}")
    print(f"target i(t)     oracle rmse {float(oracle_rmse): .3f}  ACE rmse {float(rmse): .3f}")
    return PriorResult(toy, oracle, beta_grid, beta_logp, gamma_grid, gamma_logp, y_mean, y_std, metrics)


@torch.no_grad()
def evaluate(model: ACE, args: argparse.Namespace) -> Diagnostic:
    """Run the fixed SIR diagnostic under each prior setting and print metrics."""

    return Diagnostic({kind: _evaluate_one(model, args, kind) for kind in PRIOR_KINDS})


def _plot_marginal(ax, grid, prior_unit, prior_nu, rng, oracle_probs, ace_logp, truth, title) -> None:
    """Plot one latent marginal: runtime prior, oracle posterior, ACE posterior."""

    prior_logp = beta_logprior_on_grid(grid, prior_unit, prior_nu, *rng)
    prior_p = (prior_logp - torch.logsumexp(prior_logp, dim=0)).exp()
    ace_p = (ace_logp - torch.logsumexp(ace_logp, dim=0)).exp()
    ax.plot(grid, prior_p, color="0.4", linestyle=":", label="prior")
    ax.plot(grid, oracle_probs, color="tab:green", label="oracle")
    ax.plot(grid, ace_p, color="tab:blue", linestyle="--", label="ACE")
    ax.axvline(truth, color="0.2", alpha=0.4)
    ax.set_title(title)
    ax.legend(fontsize=8)


def plot_diagnostic(diag: Diagnostic, path: str | Path) -> None:
    """Save the uniform-vs-informative prior contrast figure."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.1])

    for col, kind in enumerate(PRIOR_KINDS):
        res = diag.results[kind]
        toy = res.oracle
        ax_b = fig.add_subplot(gs[0, col])
        ax_g = fig.add_subplot(gs[1, col])
        _plot_marginal(
            ax_b,
            toy.beta_grid.detach().cpu(),
            res.toy.beta_prior_unit.detach().cpu()[0],
            res.toy.beta_prior_nu.detach().cpu()[0],
            BETA_RANGE,
            toy.beta_probs.detach().cpu(),
            res.beta_logp.detach().cpu(),
            float(res.toy.beta[0]),
            f"beta marginal ({kind})",
        )
        ax_b.set_xlabel("beta")
        _plot_marginal(
            ax_g,
            toy.gamma_grid.detach().cpu(),
            res.toy.gamma_prior_unit.detach().cpu()[0],
            res.toy.gamma_prior_nu.detach().cpu()[0],
            GAMMA_RANGE,
            toy.gamma_probs.detach().cpu(),
            res.gamma_logp.detach().cpu(),
            float(res.toy.gamma[0]),
            f"gamma marginal ({kind})",
        )
        ax_g.set_xlabel("gamma")

    res = diag.results["informative"]
    t = res.toy.t_target[0].detach().cpu()
    i_true = res.toy.y_target[0].detach().cpu()
    t_ctx = res.toy.t_context[0].detach().cpu()
    y_ctx = res.toy.y_context[0].detach().cpu()
    oracle_mean = res.oracle.y_mean.detach().cpu()
    oracle_std = res.oracle.y_std.detach().cpu()
    ace_mean = res.y_mean.detach().cpu()
    ace_std = res.y_std.detach().cpu()

    ax_pred = fig.add_subplot(gs[2, :])
    ax_pred.plot(t, i_true, color="0.25", linewidth=1.4, label="true i(t)")
    ax_pred.plot(t, oracle_mean, color="tab:green", linewidth=1.5, label="oracle mean")
    ax_pred.fill_between(t, oracle_mean - 2.0 * oracle_std, oracle_mean + 2.0 * oracle_std, color="tab:green", alpha=0.14)
    ax_pred.plot(t, ace_mean, color="tab:blue", linewidth=1.5, label="ACE mean")
    ax_pred.fill_between(t, ace_mean - 2.0 * ace_std, ace_mean + 2.0 * ace_std, color="tab:blue", alpha=0.16)
    ax_pred.scatter(t_ctx, y_ctx, color="black", s=28, zorder=3, label="context")
    ax_pred.set_title("posterior predictive (informative prior)")
    ax_pred.set_xlabel("time")
    ax_pred.set_ylabel("infected fraction")
    ax_pred.legend(loc="best", fontsize=8)

    info = diag.results["informative"].metrics
    unif = diag.results["uniform"].metrics
    fig.suptitle(
        f"beta std: uniform {unif['beta_std']:.3f} -> informative {info['beta_std']:.3f}   "
        f"gamma std: uniform {unif['gamma_std']:.3f} -> informative {info['gamma_std']:.3f}"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved diagnostic plot: {path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the SIR example."""

    p = argparse.ArgumentParser(description="Train/evaluate the nanoACE SIR SBI example.")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-context", type=int, default=12)
    p.add_argument("--min-context", type=int, default=4)
    p.add_argument("--data-targets", type=int, default=13)
    p.add_argument("--eval-points", type=int, default=120)
    p.add_argument("--bins", type=int, default=64, help="oracle/diagnostic grid bins")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--components", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--latent-weight", type=float, default=2.0)
    p.add_argument("--latent-context-prob", type=float, default=0.20)
    p.add_argument("--sigma-obs", type=float, default=SIGMA_OBS)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--plot-path", default="artifacts/sbi_sir.png")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--save-checkpoint", default="")
    p.add_argument("--load-checkpoint", default="")
    p.add_argument("--eval-only", action="store_true")
    return p.parse_args()


def main() -> None:
    """Run SIR training/evaluation from the command line."""

    args = parse_args()
    device = torch.device(args.device)
    if args.load_checkpoint:
        model = load_checkpoint(args.load_checkpoint, device)
    elif args.eval_only:
        raise SystemExit("--eval-only requires --load-checkpoint")
    else:
        model = None

    if not args.eval_only:
        model = train(args, model)
    assert model is not None

    diag = evaluate(model, args)
    if args.save_checkpoint:
        save_checkpoint(model, args.save_checkpoint, args)
    if not args.no_plot and args.plot_path:
        plot_diagnostic(diag, args.plot_path)


if __name__ == "__main__":
    main()
