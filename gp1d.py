"""Executable 1D Gaussian-process regression example for nanoACE.

Problem: infer a sampled function's GP kernel family, log lengthscale, and log
output scale from irregular 1D observations, while also predicting function
values at target locations. The continuous hyperparameter latents are bounded
and encoded to ACE token coordinates; the kernel latent is discrete. Revealed
continuous latents are zero-spread PRIOR tokens, and a revealed kernel is a
VALUE class-label token. This example does not use finite-spread runtime priors.

GP function sampling and the diagnostic oracle use CPU float64 Cholesky. ACE
training and prediction run on the selected device. The fixed diagnostic
numerically integrates over kernel and hyperparameter grids and mixes GP
posterior predictives under those weights.

File layout:
1. constants and small dataclasses;
2. `variables()` schema and token constructor;
3. GP kernels, sampler, and ACE batch construction;
4. numerical grid oracle over kernel and hyperparameters;
5. model construction, training, checkpoint helpers;
6. fixed evaluation, printed metrics, and plot;
7. CLI entry point.

Task-specific generation and diagnostics live here; reusable ACE machinery
stays in `ace.py` and `diagnostics.py`.
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
from diagnostics import normalized_moments, query_log_density, repeat_tokens


KERNELS = ("RBF", "Matern12", "Matern32", "Periodic")
LOG_LENGTHSCALE_RANGE = (math.log(0.12), math.log(0.80))
LOG_OUTPUTSCALE_RANGE = (math.log(0.25), math.log(1.00))
N_TOTAL = 64  # data observation points per instance (context + targets); the pool's [n, N_TOTAL] data width
GEN_JITTER = 1e-5  # frozen Cholesky jitter for an offline pool (also the --jitter CLI default)
EVAL_KERNEL = 3
EVAL_LOG_LENGTHSCALE = math.log(0.28)
EVAL_LOG_OUTPUTSCALE = math.log(0.75)
EVAL_SEED = 20260606


@dataclass
class GPBatch:
    """A GP-1D ACE batch plus the sampled latent values."""

    batch: Batch
    x_context: torch.Tensor
    y_context: torch.Tensor
    x_target: torch.Tensor
    y_target: torch.Tensor
    log_lengthscale: torch.Tensor
    log_outputscale: torch.Tensor
    kernel: torch.Tensor


@dataclass
class Diagnostic:
    """Model predictions for the fixed GP-1D diagnostic problem."""

    toy: GPBatch
    y_mean: torch.Tensor
    y_std: torch.Tensor
    ell_grid: torch.Tensor
    ell_logp: torch.Tensor
    scale_grid: torch.Tensor
    scale_logp: torch.Tensor
    kernel_probs: torch.Tensor
    oracle: "GPOracle"
    metrics: dict[str, float]


@dataclass
class GPOracle:
    """Grid posterior and posterior predictive for the fixed GP diagnostic."""

    kernel_log_marginal: torch.Tensor
    kernel_probs: torch.Tensor
    ell_grid: torch.Tensor
    ell_probs: torch.Tensor
    scale_grid: torch.Tensor
    scale_probs: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor


def variables() -> list[Variable]:
    """Schema for GP observations and the three task latents."""

    return [
        Variable("y", "data", "continuous"),
        Variable("log_lengthscale", "latent", "continuous", transform="log", bounds=LOG_LENGTHSCALE_RANGE),
        Variable("log_outputscale", "latent", "continuous", transform="log", bounds=LOG_OUTPUTSCALE_RANGE),
        Variable("kernel", "latent", "discrete", cardinality=len(KERNELS)),
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
    """Construct GP tokens, keeping data `x` and discrete labels explicit."""

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


def _kernel_covariance(
    x_left: torch.Tensor,
    x_right: torch.Tensor,
    kernel: torch.Tensor,
    log_lengthscale: torch.Tensor,
    log_outputscale: torch.Tensor,
) -> torch.Tensor:
    """Batch of GP cross-covariance matrices on CPU float64 tensors."""

    r = (x_left[:, :, None] - x_right[:, None, :]).abs()
    ell = log_lengthscale.exp()[:, None, None].clamp_min(1e-6)
    amp2 = log_outputscale.exp().pow(2)[:, None, None]
    mats = torch.empty_like(r)

    for idx, name in enumerate(KERNELS):
        sel = kernel == idx
        if not bool(sel.any()):
            continue
        rr = r[sel]
        ee = ell[sel]
        if name == "RBF":
            base = torch.exp(-0.5 * (rr / ee).pow(2))
        elif name == "Matern12":
            base = torch.exp(-rr / ee)
        elif name == "Matern32":
            z = math.sqrt(3.0) * rr / ee
            base = (1.0 + z) * torch.exp(-z)
        elif name == "Periodic":
            period = 1.0
            base = torch.exp(-2.0 * torch.sin(math.pi * rr / period).pow(2) / ee.pow(2))
        else:
            raise ValueError(f"unknown kernel {name}")
        mats[sel] = amp2[sel] * base

    return mats


def _kernel_matrix(
    x: torch.Tensor,
    kernel: torch.Tensor,
    log_lengthscale: torch.Tensor,
    log_outputscale: torch.Tensor,
    *,
    jitter: float,
) -> torch.Tensor:
    """Batch of GP covariance matrices on CPU float64 tensors."""

    mats = _kernel_covariance(x, x, kernel, log_lengthscale, log_outputscale)
    eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    return mats + jitter * eye


def draw_gp(
    x: torch.Tensor,
    kernel: torch.Tensor,
    log_lengthscale: torch.Tensor,
    log_outputscale: torch.Tensor,
    *,
    jitter: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw zero-mean GP values at `x` using CPU float64 Cholesky."""

    k = _kernel_matrix(x, kernel, log_lengthscale, log_outputscale, jitter=jitter)
    chol = torch.linalg.cholesky(k)
    eps = torch.randn(x.shape[0], x.shape[1], 1, dtype=x.dtype, device=x.device, generator=generator)
    return torch.bmm(chol, eps).squeeze(-1)


def draw_instances(n_instances: int, *, n_points: int, jitter: float) -> dict[str, torch.Tensor]:
    """Draw the expensive GP physics for `n_instances` tasks (CPU float64, no tokens).

    Struct-of-arrays, one row per instance, `n_points` observation points each. RNG draw
    order is `x -> log_ell -> log_scale -> kernel -> y`; keep it stable so the online path
    (post per-step reseed) matches the old monolithic sampler. The split / reveal /
    tokenization is `assemble`'s job; this physics is the only part a `data.py` pool caches.
    """

    x = 2.0 * torch.rand(n_instances, n_points, dtype=torch.float64) - 1.0
    log_ell = torch.empty(n_instances, dtype=torch.float64).uniform_(*LOG_LENGTHSCALE_RANGE)
    log_scale = torch.empty(n_instances, dtype=torch.float64).uniform_(*LOG_OUTPUTSCALE_RANGE)
    kernel = torch.randint(0, len(KERNELS), (n_instances,), dtype=torch.long)
    y = draw_gp(x, kernel, log_ell, log_scale, jitter=jitter)
    return {"x": x, "y": y, "log_ell": log_ell, "log_scale": log_scale, "kernel": kernel}


def assemble(
    inst: dict[str, torch.Tensor],
    *,
    variables: list[Variable],
    n_context: torch.Tensor,
    reveal_mask: torch.Tensor,
    max_context: int,
    device: torch.device | str,
) -> Batch:
    """Tokenize drawn GP instances into an ACE `Batch` (RNG-free).

    `n_context` (`[B]`) and `reveal_mask` (`[B, 3]`) are decided by the caller -- the online
    path draws them from the global RNG, the offline `data.py` reader from a stateless index
    hash -- so this function is deterministic given its inputs and shared by both. The first
    `max_context` points are context candidates (the first `n_context` active); **targets are
    all non-context points** (`n_target = n_points - n_context`). Tensorize with the context
    block of width `max_context` and the target block of width `n_points` (masked
    `>= n_context`), so context self-attention stays O(`max_context`^2) and only the target
    cross-attention grows. Revealed continuous latents become zero-spread PRIOR tokens, a
    revealed kernel a VALUE label; the rest are queried. CPU-native `inst` physics is moved
    to `device` as float32 here.
    """

    device = torch.device(device)
    b = int(inst["x"].shape[0])
    n_points = int(inst["x"].shape[1])
    if not 1 <= max_context < n_points:
        raise ValueError(f"need 1 <= max_context ({max_context}) < n_points ({n_points}) for >=1 target")
    x = inst["x"].float().to(device)
    y = inst["y"].float().to(device)
    log_ell = inst["log_ell"].float().to(device)
    log_scale = inst["log_scale"].float().to(device)
    kernel = inst["kernel"].to(device)
    log_ell_internal = encode_value(variables[1], log_ell)
    log_scale_internal = encode_value(variables[2], log_scale)

    reveal_ell, reveal_scale, reveal_kernel = reveal_mask[:, 0], reveal_mask[:, 1], reveal_mask[:, 2]

    # Context: first max_context points are candidates; the first n_context are active.
    ctx_t = max_context + 3
    ell_pos, scale_pos, kernel_pos = max_context, max_context + 1, max_context + 2
    ctx_var = torch.zeros(b, ctx_t, device=device, dtype=torch.long)
    ctx_var[:, ell_pos] = 1
    ctx_var[:, scale_pos] = 2
    ctx_var[:, kernel_pos] = 3
    ctx_x = torch.zeros(b, ctx_t, 1, device=device)
    ctx_x[:, :max_context, 0] = x[:, :max_context]
    ctx_value = torch.zeros(b, ctx_t, device=device)
    ctx_value[:, :max_context] = y[:, :max_context]
    ctx_value[:, ell_pos] = log_ell_internal
    ctx_value[:, scale_pos] = log_scale_internal
    ctx_value[:, kernel_pos] = kernel.float()
    ctx_index = torch.zeros(b, ctx_t, device=device, dtype=torch.long)
    ctx_index[:, kernel_pos] = kernel
    ctx_prior = torch.zeros(b, ctx_t, PRIOR_FEATURES, device=device)
    ctx_prior[:, ell_pos, 0] = log_ell_internal
    ctx_prior[:, scale_pos, 0] = log_scale_internal
    ctx_mode = torch.full((b, ctx_t), VALUE, device=device)
    ctx_mode[:, ell_pos] = PRIOR
    ctx_mode[:, scale_pos] = PRIOR
    ctx_mask = torch.zeros(b, ctx_t, device=device, dtype=torch.bool)
    ctx_ar = torch.arange(max_context, device=device)[None, :]
    ctx_mask[:, :max_context] = ctx_ar < n_context[:, None]
    ctx_mask[:, ell_pos] = reveal_ell
    ctx_mask[:, scale_pos] = reveal_scale
    ctx_mask[:, kernel_pos] = reveal_kernel
    context = make_tokens(
        var_id=ctx_var,
        x=ctx_x,
        value=ctx_value,
        value_index=ctx_index,
        mode=ctx_mode,
        mask=ctx_mask,
        prior=ctx_prior,
    )

    # Target: all non-context points (mask >= n_context) plus the 3 latent queries.
    tgt_t = n_points + 3
    tgt_var = torch.zeros(b, tgt_t, device=device, dtype=torch.long)
    tgt_var[:, 0] = 1
    tgt_var[:, 1] = 2
    tgt_var[:, 2] = 3
    tgt_x = torch.zeros(b, tgt_t, 1, device=device)
    tgt_x[:, 3:, 0] = x
    tgt_value = torch.zeros(b, tgt_t, device=device)
    tgt_value[:, 0] = log_ell_internal
    tgt_value[:, 1] = log_scale_internal
    tgt_value[:, 2] = kernel.float()
    tgt_value[:, 3:] = y
    tgt_index = torch.zeros(b, tgt_t, device=device, dtype=torch.long)
    tgt_index[:, 2] = kernel
    tgt_mask = torch.ones(b, tgt_t, device=device, dtype=torch.bool)
    tgt_mask[:, 0] = ~reveal_ell
    tgt_mask[:, 1] = ~reveal_scale
    tgt_mask[:, 2] = ~reveal_kernel
    tgt_ar = torch.arange(n_points, device=device)[None, :]
    tgt_mask[:, 3:] = tgt_ar >= n_context[:, None]
    target = make_tokens(
        var_id=tgt_var,
        x=tgt_x,
        value=tgt_value,
        value_index=tgt_index,
        mode=torch.full((b, tgt_t), QUERY, device=device),
        mask=tgt_mask,
    )
    return Batch(variables, context, target)


def online_batch(model: ACE, args: argparse.Namespace, device: torch.device | str) -> Batch:
    """Draw + assemble one online GP-1D training batch (global RNG; see `assemble`)."""

    inst = draw_instances(args.batch_size, n_points=N_TOTAL, jitter=args.jitter)
    n_context = torch.randint(args.min_context, args.max_context + 1, (args.batch_size,), device=device)
    reveal_mask = sample_reveal_mask(3, args.batch_size, q=1.0 - args.latent_context_prob, device=device)
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
        "log_lengthscale_range": list(LOG_LENGTHSCALE_RANGE),
        "log_outputscale_range": list(LOG_OUTPUTSCALE_RANGE),
        "N_TOTAL": N_TOTAL,
        "jitter": GEN_JITTER,
    }


def draw_pool(n_instances: int) -> dict[str, torch.Tensor]:
    """`draw_instances` bound to the frozen pool DGP config (used by `data.write_pool`)."""

    return draw_instances(n_instances, n_points=N_TOTAL, jitter=GEN_JITTER)


def load_checkpoint(path: str | Path, device: torch.device) -> ACE:
    """Load a GP-1D checkpoint (2-arg wrapper over `train.load_checkpoint`).

    Kept at this 2-arg signature because `playground/export_weights.py` and
    `playground/parity.py` call `gp1d.load_checkpoint(path, device)` directly.
    """

    return train.load_checkpoint(path, device, variables())


def fixed_eval_batch(vars_: list[Variable], *, device: torch.device | str, points: int, jitter: float) -> GPBatch:
    """Build the fixed GP function used by the diagnostic plot.

    The context locations include nearby pairs and triples. Sparse, evenly
    spaced points make kernel and lengthscale inference mostly guesswork.
    """

    gen = torch.Generator(device="cpu").manual_seed(EVAL_SEED)
    x_context = torch.tensor(
        [[-0.94, -0.89, -0.83, -0.62, -0.56, -0.34, -0.30, -0.05, 0.00, 0.22, 0.26, 0.53, 0.58, 0.88]],
        dtype=torch.float64,
    )
    x_target = torch.linspace(-1.0, 1.0, points, dtype=torch.float64)[None, :]
    x_all = torch.cat([x_context, x_target], dim=1)
    kernel = torch.tensor([EVAL_KERNEL], dtype=torch.long)
    log_ell = torch.tensor([EVAL_LOG_LENGTHSCALE], dtype=torch.float64)
    log_scale = torch.tensor([EVAL_LOG_OUTPUTSCALE], dtype=torch.float64)
    y_all = draw_gp(x_all, kernel, log_ell, log_scale, jitter=jitter, generator=gen)

    device = torch.device(device)
    x_context_d = x_context.float().to(device)
    x_target_d = x_target.float().to(device)
    y_context_d = y_all[:, : x_context.shape[1]].float().to(device)
    y_target_d = y_all[:, x_context.shape[1] :].float().to(device)
    log_ell_d = log_ell.float().to(device)
    log_scale_d = log_scale.float().to(device)
    kernel_d = kernel.to(device)

    context = make_tokens(
        var_id=torch.zeros(1, x_context.shape[1], device=device, dtype=torch.long),
        x=x_context_d[..., None],
        value=y_context_d,
        mode=torch.full((1, x_context.shape[1]), VALUE, device=device),
        mask=torch.ones(1, x_context.shape[1], device=device, dtype=torch.bool),
    )
    target = make_tokens(
        var_id=torch.zeros(1, points, device=device, dtype=torch.long),
        x=x_target_d[..., None],
        value=y_target_d,
        mode=torch.full((1, points), QUERY, device=device),
        mask=torch.ones(1, points, device=device, dtype=torch.bool),
    )
    return GPBatch(Batch(vars_, context, target), x_context_d, y_context_d, x_target_d, y_target_d, log_ell_d, log_scale_d, kernel_d)


def kernel_posterior(model: ACE, batch: Batch) -> torch.Tensor:
    """Evaluate ACE's posterior over the discrete kernel latent."""

    k = len(KERNELS)
    device = batch.context.value.device
    labels = torch.arange(k, device=device)
    target = make_tokens(
        var_id=torch.full((k, 1), 3, device=device),
        value=labels.float()[:, None],
        value_index=labels[:, None],
        mode=torch.full((k, 1), QUERY, device=device),
        mask=torch.ones(k, 1, device=device, dtype=torch.bool),
    )
    rep = Batch(batch.variables, repeat_tokens(batch.context, k), target)
    logp = model(rep).log_prob(target).squeeze(1)
    return (logp - torch.logsumexp(logp, dim=0)).exp()


def _moments_from_probs(grid: torch.Tensor, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean/std for normalized probability mass on a grid."""

    mean = (probs * grid).sum()
    std = (probs * (grid - mean).pow(2)).sum().sqrt()
    return mean, std


def gp_oracle(
    toy: GPBatch,
    *,
    bins: int,
    jitter: float,
    chunk: int,
) -> GPOracle:
    """Numerically integrate the GP posterior over kernel and hyperparameters."""

    if bins < 2:
        raise ValueError("GP oracle needs at least two grid bins")

    x_ctx = toy.x_context[0].detach().cpu().double()
    y_ctx = toy.y_context[0].detach().cpu().double()
    x_tgt = toy.x_target[0].detach().cpu().double()
    n = x_ctx.numel()

    ell_grid = torch.linspace(LOG_LENGTHSCALE_RANGE[0], LOG_LENGTHSCALE_RANGE[1], bins, dtype=torch.float64)
    scale_grid = torch.linspace(LOG_OUTPUTSCALE_RANGE[0], LOG_OUTPUTSCALE_RANGE[1], bins, dtype=torch.float64)
    ell_w = torch.ones(bins, dtype=torch.float64)
    scale_w = torch.ones(bins, dtype=torch.float64)
    ell_w[[0, -1]] = 0.5
    scale_w[[0, -1]] = 0.5
    ell_step = (LOG_LENGTHSCALE_RANGE[1] - LOG_LENGTHSCALE_RANGE[0]) / (bins - 1)
    scale_step = (LOG_OUTPUTSCALE_RANGE[1] - LOG_OUTPUTSCALE_RANGE[0]) / (bins - 1)
    log_cell = math.log(ell_step * scale_step) - math.log(
        (LOG_LENGTHSCALE_RANGE[1] - LOG_LENGTHSCALE_RANGE[0])
        * (LOG_OUTPUTSCALE_RANGE[1] - LOG_OUTPUTSCALE_RANGE[0])
    )
    kernel_grid = torch.arange(len(KERNELS), dtype=torch.float64)
    kk, ee, ss = torch.meshgrid(kernel_grid, ell_grid, scale_grid, indexing="ij")
    flat_kernel = kk.reshape(-1).long()
    flat_ell = ee.reshape(-1)
    flat_scale = ss.reshape(-1)
    _, ew, sw = torch.meshgrid(kernel_grid, ell_w, scale_w, indexing="ij")
    log_quad = (ew * sw).reshape(-1).log() + log_cell
    g = flat_kernel.numel()

    x_batch = x_ctx.expand(g, n)
    kcc = _kernel_matrix(x_batch, flat_kernel, flat_ell, flat_scale, jitter=jitter)
    chol = torch.linalg.cholesky(kcc)
    y = y_ctx.view(1, n, 1).expand(g, n, 1)
    alpha = torch.cholesky_solve(y, chol).squeeze(-1)
    quad = (y.squeeze(-1) * alpha).sum(dim=1)
    logdet = 2.0 * chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=1)
    log_like = -0.5 * (quad + logdet + n * math.log(2.0 * math.pi))
    log_joint = log_like + log_quad
    log_joint_by_kernel = log_joint.reshape(len(KERNELS), bins, bins)
    kernel_log_marginal = torch.logsumexp(log_joint_by_kernel.flatten(1), dim=1)
    log_post = log_joint - torch.logsumexp(log_joint, dim=0)
    weights = log_post.exp()
    post = weights.reshape(len(KERNELS), bins, bins)

    kernel_probs = post.sum(dim=(1, 2))
    ell_probs = post.sum(dim=(0, 2))
    scale_probs = post.sum(dim=(0, 1))

    p = x_tgt.numel()
    mean_acc = torch.zeros(p, dtype=torch.float64)
    second_acc = torch.zeros(p, dtype=torch.float64)
    for start in range(0, g, chunk):
        end = min(start + chunk, g)
        b = end - start
        kernel_s = flat_kernel[start:end]
        ell_s = flat_ell[start:end]
        scale_s = flat_scale[start:end]
        x_left = x_tgt.expand(b, p)
        x_right = x_ctx.expand(b, n)
        ktc = _kernel_covariance(x_left, x_right, kernel_s, ell_s, scale_s)
        solved = torch.cholesky_solve(ktc.transpose(1, 2), chol[start:end])
        mean = torch.bmm(ktc, alpha[start:end, :, None]).squeeze(-1)
        diag = (ktc * solved.transpose(1, 2)).sum(dim=-1)
        prior_var = scale_s.exp().pow(2)[:, None] + jitter
        var = (prior_var - diag).clamp_min(1e-10)
        w = weights[start:end, None]
        mean_acc += (w * mean).sum(dim=0)
        second_acc += (w * (var + mean.pow(2))).sum(dim=0)

    y_mean = mean_acc
    y_std = (second_acc - y_mean.pow(2)).clamp_min(1e-10).sqrt()
    return GPOracle(kernel_log_marginal, kernel_probs, ell_grid, ell_probs, scale_grid, scale_probs, y_mean, y_std)


@torch.no_grad()
def evaluate(model: ACE, args: argparse.Namespace) -> Diagnostic:
    """Run the fixed GP diagnostic and print compact metrics."""

    device = next(model.parameters()).device
    toy = fixed_eval_batch(model.variables, device=device, points=args.eval_points, jitter=args.jitter)
    oracle = gp_oracle(toy, bins=args.oracle_bins, jitter=args.jitter, chunk=args.oracle_chunk)
    pred = model(toy.batch)
    y_mean = pred.mean(toy.batch.target)[0]
    y_std = pred.continuous_var()[0].clamp_min(1e-8).sqrt()
    y_logp = pred.log_prob(toy.batch.target)[0]

    ell_grid = torch.linspace(LOG_LENGTHSCALE_RANGE[0], LOG_LENGTHSCALE_RANGE[1], args.oracle_bins, device=device)
    scale_grid = torch.linspace(LOG_OUTPUTSCALE_RANGE[0], LOG_OUTPUTSCALE_RANGE[1], args.oracle_bins, device=device)
    ell_model_grid = encode_value(model.variables[1], ell_grid)
    scale_model_grid = encode_value(model.variables[2], scale_grid)
    ell_logp = query_log_density(model, toy.batch, 1, ell_model_grid)
    scale_logp = query_log_density(model, toy.batch, 2, scale_model_grid)
    kernel_probs = kernel_posterior(model, toy.batch)
    ell_mean, ell_std = normalized_moments(ell_grid, ell_logp)
    scale_mean, scale_std = normalized_moments(scale_grid, scale_logp)
    oracle_ell_mean, oracle_ell_std = _moments_from_probs(oracle.ell_grid, oracle.ell_probs)
    oracle_scale_mean, oracle_scale_std = _moments_from_probs(oracle.scale_grid, oracle.scale_probs)
    log_marginal_delta = oracle.kernel_log_marginal - oracle.kernel_log_marginal.max()

    rmse = (y_mean - toy.y_target[0]).pow(2).mean().sqrt()
    nll = -y_logp.mean()
    true_kernel_prob = kernel_probs[int(toy.kernel[0])]
    oracle_true_kernel_prob = oracle.kernel_probs[int(toy.kernel[0])]
    oracle_rmse = (oracle.y_mean.to(device) - toy.y_target[0]).pow(2).mean().sqrt()
    kernel_kl = (oracle.kernel_probs * (oracle.kernel_probs.clamp_min(1e-12).log() - kernel_probs.detach().cpu().clamp_min(1e-12).log())).sum()
    metrics = {
        "y_rmse": float(rmse),
        "y_nll": float(nll),
        "oracle_y_rmse": float(oracle_rmse),
        "kernel_true_prob": float(true_kernel_prob),
        "oracle_kernel_true_prob": float(oracle_true_kernel_prob),
        "kernel_kl_oracle_model": float(kernel_kl),
        "log_lengthscale_mean": float(ell_mean),
        "log_lengthscale_std": float(ell_std),
        "oracle_log_lengthscale_mean": float(oracle_ell_mean),
        "oracle_log_lengthscale_std": float(oracle_ell_std),
        "log_outputscale_mean": float(scale_mean),
        "log_outputscale_std": float(scale_std),
        "oracle_log_outputscale_mean": float(oracle_scale_mean),
        "oracle_log_outputscale_std": float(oracle_scale_std),
    }
    metrics.update({f"oracle_log_marginal_delta_{name}": float(delta) for name, delta in zip(KERNELS, log_marginal_delta)})

    print("\nGP-1D diagnostic")
    print(f"truth kernel        {KERNELS[int(toy.kernel[0])]}")
    print(f"truth log_length    {float(toy.log_lengthscale[0]): .3f}")
    print(f"oracle log_length   mean {float(oracle_ell_mean): .3f}  std {float(oracle_ell_std): .3f}")
    print(f"ACE log_length      mean {float(ell_mean): .3f}  std {float(ell_std): .3f}")
    print(f"truth log_output    {float(toy.log_outputscale[0]): .3f}")
    print(f"oracle log_output   mean {float(oracle_scale_mean): .3f}  std {float(oracle_scale_std): .3f}")
    print(f"ACE log_output      mean {float(scale_mean): .3f}  std {float(scale_std): .3f}")
    print("oracle log marg delta " + "  ".join(f"{name} {float(delta):.2f}" for name, delta in zip(KERNELS, log_marginal_delta)))
    print("oracle kernel       " + "  ".join(f"{name} {float(prob):.3f}" for name, prob in zip(KERNELS, oracle.kernel_probs)))
    print("ACE kernel          " + "  ".join(f"{name} {float(prob):.3f}" for name, prob in zip(KERNELS, kernel_probs)))
    print(f"kernel true prob    oracle {float(oracle_true_kernel_prob): .3f}  ACE {float(true_kernel_prob): .3f}")
    print(f"target y            oracle rmse {float(oracle_rmse): .3f}  ACE rmse {float(rmse): .3f}  ACE nll {float(nll): .3f}")
    return Diagnostic(toy, y_mean, y_std, ell_grid, ell_logp, scale_grid, scale_logp, kernel_probs, oracle, metrics)


def plot_diagnostic(diag: Diagnostic, path: str | Path) -> None:
    """Save an oracle-vs-ACE diagnostic figure for the fixed GP problem."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x_ctx = diag.toy.x_context[0].detach().cpu()
    y_ctx = diag.toy.y_context[0].detach().cpu()
    x = diag.toy.x_target[0].detach().cpu()
    y = diag.toy.y_target[0].detach().cpu()
    y_mean = diag.y_mean.detach().cpu()
    y_std = diag.y_std.detach().cpu()
    oracle_y_mean = diag.oracle.y_mean.detach().cpu()
    oracle_y_std = diag.oracle.y_std.detach().cpu()
    ell_grid = diag.ell_grid.detach().cpu()
    ell_p = (diag.ell_logp - torch.logsumexp(diag.ell_logp, dim=0)).exp().detach().cpu()
    oracle_ell_grid = diag.oracle.ell_grid.detach().cpu()
    oracle_ell_p = diag.oracle.ell_probs.detach().cpu()
    scale_grid = diag.scale_grid.detach().cpu()
    scale_p = (diag.scale_logp - torch.logsumexp(diag.scale_logp, dim=0)).exp().detach().cpu()
    oracle_scale_grid = diag.oracle.scale_grid.detach().cpu()
    oracle_scale_p = diag.oracle.scale_probs.detach().cpu()
    kernel_p = diag.kernel_probs.detach().cpu()
    oracle_kernel_p = diag.oracle.kernel_probs.detach().cpu()
    true_kernel = int(diag.toy.kernel[0])

    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    ax_y = fig.add_subplot(gs[0, :])
    ax_kernel = fig.add_subplot(gs[1, 0])
    ax_latent = fig.add_subplot(gs[1, 1])

    ax_y.plot(x, y, color="0.25", linewidth=1.4, label="sampled function")
    ax_y.plot(x, oracle_y_mean, color="tab:green", linewidth=1.5, label="oracle mean")
    ax_y.fill_between(
        x,
        oracle_y_mean - 2.0 * oracle_y_std,
        oracle_y_mean + 2.0 * oracle_y_std,
        color="tab:green",
        alpha=0.14,
        label="oracle +/-2 std",
    )
    ax_y.plot(x, y_mean, color="tab:blue", linewidth=1.5, label="ACE mean")
    ax_y.fill_between(x, y_mean - 2.0 * y_std, y_mean + 2.0 * y_std, color="tab:blue", alpha=0.16, label="ACE +/-2 std")
    ax_y.scatter(x_ctx, y_ctx, color="black", s=28, zorder=3, label="context")
    ax_y.set_title("posterior predictive")
    ax_y.set_xlabel("x")
    ax_y.set_ylabel("y")
    ax_y.legend(loc="best")

    xpos = list(range(len(KERNELS)))
    width = 0.38
    oracle_bars = ax_kernel.bar([i - width / 2 for i in xpos], oracle_kernel_p, width=width, color="tab:green", label="oracle")
    ace_bars = ax_kernel.bar([i + width / 2 for i in xpos], kernel_p, width=width, color="tab:blue", label="ACE")
    for bars in (oracle_bars, ace_bars):
        bars[true_kernel].set_edgecolor("tab:orange")
        bars[true_kernel].set_linewidth(2.0)
    ax_kernel.set_ylim(0.0, 1.0)
    ax_kernel.set_title("kernel posterior")
    ax_kernel.set_xticks(xpos, KERNELS)
    ax_kernel.tick_params(axis="x", rotation=20)
    ax_kernel.legend()

    ax_latent.plot(oracle_ell_grid, oracle_ell_p, color="tab:blue", linewidth=1.5, label="oracle log_lengthscale")
    ax_latent.plot(ell_grid, ell_p, color="tab:blue", linestyle="--", linewidth=1.5, label="ACE log_lengthscale")
    ax_latent.plot(oracle_scale_grid, oracle_scale_p, color="tab:orange", linewidth=1.5, label="oracle log_outputscale")
    ax_latent.plot(scale_grid, scale_p, color="tab:orange", linestyle="--", linewidth=1.5, label="ACE log_outputscale")
    ax_latent.axvline(float(diag.toy.log_lengthscale[0]), color="tab:blue", alpha=0.35)
    ax_latent.axvline(float(diag.toy.log_outputscale[0]), color="tab:orange", alpha=0.35)
    ax_latent.set_title("latent marginals")
    ax_latent.set_xlabel("latent value")
    ax_latent.set_ylabel("posterior mass on grid")
    ax_latent.legend()

    fig.suptitle(
        f"truth kernel={KERNELS[true_kernel]}, "
        f"oracle RMSE={diag.metrics['oracle_y_rmse']:.2f}, "
        f"ACE RMSE={diag.metrics['y_rmse']:.2f}, "
        f"kernel KL={diag.metrics['kernel_kl_oracle_model']:.2f}"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved diagnostic plot: {path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the GP-1D example."""

    p = argparse.ArgumentParser(parents=[train.common_parser()], description="Train/evaluate the nanoACE GP-1D toy.")
    # Targets are all non-context points (complement-targets); N_TOTAL is the point budget.
    # `--data-targets` is inherited from common_parser but unused by GP-1D (no-op).
    p.set_defaults(
        batch_size=64,
        max_context=20,
        min_context=1,
        d_model=128,
        heads=4,
        layers=4,
        hidden=256,
        components=8,
        plot_path="artifacts/gp1d.png",
    )
    p.add_argument("--eval-points", type=int, default=160)
    p.add_argument("--oracle-bins", type=int, default=64)
    p.add_argument("--oracle-chunk", type=int, default=512)
    p.add_argument("--jitter", type=float, default=GEN_JITTER)
    p.add_argument("--pool", default="", help="train from an offline data.py pool directory instead of online")
    p.add_argument("--pool-force", action="store_true", help="reuse a pool despite a DGP config-hash mismatch")
    return train.apply_config_file(p)


def main() -> None:
    """Run GP-1D training/evaluation from the command line."""

    args = parse_args()
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
            if args.jitter != GEN_JITTER:
                raise SystemExit(
                    f"--pool freezes the DGP (pool jitter={GEN_JITTER}); --jitter {args.jitter} would only "
                    "affect diagnostics, not the cached training data. Regenerate the pool or drop --pool."
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
        plot_diagnostic(diag, args.plot_path)


if __name__ == "__main__":
    main()
