"""Executable 1D Gaussian-process example for nanoACE.

This file defines one compact GP regression task with three latents:
`log_lengthscale`, `log_outputscale`, and a discrete kernel family. It owns the
online sampler, training loop, fixed diagnostic case, checkpoint helpers, and
plot. GP sampling uses CPU float64 Cholesky; ACE itself runs on the selected
device.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ace import ACE, ACEConfig, Batch, QUERY, VALUE, Tokens, Variable
from diagnostics import normalized_moments, query_log_density, repeat_tokens


KERNELS = ("RBF", "Matern12", "Matern32", "Periodic")
LOG_LENGTHSCALE_RANGE = (math.log(0.12), math.log(0.80))
LOG_OUTPUTSCALE_RANGE = (math.log(0.25), math.log(1.00))
EVAL_KERNEL = 2
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


def variables(n_bins: int) -> list[Variable]:
    """Schema for GP observations and the three task latents."""

    return [
        Variable("y", "data", "continuous"),
        Variable("log_lengthscale", "latent", "continuous", transform="log", prior_range=LOG_LENGTHSCALE_RANGE, prior_bins=n_bins),
        Variable("log_outputscale", "latent", "continuous", transform="log", prior_range=LOG_OUTPUTSCALE_RANGE, prior_bins=n_bins),
        Variable("kernel", "latent", "discrete", cardinality=len(KERNELS)),
    ]


def make_tokens(
    *,
    var_id: torch.Tensor,
    value: torch.Tensor,
    mode: torch.Tensor,
    mask: torch.Tensor,
    bins: int,
    x: torch.Tensor | None = None,
    value_index: torch.Tensor | None = None,
) -> Tokens:
    """Construct GP tokens, keeping data `x` and discrete labels explicit."""

    b, t = var_id.shape
    device = value.device
    if x is None:
        x = torch.zeros(b, t, 1, device=device, dtype=value.dtype)
    if value_index is None:
        value_index = torch.zeros(b, t, device=device, dtype=torch.long)
    return Tokens(
        var_id=var_id.long(),
        x=x,
        value=value,
        value_index=value_index.long(),
        prior=torch.zeros(b, t, bins, device=device, dtype=value.dtype),
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


def sample_gp_batch(
    vars_: list[Variable],
    *,
    batch_size: int,
    max_context: int,
    min_context: int,
    data_targets: int,
    bins: int,
    device: torch.device | str,
    latent_context_prob: float,
    jitter: float,
) -> GPBatch:
    """Sample one online GP-1D training batch."""

    total = max_context + data_targets
    x_cpu = 2.0 * torch.rand(batch_size, total, dtype=torch.float64) - 1.0
    log_ell_cpu = torch.empty(batch_size, dtype=torch.float64).uniform_(*LOG_LENGTHSCALE_RANGE)
    log_scale_cpu = torch.empty(batch_size, dtype=torch.float64).uniform_(*LOG_OUTPUTSCALE_RANGE)
    kernel_cpu = torch.randint(0, len(KERNELS), (batch_size,), dtype=torch.long)
    y_cpu = draw_gp(x_cpu, kernel_cpu, log_ell_cpu, log_scale_cpu, jitter=jitter)

    device = torch.device(device)
    x = x_cpu.float().to(device)
    y = y_cpu.float().to(device)
    log_ell = log_ell_cpu.float().to(device)
    log_scale = log_scale_cpu.float().to(device)
    kernel = kernel_cpu.to(device)

    n_ctx = torch.randint(min_context, max_context + 1, (batch_size,), device=device)
    ar = torch.arange(max_context, device=device)[None, :]
    reveal = torch.rand(batch_size, device=device) < latent_context_prob
    reveal_which = torch.randint(0, 3, (batch_size,), device=device)
    reveal_ell = reveal & (reveal_which == 0)
    reveal_scale = reveal & (reveal_which == 1)
    reveal_kernel = reveal & (reveal_which == 2)

    ctx_t = max_context + 3
    ell_pos, scale_pos, kernel_pos = max_context, max_context + 1, max_context + 2
    ctx_var = torch.zeros(batch_size, ctx_t, device=device, dtype=torch.long)
    ctx_var[:, ell_pos] = 1
    ctx_var[:, scale_pos] = 2
    ctx_var[:, kernel_pos] = 3
    ctx_x = torch.zeros(batch_size, ctx_t, 1, device=device)
    ctx_x[:, :max_context, 0] = x[:, :max_context]
    ctx_value = torch.zeros(batch_size, ctx_t, device=device)
    ctx_value[:, :max_context] = y[:, :max_context]
    ctx_value[:, ell_pos] = log_ell
    ctx_value[:, scale_pos] = log_scale
    ctx_value[:, kernel_pos] = kernel.float()
    ctx_index = torch.zeros(batch_size, ctx_t, device=device, dtype=torch.long)
    ctx_index[:, kernel_pos] = kernel
    ctx_mask = torch.zeros(batch_size, ctx_t, device=device, dtype=torch.bool)
    ctx_mask[:, :max_context] = ar < n_ctx[:, None]
    ctx_mask[:, ell_pos] = reveal_ell
    ctx_mask[:, scale_pos] = reveal_scale
    ctx_mask[:, kernel_pos] = reveal_kernel
    context = make_tokens(
        var_id=ctx_var,
        x=ctx_x,
        value=ctx_value,
        value_index=ctx_index,
        mode=torch.full((batch_size, ctx_t), VALUE, device=device),
        mask=ctx_mask,
        bins=bins,
    )

    tgt_t = 3 + data_targets
    tgt_var = torch.zeros(batch_size, tgt_t, device=device, dtype=torch.long)
    tgt_var[:, 0] = 1
    tgt_var[:, 1] = 2
    tgt_var[:, 2] = 3
    tgt_x = torch.zeros(batch_size, tgt_t, 1, device=device)
    tgt_x[:, 3:, 0] = x[:, max_context:]
    tgt_value = torch.zeros(batch_size, tgt_t, device=device)
    tgt_value[:, 0] = log_ell
    tgt_value[:, 1] = log_scale
    tgt_value[:, 2] = kernel.float()
    tgt_value[:, 3:] = y[:, max_context:]
    tgt_index = torch.zeros(batch_size, tgt_t, device=device, dtype=torch.long)
    tgt_index[:, 2] = kernel
    tgt_mask = torch.ones(batch_size, tgt_t, device=device, dtype=torch.bool)
    tgt_mask[:, 0] = ~reveal_ell
    tgt_mask[:, 1] = ~reveal_scale
    tgt_mask[:, 2] = ~reveal_kernel
    target = make_tokens(
        var_id=tgt_var,
        x=tgt_x,
        value=tgt_value,
        value_index=tgt_index,
        mode=torch.full((batch_size, tgt_t), QUERY, device=device),
        mask=tgt_mask,
        bins=bins,
    )
    return GPBatch(Batch(vars_, context, target), x[:, :max_context], y[:, :max_context], x[:, max_context:], y[:, max_context:], log_ell, log_scale, kernel)


def build_model(args, device: torch.device) -> ACE:
    """Construct the GP-1D ACE model from CLI hyperparameters."""

    cfg = ACEConfig(
        x_dim=1,
        prior_bins=args.bins,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        mlp_hidden=args.hidden,
        head_hidden=args.hidden,
        mdn_components=args.components,
    )
    return ACE(variables(args.bins), cfg).to(device)


def train(args: argparse.Namespace, model: ACE | None = None) -> ACE:
    """Train ACE online on freshly sampled GP-1D batches."""

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = build_model(args, device) if model is None else model
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        toy = sample_gp_batch(
            model.variables,
            batch_size=args.batch_size,
            max_context=args.max_context,
            min_context=args.min_context,
            data_targets=args.data_targets,
            bins=model.cfg.prior_bins,
            device=device,
            latent_context_prob=args.latent_context_prob,
            jitter=args.jitter,
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
    """Save a lightweight GP-1D checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": asdict(model.cfg), "seed": args.seed, "state_dict": model.state_dict()}, path)
    print(f"saved checkpoint: {path}")


def load_checkpoint(path: str | Path, device: torch.device) -> ACE:
    """Load a GP-1D checkpoint saved by `save_checkpoint`."""

    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = ACEConfig(**payload["cfg"])
    model = ACE(variables(cfg.prior_bins), cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    return model


def fixed_eval_batch(vars_: list[Variable], *, bins: int, device: torch.device | str, points: int, jitter: float) -> GPBatch:
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
        bins=bins,
    )
    target = make_tokens(
        var_id=torch.zeros(1, points, device=device, dtype=torch.long),
        x=x_target_d[..., None],
        value=y_target_d,
        mode=torch.full((1, points), QUERY, device=device),
        mask=torch.ones(1, points, device=device, dtype=torch.bool),
        bins=bins,
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
        bins=model.cfg.prior_bins,
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
    toy = fixed_eval_batch(model.variables, bins=model.cfg.prior_bins, device=device, points=args.eval_points, jitter=args.jitter)
    oracle = gp_oracle(toy, bins=args.oracle_bins, jitter=args.jitter, chunk=args.oracle_chunk)
    pred = model(toy.batch)
    y_mean = pred.mean(toy.batch.target)[0]
    y_std = pred.continuous_var()[0].clamp_min(1e-8).sqrt()
    y_logp = pred.log_prob(toy.batch.target)[0]

    ell_grid = torch.linspace(LOG_LENGTHSCALE_RANGE[0], LOG_LENGTHSCALE_RANGE[1], args.oracle_bins, device=device)
    scale_grid = torch.linspace(LOG_OUTPUTSCALE_RANGE[0], LOG_OUTPUTSCALE_RANGE[1], args.oracle_bins, device=device)
    ell_logp = query_log_density(model, toy.batch, 1, ell_grid)
    scale_logp = query_log_density(model, toy.batch, 2, scale_grid)
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

    p = argparse.ArgumentParser(description="Train/evaluate the nanoACE GP-1D toy.")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bins", type=int, default=64)
    p.add_argument("--max-context", type=int, default=14)
    p.add_argument("--min-context", type=int, default=4)
    p.add_argument("--data-targets", type=int, default=32)
    p.add_argument("--eval-points", type=int, default=160)
    p.add_argument("--oracle-bins", type=int, default=64)
    p.add_argument("--oracle-chunk", type=int, default=512)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--components", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--latent-weight", type=float, default=2.0)
    p.add_argument("--latent-context-prob", type=float, default=0.20)
    p.add_argument("--jitter", type=float, default=1e-5)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--plot-path", default="artifacts/gp1d.png")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--save-checkpoint", default="")
    p.add_argument("--load-checkpoint", default="")
    p.add_argument("--eval-only", action="store_true")
    return p.parse_args()


def main() -> None:
    """Run GP-1D training/evaluation from the command line."""

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
