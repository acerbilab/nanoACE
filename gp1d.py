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
    metrics: dict[str, float]


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


def _kernel_matrix(
    x: torch.Tensor,
    kernel: torch.Tensor,
    log_lengthscale: torch.Tensor,
    log_outputscale: torch.Tensor,
    *,
    jitter: float,
) -> torch.Tensor:
    """Batch of GP covariance matrices on CPU float64 tensors."""

    r = (x[:, :, None] - x[:, None, :]).abs()
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


@torch.no_grad()
def evaluate(model: ACE, args: argparse.Namespace) -> Diagnostic:
    """Run the fixed GP diagnostic and print compact metrics."""

    device = next(model.parameters()).device
    toy = fixed_eval_batch(model.variables, bins=model.cfg.prior_bins, device=device, points=args.eval_points, jitter=args.jitter)
    pred = model(toy.batch)
    y_mean = pred.mean(toy.batch.target)[0]
    y_std = pred.continuous_var()[0].clamp_min(1e-8).sqrt()
    y_logp = pred.log_prob(toy.batch.target)[0]

    ell_grid = torch.linspace(LOG_LENGTHSCALE_RANGE[0], LOG_LENGTHSCALE_RANGE[1], args.bins, device=device)
    scale_grid = torch.linspace(LOG_OUTPUTSCALE_RANGE[0], LOG_OUTPUTSCALE_RANGE[1], args.bins, device=device)
    ell_logp = query_log_density(model, toy.batch, 1, ell_grid)
    scale_logp = query_log_density(model, toy.batch, 2, scale_grid)
    kernel_probs = kernel_posterior(model, toy.batch)
    ell_mean, ell_std = normalized_moments(ell_grid, ell_logp)
    scale_mean, scale_std = normalized_moments(scale_grid, scale_logp)

    rmse = (y_mean - toy.y_target[0]).pow(2).mean().sqrt()
    nll = -y_logp.mean()
    true_kernel_prob = kernel_probs[int(toy.kernel[0])]
    metrics = {
        "y_rmse": float(rmse),
        "y_nll": float(nll),
        "kernel_true_prob": float(true_kernel_prob),
        "log_lengthscale_mean": float(ell_mean),
        "log_lengthscale_std": float(ell_std),
        "log_outputscale_mean": float(scale_mean),
        "log_outputscale_std": float(scale_std),
    }

    print("\nGP-1D diagnostic")
    print(f"truth kernel        {KERNELS[int(toy.kernel[0])]}")
    print(f"truth log_length    {float(toy.log_lengthscale[0]): .3f}")
    print(f"model log_length    mean {float(ell_mean): .3f}  std {float(ell_std): .3f}")
    print(f"truth log_output    {float(toy.log_outputscale[0]): .3f}")
    print(f"model log_output    mean {float(scale_mean): .3f}  std {float(scale_std): .3f}")
    print(f"kernel true prob    {float(true_kernel_prob): .3f}")
    print(f"target y            rmse {float(rmse): .3f}  nll {float(nll): .3f}")
    return Diagnostic(toy, y_mean, y_std, ell_grid, ell_logp, scale_grid, scale_logp, kernel_probs, metrics)


def plot_diagnostic(diag: Diagnostic, path: str | Path) -> None:
    """Save a compact GP-1D diagnostic figure."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x_ctx = diag.toy.x_context[0].detach().cpu()
    y_ctx = diag.toy.y_context[0].detach().cpu()
    x = diag.toy.x_target[0].detach().cpu()
    y = diag.toy.y_target[0].detach().cpu()
    y_mean = diag.y_mean.detach().cpu()
    y_std = diag.y_std.detach().cpu()
    ell_grid = diag.ell_grid.detach().cpu()
    ell_p = (diag.ell_logp - torch.logsumexp(diag.ell_logp, dim=0)).exp().detach().cpu()
    scale_grid = diag.scale_grid.detach().cpu()
    scale_p = (diag.scale_logp - torch.logsumexp(diag.scale_logp, dim=0)).exp().detach().cpu()
    kernel_p = diag.kernel_probs.detach().cpu()

    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    ax_y = fig.add_subplot(gs[0, :])
    ax_kernel = fig.add_subplot(gs[1, 0])
    ax_latent = fig.add_subplot(gs[1, 1])

    ax_y.plot(x, y, color="0.25", linewidth=1.4, label="sampled function")
    ax_y.plot(x, y_mean, color="tab:blue", label="ACE mean")
    ax_y.fill_between(x, y_mean - 2.0 * y_std, y_mean + 2.0 * y_std, color="tab:blue", alpha=0.18, label="+/-2 std")
    ax_y.scatter(x_ctx, y_ctx, color="black", s=28, zorder=3, label="context")
    ax_y.set_title("GP-1D predictive")
    ax_y.set_xlabel("x")
    ax_y.set_ylabel("y")
    ax_y.legend(loc="best")

    bars = ax_kernel.bar(KERNELS, kernel_p)
    bars[int(diag.toy.kernel[0])].set_color("tab:orange")
    ax_kernel.set_ylim(0.0, 1.0)
    ax_kernel.set_title("kernel posterior")
    ax_kernel.tick_params(axis="x", rotation=20)

    ax_latent.plot(ell_grid, ell_p, label="log_lengthscale")
    ax_latent.plot(scale_grid, scale_p, label="log_outputscale")
    ax_latent.axvline(float(diag.toy.log_lengthscale[0]), color="tab:blue", alpha=0.35)
    ax_latent.axvline(float(diag.toy.log_outputscale[0]), color="tab:orange", alpha=0.35)
    ax_latent.set_title("latent marginals")
    ax_latent.set_xlabel("latent value")
    ax_latent.set_ylabel("density on grid")
    ax_latent.legend()

    fig.suptitle(f"truth kernel={KERNELS[int(diag.toy.kernel[0])]}, y RMSE={diag.metrics['y_rmse']:.2f}")
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
