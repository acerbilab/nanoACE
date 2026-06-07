"""Executable Gaussian toy example for nanoACE.

Defines the runtime-Beta-prior `(mu, log_sigma)` problem, online batch
generation, training loop, deterministic evaluation batch, analytic grid
posterior, posterior predictive density, checkpoint helpers, and diagnostic
plot.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ace import ACE, ACEConfig, Batch, PRIOR, PRIOR_FEATURES, QUERY, VALUE, Variable, encode_value
from ace_prior import beta_logprior_on_grid, draw_from_beta, known_latent_features, prior_features, sample_prior_params
from diagnostics import ar_joint_log_density, make_scalar_tokens, normalized_moments, query_log_density


MU_RANGE = (-1.5, 1.5)
LOGSIG_RANGE = (math.log(0.15), math.log(1.25))
EVAL_Y = (0.6780859231948853, 0.852228581905365, 2.016355037689209)
EVAL_TRUE_MU = 0.891850471496582
EVAL_TRUE_LOGSIG = -0.4232509136199951
EVAL_MU_PRIOR = (0.70, 20.0)
EVAL_LOGSIG_PRIOR = (0.70, 8.0)


@dataclass
class ToyBatch:
    """A Gaussian-toy ACE batch plus true latent values for diagnostics."""

    batch: Batch
    y_context: torch.Tensor
    mu: torch.Tensor
    log_sigma: torch.Tensor
    mu_prior_unit: torch.Tensor
    mu_prior_nu: torch.Tensor
    logsig_prior_unit: torch.Tensor
    logsig_prior_nu: torch.Tensor


@dataclass
class Diagnostic:
    """Posterior comparison for the deterministic evaluation batch."""

    toy: ToyBatch
    oracle: dict[str, torch.Tensor]
    mu_logp: torch.Tensor
    logsig_logp: torch.Tensor
    joint_logp: torch.Tensor
    y_grid: torch.Tensor
    oracle_y_pred: torch.Tensor
    model_y_logp: torch.Tensor
    metrics: dict[str, float]


def variables() -> list[Variable]:
    """Schema for the Gaussian toy."""

    return [
        Variable("y", "data", "continuous"),
        Variable("mu", "latent", "continuous", bounds=MU_RANGE),
        Variable("log_sigma", "latent", "continuous", transform="log", bounds=LOGSIG_RANGE),
    ]


def sample_toy_batch(
    vars_: list[Variable],
    *,
    batch_size: int,
    max_context: int,
    min_context: int,
    data_targets: int,
    device: torch.device | str,
    latent_context_prob: float = 0.0,
) -> ToyBatch:
    """Generate one online training/eval batch."""

    mu_prior_unit, mu_prior_nu = sample_prior_params((batch_size,), device=device)
    logsig_prior_unit, logsig_prior_nu = sample_prior_params((batch_size,), device=device)
    mu = draw_from_beta(mu_prior_unit, mu_prior_nu, *MU_RANGE)
    log_sigma = draw_from_beta(logsig_prior_unit, logsig_prior_nu, *LOGSIG_RANGE)
    mu_internal = encode_value(vars_[1], mu)
    logsig_internal = encode_value(vars_[2], log_sigma)
    sigma = log_sigma.exp()

    total_y = max_context + data_targets
    y = mu[:, None] + sigma[:, None] * torch.randn(batch_size, total_y, device=device)
    n_ctx = torch.randint(min_context, max_context + 1, (batch_size,), device=device)
    ar = torch.arange(max_context, device=device)[None, :]
    if latent_context_prob > 0.0:
        reveal = torch.rand(batch_size, device=device) < latent_context_prob
        reveal_mu = reveal & (torch.rand(batch_size, device=device) < 0.5)
        reveal_logsig = reveal & ~reveal_mu
    else:
        reveal_mu = torch.zeros(batch_size, device=device, dtype=torch.bool)
        reveal_logsig = torch.zeros(batch_size, device=device, dtype=torch.bool)

    ctx_t = max_context + 2
    mu_value_pos = max_context
    logsig_value_pos = max_context + 1
    ctx_var = torch.zeros(batch_size, ctx_t, device=device, dtype=torch.long)
    ctx_var[:, mu_value_pos] = 1
    ctx_var[:, logsig_value_pos] = 2
    ctx_value = torch.zeros(batch_size, ctx_t, device=device)
    ctx_value[:, :max_context] = y[:, :max_context]
    ctx_value[:, mu_value_pos] = mu_internal
    ctx_value[:, logsig_value_pos] = logsig_internal
    ctx_prior = torch.zeros(batch_size, ctx_t, PRIOR_FEATURES, device=device)
    ctx_prior[:, mu_value_pos] = prior_features(mu_prior_unit, mu_prior_nu)
    ctx_prior[:, logsig_value_pos] = prior_features(logsig_prior_unit, logsig_prior_nu)
    ctx_prior[:, mu_value_pos] = torch.where(
        reveal_mu[:, None],
        known_latent_features(mu_internal),
        ctx_prior[:, mu_value_pos],
    )
    ctx_prior[:, logsig_value_pos] = torch.where(
        reveal_logsig[:, None],
        known_latent_features(logsig_internal),
        ctx_prior[:, logsig_value_pos],
    )
    ctx_mode = torch.full((batch_size, ctx_t), VALUE, device=device)
    ctx_mode[:, mu_value_pos] = PRIOR
    ctx_mode[:, logsig_value_pos] = PRIOR
    ctx_mask = torch.zeros(batch_size, ctx_t, device=device, dtype=torch.bool)
    ctx_mask[:, :max_context] = ar < n_ctx[:, None]
    ctx_mask[:, mu_value_pos] = True
    ctx_mask[:, logsig_value_pos] = True
    context = make_scalar_tokens(
        var_id=ctx_var,
        value=ctx_value,
        prior=ctx_prior,
        mode=ctx_mode,
        mask=ctx_mask,
        x_dim=1,
    )

    tgt_t = 2 + data_targets
    tgt_var = torch.zeros(batch_size, tgt_t, device=device, dtype=torch.long)
    tgt_var[:, 0] = 1
    tgt_var[:, 1] = 2
    tgt_value = torch.zeros(batch_size, tgt_t, device=device)
    tgt_value[:, 0] = mu_internal
    tgt_value[:, 1] = logsig_internal
    tgt_value[:, 2:] = y[:, max_context:]
    tgt_mask = torch.ones(batch_size, tgt_t, device=device, dtype=torch.bool)
    tgt_mask[:, 0] = ~reveal_mu
    tgt_mask[:, 1] = ~reveal_logsig
    target = make_scalar_tokens(
        var_id=tgt_var,
        value=tgt_value,
        prior=torch.zeros(batch_size, tgt_t, PRIOR_FEATURES, device=device),
        mode=torch.full((batch_size, tgt_t), QUERY, device=device),
        mask=tgt_mask,
        x_dim=1,
    )
    return ToyBatch(
        Batch(vars_, context, target),
        y[:, :max_context],
        mu,
        log_sigma,
        mu_prior_unit,
        mu_prior_nu,
        logsig_prior_unit,
        logsig_prior_nu,
    )


def analytic_posterior(
    y_obs: torch.Tensor,
    *,
    bins: int,
    mu_prior_unit: torch.Tensor,
    mu_prior_nu: torch.Tensor,
    logsig_prior_unit: torch.Tensor,
    logsig_prior_nu: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Exact grid posterior for the Gaussian toy."""

    device = y_obs.device
    mu_grid = torch.linspace(MU_RANGE[0], MU_RANGE[1], bins, device=device)
    logsig_grid = torch.linspace(LOGSIG_RANGE[0], LOGSIG_RANGE[1], bins, device=device)
    sigma_grid = logsig_grid.exp()
    y = y_obs[None, None, :]
    mu = mu_grid[:, None, None]
    sigma = sigma_grid[None, :, None]
    loglike = (-0.5 * ((y - mu) / sigma).pow(2) - sigma.log() - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1)
    logprior_mu = beta_logprior_on_grid(mu_grid, mu_prior_unit, mu_prior_nu, *MU_RANGE)
    logprior_logsig = beta_logprior_on_grid(logsig_grid, logsig_prior_unit, logsig_prior_nu, *LOGSIG_RANGE)
    logprior = logprior_mu[:, None] + logprior_logsig[None, :]
    logpost = logprior + loglike
    post = (logpost - torch.logsumexp(logpost.reshape(-1), dim=0)).exp()
    pmu = post.sum(dim=1)
    ps = post.sum(dim=0)
    mu_mean = (pmu * mu_grid).sum()
    logsig_mean = (ps * logsig_grid).sum()
    mu_std = (pmu * (mu_grid - mu_mean).pow(2)).sum().sqrt()
    logsig_std = (ps * (logsig_grid - logsig_mean).pow(2)).sum().sqrt()
    cov = (post * (mu_grid[:, None] - mu_mean) * (logsig_grid[None, :] - logsig_mean)).sum()
    corr = cov / (mu_std * logsig_std).clamp_min(1e-8)
    return {
        "mu_grid": mu_grid,
        "logsig_grid": logsig_grid,
        "mu_mean": mu_mean,
        "mu_std": mu_std,
        "logsig_mean": logsig_mean,
        "logsig_std": logsig_std,
        "corr": corr,
        "pmu": pmu,
        "plogsig": ps,
        "post": post,
    }


def predictive_grid(
    oracle: dict[str, torch.Tensor],
    y_obs: torch.Tensor,
    *,
    points: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic posterior predictive density for a new Gaussian observation."""

    mu_grid = oracle["mu_grid"]
    logsig_grid = oracle["logsig_grid"]
    sigma_grid = logsig_grid.exp()
    post = oracle["post"]
    mu = mu_grid[:, None]
    sigma = sigma_grid[None, :]

    pred_mean = (post * mu).sum()
    pred_second = (post * (sigma.pow(2) + mu.pow(2))).sum()
    pred_std = (pred_second - pred_mean.pow(2)).clamp_min(1e-8).sqrt()
    lo = torch.minimum(y_obs.min(), pred_mean - 5.0 * pred_std)
    hi = torch.maximum(y_obs.max(), pred_mean + 5.0 * pred_std)
    y_grid = torch.linspace(float(lo), float(hi), points, device=mu_grid.device)

    y = y_grid[None, None, :]
    comp = -0.5 * ((y - mu[:, :, None]) / sigma[:, :, None]).pow(2)
    comp = comp.exp() / (sigma[:, :, None] * math.sqrt(2.0 * math.pi))
    pred = (post[:, :, None] * comp).sum(dim=(0, 1))
    return y_grid, pred


def build_model(args, device: torch.device) -> ACE:
    """Construct the toy ACE model from CLI hyperparameters."""

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
    """Train ACE online on freshly sampled Gaussian-toy batches."""

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = build_model(args, device) if model is None else model
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        toy = sample_toy_batch(
            model.variables,
            batch_size=args.batch_size,
            max_context=args.max_context,
            min_context=args.min_context,
            data_targets=args.data_targets,
            device=device,
            latent_context_prob=args.latent_context_prob,
        )
        loss = model.loss(toy.batch, latent_weight=args.latent_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0:
            print(f"step {step:5d}/{args.steps}  loss {loss.item():.4f}")
    return model


def save_checkpoint(model: ACE, path: str | Path, args) -> None:
    """Save a lightweight Gaussian example checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": asdict(model.cfg), "seed": args.seed, "state_dict": model.state_dict()}, path)
    print(f"saved checkpoint: {path}")


def load_checkpoint(path: str | Path, device: torch.device) -> ACE:
    """Load a Gaussian example checkpoint saved by `save_checkpoint`."""

    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = ACEConfig(**payload["cfg"])
    model = ACE(variables(), cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    return model


def fixed_eval_batch(vars_: list[Variable], *, device: torch.device | str) -> ToyBatch:
    """Build the fixed Gaussian evaluation batch used by `evaluate`.

    The context is the three observed `y` constants at module scope. The latent
    values are included as target labels so printed diagnostics can report the
    sampled truth for that same case.
    """

    y_obs = torch.tensor(EVAL_Y, device=device)
    mu = torch.tensor([EVAL_TRUE_MU], device=device)
    log_sigma = torch.tensor([EVAL_TRUE_LOGSIG], device=device)
    mu_internal = encode_value(vars_[1], mu)
    logsig_internal = encode_value(vars_[2], log_sigma)
    mu_prior_unit = torch.tensor([EVAL_MU_PRIOR[0]], device=device)
    mu_prior_nu = torch.tensor([EVAL_MU_PRIOR[1]], device=device)
    logsig_prior_unit = torch.tensor([EVAL_LOGSIG_PRIOR[0]], device=device)
    logsig_prior_nu = torch.tensor([EVAL_LOGSIG_PRIOR[1]], device=device)
    n = y_obs.numel()
    ctx_t = n + 2
    ctx_var = torch.zeros(1, ctx_t, device=device, dtype=torch.long)
    ctx_var[:, n] = 1
    ctx_var[:, n + 1] = 2
    ctx_value = torch.zeros(1, ctx_t, device=device)
    ctx_value[:, :n] = y_obs[None, :]
    ctx_value[:, n] = mu_internal
    ctx_value[:, n + 1] = logsig_internal
    ctx_prior = torch.zeros(1, ctx_t, PRIOR_FEATURES, device=device)
    ctx_prior[:, n] = prior_features(mu_prior_unit, mu_prior_nu)
    ctx_prior[:, n + 1] = prior_features(logsig_prior_unit, logsig_prior_nu)
    ctx_mode = torch.full((1, ctx_t), VALUE, device=device)
    ctx_mode[:, n:] = PRIOR
    context = make_scalar_tokens(
        var_id=ctx_var,
        value=ctx_value,
        prior=ctx_prior,
        mode=ctx_mode,
        mask=torch.ones(1, ctx_t, device=device, dtype=torch.bool),
        x_dim=1,
    )
    target = make_scalar_tokens(
        var_id=torch.tensor([[1, 2]], device=device),
        value=torch.stack([mu_internal, logsig_internal], dim=1),
        prior=torch.zeros(1, 2, PRIOR_FEATURES, device=device),
        mode=torch.full((1, 2), QUERY, device=device),
        mask=torch.ones(1, 2, device=device, dtype=torch.bool),
        x_dim=1,
    )
    return ToyBatch(
        Batch(vars_, context, target),
        y_obs[None, :],
        mu,
        log_sigma,
        mu_prior_unit,
        mu_prior_nu,
        logsig_prior_unit,
        logsig_prior_nu,
    )


@torch.no_grad()
def evaluate(model: ACE, *, bins: int) -> Diagnostic:
    """Compare ACE's posterior marginals and AR joint with the analytic oracle."""

    device = next(model.parameters()).device
    toy = fixed_eval_batch(model.variables, device=device)
    true = analytic_posterior(
        toy.y_context[0],
        bins=bins,
        mu_prior_unit=toy.mu_prior_unit,
        mu_prior_nu=toy.mu_prior_nu,
        logsig_prior_unit=toy.logsig_prior_unit,
        logsig_prior_nu=toy.logsig_prior_nu,
    )
    mu_grid = true["mu_grid"]
    logsig_grid = true["logsig_grid"]
    mu_model_grid = encode_value(model.variables[1], mu_grid)
    logsig_model_grid = encode_value(model.variables[2], logsig_grid)
    mu_logp = query_log_density(model, toy.batch, 1, mu_model_grid)
    logsig_logp = query_log_density(model, toy.batch, 2, logsig_model_grid)
    joint_logp = ar_joint_log_density(model, toy.batch, mu_model_grid, logsig_model_grid, first_var=1, second_var=2)
    y_grid, oracle_y_pred = predictive_grid(true, toy.y_context[0])
    model_y_logp = query_log_density(model, toy.batch, 0, y_grid)
    mu_mean, mu_std = normalized_moments(mu_grid, mu_logp)
    logsig_mean, logsig_std = normalized_moments(logsig_grid, logsig_logp)
    oracle_y_mean, oracle_y_std = normalized_moments(y_grid, oracle_y_pred.clamp_min(1e-30).log())
    model_y_mean, model_y_std = normalized_moments(y_grid, model_y_logp)
    p_joint = joint_logp.exp()
    joint_mu_mean = (p_joint.sum(dim=1) * mu_grid).sum()
    joint_s_mean = (p_joint.sum(dim=0) * logsig_grid).sum()
    metrics = {
        "mu_mean_abs_err": float((mu_mean - true["mu_mean"]).abs()),
        "mu_std_abs_err": float((mu_std - true["mu_std"]).abs()),
        "logsig_mean_abs_err": float((logsig_mean - true["logsig_mean"]).abs()),
        "logsig_std_abs_err": float((logsig_std - true["logsig_std"]).abs()),
        "joint_mu_mean_abs_err": float((joint_mu_mean - true["mu_mean"]).abs()),
        "joint_logsig_mean_abs_err": float((joint_s_mean - true["logsig_mean"]).abs()),
        "pred_y_mean_abs_err": float((model_y_mean - oracle_y_mean).abs()),
        "pred_y_std_abs_err": float((model_y_std - oracle_y_std).abs()),
        "oracle_corr": float(true["corr"]),
    }
    print("\nGaussian toy posterior moments")
    print(f"eval context    {int(toy.y_context.shape[1])} fixed observed y values")
    print(f"truth mu        {float(toy.mu[0]): .3f}")
    print(f"truth sigma     {float(toy.log_sigma[0].exp()): .3f}")
    print(f"oracle corr     {float(true['corr']): .3f}")
    print(f"oracle mu       mean {float(true['mu_mean']): .3f}  std {float(true['mu_std']): .3f}")
    print(f"model  mu       mean {float(mu_mean): .3f}  std {float(mu_std): .3f}")
    print(f"oracle log_sig  mean {float(true['logsig_mean']): .3f}  std {float(true['logsig_std']): .3f}")
    print(f"model  log_sig  mean {float(logsig_mean): .3f}  std {float(logsig_std): .3f}")
    print(f"oracle pred y   mean {float(oracle_y_mean): .3f}  std {float(oracle_y_std): .3f}")
    print(f"model  pred y   mean {float(model_y_mean): .3f}  std {float(model_y_std): .3f}")
    print(f"AR joint mean   mu {float(joint_mu_mean): .3f}  log_sig {float(joint_s_mean): .3f}")
    return Diagnostic(toy, true, mu_logp, logsig_logp, joint_logp, y_grid, oracle_y_pred, model_y_logp, metrics)


def plot_diagnostic(diag: Diagnostic, path: str | Path) -> None:
    """Save a compact posterior diagnostic figure."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mu = diag.oracle["mu_grid"].detach().cpu()
    logsig = diag.oracle["logsig_grid"].detach().cpu()
    oracle_joint = diag.oracle["post"].detach().cpu().T
    model_joint = diag.joint_logp.exp().detach().cpu().T
    oracle_mu = diag.oracle["pmu"].detach().cpu()
    oracle_s = diag.oracle["plogsig"].detach().cpu()
    model_mu = (diag.mu_logp - torch.logsumexp(diag.mu_logp, dim=0)).exp().detach().cpu()
    model_s = (diag.logsig_logp - torch.logsumexp(diag.logsig_logp, dim=0)).exp().detach().cpu()
    prior_mu_logp = beta_logprior_on_grid(
        mu,
        diag.toy.mu_prior_unit.detach().cpu()[0],
        diag.toy.mu_prior_nu.detach().cpu()[0],
        *MU_RANGE,
    )
    prior_s_logp = beta_logprior_on_grid(
        logsig,
        diag.toy.logsig_prior_unit.detach().cpu()[0],
        diag.toy.logsig_prior_nu.detach().cpu()[0],
        *LOGSIG_RANGE,
    )
    prior_mu = (prior_mu_logp - torch.logsumexp(prior_mu_logp, dim=0)).exp()
    prior_s = (prior_s_logp - torch.logsumexp(prior_s_logp, dim=0)).exp()
    y_grid = diag.y_grid.detach().cpu()
    oracle_y = diag.oracle_y_pred.detach().cpu()
    model_y = diag.model_y_logp.exp().detach().cpu()
    y_obs = diag.toy.y_context[0].detach().cpu()

    fig = plt.figure(figsize=(9, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.85])
    ax_oracle_joint = fig.add_subplot(gs[0, 0])
    ax_model_joint = fig.add_subplot(gs[0, 1])
    ax_mu = fig.add_subplot(gs[1, 0])
    ax_s = fig.add_subplot(gs[1, 1])
    ax_pred = fig.add_subplot(gs[2, :])

    extent = [float(mu[0]), float(mu[-1]), float(logsig[0]), float(logsig[-1])]
    ax_oracle_joint.imshow(oracle_joint, origin="lower", aspect="auto", extent=extent)
    ax_oracle_joint.set_title("Oracle joint")
    ax_model_joint.imshow(model_joint, origin="lower", aspect="auto", extent=extent)
    ax_model_joint.set_title("ACE AR joint")
    for ax in (ax_oracle_joint, ax_model_joint):
        ax.set_xlabel("mu")
        ax.set_ylabel("log_sigma")

    ax_mu.plot(mu, prior_mu, color="0.35", linestyle=":", label="prior")
    ax_mu.plot(mu, oracle_mu, label="oracle")
    ax_mu.plot(mu, model_mu, label="ACE")
    ax_mu.set_title("mu marginal")
    ax_mu.set_xlabel("mu")
    ax_mu.legend()

    ax_s.plot(logsig, prior_s, color="0.35", linestyle=":", label="prior")
    ax_s.plot(logsig, oracle_s, label="oracle")
    ax_s.plot(logsig, model_s, label="ACE")
    ax_s.set_title("log_sigma marginal")
    ax_s.set_xlabel("log_sigma")
    ax_s.legend()

    ax_pred.plot(y_grid, oracle_y, label="oracle")
    ax_pred.plot(y_grid, model_y, label="ACE")
    for y in y_obs:
        ax_pred.axvline(float(y), color="0.25", alpha=0.2, linewidth=1.0)
    ax_pred.set_title("posterior predictive")
    ax_pred.set_xlabel("new y")
    ax_pred.set_ylabel("density")
    ax_pred.legend()

    fig.suptitle(f"eval N={int(y_obs.numel())}, oracle corr={float(diag.oracle['corr']):.2f}")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved diagnostic plot: {path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the Gaussian example."""

    p = argparse.ArgumentParser(description="Train/evaluate the nanoACE Gaussian toy.")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bins", type=int, default=64, help="oracle/diagnostic grid bins")
    p.add_argument("--max-context", type=int, default=16)
    p.add_argument("--min-context", type=int, default=2)
    p.add_argument("--data-targets", type=int, default=4)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--components", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--latent-weight", type=float, default=2.0)
    p.add_argument("--latent-context-prob", type=float, default=0.25)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--plot-path", default="artifacts/gaussian_toy.png")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--save-checkpoint", default="")
    p.add_argument("--load-checkpoint", default="")
    p.add_argument("--eval-only", action="store_true")
    return p.parse_args()


def main() -> None:
    """Run Gaussian training/evaluation from the command line."""

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

    diag = evaluate(model, bins=args.bins)
    if args.save_checkpoint:
        save_checkpoint(model, args.save_checkpoint, args)
    if not args.no_plot and args.plot_path:
        plot_diagnostic(diag, args.plot_path)


if __name__ == "__main__":
    main()
