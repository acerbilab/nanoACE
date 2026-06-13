"""Generate parity fixtures: run the real PyTorch model on deterministic cases and
dump inputs + intermediates + outputs for the TypeScript parity test.

Each case dumps the embedded context (pre-attention), per-block ctx/tgt states,
raw head outputs (cont_raw, disc_logits), and derived log_prob/mean — so the TS
test can both verify final outputs and localize any divergence to a layer.

Run from the project venv:

    python playground/parity.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ace  # noqa: E402
import diagnostics  # noqa: E402
import bo1d  # noqa: E402
import gaussian_toy  # noqa: E402
import gp1d  # noqa: E402
import sbi_sir  # noqa: E402
from ace import PRIOR, QUERY, VALUE, Batch, Tokens, encode_value  # noqa: E402
from export_weights import quantize_fp16_inplace  # noqa: E402  (same dir as this script)
from extensions.arbuffer.arbuffer import (  # noqa: E402  (non-core extension)
    BufferedBatch,
    load_buffered_checkpoint,
    sample_joint,
)
from extensions.aline.aline import load_aline_checkpoint  # noqa: E402  (non-core extension)

OUT_DIR = Path(__file__).resolve().parent / "test" / "fixtures"

# The retained AR-buffer fine-tune (200k, concat read, K=64, joint training). On a
# retrain, re-run export_weights.py and this script together. Fixtures are skipped
# (with a note) when the artifact is absent, so the four core fixture sets stay
# regenerable.
ARBUF_CKPT = REPO_ROOT / "artifacts" / "gp1d_arbuffer.pt"

# The ALINE active-learning fine-tune (extensions/aline/; the 35k run, promoted
# to the canonical name 2026-06-13 — see extensions/aline/DEVLOG.md; the 5k
# validation artifact is preserved as gp1d_aline_5k.pt). Swap in a longer run by
# re-running export + this script together. Parity fixtures are gated like
# arbuffer's; the env fixture below is checkpoint-independent and regenerates on
# every run.
ALINE_CKPT = REPO_ROOT / "artifacts" / "gp1d_aline.pt"


class TokenBuilder:
    """Accumulate single tokens, then materialize a [1, T, ...] Tokens object."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, var_id, mode, *, mask=True, x=0.0, value=0.0, value_index=0, prior=(0.0, 0.0)):
        self.rows.append(
            dict(var_id=var_id, mode=mode, mask=bool(mask), x=[float(x)],
                 value=float(value), value_index=int(value_index), prior=[float(prior[0]), float(prior[1])])
        )
        return self

    def tokens(self, device) -> Tokens:
        r = self.rows
        return Tokens(
            var_id=torch.tensor([[t["var_id"] for t in r]], dtype=torch.long, device=device),
            x=torch.tensor([[t["x"] for t in r]], dtype=torch.float32, device=device),
            value=torch.tensor([[t["value"] for t in r]], dtype=torch.float32, device=device),
            value_index=torch.tensor([[t["value_index"] for t in r]], dtype=torch.long, device=device),
            prior=torch.tensor([[t["prior"] for t in r]], dtype=torch.float32, device=device),
            mode=torch.tensor([[t["mode"] for t in r]], dtype=torch.long, device=device),
            mask=torch.tensor([[t["mask"] for t in r]], dtype=torch.bool, device=device),
        )

    def json(self) -> dict:
        r = self.rows
        return {
            "var_id": [t["var_id"] for t in r],
            "x": [t["x"] for t in r],
            "value": [t["value"] for t in r],
            "value_index": [t["value_index"] for t in r],
            "prior": [t["prior"] for t in r],
            "mode": [t["mode"] for t in r],
            "mask": [t["mask"] for t in r],
        }


def enc(model, var_id, native) -> float:
    return float(encode_value(model.variables[var_id], torch.tensor(float(native))))


@torch.no_grad()
def run_case(model, name, ctx: TokenBuilder, tgt: TokenBuilder) -> dict:
    device = next(model.parameters()).device
    context = ctx.tokens(device)
    target = tgt.tokens(device)
    batch = Batch(model.variables, context, target)

    # Replicate ACE.forward step by step to capture intermediates.
    ce = model._embed(context)
    te = model._embed(target)
    embed_context = ce.clone()
    per_ctx, per_tgt = [], []
    c, t = ce, te
    for block in model.blocks:
        c, t = block(c, t, context.mask, target.mask)
        per_ctx.append(c.clone())
        per_tgt.append(t.clone())
    t_norm = model.final_norm(t)
    cont_raw = model.cont_head(t_norm)
    disc_logits = model.disc_head(t_norm)

    # Sanity: the official forward must match our manual replication.
    pred = model(batch)
    assert torch.allclose(cont_raw, pred.cont_raw, atol=1e-6), name
    assert torch.allclose(disc_logits, pred.disc_logits, atol=1e-6), name

    logp = pred.log_prob(target)[0].tolist()
    mean = pred.mean(target)[0].tolist()

    return {
        "name": name,
        "context": ctx.json(),
        "target": tgt.json(),
        "embed_context": embed_context[0].tolist(),
        "per_layer_ctx": [x[0].tolist() for x in per_ctx],
        "per_layer_tgt": [x[0].tolist() for x in per_tgt],
        "cont_raw": cont_raw[0].tolist(),
        "disc_logits": disc_logits[0].tolist(),
        "log_prob": logp,
        "mean": mean,
    }


def gp_cases(model) -> list[dict]:
    cases = []

    # Case 1: data-only context; query both continuous latents, the kernel, and data.
    ctx = TokenBuilder()
    for x, y in [(-0.6, 0.4), (-0.1, -0.3), (0.3, 0.8), (0.55, 0.2)]:
        ctx.add(0, VALUE, x=x, value=y)
    tgt = TokenBuilder()
    tgt.add(1, QUERY, value=enc(model, 1, gp1d.EVAL_LOG_LENGTHSCALE))
    tgt.add(2, QUERY, value=enc(model, 2, gp1d.EVAL_LOG_OUTPUTSCALE))
    tgt.add(3, QUERY, value_index=2)
    tgt.add(0, QUERY, x=0.0, value=0.1)
    tgt.add(0, QUERY, x=0.42, value=-0.2)
    cases.append(run_case(model, "gp_data_only", ctx, tgt))

    # Case 2: pinned kernel (discrete VALUE) + pinned lengthscale (zero-spread PRIOR).
    ell_int = enc(model, 1, gp1d.EVAL_LOG_LENGTHSCALE)
    ctx = TokenBuilder()
    for x, y in [(-0.5, 0.3), (0.1, -0.4), (0.6, 0.5)]:
        ctx.add(0, VALUE, x=x, value=y)
    ctx.add(3, VALUE, value=3.0, value_index=3)  # kernel = Periodic, in context
    ctx.add(1, PRIOR, value=ell_int, prior=(ell_int, 0.0))  # known lengthscale
    tgt = TokenBuilder()
    tgt.add(2, QUERY, value=enc(model, 2, gp1d.EVAL_LOG_OUTPUTSCALE))
    tgt.add(0, QUERY, x=0.25, value=0.0)
    cases.append(run_case(model, "gp_pinned_kernel_and_ell", ctx, tgt))

    # Case 3: padded context (masked tokens) to exercise key_padding_mask.
    ctx = TokenBuilder()
    ctx.add(0, VALUE, x=-0.4, value=0.2)
    ctx.add(0, VALUE, x=0.0, value=0.0, mask=False)  # padding
    ctx.add(0, VALUE, x=0.3, value=-0.5)
    ctx.add(0, VALUE, x=0.0, value=0.0, mask=False)  # padding
    tgt = TokenBuilder()
    tgt.add(1, QUERY, value=enc(model, 1, gp1d.EVAL_LOG_LENGTHSCALE))
    tgt.add(0, QUERY, x=0.15, value=0.1)
    cases.append(run_case(model, "gp_padded_context", ctx, tgt))

    return cases


def gaussian_cases(model) -> list[dict]:
    cases = []
    pf = gaussian_toy.prior_features

    def prior_vec(mu_unit, nu):
        return tuple(pf(torch.tensor(float(mu_unit)), torch.tensor(float(nu))).tolist())

    # Case 1: finite-spread Beta priors on mu & log_sigma + observed y.
    ctx = TokenBuilder()
    for y in gaussian_toy.EVAL_Y:
        ctx.add(0, VALUE, value=y)
    ctx.add(1, PRIOR, prior=prior_vec(0.70, 20.0))
    ctx.add(2, PRIOR, prior=prior_vec(0.70, 8.0))
    tgt = TokenBuilder()
    tgt.add(1, QUERY, value=enc(model, 1, 0.5))
    tgt.add(2, QUERY, value=enc(model, 2, -0.5))
    tgt.add(0, QUERY, value=0.9)
    cases.append(run_case(model, "gaussian_beta_priors", ctx, tgt))

    # Case 2: mu revealed as zero-spread PRIOR; log_sigma keeps a finite Beta prior.
    mu_int = enc(model, 1, 0.8)
    ctx = TokenBuilder()
    for y in (0.5, -0.2, 1.1):
        ctx.add(0, VALUE, value=y)
    ctx.add(1, PRIOR, prior=(mu_int, 0.0))  # known mu
    ctx.add(2, PRIOR, prior=prior_vec(0.5, 2.0))  # uniform Beta(1,1)
    tgt = TokenBuilder()
    tgt.add(2, QUERY, value=enc(model, 2, -0.3))
    tgt.add(0, QUERY, value=0.4)
    cases.append(run_case(model, "gaussian_known_mu", ctx, tgt))

    return cases


def sir_cases(model) -> list[dict]:
    cases = []
    pf = sbi_sir.prior_features

    def prior_vec(mu_unit, nu):
        return tuple(pf(torch.tensor(float(mu_unit)), torch.tensor(float(nu))).tolist())

    # Case 1: finite-spread Beta priors on beta/gamma + observed infected fractions.
    ctx = TokenBuilder()
    for t, y in [(3.0, 0.018), (8.0, 0.043), (14.0, 0.145), (22.0, 0.305)]:
        ctx.add(0, VALUE, x=float(sbi_sir.scale_time(torch.tensor(t))), value=float(sbi_sir.scale_value(torch.tensor(y))))
    ctx.add(1, PRIOR, prior=prior_vec(0.60, 12.0))
    ctx.add(2, PRIOR, prior=prior_vec(0.45, 10.0))
    tgt = TokenBuilder()
    tgt.add(1, QUERY, value=enc(model, 1, sbi_sir.EVAL_BETA))
    tgt.add(2, QUERY, value=enc(model, 2, sbi_sir.EVAL_GAMMA))
    tgt.add(0, QUERY, x=float(sbi_sir.scale_time(torch.tensor(18.0))), value=float(sbi_sir.scale_value(torch.tensor(0.20))))
    cases.append(run_case(model, "sir_beta_priors", ctx, tgt))

    # Case 2: beta revealed as zero-spread PRIOR; gamma remains finite-spread.
    beta_int = enc(model, 1, sbi_sir.EVAL_BETA)
    ctx = TokenBuilder()
    for t, y in [(4.0, 0.02), (10.0, 0.07), (16.0, 0.18)]:
        ctx.add(0, VALUE, x=float(sbi_sir.scale_time(torch.tensor(t))), value=float(sbi_sir.scale_value(torch.tensor(y))))
    ctx.add(1, PRIOR, value=beta_int, prior=(beta_int, 0.0))
    ctx.add(2, PRIOR, prior=prior_vec(0.50, 2.0))
    tgt = TokenBuilder()
    tgt.add(2, QUERY, value=enc(model, 2, sbi_sir.EVAL_GAMMA))
    tgt.add(0, QUERY, x=float(sbi_sir.scale_time(torch.tensor(24.0))), value=float(sbi_sir.scale_value(torch.tensor(0.30))))
    cases.append(run_case(model, "sir_known_beta", ctx, tgt))

    # Case 3: padded data context plus active priors.
    ctx = TokenBuilder()
    ctx.add(0, VALUE, x=float(sbi_sir.scale_time(torch.tensor(5.0))), value=float(sbi_sir.scale_value(torch.tensor(0.025))))
    ctx.add(0, VALUE, x=0.0, value=0.0, mask=False)
    ctx.add(0, VALUE, x=float(sbi_sir.scale_time(torch.tensor(12.0))), value=float(sbi_sir.scale_value(torch.tensor(0.10))))
    ctx.add(1, PRIOR, prior=prior_vec(0.50, 2.0))
    ctx.add(2, PRIOR, prior=prior_vec(0.50, 2.0))
    tgt = TokenBuilder()
    tgt.add(1, QUERY, value=enc(model, 1, 0.45))
    tgt.add(0, QUERY, x=float(sbi_sir.scale_time(torch.tensor(30.0))), value=float(sbi_sir.scale_value(torch.tensor(0.18))))
    cases.append(run_case(model, "sir_padded_context", ctx, tgt))

    return cases


def bo_cases(model) -> list[dict]:
    cases = []

    def x_prior_vec(mu_unit, nu):
        return tuple(bo1d.prior_features(torch.tensor(float(mu_unit)), torch.tensor(float(nu))).tolist())

    def y_prior_vec(mu_unit, nu):
        return tuple(bo1d.y_opt_prior_features(torch.tensor([float(mu_unit)]), torch.tensor([float(nu)]))[0].tolist())

    # Case 1: finite-spread priors on x_opt/y_opt + observed function values.
    ctx = TokenBuilder()
    for x, y in [(-0.8, 0.25), (-0.5, 0.05), (0.1, -0.28), (0.7, 0.2)]:
        ctx.add(0, VALUE, x=x, value=float(bo1d.scale_y(torch.tensor(y))))
    ctx.add(1, PRIOR, prior=x_prior_vec(0.70, 25.0))
    ctx.add(2, PRIOR, prior=y_prior_vec(0.50, 2.0))
    tgt = TokenBuilder()
    tgt.add(1, QUERY, value=enc(model, 1, bo1d.EVAL_X_OPT))
    tgt.add(2, QUERY, value=enc(model, 2, bo1d.EVAL_Y_OPT))
    tgt.add(0, QUERY, x=0.4, value=float(bo1d.scale_y(torch.tensor(-0.4))))
    cases.append(run_case(model, "bo_beta_priors", ctx, tgt))

    # Case 2: y_opt fixed as a zero-spread PRIOR; x_opt remains queried.
    y_int = enc(model, 2, bo1d.EVAL_Y_OPT)
    ctx = TokenBuilder()
    for x, y in [(-0.6, 0.18), (0.0, -0.12), (0.65, 0.22)]:
        ctx.add(0, VALUE, x=x, value=float(bo1d.scale_y(torch.tensor(y))))
    ctx.add(1, PRIOR, prior=x_prior_vec(0.50, 2.0))
    ctx.add(2, PRIOR, value=y_int, prior=(y_int, 0.0))
    tgt = TokenBuilder()
    tgt.add(1, QUERY, value=enc(model, 1, 0.25))
    tgt.add(0, QUERY, x=0.25, value=float(bo1d.scale_y(torch.tensor(-0.2))))
    cases.append(run_case(model, "bo_known_y_opt", ctx, tgt))

    # Case 3: padded data context plus active priors.
    ctx = TokenBuilder()
    ctx.add(0, VALUE, x=-0.4, value=float(bo1d.scale_y(torch.tensor(0.1))))
    ctx.add(0, VALUE, x=0.0, value=0.0, mask=False)
    ctx.add(0, VALUE, x=0.3, value=float(bo1d.scale_y(torch.tensor(-0.25))))
    ctx.add(1, PRIOR, prior=x_prior_vec(0.50, 2.0))
    ctx.add(2, PRIOR, prior=y_prior_vec(0.50, 2.0))
    tgt = TokenBuilder()
    tgt.add(2, QUERY, value=enc(model, 2, -0.5))
    tgt.add(0, QUERY, x=0.1, value=float(bo1d.scale_y(torch.tensor(0.0))))
    cases.append(run_case(model, "bo_padded_context", ctx, tgt))

    return cases


@torch.no_grad()
def gp_demo_reference(model) -> dict:
    """End-to-end reference for the GP demo's orchestration, using gp1d.py's own
    helpers on the fixed eval context (data-only, no pins). The TS `gpInfer` must
    reproduce these band/kernel/latent quantities."""

    device = next(model.parameters()).device
    toy = gp1d.fixed_eval_batch(model.variables, device=device, points=2, jitter=1e-5)
    x_ctx = toy.x_context[0]
    y_ctx = toy.y_context[0]

    ctx = TokenBuilder()
    for xi, yi in zip(x_ctx.tolist(), y_ctx.tolist()):
        ctx.add(0, VALUE, x=xi, value=yi)
    context = ctx.tokens(device)
    batch = Batch(model.variables, context, context)  # dummy target; helpers use context only

    band_x = torch.linspace(-1.0, 1.0, 161)
    bt = TokenBuilder()
    for xi in band_x.tolist():
        bt.add(0, QUERY, x=xi)
    btarget = bt.tokens(device)
    pred = model(Batch(model.variables, context, btarget))
    band_mean = pred.mean(btarget)[0].tolist()
    band_std = pred.continuous_var()[0].clamp_min(0.0).sqrt().tolist()

    kernel_probs = gp1d.kernel_posterior(model, batch).tolist()

    n = 80
    ell_grid = torch.linspace(model.variables[1].bounds[0], model.variables[1].bounds[1], n)
    scale_grid = torch.linspace(model.variables[2].bounds[0], model.variables[2].bounds[1], n)
    ell_logp = diagnostics.query_log_density(model, batch, 1, encode_value(model.variables[1], ell_grid))
    scale_logp = diagnostics.query_log_density(model, batch, 2, encode_value(model.variables[2], scale_grid))

    def norm(logp):
        return (logp - torch.logsumexp(logp, dim=0)).exp().tolist()

    return {
        "x_context": x_ctx.tolist(),
        "y_context": y_ctx.tolist(),
        "band_x": band_x.tolist(),
        "band_mean": band_mean,
        "band_std": band_std,
        "kernel_probs": kernel_probs,
        "ell_grid": ell_grid.tolist(),
        "ell_post": norm(ell_logp),
        "scale_grid": scale_grid.tolist(),
        "scale_post": norm(scale_logp),
    }


@torch.no_grad()
def gaussian_demo_reference(model) -> dict:
    """End-to-end reference for the Gaussian demo: ACE marginals/predictive (via
    gaussian_toy/diagnostics helpers) and the analytic oracle, on the fixed eval
    batch. The TS gaussInfer + oracle must reproduce these."""

    device = next(model.parameters()).device
    toy = gaussian_toy.fixed_eval_batch(model.variables, device=device)
    batch = toy.batch
    y_obs = toy.y_context[0]
    bins = 80

    mu_grid = torch.linspace(gaussian_toy.MU_RANGE[0], gaussian_toy.MU_RANGE[1], bins)
    ls_grid = torch.linspace(gaussian_toy.LOGSIG_RANGE[0], gaussian_toy.LOGSIG_RANGE[1], bins)
    y_grid = torch.linspace(-3.5, 3.5, 161)

    mu_logp = diagnostics.query_log_density(model, batch, 1, encode_value(model.variables[1], mu_grid))
    ls_logp = diagnostics.query_log_density(model, batch, 2, encode_value(model.variables[2], ls_grid))
    y_logp = diagnostics.query_log_density(model, batch, 0, y_grid)

    def norm(logp):
        return (logp - torch.logsumexp(logp, dim=0)).exp().tolist()

    true = gaussian_toy.analytic_posterior(
        y_obs,
        bins=bins,
        mu_prior_unit=toy.mu_prior_unit,
        mu_prior_nu=toy.mu_prior_nu,
        logsig_prior_unit=toy.logsig_prior_unit,
        logsig_prior_nu=toy.logsig_prior_nu,
    )
    post = true["post"]  # [mu, log_sigma]
    sig = true["logsig_grid"].exp()
    yk = y_grid[:, None, None]
    mug = true["mu_grid"][None, :, None]
    sg = sig[None, None, :]
    comp = torch.exp(-0.5 * ((yk - mug) / sg) ** 2) / (sg * math.sqrt(2.0 * math.pi))
    pred_oracle = (post[None] * comp).sum(dim=(1, 2))

    return {
        "y_obs": y_obs.tolist(),
        "mu_unit": float(toy.mu_prior_unit[0]),
        "mu_nu": float(toy.mu_prior_nu[0]),
        "ls_unit": float(toy.logsig_prior_unit[0]),
        "ls_nu": float(toy.logsig_prior_nu[0]),
        "mu_grid": mu_grid.tolist(),
        "ls_grid": ls_grid.tolist(),
        "y_grid": y_grid.tolist(),
        "mu_post_ace": norm(mu_logp),
        "ls_post_ace": norm(ls_logp),
        "pred_ace": y_logp.exp().tolist(),
        "mu_post_oracle": true["pmu"].tolist(),
        "ls_post_oracle": true["plogsig"].tolist(),
        "pred_oracle": pred_oracle.tolist(),
    }


@torch.no_grad()
def sir_demo_reference(model) -> dict:
    """End-to-end reference for the SIR demo: ACE rate marginals and predictive
    curve plus the numerical grid oracle on the fixed informative-prior case."""

    device = next(model.parameters()).device
    bins = 48
    points = 121
    toy = sbi_sir.fixed_eval_batch(model.variables, device=device, points=points, prior_kind="informative")

    pred = model(toy.batch)
    y_mean = sbi_sir.unscale_value(pred.mean(toy.batch.target)[0])
    y_std = pred.continuous_var()[0].clamp_min(1e-8).sqrt() * sbi_sir.DATA_SCALE

    beta_grid = torch.linspace(sbi_sir.BETA_RANGE[0], sbi_sir.BETA_RANGE[1], bins, device=device)
    gamma_grid = torch.linspace(sbi_sir.GAMMA_RANGE[0], sbi_sir.GAMMA_RANGE[1], bins, device=device)
    beta_logp = diagnostics.query_log_density(model, toy.batch, 1, encode_value(model.variables[1], beta_grid))
    gamma_logp = diagnostics.query_log_density(model, toy.batch, 2, encode_value(model.variables[2], gamma_grid))

    def norm(logp):
        return (logp - torch.logsumexp(logp, dim=0)).exp().tolist()

    oracle = sbi_sir.sir_oracle(toy, bins=bins, sigma_obs=sbi_sir.SIGMA_OBS)

    return {
        "t_context": toy.t_context[0].tolist(),
        "y_context": toy.y_context[0].tolist(),
        "beta_unit": float(toy.beta_prior_unit[0]),
        "beta_nu": float(toy.beta_prior_nu[0]),
        "gamma_unit": float(toy.gamma_prior_unit[0]),
        "gamma_nu": float(toy.gamma_prior_nu[0]),
        "t_grid": toy.t_target[0].tolist(),
        "beta_grid": beta_grid.tolist(),
        "gamma_grid": gamma_grid.tolist(),
        "beta_post_ace": norm(beta_logp),
        "gamma_post_ace": norm(gamma_logp),
        "y_mean_ace": y_mean.tolist(),
        "y_std_ace": y_std.tolist(),
        "beta_post_oracle": oracle.beta_probs.tolist(),
        "gamma_post_oracle": oracle.gamma_probs.tolist(),
        "y_mean_oracle": oracle.y_mean.tolist(),
        "y_std_oracle": oracle.y_std.tolist(),
    }


@torch.no_grad()
def bo_demo_reference(model) -> dict:
    """End-to-end reference for the BO demo's playground orchestration.

    There is deliberately no oracle here: this fixture only checks that the TS
    token construction and forward pass reproduce bo1d.py on the fixed case.
    """

    device = next(model.parameters()).device
    bins = 80
    points = 161
    toy = bo1d.fixed_eval_batch(model.variables, device=device, points=points, prior_kind="uniform", jitter=1e-5)

    pred = model(toy.batch)
    y_mean = bo1d.unscale_y(pred.mean(toy.batch.target)[0])
    y_std = pred.continuous_var()[0].clamp_min(1e-8).sqrt() * (0.5 * (bo1d.Y_RANGE[1] - bo1d.Y_RANGE[0]))

    x_grid = torch.linspace(bo1d.X_OPT_RANGE[0], bo1d.X_OPT_RANGE[1], bins, device=device)
    y_grid = torch.linspace(bo1d.Y_OPT_RANGE[0], bo1d.Y_OPT_RANGE[1], bins, device=device)
    x_logp = diagnostics.query_log_density(model, toy.batch, 1, encode_value(model.variables[1], x_grid))
    y_logp = diagnostics.query_log_density(model, toy.batch, 2, encode_value(model.variables[2], y_grid))

    def norm(logp):
        return (logp - torch.logsumexp(logp, dim=0)).exp().tolist()

    return {
        "x_context": toy.x_context[0].tolist(),
        "y_context": toy.y_context[0].tolist(),
        "x_prior_unit": float(toy.x_opt_prior_unit[0]),
        "x_prior_nu": float(toy.x_opt_prior_nu[0]),
        "y_prior_unit": float(toy.y_opt_prior_unit[0]),
        "y_prior_nu": float(toy.y_opt_prior_nu[0]),
        "band_x": toy.x_target[0].tolist(),
        "band_mean": y_mean.tolist(),
        "band_std": y_std.tolist(),
        "x_grid": x_grid.tolist(),
        "x_post": norm(x_logp),
        "y_grid": y_grid.tolist(),
        "y_post": norm(y_logp),
    }


@torch.no_grad()
def arbuf_packed_case(model) -> dict:
    """Packed `forward_buffered` reference with `prefix_len = arange(K)`.

    Gives the TS incremental sampler per-layer, per-token reference states: buffer
    row `j` corresponds to the TS append pass for token `j`, and target row `m`
    (which sees exactly the first `m` buffer tokens) corresponds to TS decode step
    `m`. Everything is dense/all-active, so the per-layer mask zeroing is the
    identity and the captured ctx states equal `encode_context`'s pre-zeroing ones.
    """

    device = next(model.parameters()).device
    ctx_b = TokenBuilder()
    for x, y in [(-0.6, 0.4), (-0.1, -0.3), (0.3, 0.8), (0.55, 0.2)]:
        ctx_b.add(0, VALUE, x=x, value=y)
    ctx_b.add(3, VALUE, value=2.0, value_index=2)  # kernel pinned: Matern-3/2

    buf_xy = [
        (-0.9, 0.12), (-0.72, 0.31), (-0.48, 0.05), (-0.31, -0.22),
        (-0.14, -0.41), (0.04, -0.12), (0.18, 0.47), (0.39, 0.63),
        (0.57, 0.36), (0.71, 0.08), (0.84, -0.17), (0.96, -0.33),
    ]
    buf_b = TokenBuilder()
    tgt_b = TokenBuilder()
    for x, y in buf_xy:
        buf_b.add(0, VALUE, x=x, value=y)
        tgt_b.add(0, QUERY, x=x, value=y)  # truth in value for log_prob

    context = ctx_b.tokens(device)
    buffer = buf_b.tokens(device)
    target = tgt_b.tokens(device)
    k = len(buf_xy)
    prefix = torch.arange(k, device=device)[None, :]
    bbatch = BufferedBatch(model.variables, context, buffer, target, prefix)

    # Replicate forward_buffered step by step to capture per-layer states (the same
    # pattern as run_case's manual replication of ACE.forward). Both read modes are
    # mirrored so the sanity-assert below stays meaningful for either checkpoint.
    ctx = model._embed(context)
    buf = model._embed(buffer)
    tgt = model._embed(target)
    kp_ctx = ~context.mask
    b, n = context.mask.shape
    m = tgt.shape[1]
    heads = model.cfg.n_heads
    kp_ctx_buf = torch.cat([kp_ctx, torch.zeros(b, k, dtype=torch.bool, device=device)], dim=1)
    ar = torch.arange(k, device=device)
    causal = torch.zeros(k, n + k, dtype=torch.bool, device=device)
    causal[:, n:] = ar[None, :] > ar[:, None]
    blocked = ar[None, None, :] >= prefix[:, :, None]
    if model.concat_read:
        blocked_all = torch.cat([kp_ctx[:, None, :].expand(b, m, n), blocked], dim=2)
        blocked_all = blocked_all[:, None].expand(b, heads, m, n + k)
    else:
        zero_prefix = prefix == 0
        blocked[..., 0] = blocked[..., 0] & ~zero_prefix
        prefix_mask = blocked.repeat_interleave(heads, dim=0)
        gate = (~zero_prefix).to(tgt.dtype).unsqueeze(-1)

    per_ctx, per_buf, per_tgt = [], [], []
    for blk, bblk in zip(model.blocks, model.buf_blocks):
        ctx_in = ctx
        ctx_q = blk.ctx_ln1(ctx)
        ctx_att, _ = blk.ctx_attn(ctx_q, ctx_q, ctx_q, key_padding_mask=kp_ctx, need_weights=False)
        ctx = ctx + ctx_att
        ctx = ctx + blk.ctx_mlp(blk.ctx_ln2(ctx))
        buf_q = bblk.buf_ln1(buf)
        buf_kv = bblk.buf_ln1(torch.cat([ctx_in, buf], dim=1))
        buf_att, _ = bblk.buf_attn(
            buf_q, buf_kv, buf_kv, key_padding_mask=kp_ctx_buf, attn_mask=causal, need_weights=False
        )
        buf = buf + buf_att
        buf = buf + bblk.buf_mlp(bblk.buf_ln2(buf))
        if model.concat_read:
            kv = blk.kv_ln(torch.cat([ctx, buf], dim=1))
            fmask = torch.zeros(b, heads, m, n + k, dtype=tgt.dtype, device=device)
            fmask[..., n:] = bblk.buf_bias.view(1, heads, 1, 1)
            fmask = fmask.masked_fill(blocked_all, float("-inf")).reshape(b * heads, m, n + k)
            tgt_att, _ = blk.cross_attn(blk.tgt_ln1(tgt), kv, kv, attn_mask=fmask, need_weights=False)
            tgt = tgt + tgt_att
        else:
            kv_c = blk.kv_ln(ctx)
            tgt_att, _ = blk.cross_attn(blk.tgt_ln1(tgt), kv_c, kv_c, key_padding_mask=kp_ctx, need_weights=False)
            tgt = tgt + tgt_att
            kv_b = bblk.tgt_buf_kvln(buf)
            read, _ = bblk.tgt_buf_attn(bblk.tgt_buf_qln(tgt), kv_b, kv_b, attn_mask=prefix_mask, need_weights=False)
            tgt = tgt + read * gate
        tgt = tgt + blk.tgt_mlp(blk.tgt_ln2(tgt))
        ctx = ctx * context.mask.unsqueeze(-1)
        tgt = tgt * target.mask.unsqueeze(-1)
        per_ctx.append(ctx.clone())
        per_buf.append(buf.clone())
        per_tgt.append(tgt.clone())
    cont_raw = model.cont_head(model.final_norm(tgt))

    # Sanity: the replication must match the real forward_buffered.
    pred = model.forward_buffered(bbatch)
    assert torch.allclose(cont_raw, pred.cont_raw, atol=1e-6), "arbuf packed replication drifted"

    return {
        "context": ctx_b.json(),
        "buffer": buf_b.json(),
        "target": tgt_b.json(),
        "prefix_len": prefix[0].tolist(),
        "per_layer_ctx": [s[0].tolist() for s in per_ctx],
        "per_layer_buf": [s[0].tolist() for s in per_buf],
        "per_layer_tgt": [s[0].tolist() for s in per_tgt],
        "cont_raw": pred.cont_raw[0].tolist(),
        "log_prob": pred.log_prob(target)[0].tolist(),
    }


@torch.no_grad()
def arbuf_chain_case(model) -> dict:
    """Teacher-forced `sample_joint` chain (fixed order/values, per-step log-probs):
    the exact incremental-cache semantics the TS sampler implements, end to end."""

    device = next(model.parameters()).device
    ctx_b = TokenBuilder()
    for x, y in [(-0.6, 0.4), (-0.1, -0.3), (0.3, 0.8), (0.55, 0.2)]:
        ctx_b.add(0, VALUE, x=x, value=y)
    context = ctx_b.tokens(device)

    kc = 16
    grid = torch.linspace(-1.0, 1.0, kc)
    order = [5, 12, 0, 9, 15, 3, 7, 1, 14, 10, 4, 8, 2, 13, 6, 11]
    forced = torch.stack([0.5 * torch.sin(3.0 * grid), 0.3 * torch.cos(2.0 * grid) - 0.1]).to(device)
    values, logps = sample_joint(
        model, context, grid.to(device), n_draws=2, order=order, teacher_force=forced
    )
    assert torch.equal(values, forced), "teacher-forced chain must return the forced values"

    return {
        "context": ctx_b.json(),
        "grid": grid.tolist(),
        "order": order,
        "values": forced.tolist(),
        "log_prob": logps.tolist(),
    }


@torch.no_grad()
def aline_policy_case(model) -> dict:
    """Policy-decoder reference: final trunk states + per-policy-block candidate
    streams + logits on one compact all-active case.

    Token row order matches the TS port (which omits inactive tokens instead of
    masking): context rows, then the goal-target rows the policy reads, then the
    candidate rows scored pointwise. The target deliberately mixes latent and
    data QUERY goals for embedding coverage; the tab's actual goal sets are
    subsets (covered by the chain case below).
    """

    device = next(model.parameters()).device
    ctx_b = TokenBuilder()
    for x, y in [(-0.7, 0.35), (-0.2, -0.15), (0.25, 0.6), (0.6, 0.1)]:
        ctx_b.add(0, VALUE, x=x, value=y)
    tgt_b = TokenBuilder()
    tgt_b.add(1, QUERY, value=enc(model, 1, gp1d.EVAL_LOG_LENGTHSCALE))
    tgt_b.add(2, QUERY, value=enc(model, 2, gp1d.EVAL_LOG_OUTPUTSCALE))
    tgt_b.add(3, QUERY, value_index=gp1d.EVAL_KERNEL)
    for x, y in [(-0.5, 0.2), (0.0, -0.1), (0.45, 0.3), (0.9, -0.2)]:
        tgt_b.add(0, QUERY, x=x, value=y)
    qry_b = TokenBuilder()
    for x in (-0.95, -0.65, -0.35, -0.05, 0.2, 0.5, 0.75, 0.92):
        qry_b.add(0, QUERY, x=x)

    context = ctx_b.tokens(device)
    target = tgt_b.tokens(device)
    query = qry_b.tokens(device)
    batch = Batch(model.variables, context, target)

    pred, ctx_states, tgt_states = model.forward_with_states(batch)
    plain = model(batch)
    assert torch.equal(pred.cont_raw, plain.cont_raw), "forward_with_states drifted from forward"

    # Manual policy replication (the run_case discipline): per-block candidate
    # streams so a TS divergence localizes to a block. The no_grad/detach in
    # policy_logits are gradient-only, so the values match exactly.
    qry = model._embed(query)
    per_qry = []
    for blk in model.policy_blocks:
        qry = blk(qry, ctx_states, tgt_states, query.mask, context.mask, target.mask)
        per_qry.append(qry.clone())
    logits = model.policy_head(model.policy_norm(qry)).squeeze(-1)
    logits_model = model.policy_logits(query, ctx_states, tgt_states, context.mask, target.mask)
    assert torch.allclose(logits, logits_model, atol=1e-6), "aline policy replication drifted"

    return {
        "context": ctx_b.json(),
        "target": tgt_b.json(),
        "query": qry_b.json(),
        "ctx_states": ctx_states[0].tolist(),
        "tgt_states": tgt_states[0].tolist(),
        "per_policy_qry": [s[0].tolist() for s in per_qry],
        "logits": logits_model[0].tolist(),
        "cont_raw": pred.cont_raw[0].tolist(),
        "log_prob": pred.log_prob(target)[0].tolist(),
    }


@torch.no_grad()
def aline_chain_case(model) -> dict:
    """Teacher-forced compact episode: the exact step semantics the TS tab
    implements (compact all-active tokens; the goal xi = which target tokens
    exist).

    Pool 16 + 4 predictive x* drawn in ONE joint GP draw (the training layout);
    seed observation = pool index 0; T = 6 with a mid-episode goal switch
    (xi = {log_lengthscale} for steps 0-2, then xi = predictive over the 4 x*).
    Per step: available candidates, policy logits, the argmax action (the TS
    test REPLAYS these actions rather than re-deriving argmax, so fp tie-flips
    cannot fail the build), and per-goal-token log-probs from
    `Predictions.log_prob` (not the rollout's xi-averaged log_q).
    """

    device = next(model.parameters()).device
    n_pool, m_star, t_steps, switch_at = 16, 4, 6, 3

    gen = torch.Generator().manual_seed(20260612)
    x_all = 2.0 * torch.rand(1, n_pool + m_star, generator=gen, dtype=torch.float64) - 1.0
    kernel = torch.tensor([gp1d.EVAL_KERNEL], dtype=torch.long)
    log_ell = torch.tensor([gp1d.EVAL_LOG_LENGTHSCALE], dtype=torch.float64)
    log_scale = torch.tensor([gp1d.EVAL_LOG_OUTPUTSCALE], dtype=torch.float64)
    y_all = gp1d.draw_gp(x_all, kernel, log_ell, log_scale, jitter=gp1d.GEN_JITTER, generator=gen)
    pool_x = x_all[0, :n_pool].float().tolist()
    pool_y = y_all[0, :n_pool].float().tolist()
    star_x = x_all[0, n_pool:].float().tolist()
    star_y = y_all[0, n_pool:].float().tolist()
    ell_int = enc(model, 1, float(log_ell[0]))

    def step_state(observed: list[int], available: list[int], xi: str):
        ctx_b = TokenBuilder()
        for i in observed:
            ctx_b.add(0, VALUE, x=pool_x[i], value=pool_y[i])
        tgt_b = TokenBuilder()
        if xi == "ell":
            tgt_b.add(1, QUERY, value=ell_int)
        else:
            for xs, ys in zip(star_x, star_y):
                tgt_b.add(0, QUERY, x=xs, value=ys)
        qry_b = TokenBuilder()
        for i in available:
            qry_b.add(0, QUERY, x=pool_x[i])
        context = ctx_b.tokens(device)
        target = tgt_b.tokens(device)
        query = qry_b.tokens(device)
        pred, ctx_states, tgt_states = model.forward_with_states(
            Batch(model.variables, context, target)
        )
        logits = model.policy_logits(query, ctx_states, tgt_states, context.mask, target.mask)
        return pred, target, logits

    observed = [0]  # seed point
    available = list(range(1, n_pool))
    steps = []
    for t in range(t_steps):
        xi = "ell" if t < switch_at else "pred"
        pred, target, logits = step_state(observed, available, xi)
        pick = int(logits[0].argmax())
        action = available[pick]
        steps.append(
            {
                "xi": xi,
                "available": list(available),
                "logits": logits[0].tolist(),
                "action": action,
                "observed_y": pool_y[action],
                "log_prob": pred.log_prob(target)[0].tolist(),
            }
        )
        observed.append(action)
        available.remove(action)

    # Post-episode state under the predictive goal (no action; convergence check).
    pred, target, _ = step_state(observed, available, "pred")
    final_log_prob = pred.log_prob(target)[0].tolist()

    return {
        "pool_x": pool_x,
        "pool_y": pool_y,
        "star_x": star_x,
        "star_y": star_y,
        "ell_value_internal": ell_int,
        "seed_index": 0,
        "switch_at": switch_at,
        "steps": steps,
        "final_log_prob": final_log_prob,
    }


def aline_env_fixture() -> dict:
    """Checkpoint-independent DGP fixture for the TS environment port: kernel
    matrices and Cholesky factors in float64 on two fixed x-sets (one moderate,
    one adversarial: clustered points + short lengthscale, hardest for the
    Periodic kernel), all four kernels each."""

    x_mod = torch.tensor([[-0.9, -0.55, -0.3, -0.05, 0.2, 0.45, 0.7, 0.95]], dtype=torch.float64)
    x_adv = torch.tensor([[-0.41, -0.40, -0.39, 0.10, 0.11, 0.12, 0.80, 0.805]], dtype=torch.float64)
    log_ell = torch.tensor([math.log(0.15)], dtype=torch.float64)
    log_scale = torch.tensor([math.log(0.75)], dtype=torch.float64)

    cases = []
    for set_name, x in (("moderate", x_mod), ("clustered", x_adv)):
        for k_idx, k_name in enumerate(gp1d.KERNELS):
            kern = torch.tensor([k_idx], dtype=torch.long)
            mat = gp1d._kernel_matrix(x, kern, log_ell, log_scale, jitter=gp1d.GEN_JITTER)
            chol = torch.linalg.cholesky(mat)
            cases.append(
                {
                    "set": set_name,
                    "kernel": k_name,
                    "kernel_index": k_idx,
                    "x": x[0].tolist(),
                    "log_ell": float(log_ell[0]),
                    "log_scale": float(log_scale[0]),
                    "K": mat[0].tolist(),
                    "L": chol[0].tolist(),
                }
            )
    return {
        "jitter": gp1d.GEN_JITTER,
        "period": 1.0,
        "log_lengthscale_range": list(gp1d.LOG_LENGTHSCALE_RANGE),
        "log_outputscale_range": list(gp1d.LOG_OUTPUTSCALE_RANGE),
        "kernels": list(gp1d.KERNELS),
        "cases": cases,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    gp_model = gp1d.load_checkpoint(str(REPO_ROOT / "artifacts" / "gp1d.pt"), device)
    gp_model.eval()
    quantize_fp16_inplace(gp_model)  # match the shipped fp16 weights
    gp = gp_cases(gp_model)
    (OUT_DIR / "gp1d.parity.json").write_text(json.dumps(gp))
    print(f"wrote gp1d.parity.json ({len(gp)} cases)")
    (OUT_DIR / "gp1d.demo.json").write_text(json.dumps(gp_demo_reference(gp_model)))
    print("wrote gp1d.demo.json")

    gauss_model = gaussian_toy.load_checkpoint(str(REPO_ROOT / "artifacts" / "gaussian_toy.pt"), device)
    gauss_model.eval()
    quantize_fp16_inplace(gauss_model)  # match the shipped fp16 weights
    gau = gaussian_cases(gauss_model)
    (OUT_DIR / "gaussian.parity.json").write_text(json.dumps(gau))
    print(f"wrote gaussian.parity.json ({len(gau)} cases)")
    (OUT_DIR / "gaussian.demo.json").write_text(json.dumps(gaussian_demo_reference(gauss_model)))
    print("wrote gaussian.demo.json")

    sir_model = sbi_sir.load_checkpoint(str(REPO_ROOT / "artifacts" / "sbi_sir.pt"), device)
    sir_model.eval()
    quantize_fp16_inplace(sir_model)  # match the shipped fp16 weights
    sir = sir_cases(sir_model)
    (OUT_DIR / "sbi_sir.parity.json").write_text(json.dumps(sir))
    print(f"wrote sbi_sir.parity.json ({len(sir)} cases)")
    (OUT_DIR / "sbi_sir.demo.json").write_text(json.dumps(sir_demo_reference(sir_model)))
    print("wrote sbi_sir.demo.json")

    bo_model = bo1d.load_checkpoint(str(REPO_ROOT / "artifacts" / "bo1d.pt"), device)
    bo_model.eval()
    quantize_fp16_inplace(bo_model)  # match the shipped fp16 weights
    bo = bo_cases(bo_model)
    (OUT_DIR / "bo1d.parity.json").write_text(json.dumps(bo))
    print(f"wrote bo1d.parity.json ({len(bo)} cases)")
    (OUT_DIR / "bo1d.demo.json").write_text(json.dumps(bo_demo_reference(bo_model)))
    print("wrote bo1d.demo.json")

    if ARBUF_CKPT.exists():
        arb_model = load_buffered_checkpoint(str(ARBUF_CKPT), device, gp1d.variables())
        arb_model.eval()
        quantize_fp16_inplace(arb_model)  # match the shipped fp16 weights
        payload = {
            # The TS port implements the concat read only; record the mode so the
            # TS test fails with a clear message on a separate-read fixture.
            "concat_read": bool(arb_model.concat_read),
            # plain-forward cases on the buffered checkpoint (reuses the GP case
            # builders): pins the inherited base path through the export.
            "plain": gp_cases(arb_model),
            "packed": arbuf_packed_case(arb_model),
            "chain": arbuf_chain_case(arb_model),
        }
        (OUT_DIR / "gp1d_arbuffer.parity.json").write_text(json.dumps(payload))
        print(f"wrote gp1d_arbuffer.parity.json ({len(payload['plain'])} plain cases + packed + chain)")
    else:
        print(f"skipped gp1d_arbuffer fixtures ({ARBUF_CKPT} not found)")

    # ALINE (extensions/aline/): the env fixture is checkpoint-independent and
    # regenerates on every run; the parity fixture is gated like arbuffer's.
    env = aline_env_fixture()
    (OUT_DIR / "gp1d_aline.env.json").write_text(json.dumps(env))
    print(f"wrote gp1d_aline.env.json ({len(env['cases'])} cases)")

    if ALINE_CKPT.exists():
        aline_model = load_aline_checkpoint(str(ALINE_CKPT), device, gp1d.variables())
        aline_model.eval()
        quantize_fp16_inplace(aline_model)  # match the shipped fp16 weights
        payload = {
            # plain-forward cases on the ALINE checkpoint (reuses the GP case
            # builders): pins the inherited base path through the export.
            "plain": gp_cases(aline_model),
            "policy": aline_policy_case(aline_model),
            "chain": aline_chain_case(aline_model),
        }
        (OUT_DIR / "gp1d_aline.parity.json").write_text(json.dumps(payload))
        print(f"wrote gp1d_aline.parity.json ({len(payload['plain'])} plain cases + policy + chain)")
    else:
        print(f"skipped gp1d_aline fixtures ({ALINE_CKPT} not found)")


if __name__ == "__main__":
    main()
