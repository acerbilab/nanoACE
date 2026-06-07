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
import gaussian_toy  # noqa: E402
import gp1d  # noqa: E402
from ace import PRIOR, QUERY, VALUE, Batch, Tokens, encode_value  # noqa: E402
from export_weights import quantize_fp16_inplace  # noqa: E402  (same dir as this script)

OUT_DIR = Path(__file__).resolve().parent / "test" / "fixtures"


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


if __name__ == "__main__":
    main()
