# nanoACE playground

An interactive, **fully client-side** demo of trained nanoACE models. A small
TypeScript port of `ace.py`'s forward pass (parity-tested against the real
PyTorch model) loads exported weights and runs amortized conditioning in the
browser — no server, no backend.

> This is a **non-core example**, intentionally separate from nanoACE itself.
> The core stays torch-only and legible; this folder carries the JS/TS toolchain.
> It is a frozen snapshot of the model's forward pass, kept honest by a parity test.

Two demos:

- **GP-1D regression** — click to add/drag/delete points; watch the posterior
  predictive band, the kernel posterior, and the lengthscale/outputscale
  marginals update instantly. Pin any latent (kernel / lengthscale / outputscale)
  to condition on it and predict.
- **Gaussian (μ, σ) with priors** — set Beta priors over `μ` and `log σ`, add
  observations, and watch ACE's posterior marginals and posterior predictive
  track the analytic oracle. This dramatizes runtime prior conditioning (ACEP).

## Run locally

The weight blobs are **not committed** (the hosting decision is parked — see
"Weights" below). Generate them once from the checkpoints in `artifacts/` first,
then run the dev server:

```bash
# from the repo root, using the project venv (generates public/models/*)
python playground/export_weights.py --task gp1d     --checkpoint artifacts/gp1d.pt         --out playground/public/models/gp1d
python playground/export_weights.py --task gaussian --checkpoint artifacts/gaussian_toy.pt --out playground/public/models/gaussian

cd playground
npm install
npm run dev        # http://localhost:5173
```

Other scripts:

```bash
npm run build      # tsc --noEmit + vite build  -> dist/
npm run preview    # serve the production build
npm test           # vitest: parity + orchestration + UI smoke tests
```

## How it works

- `src/ace/` — the TS port of the model: `schema.ts` (variables + bounded-latent
  coordinates), `nn.ts` (linear, LayerNorm, exact-erf GELU, attention), `model.ts`
  (`_embed` → blocks → heads), `predictions.ts` (MDN + categorical), `weights.ts`
  (manifest/blob loader), `tokens.ts` (token builder).
- `src/gp/`, `src/gaussian/` — each demo has a pure `infer.ts` (DOM-free
  inference) and a `demo.ts` (UI). `oracle.ts` is the Gaussian analytic posterior.
- `src/config.ts` — all tunable constants (OOD thresholds, view ranges, grid
  sizes) in one place.

### Out-of-distribution guardrails

The models were trained on a bounded regime (`x, y ≈ [-1, 1]`, ≤ ~14 context
points, with a random subset of latents sometimes revealed). The demos let you
roam, but flag when you leave that regime (a banner names why). Pinning multiple
latents is now in-distribution for the current multi-reveal checkpoints. For
GP-1D, a pins-only context with no observed data is still flagged because GP
training used at least four data context points.

## Weights (not committed — generate locally)

The blobs under `public/models/<task>/` are **gitignored**: the
weight-hosting decision (commit vs Git LFS vs external/HF fetch) is parked, and
the demo may grow to more examples. They are produced from the checkpoints in the
repo's `artifacts/` (also gitignored). From the project venv at the repo root:

```bash
python playground/export_weights.py --task gp1d     --checkpoint artifacts/gp1d.pt        --out playground/public/models/gp1d
python playground/export_weights.py --task gaussian --checkpoint artifacts/gaussian_toy.pt --out playground/public/models/gaussian
```

## Parity (the linchpin)

`export_weights.py` derives all constants from a live `ACE` instance, so the TS
side never re-encodes the schema by hand. `parity.py` runs the real model on
deterministic cases and dumps fixtures (`test/fixtures/`) covering every token
path (data VALUE, finite/zero-spread PRIOR, discrete VALUE, latent/data QUERY,
padding) plus per-layer intermediates. `npm test` asserts the TS forward
reproduces the PyTorch embeddings, per-layer states, raw head outputs, and
derived quantities, and that each demo's orchestration matches `gp1d.py` /
`gaussian_toy.py`.

Weights ship as **float16** (half the blob size). `export_weights.py` rounds each
parameter with torch's `.half().float()` before serializing, and `parity.py`
applies the *same* rounding before generating fixtures — so the shipped weights
and the references reflect identical values. The remaining gap is only arithmetic
(PyTorch float32 vs JS float64, where the TS loader decodes the halves), so parity
uses a combined relative+absolute tolerance rather than bit-parity. To regenerate
fixtures after a checkpoint change:

```bash
python playground/parity.py
```

## Deploy

`.github/workflows/pages.yml` builds and deploys to GitHub Pages, **manually**
(`workflow_dispatch` from the Actions tab — no auto-deploy on push). One-time
setup: repo **Settings → Pages → Source = GitHub Actions**. The production base
path is `/nanoACE/` (set automatically in CI via `GITHUB_ACTIONS`).

**Pending:** because the weights are not committed, the deploy is blocked until
the hosting decision is made. The workflow fails fast if `public/models/` is
missing, so it cannot publish a weightless demo by accident. Resolving it means
one of: commit the blobs, use Git LFS (`lfs: true` on checkout), or have the demo
fetch them at runtime. The checkpoints aren't in the repo either, so generating
weights inside CI isn't currently an option.
