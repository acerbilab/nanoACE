# nanoACE playground

An interactive, **fully client-side** demo of trained nanoACE models. A small
TypeScript port of `ace.py`'s forward pass (parity-tested against the real
PyTorch model) loads exported weights and runs amortized conditioning in the
browser — no server, no backend.

> This is a **non-core example**, intentionally separate from nanoACE itself.
> The core stays torch-only and legible; this folder carries the JS/TS toolchain.
> It is a frozen snapshot of the model's forward pass, kept honest by a parity test.

Four demos:

- **GP-1D regression** — click to add/drag/delete points; watch the posterior
  predictive band, the kernel posterior, and the lengthscale/outputscale
  marginals update instantly. Pin any latent (kernel / lengthscale / outputscale)
  to condition on it and predict.
- **Gaussian (μ, σ) with priors** — set Beta priors over `μ` and `log σ`, add
  observations, and watch ACE's posterior marginals and posterior predictive
  track the analytic oracle. This dramatizes runtime prior conditioning (ACEP).
- **SIR SBI** — edit infected-fraction observations, set Beta priors over
  `beta` and `gamma`, and compare ACE's posterior/predictive curve against a
  live browser-side numerical SIR grid oracle.
- **BO-1D** - edit black-box function observations, set or fix priors over
  `x_opt` and `y_opt`, and inspect the optimum-location/value marginals
  overlaid on the regression plot. BO intentionally has no oracle.

## Run locally

The weight blobs are **not committed to nanoACE**. For local development, either
copy/fetch them from the separate
[nanoACE-playground-weights](https://github.com/lacerbi/nanoACE-playground-weights)
repository into `playground/public/models/`, or regenerate them from checkpoints
in `artifacts`:

```bash
# from the repo root, using the project venv (generates public/models/*)
python playground/export_weights.py --task gp1d     --checkpoint artifacts/gp1d.pt         --out playground/public/models/gp1d
python playground/export_weights.py --task gaussian --checkpoint artifacts/gaussian_toy.pt --out playground/public/models/gaussian
python playground/export_weights.py --task sbi_sir  --checkpoint artifacts/sbi_sir.pt      --out playground/public/models/sbi_sir
python playground/export_weights.py --task bo1d     --checkpoint artifacts/bo1d.pt         --out playground/public/models/bo1d

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
- `src/gp/`, `src/gaussian/`, `src/sir/` — each demo has a pure `infer.ts`
  (DOM-free inference) and a `demo.ts` (UI). `oracle.ts` is the Gaussian analytic
  posterior or SIR numerical grid oracle where applicable.
  BO lives in `src/bo/` and intentionally has no oracle.
- `src/config.ts` — all tunable constants (OOD thresholds, view ranges, grid
  sizes) in one place.

### Out-of-distribution guardrails

The models were trained on a bounded regime (`x, y ≈ [-1, 1]`, ≤ ~14 context
points, with a random subset of latents sometimes revealed). The demos let you
roam, but flag when you leave that regime inside the main plot. Pinning multiple
latents is now in-distribution for the current multi-reveal checkpoints. For
GP-1D, a pins-only context with no observed data is still flagged because GP
training used at least four data context points. SIR flags very sparse
observation sets because the training sampler used at least four data points.
BO uses the same point-count/value guardrail style as GP, with prior-only
contexts flagged because BO training used at least one observed point.

## Weights

The blobs under `public/models/<task>/` are **gitignored in nanoACE** and hosted
separately in
[lacerbi/nanoACE-playground-weights](https://github.com/lacerbi/nanoACE-playground-weights)
so ordinary nanoACE clones stay small. The Pages workflow checks out that
repository beside the app and copies the model directories into
`playground/public/models/` before building.

To regenerate local blobs from checkpoints, use the project venv at the repo
root:

```bash
python playground/export_weights.py --task gp1d     --checkpoint artifacts/gp1d.pt        --out playground/public/models/gp1d
python playground/export_weights.py --task gaussian --checkpoint artifacts/gaussian_toy.pt --out playground/public/models/gaussian
python playground/export_weights.py --task sbi_sir  --checkpoint artifacts/sbi_sir.pt      --out playground/public/models/sbi_sir
python playground/export_weights.py --task bo1d     --checkpoint artifacts/bo1d.pt         --out playground/public/models/bo1d
```

## Parity (the linchpin)

`export_weights.py` derives all constants from a live `ACE` instance, so the TS
side never re-encodes the schema by hand. `parity.py` runs the real model on
deterministic cases and dumps fixtures (`test/fixtures/`) covering every token
path (data VALUE, finite/zero-spread PRIOR, discrete VALUE, latent/data QUERY,
padding) plus per-layer intermediates. `npm test` asserts the TS forward
reproduces the PyTorch embeddings, per-layer states, raw head outputs, and
derived quantities, and that each demo's orchestration matches `gp1d.py`,
`gaussian_toy.py`, or `sbi_sir.py`.

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

The deploy workflow checks out
[lacerbi/nanoACE-playground-weights](https://github.com/lacerbi/nanoACE-playground-weights)
beside the app, copies the model directories into `public/models/`, then fails
fast if any expected manifest/blob is missing.
To update the deployed models after retraining, regenerate/export weights and
parity fixtures together, push the new blobs to the weights repo, then trigger
the manual Pages workflow in nanoACE.
