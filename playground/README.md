# nanoACE playground

An interactive, **fully client-side** demo of trained nanoACE models. A small
TypeScript port of `ace.py`'s forward pass (parity-tested against the real
PyTorch model) loads exported weights and runs amortized conditioning in the
browser — no server, no backend.

> This is a **non-core example**, intentionally separate from nanoACE itself.
> The core stays torch-only and legible; this folder carries the JS/TS toolchain.
> It is a frozen snapshot of the model's forward pass, kept honest by a parity test.

Six demos:

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
- **GP-1D joint draws (AR)** — the `extensions/arbuffer/` buffered GP model:
  edit context points / pin latents like the GP tab, then watch a few
  **coherent joint function draws** decode autoregressively from one cached
  context encoding (animated), next to the diagonal band and independent
  per-point marginal samples. A sampler toggle reruns the chain as slow AR
  (full context re-encoding per step, one draw) so the per-draw cost
  difference is felt directly. The weights are the retained 200k concat-read
  fine-tune (K=64, joint training); the tab and its tests self-skip/notice
  gracefully when the blob is absent locally.
- **GP-1D active learning (ALINE)** — the `extensions/aline/` model: a hidden
  GP function (sampled in-browser by an exact port of the gp1d DGP) answers
  your queries, and you only choose **where** to sample — following or
  ignoring ALINE's acquisition advice, rendered as the policy distribution
  π(x | data, goal) along the bottom axis next to the classical
  uncertainty-sampling pick. The goal selector (predict the function vs infer
  lengthscale / outputscale / kernel) is live, including mid-episode;
  "Follow policy" lets the learned policy drive the episode (animated), with
  RMSE / log q(θ_true) tracked against the hidden truth. A secondary
  "your own data" mode gives free point editing with the advice still live.
  **Local-only** (not deployed): the current weights are the extension's 5k
  validation fine-tune — the policy beats random but its goal-targeting is
  still subtle; swap in a longer-trained checkpoint by re-running export +
  parity together. The tab and its tests self-skip/notice gracefully when
  the blob is absent.

## Run locally

The weight blobs are **not committed to nanoACE**. For local development, either
copy/fetch them from the separate
[nanoACE-playground-weights](https://github.com/acerbilab/nanoACE-playground-weights)
repository into `playground/public/models/`, or regenerate them from checkpoints
in `artifacts`:

```bash
# from the repo root, using the project venv (generates public/models/*)
python playground/export_weights.py --task gp1d     --checkpoint artifacts/gp1d.pt         --out playground/public/models/gp1d
python playground/export_weights.py --task gaussian --checkpoint artifacts/gaussian_toy.pt --out playground/public/models/gaussian
python playground/export_weights.py --task sbi_sir  --checkpoint artifacts/sbi_sir.pt      --out playground/public/models/sbi_sir
python playground/export_weights.py --task bo1d     --checkpoint artifacts/bo1d.pt         --out playground/public/models/bo1d
# AR-buffer tab (extensions/arbuffer/ retained concat-read checkpoint)
python playground/export_weights.py --task gp1d_arbuffer --checkpoint artifacts/gp1d_arbuffer.pt --out playground/public/models/gp1d_arbuffer
# ALINE tab (extensions/aline/ checkpoint; local-only — not in the deploy workflow)
python playground/export_weights.py --task gp1d_aline --checkpoint artifacts/gp1d_aline.pt --out playground/public/models/gp1d_aline

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
- `src/ace/buffered.ts` + `src/arbuf/` — the AR-buffer tab: a TS port of the
  `extensions/arbuffer/` incremental sampler in its **concat-read** form
  (`encode_context` + per-step decode; one softmax over `[context, buffer]`
  keys with the learned per-head `buf_bias` soft gate; projected-KV caches
  instead of Python's reproject-per-read — same math, same fixtures) on top of
  the untouched base port, plus the tab's DOM-free `infer.ts` (context builder,
  static band pass, step-driveable `JointSampler`) and `demo.ts`. Separate-read
  checkpoints are rejected at load with a clear error.
- `src/ace/aline.ts` + `src/aline/` — the ALINE tab: `ALINEModel` extends the
  base port without touching it (`forwardWithStates` re-reads the inherited
  forward's per-layer stacks and re-applies `final_norm`; the policy decoder
  runs on `nn.ts` primitives, scoring only the supplied candidates — omission
  replaces masking exactly, since neither targets nor candidates attend to
  each other). `env.ts` is a float64 port of the gp1d data-generating process
  (kernels, hyperprior, jitter, Cholesky) that samples the hidden episode
  functions; `infer.ts` packs goal + latent + band + US-candidate rows into
  one forward per step and slices only the goal rows for the policy;
  `demo.ts` owns the episode UI.
- `src/config.ts` — all tunable constants (OOD thresholds, view ranges, grid
  sizes) in one place.
- `src/explain.ts` — the per-tab "?" explainer: each tab's hint line ends with a
  button opening a short didactic modal (the task / what ACE is doing / how it
  compares to the classical approach, plus paper attribution). Content lives in
  each tab's `demo.ts`; the modal chrome is shared with the global "What is
  ACE?" dialog.

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
[acerbilab/nanoACE-playground-weights](https://github.com/acerbilab/nanoACE-playground-weights)
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

The AR-buffer tab has its own fixture set (`gp1d_arbuffer.parity.json`): plain
forward on the buffered checkpoint (the frozen-base invariant through the
export), a packed `forward_buffered` pass with per-layer states (buffer row j ↔
TS append pass j, target row m ↔ decode step m, so divergence localizes to a
layer), and a teacher-forced `sample_joint` chain (the exact incremental
semantics the TS sampler implements). `parity.py` skips this block, and the TS
tests self-skip, when the checkpoint/blob is absent — so `npm test` stays green
on clones without local exports.

The ALINE tab follows the same scheme (`gp1d_aline.parity.json`): plain forward
(the inference path is the unchanged ACE forward), a policy case with
per-policy-block candidate streams, and a teacher-forced compact episode with a
mid-episode goal switch — the TS test replays the recorded actions (immune to
argmax tie-flips) while asserting logits and per-goal-token log-probs. A
separate, checkpoint-independent `gp1d_aline.env.json` pins the browser GP
environment (kernel matrices + Cholesky factors in float64) against
`gp1d.draw_gp`; its suite runs even without the blob.

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
[acerbilab/nanoACE-playground-weights](https://github.com/acerbilab/nanoACE-playground-weights)
beside the app, copies the model directories into `public/models/`, then fails
fast if any expected manifest/blob is missing or if `weights.bin` is still a Git
LFS pointer. It also validates each manifest against its blob size, records
resolved weight hashes in the run summary, runs `npm test`, and then builds. The
manual workflow input `weights_ref` defaults to `main`; use a tag or commit SHA
when you want the deployment to be reproducible.
To update the deployed models after retraining, regenerate/export weights and
parity fixtures together, push the new blobs to the weights repo, then trigger
the manual Pages workflow in nanoACE.
