# nanoACE

nanoACE is a small, readable, and fully operational implementation of the
[Amortized Conditioning Engine (ACE)](https://acerbilab.github.io/amortized-conditioning-engine/):
treat data, interpretable latents, and runtime prior information as tokens;
condition on one token set; predict distributions over another token set.

The goal is a reasonably self-contained source that a human or coding agent can
read end to end and extend. The original research code is stored in
[this other repo](https://github.com/acerbilab/amortized-conditioning-engine/).

## Reference

This project is based on:

```bibtex
@article{chang2025amortized,
  title={Amortized Probabilistic Conditioning for Optimization, Simulation and Inference},
  author={Chang, Paul E and Loka, Nasrulloh and Huang, Daolang and Remes, Ulpu and Kaski, Samuel and Acerbi, Luigi},
  journal={28th Int. Conf. on Artificial Intelligence & Statistics (AISTATS 2025)},
  year={2025}
}
```

Local paper markdown is in [paper/](paper/).

## Current Status

Implemented modules:

- [ace.py](ace.py): core `Variable`, `Tokens`, `Batch`, bounded-latent
  coordinate helpers, ACE transformer, shared continuous MDN head, shared masked
  categorical head, prediction object, loss, and autoregressive sampling helper.
- [ace_prior_beta.py](ace_prior_beta.py): Beta-specific ACEP runtime-prior
  helpers that build and score the two-feature `(mean, spread)` information
  tokens used by the prior-conditioning examples. Model-side PRIOR token
  semantics live in [ace.py](ace.py).
- [train.py](train.py): shared training infrastructure for the examples — the
  Adam + grad-clip loop (`fit`), cosine LR (default) or constant, simple resume,
  checkpoint save/load, model construction, and a `common_parser()` argparse
  parent with a light-YAML `--config`. `fit` reseeds the global RNG with
  `mix_seed(seed, step)` each step, so the training stream is a pure function of
  `(seed, step)` — reproducible and resume-exact. Each example keeps its own
  `main()`, sampler, and diagnostics; `fit` takes a `(step) -> Batch` thunk, which
  the online samplers and the `data.py` `PoolReader` both satisfy (one training path).
- [data.py](data.py): optional offline sharded data pool (generate → save → train)
  for the expensive examples (GP-1D, BO). `write_pool` caches only the per-instance
  physics; `PoolReader` reads it back through the same `assemble` the online path
  uses, recomputing the context/target split and reveal mask from a stateless
  `(seed, position)` hash (batch-size-independent, resume-exact). It lazy-loads shards
  through a bounded cache and prefetches upcoming batch shards. A manifest carries the
  `variables()` schema and a DGP config-hash as the staleness guard. Build with
  `python data.py <gp1d|bo1d> --out DIR --pool-size N`.
- [gaussian_toy.py](gaussian_toy.py): Gaussian ACEP toy with two bounded
  continuous latents, runtime Beta information tokens, online
  training/evaluation CLI, analytic grid posterior, posterior predictive,
  checkpoint helpers, and plotting.
- [gp1d.py](gp1d.py): GP-1D regression example with continuous kernel
  hyperparameter latents, discrete kernel selection, online CPU float64 GP
  sampling, numerical grid posterior oracle, and a fixed diagnostic plot.
- [sbi_sir.py](sbi_sir.py): SIR simulation-based-inference example with two
  continuous rate latents (`beta`, `gamma`), online deterministic-ODE simulation
  with Gaussian observation noise, runtime Beta prior injection, a numerical
  `(beta, gamma)` grid posterior oracle, and a fixed diagnostic that contrasts a
  uniform against an informative runtime prior.
- [bo1d.py](bo1d.py): 1D Bayesian optimization example with the global optimum's
  location `x_opt` and value `y_opt` as latents (properties of the specific
  sampled function), an optimum-planting GP data-generating process, runtime Beta
  prior injection over the optimum location with an epsilon-contamination
  ("robust prior") floor, and a fixed diagnostic contrasting a uniform, a correct,
  and a wrong runtime prior. This is the one example with no oracle.
- [diagnostics.py](diagnostics.py): reusable grid-query helpers for marginal and
  two-variable AR diagnostics.
- [playground/](playground/): a **non-core**, fully in-browser TypeScript demo
  (separate toolchain) where trained models run client-side — GP-1D, Gaussian,
  SIR, and BO-1D, with interactive conditioning, latent/prior controls, and
  oracle overlays where practical. BO-1D stays no-oracle and overlays
  optimum-location/value marginals on the editable regression plot. A fifth
  tab runs the AR-buffer extension's coherent joint sampling, and a sixth
  (local-only) runs the ALINE extension's active-learning loop (see below).
  See [playground/README.md](playground/README.md). The Python core stays
  torch-only; the playground is an example built on a parity-tested TS port of
  `ace.py`'s forward pass.
- [extensions/arbuffer/](extensions/arbuffer/): an extension adding
  the causal autoregressive buffer of Hassan et al. (2026) on top of a trained
  GP-1D checkpoint. Two target-read variants: a separate zero-init gated read
  (bit-exact warm start, frozen base) and the retained paper-style
  `--concat-read` (one softmax over `[context, buffer]` keys, joint fine-tune;
  ~95% of the slow-AR joint-density gap recovered at the 20k validation
  budget). Encodes the context once and draws many coherent joint
  function samples from the cached encoding (vs `sample_ar`'s per-step
  re-encoding), plus one-pass joint density evaluation. Also the repository's
  extensibility demo (no core file changes). A playground tab runs its
  incremental sampler in the browser — the TS port follows the retained
  concat-read architecture and serves the retained 200k weights. See
  [extensions/arbuffer/README.md](extensions/arbuffer/README.md).
- [extensions/aline/](extensions/aline/): an extension implementing
  ALINE (Huang et al., 2025) — joint amortized Bayesian inference and active
  data acquisition — on the GP-1D task, warm-started from a trained GP-1D
  checkpoint. The inference network is the unchanged core ACE (parameter and
  predictive targets are QUERY tokens; the acquisition goal is *which target
  tokens are active*); the only new model part is a small read-only policy
  decoder that scores a candidate pool, trained with REINFORCE on
  self-estimated information gain, with a structural gradient firewall between
  the two. The inference path is asserted bit-equal to the base ACE forward.
  A **local-only** playground tab runs the full acquisition loop in-browser
  against a hidden GP function (the user picks where to sample; the policy
  advises; goals switch live). See
  [extensions/aline/README.md](extensions/aline/README.md).
- [DEVLOG.md](DEVLOG.md): design decisions and rationale. Read this before
  changing architecture or scope.

Current playground weights are hosted [outside this repo](https://github.com/acerbilab/nanoACE-playground-weights).
They are exported from retained runs under the shared multi-latent reveal DGP:
Gaussian 80k steps, GP-1D 200k, SIR 100k, and BO-1D 200k — plus the AR-buffer
model (a 200k concat-read fine-tune on top of the GP-1D checkpoint).
Local `artifacts/` and `playground/public/models/` remain gitignored in nanoACE.

Next work: inspect the deployed Pages build against the public weights (now
five models, including the AR-buffer tab), add manifest-level training
provenance on the next export, and consider whether the shared prior path
warrants a discrete-latent runtime prior.

## Setup

Use a local virtual environment. The current requirements pin the PyTorch CUDA
wheel that has been tested on this workstation:

- `torch==2.11.0+cu128`
- PyTorch CUDA runtime 12.8
- NVIDIA RTX 4060 Laptop GPU
- `PyYAML` (only used by `train.py`'s optional `--config`; pulled in by `requirements.txt`)

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Examples

Each example is a standalone script: it trains online, prints fixed diagnostic
summaries, and usually writes an artifact plot/checkpoint when requested.

### Shared training options

All four examples share the training flags defined in [train.py](train.py):

- `--lr-schedule {cosine,constant}` (default `cosine`) and `--warmup N`.
- `--resume <ckpt>` and `--ckpt-every N` for simple resume. A resumable checkpoint
  carries optimizer/scheduler/step; resume with the **same** `--steps` (and batch size)
  the run was started with (the cosine curve is sized to the total budget). The data
  stream is a pure function of `(seed, step)`, so a resumed run replays the exact same
  batches as an uninterrupted one.
- `--config run.yaml` loads defaults from a YAML file; explicit CLI flags still win
  (precedence: example defaults < `--config` < CLI). YAML keys are the argument names
  with underscores; unknown keys are rejected and values are coerced/validated like CLI
  args. (One asymmetry: `store_true` flags such as `no_plot`/`eval_only` can be turned
  _on_ from YAML but not back _off_ from the CLI.) Example:

  ```yaml
  # run.yaml
  steps: 10000
  lr: 3.0e-4
  lr_schedule: cosine
  warmup: 500
  latent_context_prob: 0.5
  ```

  ```powershell
  .\.venv\Scripts\python.exe gp1d.py --config run.yaml --save-checkpoint artifacts\gp1d.pt
  ```

The final `--save-checkpoint` is model-only (`cfg`/`seed`/`state_dict`) plus a `config`
provenance record; it stays compatible with the playground exporter and older checkpoints.

### Offline data generation (GP-1D, BO)

The GP-1D and BO examples can train from a pre-generated **offline pool** instead of
sampling online — the generate → save → train pattern, for the two examples whose
per-instance physics (GP Cholesky / optimum planting) is the expensive part. Gaussian
and SIR are cheap and stay online-only.

```powershell
# generate a pool (CPU; shards + a manifest under the output dir)
.\.venv\Scripts\python.exe data.py gp1d --out artifacts\pool_gp --pool-size 100000
# train from it (identical diagnostics; --pool replaces online sampling)
.\.venv\Scripts\python.exe gp1d.py --pool artifacts\pool_gp --steps 20000 --save-checkpoint artifacts\gp1d.pt
```

Only the expensive physics draws are cached; the context/target split and the reveal
mask are recomputed at read time from a stateless `(seed, position)` hash, so the pool is
independent of batch size and the reveal strategy, and a pooled run is **resume-exact**.
The manifest records the `variables()` schema (a hard compatibility gate) and a hash of
the data-generating constants; training from a pool built under different DGP constants is
refused unless you pass `--pool-force`. Pools live under gitignored `artifacts/`; size
them so `steps * batch_size` is a few passes over `--pool-size`. `PoolReader` does not
load the full pool into RAM: it keeps a bounded shard cache (`--pool-cache-shards`,
default 4) and prefetches upcoming batch shards (`--pool-prefetch-batches`, default 1;
set to 0 to disable).

### Gaussian ACEP

Run the Gaussian example:

```powershell
.\.venv\Scripts\python.exe gaussian_toy.py
```

The Gaussian example trains online with runtime Beta priors over `mu` and
`log_sigma`, prints posterior moment diagnostics against the matching analytic
oracle, and saves a plot to `artifacts/gaussian_toy.png` by default. The fixed
diagnostic plot overlays the runtime prior, oracle posterior, and ACE posterior
on the latent marginal panels.
For comparability, evaluation always uses the same deterministic batch: three
observed `y` values, plus the sampled `mu` and `log_sigma` used only for printed
diagnostics. The constants live in [gaussian_toy.py](gaussian_toy.py), so rerunning the
same checkpoint regenerates the same plotted case.
The plot also compares the posterior predictive density for a new `y`; the
analytic predictive is computed by marginalizing over the posterior grid, not by
plugging posterior moments into a Gaussian.
Training sometimes reveals a random subset of latents as zero-spread information
tokens and queries the rest, so exact multi-latent conditioning is now
in-distribution. The fixed diagnostic uses `EVAL_MU_PRIOR = (0.70, 20.0)` and
`EVAL_LOGSIG_PRIOR = (0.70, 8.0)` in unit-mean/concentration coordinates.

Useful Gaussian controls:

```powershell
.\.venv\Scripts\python.exe gaussian_toy.py --latent-context-prob 0.25
```

Common artifact names used by the Gaussian example:

- `artifacts/gaussian_toy.pt`
- `artifacts/gaussian_toy.png`

Regenerate the longer-run diagnostic and checkpoint pair:

```powershell
.\.venv\Scripts\python.exe gaussian_toy.py --steps 10000 --save-checkpoint artifacts\gaussian_toy.pt --plot-path artifacts\gaussian_toy.png
```

For a short run that verifies the script starts and completes:

```powershell
.\.venv\Scripts\python.exe gaussian_toy.py --steps 20 --batch-size 32
```

To force CPU:

```powershell
.\.venv\Scripts\python.exe gaussian_toy.py --device cpu --steps 20
```

Save and reuse a small Gaussian checkpoint:

```powershell
.\.venv\Scripts\python.exe gaussian_toy.py --save-checkpoint artifacts/gaussian_toy.pt
.\.venv\Scripts\python.exe gaussian_toy.py --eval-only --load-checkpoint artifacts/gaussian_toy.pt
```

### GP-1D

Run the GP-1D example:

```powershell
.\.venv\Scripts\python.exe gp1d.py
```

The GP-1D example trains on functions sampled online from four kernels: RBF,
Matern-1/2, Matern-3/2, and periodic. Its diagnostic computes a numerical grid
oracle for the fixed context: it scores every kernel and
`log_lengthscale`/`log_outputscale` grid point by the GP marginal likelihood,
normalizes those quadrature weights, and reports the resulting kernel posterior,
continuous latent marginals, and posterior predictive moments. The predictive
oracle is the mixture of conditional GP predictives over the posterior grid, not
a single GP at plugged-in hyperparameter means. The fixed diagnostic uses
irregular, clustered context locations so nearby observations can reveal local
roughness; evenly spaced sparse points made kernel and lengthscale inference
mostly uninformative.

Common artifact names used by the GP-1D example:

- `artifacts/gp1d.pt`
- `artifacts/gp1d.png`

For a short GP-1D run:

```powershell
.\.venv\Scripts\python.exe gp1d.py --steps 20 --batch-size 16
```

Reuse a saved GP-1D checkpoint and regenerate the oracle comparison plot:

```powershell
.\.venv\Scripts\python.exe gp1d.py --eval-only --load-checkpoint artifacts\gp1d.pt --plot-path artifacts\gp1d.png
```

### SIR SBI

Run the SIR SBI example:

```powershell
.\.venv\Scripts\python.exe sbi_sir.py
```

The SIR example is the simulation-based-inference task: recover the contact rate
`beta` and recovery rate `gamma` of an epidemic from a noisily observed infected
fraction over time. Functions are simulated online from the deterministic SIR
ODE (RK4 in fraction coordinates) plus Gaussian observation noise. Training
samples runtime Beta priors over `beta` and `gamma`, draws the true rates from
those priors, and always emits one prior token per rate (ACEP); `Beta(1, 1)` is
the uninformative case. Because the trajectory is deterministic given the rates,
the diagnostic computes an exact-up-to-grid `(beta, gamma)` posterior by scoring
every grid point's Gaussian observation likelihood times the Beta prior, and a
posterior-predictive epidemic curve as the mixture of deterministic trajectories
over that posterior grid. ACE itself only ever sees simulated draws, never the
likelihood.

The fixed diagnostic uses sparse, rise-phase observations on purpose: early
epidemic data pins down the growth rate but leaves a broad `beta`/`gamma` ridge,
so the runtime prior visibly tightens and shifts the posterior. The plot shows
the same observation under a uniform and an informative prior side by side, plus
the forecast epidemic curve.

Common artifact names used by the SIR example:

- `artifacts/sbi_sir.pt`
- `artifacts/sbi_sir.png`

For a short SIR run, or to force CPU:

```powershell
.\.venv\Scripts\python.exe sbi_sir.py --steps 20 --batch-size 16
.\.venv\Scripts\python.exe sbi_sir.py --device cpu --steps 20 --batch-size 16
```

Reuse a saved SIR checkpoint and regenerate the prior-contrast plot:

```powershell
.\.venv\Scripts\python.exe sbi_sir.py --eval-only --load-checkpoint artifacts\sbi_sir.pt --plot-path artifacts\sbi_sir.png
```

### BO-1D

Run the BO-1D example:

```powershell
.\.venv\Scripts\python.exe bo1d.py
```

The BO example is the Bayesian-optimization task: recover the location `x_opt`
and value `y_opt` of the global minimum of a black-box 1D function from a few
samples, and optionally inject a runtime Beta prior over the optimum location
(the paper's prior-injection BO). Unlike the GP example, whose latents describe
the function _class_, here the latents are properties of the _specific_ sampled
function -- exactly the quantities BO normally needs bespoke acquisition machinery
to reason about. Functions are generated online by a planting data-generating
process: sample GP hyperparameters (nuisance, not predicted), draw `x_opt`/`y_opt`
from epsilon-contaminated Beta priors, sample a GP draw conditioned on the optimum
geometry, then fold and add a convex envelope so the chosen optimum is the exact,
unique global minimum. There is **no oracle** (the fold destroys Gaussianity, and
the other three examples already carry grid oracles); the fixed diagnostic instead
plots the true function and true optimum as the reference.

The headline is **robust prior injection**. The effective prior is
`(1 - eps) * Beta + eps * Uniform`, so a confidently _wrong_ user prior cannot
starve the true optimum of probability mass. The fixed diagnostic shows the same
observation under three runtime priors side by side: uniform, a correct
informative prior (which tightens the `x_opt` posterior toward truth), and a wrong
informative prior (which the data overrides). Each column also shows the `y_opt`
marginal and the conditional `p(x_opt | y_opt, D)` (the Thompson-sampling query).

Common artifact names used by the BO example:

- `artifacts/bo1d.pt`
- `artifacts/bo1d.png`

Check only the data-generating-process scale (no training), run a short BO run,
or force CPU:

```powershell
.\.venv\Scripts\python.exe bo1d.py --scale-check
.\.venv\Scripts\python.exe bo1d.py --steps 20 --batch-size 16
.\.venv\Scripts\python.exe bo1d.py --device cpu --steps 20 --batch-size 16
```

Reuse a saved BO checkpoint and regenerate the prior-contrast plot:

```powershell
.\.venv\Scripts\python.exe bo1d.py --eval-only --load-checkpoint artifacts\bo1d.pt --plot-path artifacts\bo1d.png
```

## Design Notes

nanoACE keeps the ACE conditioning semantics, but the paper math is a starting
point rather than a constraint. The invariants are:

- variables are tokens;
- data values, latent values, and latent priors can appear in context;
- target tokens request predictive distributions;
- the training path is type-agnostic through `dist.log_prob`;
- the model uses separated context self-attention and target-to-context
  cross-attention.

The internal token representation is intentionally explicit. Data values stay in
task coordinates. Bounded continuous latent values are encoded to internal
`[-1, 1]` coordinates at token boundaries; native-coordinate prediction helpers
on `Predictions` decode means/variances/samples and add the affine density
Jacobian when needed.

```python
Batch(
    variables: list[Variable],
    context: Tokens,
    target: Tokens,
)

Tokens(
    var_id: LongTensor[B, T],
    x: FloatTensor[B, T, x_dim],
    value: FloatTensor[B, T],
    value_index: LongTensor[B, T],
    prior: FloatTensor[B, T, 2],   # bounded latent info: mean, spread
    mode: LongTensor[B, T],   # VALUE | PRIOR | QUERY
    mask: BoolTensor[B, T],
)
```
