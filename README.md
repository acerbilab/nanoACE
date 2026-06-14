# nanoACE

nanoACE is a small, readable, and fully operational implementation of the
[Amortized Conditioning Engine (ACE)](https://acerbilab.github.io/amortized-conditioning-engine/)
(Chang et al., [AISTATS 2025](#references)):
treat data, interpretable latents, and runtime prior information as tokens;
condition on one token set; predict distributions over another token set.

The goal is a reasonably self-contained source that a human or coding agent can
read end to end and extend. The original research code is stored in
[this other repo](https://github.com/acerbilab/amortized-conditioning-engine/).

## What's inside

nanoACE is meant to be **read and run**. The quickest taste needs no install:
play with the trained models in the [live playground](https://acerbilab.github.io/nanoACE/).
To run things yourself, install (below) and run any example. For the _why_ behind
each design decision, read [DEVLOG.md](DEVLOG.md) and the code.

### Runnable examples

Standalone scripts — each trains online, prints a fixed diagnostic against an
oracle (where one exists), and optionally saves a plot/checkpoint:

- **[Gaussian ACEP](#gaussian-acep)** — infer a Gaussian's `mu`/`log_sigma` with
  runtime Beta priors, against an analytic oracle.
- **[GP-1D](#gp-1d)** — GP regression with kernel hyperparameters and a discrete
  kernel choice as latents, against a grid oracle.
- **[SIR SBI](#sir-sbi)** — simulation-based inference of epidemic rates, with a
  uniform-vs-informative prior contrast.
- **[BO-1D](#bo-1d)** — 1D Bayesian optimization with the optimum location/value
  as latents and robust runtime prior injection (the one example with no oracle).

### Playground

An interactive, fully **in-browser** demo where trained models run client-side —
all four examples plus the two extensions, with live conditioning and prior
controls. **Try it live: [acerbilab.github.io/nanoACE](https://acerbilab.github.io/nanoACE/).**
To run or build it locally, see [playground/README.md](playground/README.md).

### Extensions

Non-core add-ons built on a trained checkpoint, each self-contained and changing
no core file (more in [Examples → Extensions](#extensions-1) below):

- **[arbuffer](extensions/arbuffer/README.md)** — fast coherent joint function
  sampling via a causal autoregressive buffer (Hassan et al., 2026).
- **[aline](extensions/aline/README.md)** — joint amortized inference + active
  data acquisition, ALINE (Huang et al., 2025).

### Core modules

For how it works, read these (and [DEVLOG.md](DEVLOG.md) for the why):
`ace.py` is the model (schema, embedder, attention, heads, loss, AR sampler),
`ace_prior_beta.py` the Beta runtime-prior helpers, `train.py` the shared
training/checkpoint/CLI spine, `data.py` the optional offline data pool, and
`diagnostics.py` the grid-query helpers.

### Trained weights

There are two kinds of weights, in two homes:

- **Full-precision source checkpoints** (the `.pt` files written by
  `--save-checkpoint`) live on Hugging Face at
  [`lacerbi/nanoACE`](https://huggingface.co/lacerbi/nanoACE). Each loads straight
  back into its example script:

  | Checkpoint | Example / extension | Steps |
  |---|---|---|
  | [`gaussian_toy.pt`](https://huggingface.co/lacerbi/nanoACE/resolve/main/gaussian_toy.pt) | Gaussian ACEP | 80k |
  | [`gp1d.pt`](https://huggingface.co/lacerbi/nanoACE/resolve/main/gp1d.pt) | GP-1D | 200k |
  | [`sbi_sir.pt`](https://huggingface.co/lacerbi/nanoACE/resolve/main/sbi_sir.pt) | SIR SBI | 100k |
  | [`bo1d.pt`](https://huggingface.co/lacerbi/nanoACE/resolve/main/bo1d.pt) | BO-1D | 200k |
  | [`gp1d_arbuffer.pt`](https://huggingface.co/lacerbi/nanoACE/resolve/main/gp1d_arbuffer.pt) | [arbuffer](extensions/arbuffer/README.md) | 200k |
  | [`gp1d_aline.pt`](https://huggingface.co/lacerbi/nanoACE/resolve/main/gp1d_aline.pt) | [aline](extensions/aline/README.md) | 35k |

  Fetching one needs only the optional `huggingface_hub` (`pip install
  huggingface_hub` — the core stays torch-only; this is never required to read,
  run, or train):

  ```python
  from huggingface_hub import hf_hub_download
  import gp1d
  path = hf_hub_download("lacerbi/nanoACE", "gp1d.pt")
  model = gp1d.load_checkpoint(path, "cpu")   # ready to evaluate / sample
  ```

- **fp16 browser blobs** (a manifest + `weights.bin` per model) for the playground
  live in the separate
  [acerbilab/nanoACE-playground-weights](https://github.com/acerbilab/nanoACE-playground-weights)
  repo and are derived from the source checkpoints above.

Local `artifacts/` and `playground/public/models/` stay gitignored.

## Setup

Use a local virtual environment. nanoACE depends only on PyTorch, plus
`matplotlib` (plots) and `PyYAML` (the optional `--config`). The models are small
enough to train and run on CPU, so the default install is cross-platform:

```bash
python -m venv .venv
# then activate it:
#   bash:        source .venv/bin/activate
#   PowerShell:  .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

That pulls the default PyTorch wheel for your platform — CPU on Windows, MPS on
macOS, CUDA-bundled on Linux. For an **NVIDIA GPU on Windows**, install the tested
CUDA 12.8 build instead (Linux already has CUDA above; macOS has no CUDA):

```bash
pip install -r requirements-cuda.txt
```

The commands below assume an activated venv, so they call `python` directly. On
Windows PowerShell, if you'd rather not activate, call `.\.venv\Scripts\python.exe`
in place of `python`.

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

  ```bash
  python gp1d.py --config run.yaml --save-checkpoint artifacts/gp1d.pt
  ```

The final `--save-checkpoint` is model-only (`cfg`/`seed`/`state_dict`) plus a `config`
provenance record; it stays compatible with the playground exporter and older checkpoints.

### Offline data generation (GP-1D, BO)

The GP-1D and BO examples can train from a pre-generated **offline pool** instead of
sampling online — the generate → save → train pattern, for the two examples whose
per-instance physics (GP Cholesky / optimum planting) is the expensive part. Gaussian
and SIR are cheap and stay online-only.

```bash
# generate a pool (CPU; shards + a manifest under the output dir)
python data.py gp1d --out artifacts/pool_gp --pool-size 100000
# train from it (identical diagnostics; --pool replaces online sampling)
python gp1d.py --pool artifacts/pool_gp --steps 20000 --save-checkpoint artifacts/gp1d.pt
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

```bash
python gaussian_toy.py
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

```bash
python gaussian_toy.py --latent-context-prob 0.25
```

Common artifact names used by the Gaussian example:

- `artifacts/gaussian_toy.pt`
- `artifacts/gaussian_toy.png`

Regenerate the longer-run diagnostic and checkpoint pair:

```bash
python gaussian_toy.py --steps 10000 --save-checkpoint artifacts/gaussian_toy.pt --plot-path artifacts/gaussian_toy.png
```

For a short run that verifies the script starts and completes:

```bash
python gaussian_toy.py --steps 20 --batch-size 32
```

To force CPU:

```bash
python gaussian_toy.py --device cpu --steps 20
```

Save and reuse a small Gaussian checkpoint:

```bash
python gaussian_toy.py --save-checkpoint artifacts/gaussian_toy.pt
python gaussian_toy.py --eval-only --load-checkpoint artifacts/gaussian_toy.pt
```

### GP-1D

Run the GP-1D example:

```bash
python gp1d.py
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

```bash
python gp1d.py --steps 20 --batch-size 16
```

Reuse a saved GP-1D checkpoint and regenerate the oracle comparison plot:

```bash
python gp1d.py --eval-only --load-checkpoint artifacts/gp1d.pt --plot-path artifacts/gp1d.png
```

### SIR SBI

Run the SIR SBI example:

```bash
python sbi_sir.py
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

```bash
python sbi_sir.py --steps 20 --batch-size 16
python sbi_sir.py --device cpu --steps 20 --batch-size 16
```

Reuse a saved SIR checkpoint and regenerate the prior-contrast plot:

```bash
python sbi_sir.py --eval-only --load-checkpoint artifacts/sbi_sir.pt --plot-path artifacts/sbi_sir.png
```

### BO-1D

Run the BO-1D example:

```bash
python bo1d.py
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

```bash
python bo1d.py --scale-check
python bo1d.py --steps 20 --batch-size 16
python bo1d.py --device cpu --steps 20 --batch-size 16
```

Reuse a saved BO checkpoint and regenerate the prior-contrast plot:

```bash
python bo1d.py --eval-only --load-checkpoint artifacts/bo1d.pt --plot-path artifacts/bo1d.png
```

### Extensions

Two non-core extensions build on a trained checkpoint without changing any core
file. Each has its own README with the full run recipe, and a local DEVLOG with
the design rationale:

- **[extensions/arbuffer/](extensions/arbuffer/README.md)** — the causal
  autoregressive buffer of Hassan et al. (2026). Encodes a GP context once, then
  draws many coherent joint function samples from the cache (vs `sample_ar`'s
  per-step re-encoding), plus one-pass joint density evaluation. Warm-started
  from a GP-1D checkpoint; also the repository's extensibility demo.
- **[extensions/aline/](extensions/aline/README.md)** — ALINE (Huang et al.,
  2025): joint amortized inference + active data acquisition on GP-1D. The
  inference network is the unchanged core ACE; a small read-only policy decoder
  picks where to sample next, trained with REINFORCE. Warm-started from a GP-1D
  checkpoint.

Both also appear as playground tabs.

## Design Notes

nanoACE keeps the ACE conditioning semantics, but the paper math is a starting
point rather than a constraint. The invariants are:

- variables are tokens;
- data values, latent values, and latent priors can appear in context;
- target tokens request predictive distributions;
- the training path is type-agnostic through `dist.log_prob`;
- the model uses separated context self-attention and target-to-context
  cross-attention.

The central data structure is the token batch below. Data values stay in task
coordinates; bounded continuous latent values are encoded to internal `[-1, 1]`
coordinates at token boundaries (native-coordinate prediction helpers on
`Predictions` decode means/variances/samples and add the affine density Jacobian
when needed).

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

For the full cross-file picture — how `_embed` turns `mode` into a payload, the
`ACEBlock` attention, the shared MDN + categorical heads, `sample_ar`, and the
`train.py` spine — see the
[**Architecture (the cross-file picture)**](AGENTS.md#architecture-the-cross-file-picture)
section of `AGENTS.md`; the design decisions and their rationale start with the
[**Initial design**](DEVLOG.md#2026-06-06--initial-design) entry in `DEVLOG.md`.

## References

The work in this repository is based on the following papers. The core model is
the [Amortized Conditioning Engine (ACE)](https://acerbilab.github.io/amortized-conditioning-engine/):

```bibtex
@inproceedings{chang2025amortized,
  title={Amortized Probabilistic Conditioning for Optimization, Simulation and Inference},
  author={Chang, Paul E and Loka, Nasrulloh and Huang, Daolang and Remes, Ulpu and Kaski, Samuel and Acerbi, Luigi},
  booktitle={The Twenty-eighth International Conference on Artificial Intelligence and Statistics (AISTATS 2025)},
  year={2025}
}
```

The two extensions in [extensions/](extensions/) build on further work — the
[causal autoregressive buffer](https://www.conorhassan.com/projects/artnp/) (arbuffer) and [ALINE](https://www.huangdaolang.com/aline/):

```bibtex
@inproceedings{hassan2026efficient,
  title={Efficient Autoregressive Inference for Transformer Probabilistic Models},
  author={Conor Hassan and Nasrulloh Ratu Bagus Satrio Loka and Cen-You Li and Daolang Huang and Paul Edmund Chang and Yang Yang and Francesco Silvestrin and Samuel Kaski and Luigi Acerbi},
  year={2026},
  booktitle={The Fourteenth International Conference on Learning Representations (ICLR 2026)},
}
```

```bibtex
@inproceedings{huang2025aline,
  title={ALINE: Joint Amortization for Bayesian Inference and Active Data Acquisition},
  author={Daolang Huang and Xinyi Wen and Ayush Bharti and Samuel Kaski and Luigi Acerbi},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025)},
  year={2025},
}
```

Local paper markdown for ACE is in [paper/](paper/); each extension keeps its own
paper under `extensions/<name>/paper/`.

## Acknowledgments

nanoACE is developed by the [Machine and Human Intelligence (MHI) group](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence)
at the University of Helsinki, with extensive assistance from AI coding agents
(Claude Code and Codex). Work on nanoACE was supported by the [Research Council
of Finland](https://www.aka.fi/en/) (Flagship programme: Finnish Center for Artificial Intelligence FCAI;
and grants 358980 and 356498) and by the research environment provided by [ELLIS
Institute Finland](https://www.ellisinstitute.fi/).
