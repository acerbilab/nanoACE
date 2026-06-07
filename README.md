# nanoACE

nanoACE is a small, readable implementation of the core ideas behind the
Amortized Conditioning Engine (ACE): treat data, interpretable latents, and
runtime prior information as tokens; condition on one token set; predict
distributions over another token set.

The goal is legible source that a human or coding agent can read end to end and
extend. It is not a packaged ACE runtime, not a benchmark suite, and not a clone
of the original research code.

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

Local paper markdown is in [paper/](paper/). The upstream paper page links to
the paper, markdown, and original ACE code:
[chang2025amortized_overview.md](paper/chang2025amortized_overview.md).

## Current Status

Implemented modules:

- [ace.py](ace.py): core `Variable`, `Tokens`, `Batch`, bounded-latent
  coordinate helpers, ACE transformer, shared continuous MDN head, shared masked
  categorical head, prediction object, loss, and autoregressive sampling helper.
- [ace_prior.py](ace_prior.py): shared ACEP runtime-prior helpers that build and
  score the two-feature `(mean, spread)` Beta information tokens, reused by the
  prior-conditioning examples.
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
- [diagnostics.py](diagnostics.py): reusable grid-query helpers for marginal and
  two-variable AR diagnostics.
- [DEVLOG.md](DEVLOG.md): design decisions and rationale. Read this before
  changing architecture or scope.

Next work: tune the SIR diagnostic and model size against its grid oracle, and
consider whether the shared prior path warrants a discrete-latent runtime prior.

## Setup

Use a local virtual environment. The current requirements pin the PyTorch CUDA
wheel that has been tested on this workstation:

- `torch==2.11.0+cu128`
- PyTorch CUDA runtime 12.8
- NVIDIA RTX 4060 Laptop GPU

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

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
Training sometimes reveals one latent as a zero-spread information token and
asks for the other, so the autoregressive diagnostic is trained on the
conditional latent queries it uses at evaluation time. The fixed diagnostic
uses `EVAL_MU_PRIOR = (0.70, 20.0)` and
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

## Agent Notes

Before making architectural changes, read [DEVLOG.md](DEVLOG.md). The project
values local, readable code over benchmark machinery. In particular:

- keep `ace.py` as the main readable implementation file;
- if a gitignored `temp/` directory is present, treat it as archived external
  experiment code and copy ideas only when they clearly fit this repository;
- keep examples small and diagnostic;
- prefer changing the implementation when a simpler or more robust tweak serves
  ACE's conditioning interface better than paper fidelity.

Generated directories such as `.venv/`, `__pycache__/`, `temp/`, and
`artifacts/` are ignored.
