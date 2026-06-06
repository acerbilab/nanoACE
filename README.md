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

- [ace.py](ace.py): core `Variable`, `Tokens`, `Batch`, ACE transformer, shared
  continuous MDN head, shared masked categorical head, prediction object, loss,
  and autoregressive sampling helper.
- [gaussian_toy.py](gaussian_toy.py): Gaussian toy with two continuous latents, fixed
  latent priors, online training/evaluation CLI, analytic grid posterior,
  posterior predictive, checkpoint helpers, and plotting.
- [gp1d.py](gp1d.py): GP-1D regression example with continuous kernel
  hyperparameter latents, discrete kernel selection, online CPU float64 GP
  sampling, and a fixed diagnostic plot.
- [diagnostics.py](diagnostics.py): reusable grid-query helpers for marginal and
  two-variable AR diagnostics.
- [DEVLOG.md](DEVLOG.md): design decisions and rationale. Read this before
  changing architecture or scope.

Next work: train and inspect the GP-1D example long enough to decide whether its
diagnostic is informative or needs a stronger oracle.

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

The Gaussian example trains online, prints posterior moment diagnostics against
the analytic oracle, and saves a plot to `artifacts/gaussian_toy.png` by default.
For comparability, evaluation always uses the same deterministic batch: three
observed `y` values, plus the sampled `mu` and `log_sigma` used only for printed
diagnostics. The constants live in [gaussian_toy.py](gaussian_toy.py), so rerunning the
same checkpoint regenerates the same plotted case.
The plot also compares the posterior predictive density for a new `y`; the
analytic predictive is computed by marginalizing over the posterior grid, not by
plugging posterior moments into a Gaussian.
Training sometimes reveals one latent as a context value and asks for the other,
so the autoregressive diagnostic is trained on the conditional latent queries it
uses at evaluation time.

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
Matern-1/2, Matern-3/2, and periodic. Its diagnostic is not an exact posterior
oracle. It plots a fixed sampled function, ACE's predictive mean/uncertainty,
the posterior over the discrete kernel latent, and marginals for
`log_lengthscale` and `log_outputscale`. The fixed diagnostic uses irregular,
clustered context locations so nearby observations can reveal local roughness;
evenly spaced sparse points made kernel and lengthscale inference mostly
uninformative.

Common artifact names used by the GP-1D example:

- `artifacts/gp1d.pt`
- `artifacts/gp1d.png`

For a short GP-1D run:

```powershell
.\.venv\Scripts\python.exe gp1d.py --steps 20 --batch-size 16
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

The internal token representation is intentionally explicit:

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
    prior: FloatTensor[B, T, n_bins],
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
