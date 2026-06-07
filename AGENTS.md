## Project

nanoACE is a small, readable implementation of the Amortized Conditioning Engine (ACE):
data, interpretable latents, and runtime prior information are all **tokens**; the model
conditions on one token set and predicts distributions over another.

**Read [DEVLOG.md](DEVLOG.md) before any architectural or scope change.** It records the
design decisions and their rationale, and it is the source of truth for what is in scope.
[README.md](README.md) covers setup and the public-facing summary. The paper this is based
on is in [paper/](paper/) as markdown.

## Commands

Windows / PowerShell, using the project virtualenv explicitly:

```powershell
# setup (CUDA wheel pinned in requirements.txt: torch 2.11.0+cu128)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# run the Gaussian example (trains, then prints oracle-vs-model posterior moments)
.\.venv\Scripts\python.exe gaussian_toy.py

# run the GP-1D example (trains, then prints oracle-vs-model GP diagnostics)
.\.venv\Scripts\python.exe gp1d.py

# run the SIR SBI example (trains, then prints oracle-vs-model posteriors under
# a uniform vs an informative runtime prior)
.\.venv\Scripts\python.exe sbi_sir.py

# run the 1D Bayesian optimization example (trains, then prints x_opt/y_opt
# posteriors under a uniform, a correct, and a wrong runtime prior; no oracle)
.\.venv\Scripts\python.exe bo1d.py

# short run that verifies the script starts and completes
.\.venv\Scripts\python.exe gaussian_toy.py --steps 20 --batch-size 32
.\.venv\Scripts\python.exe gp1d.py --steps 20 --batch-size 16
.\.venv\Scripts\python.exe sbi_sir.py --steps 20 --batch-size 16
.\.venv\Scripts\python.exe bo1d.py --steps 20 --batch-size 16

# verify only the BO data-generating process scale (no training)
.\.venv\Scripts\python.exe bo1d.py --scale-check

# force CPU
.\.venv\Scripts\python.exe gaussian_toy.py --device cpu --steps 20
.\.venv\Scripts\python.exe gp1d.py --device cpu --steps 20 --batch-size 16
.\.venv\Scripts\python.exe sbi_sir.py --device cpu --steps 20 --batch-size 16
.\.venv\Scripts\python.exe bo1d.py --device cpu --steps 20 --batch-size 16
```

There is no separate test suite, linter, or build step. **Verification = run `gaussian_toy.py` and
check the printed model posterior moments track the analytic `oracle` moments**. The
Gaussian toy has an analytic grid posterior in the same file. Keep that check loose, not
a strict quality gate (see DEVLOG "Open questions"). For `gp1d.py`, verification is a
short run plus visual inspection of the fixed diagnostic plot. The GP diagnostic has a
numerical grid oracle over kernel and hyperparameters; treat it as exact up to grid
resolution, bounded hyperparameter ranges, and Cholesky jitter, not as a closed-form
analytic posterior. For `sbi_sir.py`, verification is a short run plus the fixed
diagnostic: a numerical `(beta, gamma)` grid oracle (deterministic ODE + Gaussian
observation likelihood), shown for a uniform and an informative runtime prior so the
prior conditioning is visible; treat it as exact up to grid resolution, the bounded rate
ranges, and the RK4/interpolation step. For `bo1d.py`, there is **no oracle** (the `|.|`
fold in the optimum-planting DGP destroys Gaussianity, and the other three examples
already carry grid oracles): verification is `--scale-check` (data token values sit
in `[-1, 1]`) plus the fixed three-prior diagnostic, read structurally -- the `x_opt`
posterior should tighten/shift toward truth from the uniform to the correct prior, and
the wrong prior should be *resisted* (posterior stays near the data, not the wrong prior)
thanks to the epsilon-contaminated prior. The true function and true `(x_opt, y_opt)` are
plotted as the reference.

## Architecture (the cross-file picture)

Everything routes through one idea: **variables as tokens**. The model is in
`ace.py`; `gaussian_toy.py`, `gp1d.py`, `sbi_sir.py`, and `bo1d.py` are the current
executable task examples built on top of it.

- **Data model (`ace.py`).** `Variable` is the static schema (name, `kind` data/latent,
  continuous/discrete + `cardinality`, `transform`, optional bounded continuous-latent
  `bounds`). `Tokens` is a
  padded, field-parallel struct of `[B, T]` tensors (`var_id, x, value, value_index,
prior, mode, mask`). `Batch` = `variables + context: Tokens + target: Tokens`. Device
  movement lives on these objects (`.to`); `cat_tokens` / `Tokens.column` / `with_values`
  are the manipulation primitives.
- **`mode` drives the embedder.** Each token is `VALUE`, `PRIOR`, or `QUERY` (constants in
  `ace.py`). `ACE._embed` builds `var_embed + mode_embed + x_embed + payload`, where the
  payload is selected by mode: spread-gated continuous-latent information payload for `PRIOR`, the learned `unknown`
  parameter for `QUERY` (so target truth is ignored even when present), else the value
  embedding. Latent tokens zero out `x`. Continuous values go through an MLP; discrete
  values index one shared embedding table via per-variable offsets (`disc_offsets`).
- **Attention (`ACEBlock`).** Separated **context self-attention** then **target→context
  cross-attention**, pre-LN, no directional mask — conditioning direction is structural
  (targets read context, never each other). Padding is handled by `key_padding_mask` from
  the context `mask`. This is deliberate; do not replace it with a single masked
  `(N+M)²` stream (see DEVLOG "Attention").
- **Heads + dispatch (`Predictions`).** `ACE.forward` returns a `Predictions` object
  wrapping a shared continuous **MDN head** (`3*K` outputs → log-weights/means/scales) and
  a shared **categorical head** (`max_cardinality` logits, masked per-token to the
  variable's `cardinality`). `Predictions.log_prob/.mean/.sample` dispatch per token by
  `is_discrete[var_id]`, so callers never branch on type.
- **Loss is type-agnostic.** `ACE.loss` is just `-log_prob` over active target tokens,
  weighted by scalar `data_weight` / `latent_weight` and the target `mask`.
- **Autoregression is a helper, not architecture.** The base model is a _diagonal_
  prediction map (independent 1-D marginals). `sample_ar` builds joint samples by
  predicting one target, sampling it, appending data/discrete samples as `VALUE` tokens
  and bounded continuous latent samples as zero-spread `PRIOR` tokens, then repeating.

## Conventions and gotchas

- **Continuous latent coordinates are internal.** Bounded continuous latent token values
  live in `[-1, 1]`; examples keep native semantic coordinates for sampling, oracles,
  printing, and plotting, and use `encode_value` / `decode_value` at token boundaries.
- **`Tokens.prior` has two features.** For bounded continuous latent `PRIOR` tokens,
  `prior[..., 0]` is the internal-coordinate mean/location and `prior[..., 1]` is
  internal-coordinate spread. Spread zero is an exact known latent value. Gaussian and
  SIR emit finite-spread Beta information tokens (ACEP); GP-1D emits no finite-spread
  priors, only zero-spread tokens when a continuous latent is revealed. The shared Beta
  prior-token helpers live in `ace_prior.py`.
- **Latent reveal uses uniform non-empty subsets.** `sample_reveal_mask` in `ace.py`
  chooses no reveal with probability `q`; otherwise it samples a uniform non-empty
  subset of the task latents. Gaussian and GP-1D are both trained under this multi-reveal
  DGP, so exact multi-pin conditioning in the playground is in-distribution.
- **Data values remain task-scaled.** Data values should generally be scaled around
  `[-1, 1]` at generation time. This is a soft convention, not clipping: Gaussian and
  GP samples can have stochastic tails outside that range, which may matter when reading
  predictive calibration.
- **Target tokens may carry truth** in `value`/`value_index` while `mode == QUERY`; the
  embedder ignores it and the loss uses it. For prediction-only calls, pass dummy values
  and simply don't call `.log_prob`.
- **`temp/` is not part of nanoACE.** If a gitignored `temp/` directory is present, treat
  it as archived external experiment code. It may contain useful ideas, but large
  experiment-management machinery such as Slurm scripts, cache provenance, prefetch
  systems, and resume matrices should not be copied into this repository.
- **Currently implemented:** `ace.py` for the model, `ace_prior.py` for the shared ACEP
  Beta prior-token helpers (including `sample_contaminated` for robust priors),
  `gaussian_toy.py` for the executable Gaussian example and analytic oracle, `gp1d.py` for
  the executable GP regression example, `sbi_sir.py` for the executable SIR
  simulation-based-inference example, `bo1d.py` for the executable 1D Bayesian
  optimization example (optimum latents + runtime prior injection, no oracle), and
  `diagnostics.py` for grid queries. `data.py` / `train.py` are planned in DEVLOG "Layout"
  but not yet built.
- **`playground/` is a non-core example, not part of the core.** It is a Vite + TypeScript
  in-browser demo that reimplements `ace.py`'s forward pass in TS (parity-tested against
  the PyTorch model) so trained checkpoints run client-side. The core stays torch-only and
  legible; do not let the JS toolchain or web concerns bleed into `ace.py` or the examples.
  Treat it like `temp/` in spirit (separate, optional), but unlike `temp/` it *is* checked
  in and maintained. See `playground/README.md` and the DEVLOG "Web playground" entry.
