## Project

nanoACE is a small, readable implementation of the Amortized Conditioning Engine (ACE):
data, interpretable latents, and runtime prior information are all **tokens**; the model
conditions on one token set and predicts distributions over another.

**Read [DEVLOG.md](DEVLOG.md) before any architectural or scope change.** It records the
design decisions and their rationale, and it is the source of truth for what is in scope.
[README.md](README.md) covers setup and the public-facing summary. Historical
implementation plans live in [docs/plans/](docs/plans/); use them for rationale and
checklists, but if a plan conflicts with the code or DEVLOG, the code and DEVLOG win.
The paper this is based on is in [paper/](paper/) as markdown.

**Do not use assistant/session memory systems for this project.** Anything durable
belongs in the repository docs, as appropriate: design decisions and open items in
`DEVLOG.md` (or an extension's local DEVLOG), public-facing summaries in the READMEs,
implementation plans in `docs/plans/`.

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
`ace.py`; `train.py` holds the shared training loop, checkpointing, and CLI/config
plumbing; `gaussian_toy.py`, `gp1d.py`, `sbi_sir.py`, and `bo1d.py` are the current
executable task examples built on top of both.

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
- **Training spine (`train.py`).** The four examples share one loop, checkpoint format,
  `build_model`, and a `common_parser()` argparse parent (with a light-YAML `--config`
  layered under explicit CLI flags). `fit` reseeds the global RNG with `mix_seed(seed, step)`
  each step (so each batch is a pure function of `(seed, step)`: reproducible + resume-exact)
  and takes a `(step) -> Batch` thunk, which the online samplers and the `data.py` `PoolReader`
  both satisfy (one training path). It runs Adam + grad-clip with cosine LR (default;
  `--lr-schedule constant` reproduces the old loop) and supports simple resume
  (`--resume`/`--ckpt-every`). Each example keeps its own `main()`, sampler,
  `evaluate`, `plot_diagnostic`, and a 2-arg `load_checkpoint(path, device)` wrapper (the
  contract the playground calls). Checkpoints are `{cfg, seed, state_dict}` plus optional
  `config` provenance and optional `{optimizer, scheduler, step}` for resume — all
  additive, so legacy files still load. `fit` itself stays synchronous; the optional
  offline `PoolReader` owns shard-level prefetching. `main()` is intentionally not
  centralized (examples stay readable end-to-end).

## Conventions and gotchas

- **Continuous latent coordinates are internal.** Bounded continuous latent token values
  live in `[-1, 1]`; examples keep native semantic coordinates for sampling, oracles,
  printing, and plotting, and use `encode_value` / `decode_value` at token boundaries.
- **`Tokens.prior` has two features.** For bounded continuous latent `PRIOR` tokens,
  `prior[..., 0]` is the internal-coordinate mean/location and `prior[..., 1]` is
  internal-coordinate spread. Spread zero is an exact known latent value. Gaussian and
  SIR emit finite-spread Beta information tokens (ACEP); GP-1D emits no finite-spread
  priors, only zero-spread tokens when a continuous latent is revealed. The shared
  Beta prior-token helpers live in `ace_prior_beta.py`; model-side PRIOR token
  semantics live in `ace.py`.
- **Latent reveal uses a shared mixture DGP.** `sample_reveal_mask` in `ace.py` picks no
  reveal with probability `q`; otherwise it splits 50/50 between a uniform non-empty subset
  and a uniform count (`k` in `1..L`) then a uniform size-`k` subset. All four examples
  (`gaussian_toy`, `gp1d`, `sbi_sir`, `bo1d`) share it via `latent_context_prob`
  (= P(reveal anything), default 0.5), so conditioning on any subset of latents — including
  multi-pin — is in-distribution. The offline `data.py` reader uses `reveal_mask_from_index`,
  a stateless index-keyed variant that reproduces the same distribution.
- **Offline pool: frozen physics vs free split.** A `data.py` pool caches only an example's
  `draw_instances` physics plus its `gen_config` (the DGP constants — hashed; changing them
  needs a regenerate). The context/target split (GP-1D and BO use **complement-targets**:
  every non-context point is a target, `n_target = N_TOTAL - n_context`) and the reveal mask
  are recomputed at read time from a stateless `(seed, position)` hash, so batch size,
  `--steps`, and the reveal strategy can all change without regenerating. `N_TOTAL` counts
  data points only; latent tokens are separate columns on top. `PoolReader` lazy-loads
  touched shards through a bounded LRU cache and a one-thread prefetcher; it does not load
  the full pool into RAM.
- **Data values remain task-scaled.** Data values should generally be scaled around
  `[-1, 1]` at generation time. This is a soft convention, not clipping: Gaussian and
  GP samples can have stochastic tails outside that range, which may matter when reading
  predictive calibration.
- **Target tokens may carry truth** in `value`/`value_index` while `mode == QUERY`; the
  embedder ignores it and the loss uses it. For prediction-only calls, pass dummy values
  and simply don't call `.log_prob`.
- **`temp/` is not part of nanoACE.** If a gitignored `temp/` directory is present, treat
  it as archived external experiment code. It may contain useful ideas, but large
  experiment-management machinery such as Slurm scripts, cache provenance, elaborate
  prefetch systems, and resume matrices should not be copied into this repository.
- **Currently implemented:** `ace.py` for the model, `ace_prior_beta.py` for the
  shared Beta-specific ACEP prior-token helpers (including `sample_contaminated`
  for robust priors), `train.py` for the shared training loop / checkpointing /
  argparse parent + light-YAML config / cosine LR / simple resume,
  `gaussian_toy.py` for the executable Gaussian example and analytic oracle, `gp1d.py` for
  the executable GP regression example, `sbi_sir.py` for the executable SIR
  simulation-based-inference example, `bo1d.py` for the executable 1D Bayesian
  optimization example (optimum latents + runtime prior injection, no oracle), and
  `diagnostics.py` for grid queries, and `data.py` for the optional offline sharded pool
  (`write_pool` + `PoolReader`, GP-1D and BO only; generate → save → train). GP-1D and BO
  split their sampler into `draw_instances` (cached physics) + `assemble` (RNG-free
  tokenization shared by the online and pooled paths); Gaussian and SIR stay online-only.
- **`playground/` is a non-core example, not part of the core.** It is a Vite + TypeScript
  in-browser demo that reimplements `ace.py`'s forward pass in TS (parity-tested against
  the PyTorch model) so trained checkpoints run client-side. Current tabs cover GP-1D,
  Gaussian, SIR, BO-1D, an AR-buffer joint-sampling tab (the
  `extensions/arbuffer/` model through a parity-tested TS port of its incremental
  sampler; its tests self-skip when the local blob is absent), and a **local-only**
  ALINE active-learning tab (the `extensions/aline/` model + a fixture-pinned TS
  port of the gp1d DGP as the hidden-function environment; same self-skip
  discipline, not in the deploy workflow). The core stays
  torch-only and legible; do not let the JS toolchain or
  web concerns bleed into `ace.py` or the examples.
  User-facing playground text (hints, explainer modals, README) is didactic and
  matter-of-fact: state the task, what ACE does, and the honest trade-offs — no
  hyperbole, and no dismissiveness toward classical methods.
  See `playground/README.md` and the DEVLOG "Web playground" entry.
- **`extensions/` holds non-core model extensions.** Each child is self-contained
  (own `README.md` + `DEVLOG.md`; root docs carry only pointers), torch-only, and changes
  **no core file** — it imports the core (and may subclass its internals) from the repo
  root via the playground's `sys.path` bootstrap pattern. Currently:
  `extensions/arbuffer/`, a causal autoregressive buffer (Hassan et al., 2026) warm-started
  from a trained GP-1D checkpoint — fast coherent joint function sampling via one cached
  context encoding. Two target-read modes: the default separate zero-init gated read
  (bit-exact warm start, frozen base) and the retained paper-style `--concat-read`
  (one softmax over `[context, buffer]` keys + a learned per-head logit bias, joint
  fine-tune with the 50/50 curriculum). An automatic step-0 parity check guards the
  coupling to core internals (bitwise for the plain forward in both modes and for the
  buffered forward in separate-read mode; reported drift for the concat read): if
  `ace.py`'s forward changes, the warm start fails loudly. And `extensions/aline/`,
  ALINE (Huang et al., 2025): joint amortized inference + active data acquisition on
  GP-1D, warm-started from a trained GP-1D checkpoint — the unchanged core ACE is the
  inference network (the acquisition goal ξ = which target tokens are active; query
  candidates are data QUERY tokens), plus a small read-only policy decoder over the
  candidate pool trained with REINFORCE on self-estimated information gain. Alternating
  NLL/policy phases with a structural φ/ψ gradient firewall; the inference path is
  asserted bit-equal to the base ACE forward (a permanent parity guard, not just
  step-0). A local-only playground tab runs its acquisition loop in-browser.
