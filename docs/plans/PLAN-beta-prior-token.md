# Plan: bounded latent coordinates + Beta information tokens

Created: 2026-06-07
Status: COMPLETE (historical implementation plan)

## Completion Notes
Implemented on 2026-06-07. This plan records the schema migration that landed
bounded continuous-latent coordinates, two-feature information tokens, native
prediction helpers, Gaussian ACEP, and GP-1D compatibility.

Current source of truth is the code plus `DEVLOG.md`. Artifact freshness has
since been superseded by the later shared-reveal DGP work: any Gaussian/GP
artifact regeneration mentioned below is historical for this schema migration,
while the current retrain/export status lives in `DEVLOG.md` under "Single
shared multi-latent reveal strategy".

Later examples reuse the same representation: `sbi_sir.py` and `bo1d.py` emit
finite-spread Beta information tokens for their bounded continuous latents, while
`gp1d.py` remains finite-prior-free and uses only zero-spread known-latent
tokens for continuous latent reveals.

## Summary
Replace the histogram prior encoder with a compact continuous-latent information
token and, at the same time, make bounded continuous latents live in ACE's
internal `[-1, 1]` coordinate system at token boundaries. Examples still sample,
compute oracles, print, and plot in native task coordinates (`mu`, `log_sigma`,
`log_lengthscale`, etc.), but every bounded continuous latent value inserted into
ACE tokens is encoded before it reaches the model.

This is a schema change, not a small compatibility patch. It removes histogram
prior grids, adds explicit latent bounds to `Variable`, centralizes
native<->internal coordinate helpers, and updates the Gaussian toy into a real
ACEP demo with always-on runtime Beta priors. Exact known continuous latents are
represented as zero-spread information tokens, not as separate VALUE-mode
latents.

## Scope
- **In scope**: bounded-latent coordinate helpers in `ace.py`; a 2-feature
  continuous-latent information token; zero-spread exact continuous-latent
  observations; Gaussian ACEP integration; semantic diagnostics and GP-1D
  updates for the new latent-coordinate representation; docs (`DEVLOG.md`,
  `AGENTS.md`, `README.md`, `ace.py` docstrings); explicit native-coordinate
  prediction helpers; short verification runs; then-current Gaussian and GP
  artifact regeneration for the schema migration.
- **Out of scope**:
  - Exact bounded-output distributions. The MDN still predicts on `[-1, 1]` and
    may place small Gaussian-mixture mass outside that interval.
  - Discrete-latent prior tokens. GP-1D's `kernel` latent remains prior-free.
  - Normalizing data variables (`x`, `y`) in the ACE core. Examples remain
    responsible for keeping data values on a reasonable scale.
  - Heavy convergence tuning beyond producing working example artifacts.

## Architectural Decisions

### Token Coordinate Invariant
`Tokens.value` stores model-ready values.

- Data variables: unchanged. `x` and `y` are in task coordinates.
- Discrete variables: unchanged local integer labels via `value_index`.
- Bounded continuous latent QUERY targets: truth in `Tokens.value` is internal
  `[-1, 1]`.
- Bounded continuous latent context information: stored in `Tokens.prior`, not
  as VALUE-mode continuous latent tokens.

The user/example-facing coordinate remains the semantic coordinate named by the
variable, e.g. `mu`, `log_sigma`, `log_lengthscale`. The affine map is:

```text
u = 2 * (theta - lo) / (hi - lo) - 1
theta = lo + 0.5 * (u + 1) * (hi - lo)
```

`Variable.transform` remains semantic metadata (`identity`, `log`, `logit`);
`Variable.bounds` is the hard support in that transformed/native coordinate.
This is deliberately loud: for transformed latents, bounds are in the transformed
coordinate. Example: `log_sigma` bounds are bounds on `log_sigma`, not on
`sigma`.

### Variable Schema
- Remove `Variable.prior_range` and `Variable.prior_bins`.
- Add `Variable.bounds: tuple[float, float] | None = None`.
- Require finite, ordered bounds for continuous latent variables.
- Data variables may leave `bounds=None`; the ACE core does not normalize them.

### Central Coordinate Helpers
Add helpers in `ace.py` so examples do not reimplement affine math:

- `encode_value(variable, value)` for one variable.
- `decode_value(variable, value)` for one variable.
- `encode_token_values(variables, var_id, value)` for token-shaped mixed
  variable tensors.
- `decode_token_values(variables, var_id, value)` for token-shaped mixed
  variable tensors.

These helpers only transform continuous latent variables with bounds; otherwise
they return values unchanged. They must be tensor-friendly and preserve device
and dtype.

### Continuous-Latent Information Token
Keep the existing `PRIOR` mode name, but broaden its meaning for continuous
latents: a PRIOR token is any probabilistic information about a bounded
continuous latent. A finite-spread token is a prior; a zero-spread token is an
exact known value.

- Add `PRIOR_FEATURES = 2` in `ace.py`.
- `Tokens.prior: FloatTensor[B, T, 2]`.
- Continuous latent PRIOR token:

```text
prior[..., 0] = mean in internal [-1, 1] coordinates
prior[..., 1] = spread in internal [-1, 1] coordinates
```

Finite Beta prior:

```text
mean_internal = 2 * mu_unit - 1
spread_internal = sqrt((1 - mean_internal^2) / (nu + 1))
```

where `mu_unit` is the Beta mean on `[0, 1]` and `nu = alpha + beta`.
This is exactly the Beta standard deviation after mapping to `[-1, 1]`.
So `prior[..., 1]` is a standard deviation in ACE's internal coordinate, not
`log nu`, not a variance, and not a native-coordinate width.

Exact known continuous latent:

```text
mean_internal = encoded latent value
spread_internal = 0
```

This implements the DEVLOG's "known value as a limit" design. Exact continuous
latents are zero-spread members of the same PRIOR-token family as finite priors.
They are not separately-moded continuous latent VALUE tokens. VALUE mode remains
for data values and discrete latent labels.

Use at most one information token per bounded continuous latent in a context.
The token is finite-spread when only prior information is available, and the same
slot becomes zero-spread when the latent is known exactly. Exact information
supersedes finite prior information for that latent; finite and exact tokens for
the same continuous latent should not coexist.

### Embedder
Replace the histogram prior MLP with a spread-gated residual around the value
embedding:

```python
prior_input = tokens.prior[..., :2]
prior_payload = self.value_embed(prior_input[..., 0:1])
prior_payload = prior_payload + prior_input[..., 1:2] * self.spread_embed(prior_input)
```

Then select `prior_payload` when `mode == PRIOR`. At `spread == 0`,
`prior_payload == value_embed(mean_internal)` by construction. This is the
representation-level value limit from the DEVLOG. VALUE tokens use
`value_embed(tokens.value)` for data values; continuous latent observations use
PRIOR mode with spread zero.

### Prediction Semantics
`Predictions.log_prob`, `.mean`, `.sample`, and `.continuous_var` operate in
token/model coordinates. For bounded latents, that means `[-1, 1]`.

Training uses those token-space methods. Native-coordinate use is explicit:
`Predictions` should also expose helper methods for user-facing diagnostics and
future APIs:

- `log_prob_native(tokens)`: token-space log probability plus the affine
  Jacobian for bounded continuous latents,
  `log(2 / (hi - lo))`.
- `mean_native(tokens)`: decode bounded continuous latent means back to native
  coordinates; data and discrete variables are unchanged.
- `continuous_var_native(tokens)`: scale bounded continuous latent variances by
  `((hi - lo) / 2)^2`; data variables are unchanged.
- `sample_native(tokens)`: sample in token space, then decode bounded
  continuous latent values.

The raw methods stay token-space so training loss and AR context construction do
not hide coordinate conversions.

`sample_ar` should turn sampled continuous latent targets into zero-spread PRIOR
tokens for context. If a finite-spread information token for that same latent is
already present, replace it; append only when no existing information token for
that latent is present. Data and discrete samples remain VALUE tokens.

`Predictions` needs explicit bounds plumbing. `ACE.__init__` registers buffers
derived from `variables`, and `ACE.forward` passes them to `Predictions`:

```python
is_latent: BoolTensor[n_vars]
is_discrete: BoolTensor[n_vars]
has_bounds: BoolTensor[n_vars]
bound_lo: FloatTensor[n_vars]
bound_hi: FloatTensor[n_vars]
```

These buffers support native helper methods and any context-token conversion
that depends on variable type and bounds.

### Settled Example Semantics
- **Helper API names**: use `encode_value`/`decode_value` for one-variable
  transforms and `encode_token_values`/`decode_token_values` for token-shaped
  mixed-variable tensors. Avoid vague plural names like `encode_values`.
- **Native helper placement**: implement native-coordinate helpers as methods on
  `Predictions`, not as loose example utilities. `Predictions` should carry the
  variable metadata or equivalent tensors needed to identify bounded continuous
  latents and their affine widths.
- **GP-1D stays prior-free**: GP-1D emits no finite-spread runtime prior tokens.
  It only emits zero-spread PRIOR tokens when a continuous latent is revealed.
  The default latent prior is learned from the training distribution over the
  declared bounds.
- **ACEP examples**: Gaussian always emits one information token per
  continuous latent. For `mu` and `log_sigma`, that token is finite-spread prior
  information by default; if the latent is revealed, the same slot is replaced
  by a zero-spread exact-value token. Later SIR and BO examples use the same
  finite-spread Beta information-token representation for their bounded
  continuous latents.
- **Spread feature**: the second prior feature is internal-coordinate standard
  deviation, not `log nu`. `log nu` may be used when sampling hyperpriors, but it
  is not passed to ACE.
- **Gaussian eval prior strength**: use moderately informative fixed eval
  priors so the diagnostic visibly exercises runtime prior conditioning while
  keeping `alpha,beta >= 1`. These are diagnostic defaults, not architectural
  constants:
  - `EVAL_MU_PRIOR = (mu_unit=0.70, nu=20.0)`.
  - `EVAL_LOGSIG_PRIOR = (mu_unit=0.70, nu=8.0)`.

### Coordinate Placement Table
This table pins where native/internal conversions happen. It is the correctness
guard for diagnostics.

| Site | Oracle Grid / Quantity | Model Query | Model Output | Comparison / Display |
| --- | --- | --- | --- | --- |
| Gaussian `mu` marginal | native `mu_grid` | `encode_value(mu_var, mu_grid)` | internal log density on encoded grid | normalize model log density over grid; compute moments/plot against native `mu_grid`; Jacobian cancels for normalized mass |
| Gaussian `log_sigma` marginal | native `logsig_grid` | `encode_value(logsig_var, logsig_grid)` | internal log density on encoded grid | normalize over grid; compute moments/plot against native `logsig_grid`; Jacobian cancels |
| Gaussian AR joint | native `mu_grid`, `logsig_grid` | encoded grids; conditional context replaces that latent's finite-spread PRIOR token with a zero-spread PRIOR token | internal joint log density on encoded grid | normalize joint mass; moments/plot use native mesh grids; constant Jacobians cancel |
| Gaussian predictive `y` | native `y_grid` | native `y_grid` because `y` is data | native data log density | compare directly to native oracle predictive |
| GP `log_lengthscale` marginal | native `ell_grid` | `encode_value(ell_var, ell_grid)` | internal log density on encoded grid | normalize over grid; compute moments/plot against native `ell_grid`; Jacobian cancels |
| GP `log_outputscale` marginal | native `scale_grid` | `encode_value(scale_var, scale_grid)` | internal log density on encoded grid | normalize over grid; compute moments/plot against native `scale_grid`; Jacobian cancels |

Absolute native density comparisons must use `log_prob_native` or manually add
the affine Jacobian. Normalized posterior curves and moments do not need the
Jacobian because each latent's affine Jacobian is constant across its grid.

## Exhaustive Call-Site Checklist

Removing `cfg.prior_bins` and changing latent token coordinates touches these
areas.

- `ace.py`
  - Add `PRIOR_FEATURES`.
  - Update `Variable` fields and validation.
  - Remove `ACEConfig.prior_bins`.
  - Add encode/decode helpers.
  - Remove `prior_embed`; add `spread_embed`.
  - Update `_embed` for 2-feature continuous-latent information tokens.
  - Update sample-token helpers / `sample_ar` so sampled continuous latents
    become zero-spread PRIOR tokens and replace any existing finite-spread
    token for the same latent.
  - Add `Predictions.sample_as_context_tokens(tokens)`:
    - bounded continuous latent samples -> PRIOR tokens with
      `prior=(sample_internal, 0)`;
    - data and discrete samples -> VALUE tokens.
  - Add a small context helper for replacing an existing continuous-latent
    information token by `var_id`, used by diagnostics and AR sampling.
  - Add explicit native-coordinate prediction helpers / methods with the affine
    density Jacobian.
  - Register and pass `has_bounds`, `bound_lo`, and `bound_hi` buffers into
    `Predictions`.
  - Update docstrings for `Tokens.prior` and bounded latent coordinates.
- `diagnostics.py`
  - Import `PRIOR_FEATURES`.
  - Allocate `prior=torch.zeros(..., PRIOR_FEATURES)`.
  - Grid-query helpers should accept grids in token coordinates. Callers encode
    native latent grids before querying.
  - Conditional latent context should use zero-spread PRIOR tokens for
    continuous latents, replacing an existing finite-spread information token
    for that latent when one is present.
  - Replace/rename `value_token` with a less misleading known-context helper:
    it emits zero-spread PRIOR tokens for bounded continuous latents and VALUE
    tokens for data/discrete variables.
- `gaussian_toy.py`
  - `variables()` adds `bounds=MU_RANGE` and `bounds=LOGSIG_RANGE`; no
    `prior_bins`.
  - Sampler draws native latents, then encodes latent QUERY token values.
  - Always emit two finite-spread PRIOR tokens, with prior means encoded into
    `[-1, 1]`.
  - Revealed continuous latents replace their finite-spread PRIOR token with a
    zero-spread PRIOR token, not a VALUE token.
  - Analytic oracle remains native.
  - Model queries use encoded latent grids; plots/printed moments decode or use
    native grids for display.
  - `--bins` remains oracle/diagnostic grid resolution only.
- `gp1d.py`
  - `variables()` adds bounds for `log_lengthscale` and `log_outputscale`; no
    `prior_bins`.
  - Sampler and oracle remain native.
  - Encode latent QUERY token values.
  - Revealed continuous latents are zero-spread PRIOR tokens. The discrete
    kernel reveal remains a VALUE token via `value_index`.
  - Query encoded latent grids; decode/label plots in native units.
  - Remove `--bins`; keep `--oracle-bins`.
- Docs
  - README/AGENTS token field docs: `prior: [B, T, 2]`.
  - DEVLOG: add implementation entry and mark the old prior-redesign note as
    resolved by this implementation.

## Phases

### Phase 1: Core Schema and Coordinates (`ace.py`)
**Goal**: ACE has explicit bounded-latent coordinates and no histogram prior
schema.

**Work**:
- Add `PRIOR_FEATURES = 2`.
- Replace `Variable.prior_range`/`prior_bins` with `Variable.bounds`.
- Validate that continuous latent variables have finite ordered bounds.
- Remove `ACEConfig.prior_bins`.
- Add encode/decode helpers.
- Add helper(s) for affine density Jacobians for bounded continuous latents.
- Register `has_bounds`, `bound_lo`, and `bound_hi` buffers in `ACE.__init__`.
- Update `Tokens` and `Batch` docstrings to state that bounded latent values are
  tokenized in internal coordinates.

**Verification**:
- [x] `python -c "import ace"` imports cleanly.
- [x] No `prior_bins`, `prior_range`, or `prior_embed` references remain in
      `ace.py`.

### Phase 2: Core Prior Embedder (`ace.py`)
**Goal**: ACE consumes a 2-feature continuous-latent information token with an
honest zero-spread value limit.

**Work**:
- Replace `self.prior_embed = _mlp(...)` with
  `self.spread_embed = _mlp(2, cfg.mlp_hidden, cfg.d_model)`.
- Update `_embed`:

```python
prior_input = tokens.prior[..., :2]
prior_payload = self.value_embed(prior_input[..., 0:1])
prior_payload = prior_payload + prior_input[..., 1:2] * self.spread_embed(prior_input)
payload = torch.where((tokens.mode == PRIOR).unsqueeze(-1), prior_payload, val)
```

At `spread == 0`, the PRIOR payload exactly reduces to the embedded latent
location.

Add explicit native-coordinate prediction methods on `Predictions`:

```python
pred.log_prob_native(tokens)
pred.mean_native(tokens)
pred.continuous_var_native(tokens)
pred.sample_native(tokens)
```

These are methods on `Predictions`; `Predictions` carries enough variable
metadata, or equivalent registered tensors, to identify bounded continuous
latents and their affine widths.

Add:

```python
pred.sample_as_context_tokens(tokens)
```

This replaces using `sample_as_tokens` for AR context construction. It emits
zero-spread PRIOR tokens for bounded continuous latent samples and VALUE tokens
for data/discrete samples. AR context construction must replace an existing
information token for the same continuous latent when present, not create a
duplicate.

**Verification**:
- [x] Small synthetic forward pass with one VALUE, one PRIOR, and one QUERY token
      works.
- [x] A synthetic bounded latent has
      `log_prob_native = log_prob + log(2 / (hi - lo))`.
- [x] `Beta(1,1)` maps to `mean_internal = 0` and
      `spread_internal = 1 / sqrt(3)`, matching `Uniform[-1,1]`.
- [x] A spread-zero PRIOR token embeds exactly as `value_embed(mean_internal)`
      before adding variable/mode/x embeddings.

### Phase 3: Diagnostics Compatibility (`diagnostics.py`)
**Goal**: reusable grid-query helpers work with the new prior shape and explicit
model-coordinate grids before Gaussian evaluation uses them.

**Work**:
- Replace `model.cfg.prior_bins` with `PRIOR_FEATURES`.
- Update docstrings to say `query_log_density` expects `values` in token/model
  coordinates.
- Replace/rename `value_token` with a known-context helper that emits:
  - zero-spread PRIOR tokens for bounded continuous latents;
  - VALUE tokens for data/discrete variables.
- Change `conditional_log_density` to use the known-context helper and replace
  an existing finite-spread information token for the conditioned latent when
  present.
- Keep `normalized_moments` generic; callers choose which coordinate their grid
  represents.

**Verification**:
- [x] `python -c "import diagnostics"` imports cleanly.

### Phase 4: Gaussian Toy ACEP (`gaussian_toy.py`)
**Goal**: Gaussian toy trains with always-on runtime Beta priors and compares
against a Beta-aware analytic oracle.

**Work**:
- Add Beta helpers:
  - `sample_prior_params(shape, device) -> (mu_unit, nu)`.
  - `beta_alpha_beta(mu_unit, nu) -> (alpha, beta)`.
  - `prior_features(mu_unit, nu) -> [..., 2]`, with mean
    `2 * mu_unit - 1` and spread
    `sqrt((1 - mean_internal^2) / (nu + 1))`.
  - `known_latent_features(value_internal) -> [..., 2]`, with spread `0`.
  - `draw_from_beta(mu_unit, nu, lo, hi, device) -> native value`.
  - `beta_logprior_on_grid(grid_native, mu_unit, nu, lo, hi) -> [G]`.
- Remove the old `fixed_prior` helper; the oracle must not silently retain a
  uniform prior once eval/runtime Beta priors are present.
- Keep native `mu` and `log_sigma` in `ToyBatch` fields for diagnostics.
- Encode latent target values via `encode_value` or `encode_token_values`,
  depending on construction shape.
- Context layout:
  - y VALUE tokens.
  - one mu PRIOR/info slot: finite-spread prior by default, zero-spread exact
    value if `mu` is revealed.
  - one log_sigma PRIOR/info slot: finite-spread prior by default, zero-spread
    exact value if `log_sigma` is revealed.
- Target layout:
  - mu QUERY token with encoded truth.
  - log_sigma QUERY token with encoded truth.
  - y QUERY tokens in native data units.
- Fixed eval batch uses fixed informative Beta priors:
  `EVAL_MU_PRIOR = (0.70, 20.0)` and
  `EVAL_LOGSIG_PRIOR = (0.70, 8.0)`. These are tunable diagnostic defaults.
- Analytic posterior uses native grids and Beta priors.
- Model marginal queries use encoded grids; printed/plot values use native grids
  and native-coordinate prediction helpers where convenient.
- AR-joint diagnostic follows the coordinate placement table: native grids are
  encoded before model queries; conditional known latent context replaces the
  relevant finite-spread PRIOR token with zero-spread PRIOR mode; displayed
  moments use native mesh grids.
- `build_model` and checkpoint loading use `ACE(variables(), cfg)`.
- `--bins` help text becomes "oracle/diagnostic grid bins".

**Verification**:
- [x] `.\.venv\Scripts\python.exe gaussian_toy.py --steps 20 --batch-size 32`
      completes.
- [x] `.\.venv\Scripts\python.exe gaussian_toy.py --device cpu --steps 20`
      completes.
- [x] A short run prints posterior moments against the oracle; convergence
      tuning is deferred beyond this schema migration.

### Phase 5: GP-1D Compatibility (`gp1d.py`)
**Goal**: GP-1D remains prior-free but runs with bounded continuous latent
coordinates.

**Work**:
- Update `variables()` with bounds for `log_lengthscale` and `log_outputscale`.
- Import `PRIOR_FEATURES` and encode/decode helpers.
- Update `make_tokens` to allocate `[B, T, PRIOR_FEATURES]`.
- Remove all `bins` plumbing from token construction and model config.
- Encode latent target values.
- Represent revealed continuous latents as zero-spread PRIOR tokens. The discrete
  kernel reveal remains a VALUE token via `value_index`.
- Evaluate latent posterior marginals by encoding native oracle grids before
  calling `query_log_density`.
- Print and plot in native coordinates; use native prediction helpers for
  bounded latent summaries where direct model moments are displayed.
- Remove the obsolete GP `--bins` argument.

**Verification**:
- [x] `python -c "import gp1d"` imports cleanly.
- [x] `.\.venv\Scripts\python.exe gp1d.py --steps 20 --batch-size 16`
      completes.

### Phase 6: Documentation
**Goal**: docs describe the implemented coordinate and prior semantics.

**Work**:
- `DEVLOG.md`: add a dated implementation entry:
  - bounded continuous latents are tokenized to `[-1, 1]`;
  - continuous-latent PRIOR token is `(mean_internal, spread_internal)`;
  - exact known continuous latents use spread zero;
  - native-coordinate prediction helpers apply decode maps and the affine
    density Jacobian explicitly;
  - histogram prior grids removed;
  - Gaussian is now an ACEP demo;
  - GP remains prior-free but uses encoded latent tokens;
  - exact bounded MDN and discrete priors deferred.
- Mark the old prior-redesign entry as resolved by this implementation.
- `AGENTS.md`: update architecture bullets and gotchas.
- `README.md`: update current status and `Tokens` schema.
- `PLAN-beta-prior-token.md`: mark phases as complete as work lands. The file
  now lives under `docs/plans/` as a historical development artifact.

**Verification**:
- [x] Grep docs for stale `prior_bins`/histogram claims; only historical DEVLOG
      reasoning should remain.
- [x] No README/AGENTS command examples mention GP `--bins`.

### Phase 7: Artifacts and Final Checks
**Goal**: leave the repo runnable end to end.

**Current note**: this phase describes the artifact state at the end of the
Beta-token schema migration. Later shared-reveal DGP work changed the training
distribution and made the current Gaussian/GP retrain/export status a separate
follow-up tracked in `DEVLOG.md`.

**Work**:
- Treat all existing checkpoints as stale after the schema change.
- Regenerate the Gaussian artifact pair:
  - `artifacts/gaussian_toy.pt`
  - `artifacts/gaussian_toy.png`
- Regenerate the GP-1D artifact pair:
  - `artifacts/gp1d.pt`
  - `artifacts/gp1d.png`
- Verify both checkpoint reload paths:
  - `gaussian_toy.py --eval-only --load-checkpoint artifacts/gaussian_toy.pt`
  - `gp1d.py --eval-only --load-checkpoint artifacts/gp1d.pt`

**Verification**:
- [x] `rg -n "prior_bins|prior_range|prior_embed"` has no live-code or
      current-doc hits.
- [x] Gaussian short run passes.
- [x] GP short run passes.
- [x] Gaussian eval-only checkpoint reload passes.
- [x] GP eval-only checkpoint reload passes.

## Risks
- **Coordinate confusion**: native values and token values now differ for
  bounded continuous latents. Mitigation: central helpers, explicit docstrings,
  and example dataclasses keep native values for diagnostics.
- **Density interpretation**: raw model log probabilities are in token space for
  bounded latents. Native-space density is exposed only through explicit helper
  methods that add the affine Jacobian.
- **Trainability**: runtime priors previously made Gaussian harder under
  histograms. The new Beta token is compact, shares the value embedding for
  location, and has an exact value limit.
- **Endpoint divergence**: Beta priors with alpha or beta below 1 diverge at
  bounds. The oracle clamps unit-grid values; fixed eval priors should use
  alpha,beta >= 1 unless we specifically want to inspect edge behavior.
- **Information-token replacement bookkeeping**: Gaussian ACEP keeps one
  continuous-latent information slot per latent. Reveal and AR-conditioning code
  must replace the finite-spread token with a zero-spread token for the same
  `var_id`, not append a duplicate.
- **Checkpoint break**: old `.pt` files will not load. This is accepted; both
  Gaussian and GP artifacts are regenerated as part of this work.

## Open Questions
None. The remaining plan choices are settled above.
