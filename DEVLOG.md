# nanoACE DEVLOG

A running log of design decisions and their rationale. The *why* matters as much as
the *what* — this file exists so a human or coding agent can reconstruct the reasoning
behind the code without spelunking git history.

Reference: Chang et al. (2025), *Amortized Probabilistic Conditioning for Optimization,
Simulation and Inference* (AISTATS 2025). Paper markdown lives in `paper/`.

---

## 2026-06-07 — 1D Bayesian optimization example (`bo1d.py`)

Full design in [PLAN-bo1d.md](PLAN-bo1d.md). Status: **built and run**. The plan
was checked by two reviewers, revised, then implemented and validated (CPU run,
torch 2.12.0). The DGP, training, and three-prior diagnostic work end to end;
`--scale-check` confirms data token values sit in `[-1, 1]` (~0.5% tail spill).
The structural checks pass: uniform→correct tightens/shifts `p(x_opt | D)` toward
truth, and the wrong prior is resisted (posterior stays near the data, not the
wrong prior, thanks to the ε floor). The effect is directionally correct but
modest (the chosen fixed case is deliberately hard -- the true optimum sits
unobserved between context points -- and ε=0.1 caps prior influence); sharpening
it is optional loose tuning, recorded in the plan's Status. This entry reflects
the revised plan (see PLAN-bo1d.md "Review notes").

- **Fourth example: `bo1d.py`.** 1D Bayesian optimization. The two latents are the
  global optimum's **location** `x_opt` and **value** `y_opt`. The headline is that
  ACE amortizes `p(x_opt | D)` and `p(y_opt | D)` directly (normally intractable,
  the reason BO needs bespoke acquisition machinery) and accepts a runtime Beta
  prior over the optimum location (the paper's πBO / ACEP-TS). It is a mix:
  GP function sampling + sampled kernel/hyperparameters from `gp1d.py`, ACEP Beta
  prior tokens + the gaussian/sir reveal mechanism + observation noise, and a
  new optimum-planting DGP. Both latents are bounded continuous and reuse the
  existing PRIOR-token path, so there is **no new `ace.py` machinery**.
- **Latents are instance properties, not class properties.** Unlike `gp1d.py`
  (kernel/hyperparameters describe the function *class*), `x_opt`/`y_opt` describe
  the *specific sampled function*. This is the paper's BO headline and what makes
  `p(x_opt | D)` worth amortizing. The kernel/lengthscale/outputscale are sampled
  **nuisance**, not predicted -- `gp1d.py` already covers the discrete/kernel path,
  so a kernel latent here would be redundant.
- **DGP adapted from Appendix C.3.1** (not faithful -- the fold operand and the
  role of the min-value distribution differ). Sample hyperparameters; draw `x_opt`
  from the (contaminated) prior; draw the natural optimum depth `d` from the
  min-value distribution (min of `N = ceil(2/ℓ)` Gaussian draws, with `p=0.1` an
  extra `Exp(1)` "unexpectedly low" kick), then clamp `d = min(d, 0)` and cap
  `|d|`; sample a GP draw conditioned on `g_c(x_opt)=d` via **Matheron's rule** (a
  true posterior sample, not a mean-shift); then fold+envelope
  `f(x) = |g_c(x) − d| + (1/5)(x − x_opt)² + y_opt`. Both added terms are `≥ 0`
  and vanish together only at `x = x_opt`, and the envelope is strictly positive
  off `x_opt`, so `x_opt` is the **exact, unique** global minimum with value
  `y_opt` -- level-`d` re-crossings of the fold are lifted above `y_opt` by the
  envelope (they create the kinked Fig.-S12 geometry / multi-basin structure, not
  spurious minima). The min-value machinery shapes the *local geometry/depth* (the
  conditioning bump's depth is set by `d`, its width by the lengthscale); **our
  prior is the leveling shift `y_opt`**, replacing the paper's final `U[-5, 5]`
  offset. This honors keeping the full min-value/`Exp(1)` machinery: it shapes the
  function; the prior only sets the absolute level.
- **No oracle (deliberate).** The `|·|` fold destroys Gaussianity, so there is no
  closed-form posterior, and the other three examples already carry numerical grid
  oracles -- this one demonstrates the no-oracle case. A Monte-Carlo simulator
  posterior was considered and declined. The gate is **structural + qualitative**:
  short run + a token-scale + contamination-marginal histogram check + a fixed
  diagnostic plotting the true function, marked true `(x_opt, y_opt)`, and ACE's
  marginals. The three prior columns give real falsifiability -- uniform→correct
  must tighten (the model uses priors) and correct→wrong must recover (it does not
  blindly follow). Recorded here and in `AGENTS.md` as a departure from the oracle
  convention.
- **ε-contamination ("robust prior").** The effective generative prior is
  `(1−ε)·Beta + ε·Uniform` (ε≈0.1, a classic robust-Bayes / ε-contamination
  prior), applied to both latents. **Why it is not redundant with the existing
  `sample_prior_params` mixture:** that helper always draws truth from the *same*
  Beta the token encodes -- the token never lies -- so a model trained on it learns
  to trust a concentrated token fully and the wrong-prior column would fail.
  ε-contamination's job is to **decouple truth from the token** a fraction ε of the
  time, the one thing `sample_prior_params` cannot do. So `bo1d` reuses
  `sample_prior_params` for the token and applies contamination *only at the
  truth-draw* (new `ace_prior.sample_contaminated`). It lives entirely in the DGP
  truth-draw and plot overlays, not in the token or model (a single Beta token
  cannot represent a mixture, and need not). Corrected framing: the model does not
  learn "a global discount knob" -- the contaminated prior is the true generative
  prior, NLL learns the Bayes-optimal posterior under it, and the uniform floor
  keeps that posterior from ever fully committing to a confident-but-wrong
  location; the override strength is data-dependent, not constant. ε must be fixed
  (and not in the token) for this to be robust-Bayes rather than an averaged
  hyperprior. The plot-only `mixture_logprior_on_grid` stays local to `bo1d.py`.
  The diagnostic makes this visible with three columns: uniform /
  correct-informative / **wrong-informative**.
- **Scaling is the real work.** `y_opt` and data `y` are the same physical quantity
  (function values), so they share one affine, written explicitly as
  `scale_y(y) = encode_value(y_opt_var, y)` over a **frozen** module constant
  `Y_RANGE` (frozen for checkpoint compatibility; not a CLI arg). `Y_RANGE` covers
  the full native y range and is both the `y_opt` bounds and the data-`y` scaling,
  so the model sees both on one ruler and `y_opt ≤ all y` is legible. Corrected
  budget: away from `x_opt`, `|g_c − d| ≈ |g| + |d|`, so the natural depth `|d|`
  inflates the *whole* function height (not just the dip) -- hence the `|d|` cap and
  a tamed `σ_f`. All scale numbers are provisional; the first implementation step
  is a histogram check that token values sit ~`[-1, 1]` and that the drawn-`x_opt`
  marginal matches the contaminated prior. Stochastic tails outside `[-1, 1]` are
  accepted (soft convention).
- **Observation noise + reveal.** Data `y` carry small Gaussian noise
  (`--sigma-obs`), matching the continuous MDN and BO realism. Reveals use the
  gaussian/sir pattern (replace a finite-spread PRIOR token with a zero-spread
  known one, token stays active), required so `conditional_log_density` finds an
  active PRIOR slot.
- **No BO loop.** No iterative acquisition rollout; the conditional
  `p(x_opt | y_opt = v, D)` is shown (via `conditional_log_density`, no
  `sample_ar`) to gesture at Thompson sampling, nothing more.
- **Bigger default network than the other examples.** BO is the hardest task
  (instance-level latents, multimodal `p(x_opt | D)`), and the paper's 1D-BO model
  is correspondingly larger (`d_emb=256`, 6 layers, 16 heads, `K=20`). `bo1d`
  defaults to `d_model=192`, **6 transformer blocks**, 16 heads, mlp 384, `K=12`
  (~3.9M params, vs the ~1.2M `gp1d`/Gaussian defaults). Six blocks were adopted
  deliberately (the 4-block default underfit the prior-integration); `K` is raised
  from 8 to 12 for the multimodal location posterior. Still well short of the
  paper's `5e5`-step budget -- CPU validation runs are necessarily undertrained;
  real training is a GPU run. Observed at equal small CPU budgets the ~1.2M / 4-block
  model used the runtime prior *more* than the ~3.9M / 6-block one (the bigger model
  is more undertrained at equal steps), which confirms the bottleneck is training
  budget, not capacity. The committed CPU artifact is therefore a smoke test, not a
  quality result; the `ε=0.05`-vs-`0.10` choice is likewise deferred to a proper
  GPU run. Defaults kept faithful (6 blocks, `ε=0.05`) for that run.
- **Scope note.** This is example #4; "nano ships exactly two" (initial design) is
  already stretched by SIR. BO earns its place: it adds instance-level latents, the
  optimum-posterior headline, and the robust prior-injection mechanism, none of
  which the other examples cover. Recorded as a deliberate decision.

---

## 2026-06-07 — Web playground (in-browser TS port)

- **A non-core interactive demo lives in `playground/`.** It is an *example*, not
  part of nanoACE: the core stays torch-only and legible, while `playground/`
  carries a Vite + TypeScript toolchain. It reimplements `ace.py`'s forward pass
  in TS so trained models run fully client-side (GitHub Pages, no server). Two
  demos: GP-1D (add/drag points, infer the kernel, **pin a latent and predict**)
  and Gaussian (Beta-prior sliders + observed `y`, with the analytic oracle
  overlaid). The headline is that amortized conditioning is instant — a forward
  pass per interaction — which is exactly what an interactive demo makes visible.
- **The port is a frozen snapshot kept honest by a parity test.** `export_weights.py`
  derives every constant from a live `ACE` instance (no hand re-encoding of the
  schema) and writes plain float16 arrays + a JSON manifest — readable weights,
  not a mystery binary, regenerable from a checkpoint. Weights are float16 to halve
  the blobs (~3.6 MB total); parameters are rounded with `.half().float()` in
  *both* the exporter and `parity.py`, so the shipped weights and the references
  reflect identical values (only float32-vs-float64 arithmetic differs). `parity.py` dumps the real
  model's embeddings, per-layer states, raw head outputs, and derived quantities
  on deterministic cases covering every token path; `npm test` asserts the TS
  forward reproduces them (and that each demo's orchestration matches `gp1d.py` /
  `gaussian_toy.py`). PyTorch-float32 vs JS-float64 means the gate is a combined
  relative+absolute tolerance, not bit-parity. If `ace.py`'s forward changes, the
  parity test fails loudly and localizes the drift; re-port and regenerate.
- **Gotcha — fixtures + blob can go collectively stale vs the checkpoint.** Both
  `export_weights.py` and `parity.py` derive from the checkpoint loaded at run time,
  and the parity test compares the TS forward only against those fixtures — *not*
  against the current checkpoint. So if the checkpoint is retrained but the blob and
  fixtures aren't regenerated together, the demo silently loads the old model while
  `npm test` stays green. (This happened 2026-06-07: a GP retrain at 12:52 — after a
  12:49 export — left the demo predicting a flat mean from a stale blob; the fix was
  to re-run export + parity.) **Always re-run `export_weights.py` and `parity.py`
  together after retraining**; a Python sanity assert on the freshly exported model
  (e.g. it tracks a smooth function) would catch this class of bug.
- **Weight hosting is an open decision; blobs are not committed (parked).** The
  exported fp16 blobs (`playground/public/models/`) are gitignored for now.
  Options: commit them (~3.6 MB today, but binary churn grows with retrains and
  more examples), Git LFS (keeps `.git` lean, still same-origin), or runtime fetch
  from an external host such as HF (adds a CORS + browser-caching dependency and
  risks the fetched weights drifting out of sync with the parity-pinned code, the
  one integrity property this design leans on). Regenerate locally via
  `export_weights.py` meanwhile. The Pages deploy is blocked until this resolves.
- **Multi-latent reveal (DONE 2026-06-07) — playground multi-pin is now in-distribution.**
  Previously both samplers revealed *at most one* latent as context per example, so
  pinning two or three latents in the playground was out-of-distribution. The DGP
  below was implemented and both examples retrained (Gaussian 30k, GP 100k), so
  multi-pin is now a real conditional and the ≥2-pin OOD banner has been removed.
  - **Why:** the playground's headline interaction is "pin latents and predict";
    multi-pin should be a real conditional, not OOD.
  - **Resolved DGP — the reveal/conditioning distribution (decided 2026-06-07).**
    The task generators are otherwise unchanged (latent priors/ranges, kernels,
    function sampling); only how each task chooses revealed-vs-queried latents changes.
    - **Reveal mask (shared helper, both examples).** Per task: with prob `q` reveal
      *nothing*; otherwise reveal a **uniform random non-empty subset** of the latents.
      Implement once, e.g. `sample_reveal_mask(n_latents, batch, q, device) -> bool[B, L]`:
      `reveal_any ~ Bernoulli(1-q)`; if revealing, sample one integer bitmask uniformly
      from `1..2^L-1` and decode it to latent booleans, else all-False. Default
      `q ≈ 0.5`, tunable per example — keeps the headline 0-reveal case at ~half the
      mass while the rest covers every multi-pin combination.
    - **Mental model.** Every latent's prior token has a spread running from informative
      (Beta) → zero (exact). The reveal mask picks *which latents have zero spread (exact)*.
    - **GP (`gp1d.sample_gp_batch`) — exact-only, stays finite-prior-free.** Drop
      `reveal_which` (the `randint(0,3)` single pick); use the shared mask. A *revealed*
      latent enters context as a zero-spread token (continuous → zero-spread `PRIOR`,
      `kernel` → `VALUE` class label) and is dropped from the target; a *non-revealed*
      latent has **no** prior token and is queried. The kernel may be revealed together
      with continuous latents.
    - **Gaussian (`gaussian_toy.sample_toy_batch`) — keep the always-present Beta priors.**
      Both latents always carry a runtime Beta prior token (the latent is drawn from it),
      exactly as today — that is the ACEP demo and is already in-distribution. Replace
      `reveal_mu` xor `reveal_logsig` with the shared mask: a *revealed* latent collapses
      its Beta slot to a zero-spread exact token and leaves the target; a *non-revealed*
      latent keeps its finite Beta prior and is queried. (The current Beta-slider demo
      needs no retrain; this mainly makes exact multi-"pin" in-distribution and unifies
      the two samplers under one helper.)
  - **Follow-up completed after retraining:** re-ran `playground/export_weights.py`
    for the affected task(s), regenerated fixtures with `playground/parity.py`
    (they pin the *current* checkpoint — see the fixtures+blob staleness gotcha
    above), and removed the ≥2-pin OOD trigger in the playground.
  - **Status (2026-06-07): DONE.** `sample_reveal_mask` in `ace.py`, wired into
    `sample_gp_batch` / `sample_toy_batch`, default `--latent-context-prob` (P(reveal
    any)) = 0.5. Both examples retrained under the new DGP (Gaussian 30k, GP 100k),
    blobs re-exported and fixtures regenerated, and the playground ≥2-pin OOD trigger
    removed (`PIN_OOD_MIN` deleted from `config.ts`; pin branch dropped from
    `oodReasons`). Diagnostics still track the oracle (Gaussian μ/log σ ≈ oracle; GP
    predictive RMSE 0.36 vs oracle 0.345). Note GP is a bit *overconfident* on kernel
    identity (Periodic 0.81 vs oracle 0.50) — acceptable for the demo, worth a glance
    if kernel calibration matters later. GP pins-only contexts with no observed data
    remain flagged OOD in the playground because training used at least four data
    context points.

---

## 2026-06-07 — SIR simulation-based-inference example + `ace_prior.py`

- **Third example: `sbi_sir.py`.** This is the SBI task from the paper's third
  application area: infer the contact rate `beta` and recovery rate `gamma` of an
  epidemic from a noisily observed infected fraction over time. It is deliberately a
  *fusion* of the two existing examples, not new architecture — runtime Beta prior
  injection (ACEP) from `gaussian_toy.py` plus online time-series simulation, 1D
  time-indexed data tokens, and a grid oracle from `gp1d.py`. The only genuinely new
  machinery is a small batched RK4 integrator.
- **Simulator: deterministic SIR ODE + Gaussian observation noise.** Chosen over the
  paper's Binomial/Poisson counts because the deterministic-trajectory-given-`(beta,
  gamma)` property makes the marginal likelihood a product of Gaussian observation
  densities, so the `(beta, gamma)` grid posterior is tractable — the same recipe as
  `gp_oracle`. ACE only ever sees simulator draws, never the likelihood, so it is
  honestly simulation-based; the grid oracle is the reference. Gaussian observation
  noise also matches the continuous MDN head exactly (no continuous-vs-discrete
  predictive caveat).
- **Continuous-only latents.** `beta`, `gamma` are bounded continuous latents with
  identity transform, so the uniform generative prior is exactly `Beta(1, 1)`, matching
  `gaussian_toy.py`. No discrete latent — `gp1d.py` already exercises that path, and a
  discrete latent here would be redundant and complicate the oracle.
- **Data scaling is explicit.** The infected fraction is small (~0–0.4) while the
  embedders assume values near `[-1, 1]`, so the example applies a fixed affine
  `scale_value`/`unscale_value` (and `scale_time`) at the token boundary. The oracle
  works entirely in native fraction space; ACE predictive moments are decoded before
  comparison. `"y"` is a data variable, so the core `encode_value`/`decode_value` (which
  only touch bounded latents) do not apply — the affine is local to this example. This is
  the "scaling happens at generation" convention made explicit for a small-valued series.
- **Diagnostic: uniform-vs-informative prior contrast.** The headline SBI feature is
  prior conditioning, so the fixed case is evaluated twice on the *same* observation —
  once with `Beta(1, 1)` and once with an informative Beta — and both posteriors are
  plotted. To make the prior matter, the fixed context is sparse rise-phase data: early
  epidemic observations identify the growth rate but leave a broad, near-degenerate
  `beta`/`gamma` ridge (oracle corr ≈ 0.99), so the informative prior visibly tightens
  the marginals (β std ≈ 0.080 → 0.050, γ ≈ 0.057 → 0.035) and pulls them toward truth.
  With many clean observations the likelihood dominates and the contrast vanishes; that
  is the correct behavior, just not an interesting demo.
- **Shared prior helpers extracted to `ace_prior.py`.** The Beta prior-token helpers
  (`sample_prior_params`, `beta_alpha_beta`, `prior_features`, `known_latent_features`,
  `draw_from_beta`, `beta_logprior_on_grid`) were a pure move out of `gaussian_toy.py`;
  both prior-conditioning examples now import them. The `(mean, spread)` prior token is a
  core ACEP representation, so a shared module is the right home. Audit: the model-side
  prior code (`PRIOR_FEATURES`, `spread_embed`, the `_embed` payload) and the general
  `encode_value`/`decode_value` coordinate maps stay in `ace.py`; `ace_prior.py` imports
  `PRIOR_FEATURES` from the core and asserts the two-feature layout.
- **Deferred.** No discrete-latent runtime prior token (still deferred from the prior
  redesign). The SIR AR two-latent joint heatmap is computed-capable but not plotted, to
  keep the uniform-vs-informative contrast legible; the marginals carry the story.
- **TODO — migrate SIR to the shared multi-reveal DGP (`sample_reveal_mask`).** `sbi_sir.py`
  predates the multi-latent reveal work (it landed on a branch that split before it), so its
  sampler still uses the old single-reveal logic: with prob `latent_context_prob` it reveals
  *exactly one* of `beta`/`gamma` (`reveal_beta` xor `reveal_gamma`), never both. It runs fine
  as-is and the current diagnostic doesn't pin latents, so nothing is broken. But it means a
  two-pin SIR context (both `beta` and `gamma` revealed) is out-of-distribution, unlike Gaussian
  and GP-1D. Follow-up for consistency: replace the xor with `sample_reveal_mask(2, batch, q=1
  - latent_context_prob, device)` (the same swap done in `sample_toy_batch` / `sample_gp_batch`),
  then retrain the SIR checkpoint so exact multi-pin conditioning is in-distribution and all
  three examples share one reveal helper. Not required until SIR grows an interactive/pinning
  demo.

---

## 2026-06-07 — Bounded latent coordinates and Beta information tokens

- **Bounded continuous latents are tokenized to `[-1, 1]`.** `Variable.bounds`
  is now the hard support in the variable's semantic/transformed coordinate
  (`log_sigma` bounds are on `log_sigma`, not `sigma`). Data values remain in
  task coordinates. The helpers `encode_value` / `decode_value` and
  `encode_token_values` / `decode_token_values` centralize the affine map so
  examples do not carry hand-rolled conversions.
- **Histogram prior grids are removed.** `ACEConfig.prior_bins`,
  `Variable.prior_range`, `Variable.prior_bins`, and the histogram
  `prior_embed` MLP are gone. `Tokens.prior` is now a fixed two-feature tensor:
  `(mean_internal, spread_internal)` for bounded continuous latent information.
- **Known continuous latents are zero-spread information tokens.** A finite
  spread token represents a runtime prior; spread zero represents an exact
  known latent. The embedder uses
  `value_embed(mean_internal) + spread * spread_embed(mean, spread)`, so the
  zero-spread payload is exactly the same location payload used for a scalar
  value before mode/variable embeddings are added. Exact continuous latent
  observations use `PRIOR` mode, not `VALUE` mode.
- **Native-coordinate prediction is explicit.** Raw `Predictions.log_prob`,
  `.mean`, `.sample`, and `.continuous_var` remain in token coordinates because
  training and AR context construction live there. `log_prob_native`,
  `mean_native`, `continuous_var_native`, and `sample_native` decode bounded
  continuous latents and apply the affine density Jacobian when needed.
- **Gaussian is now the ACEP demo.** `gaussian_toy.py` always emits one
  information token per continuous latent. Training samples runtime Beta priors,
  draws the true latent from those priors, and compares against a Beta-aware
  analytic grid posterior. Revealed continuous latents replace the finite-spread
  prior slot with a zero-spread exact slot.
- **Gaussian diagnostic priors are visible.** The fixed Gaussian diagnostic uses
  moderately informative runtime priors `EVAL_MU_PRIOR = (mu_unit=0.70,
  nu=20.0)` and `EVAL_LOGSIG_PRIOR = (mu_unit=0.70, nu=8.0)`. The `mu` prior is
  shifted slightly left from the first implementation while keeping a simple
  concentration; the `log_sigma` prior is deliberately broader.
  The plot overlays these prior curves on the corresponding posterior marginal
  panels, so the runtime prior information is visible next to the oracle and ACE
  posterior curves.
- **GP-1D stays finite-prior-free.** `gp1d.py` uses encoded bounded continuous
  latent targets and zero-spread `PRIOR` tokens when lengthscale/outputscale are
  revealed. The discrete `kernel` latent remains a VALUE-mode class label when
  revealed, and no categorical prior token is implemented.
- **Deferred.** The MDN still predicts unconstrained Gaussian-mixture mass in
  token space, so exact bounded output distributions are not implemented.
  Discrete-latent runtime prior tokens are also deferred. Existing checkpoints
  from the histogram schema are stale and should be regenerated.

---

## 2026-06-06 — Prior representation: candidate redesign (resolved 2026-06-07)

This section records the reasoning that led to the 2026-06-07 implementation.
The shipped code no longer uses the histogram→MLP prior encoder. The old design
spent a 64-dim, discretization-noisy vector to encode what is almost always two
numbers, and it dragged in scale coupling — `Variable.prior_range`,
per-variable `prior_bins`, and the global-`prior_bins` validation in
`ACE.__init__` existed only to pin the grid. It is also what made the Gaussian
toy harder, which is why runtime priors were initially removed from that
example.

- **Preferred candidate: a rescaled Beta prior token.** Encode a continuous latent's prior
  as a Beta on the latent's bounded range, user-facing `(mean, SD)`. This fits the toys
  specifically because their latents are bounded-uniform by construction (`MU_RANGE`,
  `LOGSIG_RANGE`, `LOG_LENGTHSCALE_RANGE`, `LOG_OUTPUTSCALE_RANGE`), so a bounded prior
  matches the true support and the no-information case is exactly `Beta(1,1) = uniform`
  (today's generative prior). Two numbers buy uniform, Gaussian-like bumps, skew (`α≠β`),
  and edge-peaked/U-shaped priors (`α,β<1`).
- **Parameterization: user `(mean, SD)`, internal `(μ, ν)`, location embedded like a value.**
  User interface `(mean, SD)` for interpretability; sample/store internally as `(μ, ν)` with
  concentration `ν = α + β` — constraint-free, since every `μ∈(0,1), ν>0` is a valid Beta.
  For the embedding, feed the prior **mean directly** (rescaled to the latent's range / the
  `[-1,1]` convention), ideally through the existing `value_embed` so a prior's location
  enters exactly like an observed value, and carry the **concentration** as a separate
  channel. `log ν` is the natural concentration coordinate (a positive multiplicative scale
  spanning ≈2 to 1000); **`logit μ` is not** worth it — `μ` is a bounded location with no
  positivity or scale argument, and `d/dμ logit(μ) = 1/(μ(1−μ))` over-resolves the
  degenerate endpoints while compressing the mid-range where informative priors usually sit.
  (For the value-limit variant below, carry concentration as the `σ`-gated term instead of
  raw `log ν`.) Recover shape params when needed: with variance `v`, `ν = μ(1−μ)/v − 1`,
  `α = μν`, `β = (1−μ)ν`, valid iff `σ² < μ(1−μ)` (the Bernoulli-SD cap); clamp/validate the
  user's `(μ, σ)` against that bound.
- **Coordinate convention (avoid space confusion).** Three spaces: the user specifies
  `(mean, SD)` in the latent's **original** range `[a,b]`; the **Beta math** (recover
  `α,β`/`ν`, check validity `SD² < μ(1−μ)`) lives in **unit `[0,1]`**; the **embedding** uses
  the `[-1,1]` convention. Maps are affine (width `w = b−a`): to unit, `μ_u = (m−a)/w`,
  `SD_u = SD/w`; to `[-1,1]`, `m_± = 2μ_u − 1`, `SD_± = 2·SD_u` (mean through the affine map,
  SD times the slope). Crucially **`ν` is coordinate-free** — convert `SD ↔ ν` *once* in unit
  space, then carry `(location, ν)` and only ever transform the *location*; `ν` never changes
  (equivalently `SD_±² = (1−m_±²)/(ν+1)`). The prior mean in `[-1,1]` lands in the same
  coordinate as a VALUE token for that latent, which is what makes the `value_embed` reuse
  coherent. Worked example: `[a,b]=[-2,8]`, user `mean=2, SD=2` → unit `μ=0.4, SD=0.2` (not
  `0.5/0.25`) → `Beta(2,3)`, `ν=5` → `[-1,1]` location `−0.2`, SD `0.4`.
- **Known-value as a limit, done honestly.** An earlier claim that a prior token "collapses
  to a VALUE token" as spread→0 is *not* enforced by the current representation: the spread
  MLP has no zero limit, and the additive `mode_embed(PRIOR)` vs `mode_embed(VALUE)` is a
  constant offset that never closes. To make it real you must (a) gate the spread so it
  vanishes by construction — e.g. payload `value_embed(mean) + σ·h(mean, σ)`, feeding `σ`
  not `log σ` so the input stays bounded — and (b) treat a known latent as the zero-spread /
  `ν→∞` prior token rather than a separately-moded VALUE token, so "value" *is* the boundary
  of "prior." With Beta this falls out naturally: `ν→∞` is a spike at `μ`. Even then this is
  representation continuity only; the model behaves like exact conditioning in the limit only
  if training visits small spreads.
- **Discrete latents stay separate.** Beta covers continuous bounded latents. The discrete
  `kernel` latent's runtime prior is a categorical pmf — a small "K-probabilities" prior
  token, not a Beta.
- **Training implication.** Sample a Beta hyperprior `(α, β)`, draw the true latent from it,
  build the prior token, NLL as usual. This is much simpler than the paper's
  MoG/geometric-K/Dirichlet prior-generating machinery (already slated for slimming) and is
  the actual distribution the model would amortize over.
- **Default hyperprior over `(μ, ν)`.** The prior-generating mixture, defined in unit
  `(μ, ν)` space so it is range-agnostic and reused across all continuous latents:
  - `1/3`: `μ = 0.5, ν = 2` → exactly `Beta(1,1)` (uniform; the uninformative case).
  - `1/6`: `μ ~ U(0,1), ν ~ U(0.1, 2)` → broad / skewed / U-shaped (`ν<1`).
  - `1/2`: `μ ~ U(0,1), log ν ~ U(log 2, log 1000)` → concentrated, up to near-spike.

  Clamp `μ` to `(ε, 1−ε)` so `α,β > 0`. Weights and bounds are tunable. The heavy `1/2`
  mass on concentrated priors (with `ν` to 1000) is deliberate — it is where prior
  conditioning matters most and where the `ν→∞ ≈ known-value` limit is learned. The `1/6`
  U-shaped band is the most exotic slice and the first dial to turn if it hurts (raise
  `ν_min`, e.g. 0.1→0.5, or lower the weight). Note `log ν ~ U(2, 1e3)` taken literally
  overflows (`e^1000`); the intended range is `log ν ~ U(log 2, log 1000)`.
- **ACEP always emits prior tokens; "no prior" = uniform.** The architecture does not (and
  need not) enforce "absent prior token ≡ uniform prior," so do not train over a
  present/absent mixture. The prior-conditioning model (ACEP) always carries one prior token
  per continuous latent, using `Beta(1,1)` when there is no information. This is benign here
  because the latents are bounded-uniform by construction, so the uniform token is the honest
  "no extra info" prior rather than the hard amortization burden that arbitrary histograms
  were. Prior-free ACE and always-prior ACEP are then simply two variants with fixed token
  layouts (supersedes the earlier idea of a separate `prior_prob` present/absent knob).
- **Fallbacks if more expressiveness is wanted.** Plain moments `(mean, log_std)` if even
  Beta is more than needed (but it cannot express bounded support or skew, and a Gaussian
  encoder leaks past the latent range); a small fixed quantile vector (e.g. 5/25/50/75/95th
  percentiles) for asymmetric unimodal priors; a handful of orthogonal-basis coefficients
  (Hermite/Legendre) only if arbitrary/multimodal priors become an explicit demo. All three
  still beat a 64-bin histogram on compactness and trainability.
- **What adopting any of these deletes.** `Tokens.prior` shrinks from `[B, T, prior_bins]`
  to a 2–3 vector; `Variable.prior_range`, per-variable `Variable.prior_bins`, and the
  `prior_bins != cfg.prior_bins` check all go away (a fixed small vector is never ragged);
  "no prior" becomes "do not emit a PRIOR token" instead of an encoded uniform histogram.

---

## 2026-06-06 — GP-1D executable example

- **Standalone GP file.** `gp1d.py` owns the GP regression example end to end:
  online function sampling, command-line training/evaluation, checkpoint helpers, and a
  fixed diagnostic plot. It follows the same local-file pattern as `gaussian_toy.py`
  rather than introducing a generic training framework.
- **Latents.** The example uses two continuous latents, `log_lengthscale` and
  `log_outputscale`, plus one discrete `kernel` latent with four classes: RBF,
  Matern-1/2, Matern-3/2, and periodic. This exercises the shared continuous MDN head,
  the shared `Kmax` categorical head, and discrete value embeddings when a kernel latent
  is occasionally revealed as context.
- **Sampler.** GP functions are generated online. Kernel matrices and Cholesky sampling
  run on CPU float64, then sampled tensors move to the selected ACE device. There is no
  sharded data layer yet; add one only when a real GP training run needs saved pools.
  Function values are not clipped: with `log_outputscale` up to `log(1.0)`, sampled `y`
  values can naturally leave the loose `[-1, 1]` convention used by the embedders. Treat
  that as a calibration caveat, not a bug. The default Cholesky jitter is `1e-5`; increase
  it if clustered x-locations ever make Matern-1/2 or periodic kernel matrices unstable.
- **Diagnostic.** GP-1D now computes a numerical grid oracle for the fixed context.
  For each kernel and each `log_lengthscale`/`log_outputscale` grid point, it evaluates
  the GP marginal likelihood of the observed context, applies uniform-prior quadrature
  weights, and normalizes over the full kernel-by-hyperparameter grid. The diagnostic
  reports per-kernel integrated marginal likelihood deltas, the kernel posterior,
  continuous latent marginals, and the posterior predictive moments formed by mixing
  conditional GP predictives over the grid. This is not a closed-form analytic posterior;
  it is an oracle up to grid resolution, the bounded hyperparameter ranges, and the
  Cholesky jitter used by the sampler. The fixed context locations are irregular and
  clustered, including nearby pairs/triples, because sparse evenly spaced points do not
  say much about local roughness and make kernel/lengthscale inference mostly guesswork.
- **Oracle grid sanity check.** The fixed GP diagnostic was rerun from the retained
  checkpoint with `--oracle-bins 32`, `64`, and `96`. Kernel posterior probabilities,
  latent posterior moments, and predictive RMSE were stable to the printed precision,
  so the default 64-bin oracle is adequate for this diagnostic. The RBF integrated
  marginal likelihood delta moves by about 0.1 log units across those grids, but RBF has
  posterior mass near 0.002 in this case, so that movement is not practically relevant.
  This is a one-case numerical check, not a benchmark harness.
- **Current retained GP artifact.** The retained `artifacts/gp1d.pt` / `artifacts/gp1d.png`
  pair is a post-Beta-token-schema 100k-step online run. The fixed diagnostic uses a periodic generating
  kernel. The numerical oracle still leaves real uncertainty: posterior mass is roughly
  RBF 0.002, Matern-1/2 0.377, Matern-3/2 0.121, periodic 0.500. The checkpoint gives
  roughly RBF 0.007, Matern-1/2 0.152, Matern-3/2 0.083, periodic 0.759. The
  oracle-vs-ACE kernel KL is about 0.18; oracle predictive RMSE is about 0.35 and ACE
  predictive RMSE is about 0.39. Treat the plot as an oracle-calibrated ambiguous
  posterior, not as a hard truth-recovery example.

---

## 2026-06-06 — Gaussian toy implementation

- **Core implementation.** `ace.py` now contains the current source-of-truth
  implementation: `Variable`, `Tokens`, `Batch`, separated context/target transformer
  blocks, shared continuous MDN head, shared masked `Kmax` categorical head, prediction
  object, target NLL loss, and `sample_ar`.
- **Executable Gaussian toy file.** `gaussian_toy.py` owns the Gaussian example end to end:
  online batch generation, command-line training/evaluation, oracle math, checkpoint
  helpers, and plotting. There is no separate generic command-line wrapper because that
  wrapper would still depend on the Gaussian file to be understood. Reusable grid query
  helpers live in `diagnostics.py`.
- **Gaussian diagnostic.** `gaussian_toy.py` trains the Gaussian toy online under fixed latent priors,
  compares ACE marginals to an analytic grid posterior, and includes an autoregressive
  two-latent joint diagnostic. The Gaussian example no longer injects random runtime prior
  tokens: amortizing over arbitrary prior histograms made the small Gaussian example much
  harder than the posterior-learning behavior this example is meant to inspect. Runtime
  prior tokens remain in the core ACE representation for examples that explicitly train prior
  conditioning. The AR diagnostic averages the two
  factorizations `p(mu)p(log_sigma|mu)` and `p(log_sigma)p(mu|log_sigma)` in probability
  space so the displayed joint posterior is not tied to one arbitrary order. The
  deterministic evaluation batch is defined in `gaussian_toy.py`: three observed `y` values,
  plus latent truth for printed diagnostics. Keeping that batch constant makes plots
  comparable across runs without a separate evaluation-case selector.
  The plot includes a posterior predictive panel for a new `y`: the analytic curve is the
  posterior mixture `sum p(mu, log_sigma | D) Normal(y | mu, sigma)`, not a Gaussian
  plug-in approximation.
  Training now mixes in latent-value context cases (`--latent-context-prob`, default
  `0.25`): one latent is sometimes revealed as a `VALUE` token and the other remains a
  target. Without that, AR conditional queries are syntactically valid but
  out-of-distribution for the toy model.
- **Artifacts are optional and regenerable.** `gaussian_toy.py` can save a diagnostic plot
  and a lightweight checkpoint under `artifacts/`, but these are convenience outputs
  rather than load-bearing repository assets. `--eval-only --load-checkpoint` verifies a
  saved checkpoint.
- **Environment.** The project currently pins the PyTorch CUDA wheel tested on this
  workstation: `torch==2.11.0+cu128`, PyTorch CUDA runtime 12.8, and the RTX 4060 Laptop
  GPU.

---

## 2026-06-06 — Initial design

### Vision

- **What it is.** A single-purpose, legible implementation of ACE that a coding agent
  can read end-to-end and extend. The optimization target is the *source* — legibility
  and extensibility — not a packaged runtime artifact (pretrained weights, a stable
  calling contract, an MCP tool). That packaging is outside the current scope, not an
  anti-goal: a clean `condition().predict()` surface is part of legible source anyway,
  so doing this well keeps that path open rather than foreclosing it.
- **What "nano" means.** Minimal *concepts* and maximal *locality* — not minimal lines,
  and not a trivial toy. nanoACE should be legible **and real** (you can train real
  models with it), exactly as nanoGPT is small/readable yet genuinely trains GPT-2.
  The bar a feature must clear is conceptual surface, not compute.

### Guiding principles

- **Paper-faithful starting point, not paper-bound math.** Start from the paper where it
  preserves ACE's core idea, but treat the equations as an existence proof rather than a
  constraint. The invariants are the conditioning semantics: variables as tokens, data /
  latent / prior information in one context set, target tokens that request predictive
  distributions, and a type-agnostic `dist.log_prob` training path. Attention details,
  output-head parameterization, prior features, latent transforms, and training mixtures
  can change when a tweak is simpler, more robust, or makes the conditioning interface
  clearer.
- **Tiebreaker when simplicity vs. performance conflict.** Simplicity is the default. A
  feature *can* buy its way in if it improves predictions or real-training throughput —
  but it pays rent in legibility (inline shape contracts, an obvious why). Large
  experiment-management complexity (clusters, multi-seed pools) is outside this
  repository's scale target.
- **"naïve" is not a synonym for "legible."** Evaluate case by case (see Attention).

### Model / architecture

- **Discrete in the core from day one.** A `Variable` schema (type, cardinality; ~8
  lines) travels with each token. Discrete value embedding uses an embedding matrix
  (`E_val`); discrete output uses one shared categorical head with `Kmax =
  max(cardinality)`. At distribution construction, mask logits beyond the target
  variable's cardinality and read only local labels `0..k-1`. Rationale: the paper's
  headline is "unifies discrete + continuous," and this keeps the code path shared rather
  than introducing per-variable heads.
- **Heads return distribution objects** (`.log_prob` / `.sample`). The training loss is
  then `-dist.log_prob(z)` for *everything*, so the loop is type-agnostic — the GMM vs.
  categorical distinction never reaches `train.py`.
- **Embedder (paper-shaped default).**
  - data:         `f_x(x) + f_val(y) + e_data`
  - latent:       `f_val(θ_l) + e_l`
  - prior-latent: `f_prob(p_l) + e_l`
  - target value branch replaced by the learned unknown embedding `e_?`.
  - discrete `f_val` → embedding matrix `E_val`.
- **Attention: separated context self-attn + target→context cross-attn. No mask.**
  This *is* the paper's choice (§3.2, `O(N²+NM)`), and it's both cheaper and more
  legible than the alternative. The "naïve" single masked stream materializes the full
  `(N+M)²` matrix and masks ~half away, *and* buries the key invariant ("targets read
  context, never each other") inside a fiddly block mask. The separated version needs no
  mask — the conditioning structure falls out of which sets you pass in:
  ```
  ctx = ctx + self_attn(q=ctx, kv=ctx)    # N×N
  tgt = tgt + cross_attn(q=tgt, kv=ctx)   # M×N
  ```
- **Block: standard pre-LN default.** Use the stable transformer form
  `x = x + attn(LN(x))`; `x = x + mlp(LN(x))`. No scalar-channel plumbing. If training
  stability argues for a different normalization placement, change the block rather than
  preserving paper/formula fidelity for its own sake.
- **Single MDN head, allowed to be slightly stronger.** Default to a shared
  `MLP(D, hidden) → Linear(3K)` → (raw weight, mean, log-std), with
  softmax/identity/softplus. *Not* K separate per-component MLPs + a global raw bias
  (that's more apparatus than the idea needs). A bare `Linear(D, 3K)` is acceptable only
  if the toy posterior still behaves.
- **Prior: keep the runtime-conditioning idea, not the exact representation.** The
  histogram→MLP encoder (`f_prob`, N_bins grid) is the first implementation because it is
  direct and visible. If moments, CDF features, learned basis coefficients, or another
  compact prior summary is simpler or trains better, use that. The elaborate prior
  *generative* process (80% MoG / geometric-K / Dirichlet / three mean-std configs) is
  slimmed to a simple sampler; it demonstrates injection just as well.
- **Latent loss weight: a single scalar `latent_weight`** — not the
  `((n_total − ½(max_ctx+min_ctx))/n_latent)^T` formula + `T` grid search (a
  benchmark-tuning artifact).
- **Cheap wins taken by default:** log-space latent prediction; multi-task loss balancing.

### Interface / token representation

- **Canonical internal object: two token sets.** The model receives a small `Batch` with
  `context: Tokens` and `target: Tokens`. Context tokens are known values or latent priors;
  target tokens are queries whose true values may be present for loss. Use padding masks
  for ragged batches, but no directional attention mask: conditioning direction comes from
  `ctx = self_attn(ctx)` and `tgt = cross_attn(tgt, ctx)`.
- **Shape contract.** The first code pass should implement this directly:
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
  Target truth sits in `value` / `value_index` for loss, but the embedder ignores it
  when `mode == QUERY`.
- **`Tokens` is padded, explicit, and boring.** Fields:
  - `var_id: LongTensor[B, T]` indexes the `Variable` schema.
  - `x: FloatTensor[B, T, x_dim]` stores data covariates; latent tokens use zeros.
  - `value: FloatTensor[B, T]` stores transformed continuous values or dummy zeros.
  - `value_index: LongTensor[B, T]` stores discrete labels or dummy zeros.
  - `prior: FloatTensor[B, T, n_bins]` stores latent-prior histograms, zeros otherwise.
  - `mode: LongTensor[B, T]` is `VALUE`, `PRIOR`, or `QUERY`.
  - `mask: BoolTensor[B, T]` marks active padded tokens.
  Target truth lives in `value` / `value_index`; the embedder ignores it when
  `mode == QUERY`, while the loss uses it where `mask` is true.
- **`Variable` owns semantics.** Each variable has `name`, `kind` (`data` / `latent`),
  value type (`continuous` / `discrete`), optional discrete `cardinality`, transform
  (`identity`, `log`, maybe `logit`), and optional prior-grid range/bin count. Prior-grid
  ownership lives on latent variables, not in a global config.
- **Heads are hidden behind a prediction object.** `ACE.forward(batch)` returns an object
  with `.log_prob(batch.target)`, `.sample(batch.target, ...)`, and `.mean(batch.target)`.
  Continuous targets use the shared MDN path. Discrete targets use the shared
  `Kmax`-logit categorical path, masking invalid logits by `cardinality[var_id]`.
  The training loop should not branch on variable type.
- **Autoregression is a helper, not architecture.** Joint samples come from
  `sample_ar(model, batch, order=None)`, which repeatedly predicts one target,
  samples it, appends it to context as a `VALUE` token, and continues. The base model stays
  a diagonal prediction map. With `order=None`, the helper randomizes target order by
  default; pass an explicit order for deterministic probes.
- **Continuous head is shared.** Use one shared MDN head for all continuous variables;
  `var_id` / variable embeddings tell the model which scalar is being predicted. Do not
  add per-variable continuous heads unless a concrete task shows shared capacity is the
  bottleneck.
- **Discrete value embeddings use local labels plus offsets.** Each discrete variable's
  labels are local integers `0..k-1`. For context value embeddings, use one global
  embedding table with per-variable offsets so labels from different variables do not
  collide. For target predictions, use the shared `Kmax` head described above.
- **One `n_bins` initially.** Latent variables own their prior ranges, but the first
  implementation uses a global prior-bin count to avoid ragged prior tensors and nuisance
  padding. Different bin counts can wait until a real example needs them.
- **Inference targets may omit truth.** It is valid to pass target tokens with dummy
  `value` / `value_index` for prediction-only calls. `.log_prob(target)` is only called
  when target truth is present; example/train code are responsible for that distinction.
- **Loss weighting stays minimal.** Start with `data_weight` and `latent_weight` scalars,
  then average per active target token. No benchmark-derived formulas or per-task grids.
- **Device movement lives on the data objects.** Add `Tokens.to(device)` and
  `Batch.to(device)` immediately so training scripts do not accumulate tensor
  plumbing.

### Tighten before coding

- **Implement in dependency order.** Build `ace.py` and the Gaussian toy path first, prove
  the interface with `gaussian_toy.py`, then add GP-1D save/load shards. Do not build a generic
  sharded data layer before the model and toy diagnostic work.
- **Keep `Variable` boring.** The schema should be explicit and local: `name`, `kind`
  (`data` / `latent`), value type (`continuous` / `discrete`), cardinality for discrete
  variables, and optional transform metadata. Avoid clever token abstractions until the
  examples force them.
- **Restrict priors to latents initially.** ACE's useful runtime-prior story is about
  task-relevant latent variables. Do not generalize prior tokens to arbitrary data values
  until there is a concrete example that needs it.
- **Let the Gaussian toy be the first correctness oracle.** It should check that posterior
  moments and autoregressive two-latent predictions are sane before GP-1D adds Cholesky
  and dataset plumbing.
- **Use archived experiments selectively.** If a gitignored `temp/` directory is present,
  treat it as external experiment code. Reuse ideas only when they fit nanoACE directly:
  rectangular attention, CPU float64 GP sampling, and simple train-loop habits are useful;
  cache provenance, Slurm, prefetch, reeval schemas, and large diagnostic suites are out
  of scope.
- **Short verification runs stay loose.** The Gaussian toy oracle should catch broken
  heads and obviously wrong posterior moments, but it should not become a brittle
  strict-quality gate around stochastic training.
- **Checkpoint artifact deferred.** A tiny toy `state_dict` can ship only after the toy
  trains reliably and the blob is genuinely small. Design the code so retraining is the
  fallback; do not design the repository around a checkpoint.

### Data layer

- **Native path: generate → save → train** (the nanoGPT `prepare`→`train` pattern).
  Decoupled (`data.py` writes, `train.py` reads), inspectable (the dataset is an artifact
  you can open), reproducible (not an RNG side-effect of the loop).
- **Dataset format: minimal `.pt` shards plus a tiny manifest.** Shards store tensors in
  the canonical `Batch`/`Tokens` shape or enough arrays to reconstruct it. The manifest
  records schema version, variable definitions, normalization/transform metadata, and
  shard list. Avoid cache fingerprints, optimizer provenance, or resume-compatibility
  matrices.
- **Sharded pool on disk** so it scales past RAM. This is the regime that actually
  produced results at scale (a large finite pool, a few passes), not a toy shortcut.
- **Deterministic `(seed, pass)` reconstruction** of `n` and the context/target split;
  independent of batch size.
- **Shuffle is fixed to "both"** (permute shard order + rows within shards), seeded by
  `(seed, pass)`. No `shuffle_mode` enum — drops the field from any manifest and removes
  one axis from resume checks.
- **Online generation is not a mode.** Generators (`gaussian_toy`, `gp1d`, ...) are pure
  functions returning a batch. "Online" means calling the generator inside the training
  loop; it needs no config flag. The boundary is the generator function itself, so
  `train.py` does not need separate online and cached code paths.
- **Scaling happens at generation.** Examples should keep values roughly around
  `[-1,1]`, because the embedders/head bias assume that scale. This is not a hard bound:
  stochastic observations can have tails outside the range, and calibration issues should
  be checked before adding clipping or rescaling.

### Training / ops

- **Keep:** checkpoint + simple resume; cosine LR; Adam; grad clip; GPU model with a
  **CPU float64 data/oracle split** (Cholesky on the Win/WSL2 GPU path can trip the host
  watchdog).
- **Drop prefetch.** Generate→save moved the expensive Cholesky *off* the training loop
  (it now runs once, in the save step), so prefetch no longer earns its keep. `train.py`
  reads shards synchronously. **Leave a code comment** explaining this, so the absence
  reads as a decision, not an oversight. (If the GPU ever starves on H2D, add
  pinned-memory + a copy stream as a later, local change.)
- **Cut entirely (large experiment management):** Slurm/HPC layer; finite-pool caching machinery
  beyond the simple sharded save (manifest, DGP-hash fingerprint, shuffle enum);
  multi-axis resume-provenance guards; schema-migration `reeval`; mechanistic
  probes / partial-R² / causal interventions (those served a separate research question).

### Examples / scope

- **Example 1: Gaussian (`mu`, `sigma`) toy.** Two latents, fixed latent priors, analytic
  Bayesian posterior on a grid, and a matching posterior predictive diagnostic. Runs
  online and remains small; longer local runs can save a checkpoint/plot for inspection.
  - **An optional pretrained checkpoint is useful** because it would make `gaussian_toy.py`
    instant and provide a stable regression anchor. Distribution is deferred until the
    trained artifact is small and worth keeping. See Open questions.
- **Example 2: GP-1D regression.** Lengthscale/outputscale as continuous latents,
  **plus kernel selection as a discrete latent** (this is what exercises the discrete
  path so it isn't dead code). Use a **deliberately diverse kernel set** (RBF, Matern,
  periodic, ...) so the sampled functions look genuinely different and classification is
  meaningful. The first implementation is online in `gp1d.py`; use the generate/save/pool
  path only if longer training makes online Cholesky sampling the bottleneck.
- **"Add a third task" is the canonical user/agent extension** — nano ships exactly two.
  Generality lives in the *architecture* (free: attention over a token stream), not in
  the *example surface*.

### Layout

- `ace.py`         — config → embedder → block → forward → heads → loss   (THE file)
- `diagnostics.py` — reusable grid-query helpers for marginal/AR diagnostics
- `gaussian_toy.py` — executable Gaussian toy: generator, training, oracle, eval, checkpoint, plot
- `gp1d.py`        — executable GP-1D example: generator, training, eval, checkpoint, plot
- `data.py`        — generators + save/load (future sharded saved-pool path)
- `train.py`       — loop + resume (future sharded saved-pool path)
- Dependencies: **torch only** in the core; plotting imports are isolated to `gaussian_toy.py`.
  Match the pinned workstation-tested stack unless there is a specific reason to change it:
  `torch==2.11.0+cu128` via `https://download.pytorch.org/whl/cu128` on the RTX 4060
  Laptop GPU. Single dataclass config, no config framework.

### Open questions

- **Correctness oracle as an automated test?** The Gaussian toy's analytic grid posterior
  is kept as a first-class *diagnostic* (`gaussian_toy.py` computes/shows predicted-vs-true
  moments). Whether it also becomes an automated check is undecided — leaning at most a
  loose check that catches catastrophic breakage, not a strict R²-threshold quality
  gate (brittle against stochastic training).
- **Online gen as a one-flag option** — resolved: not a flag; it's the generator-as-
  function (see Data layer).
- **How to distribute optional checkpoints** (deferred — decide once there is a real
  trained artifact). Options on the table: a tiny toy checkpoint committed in-repo (keeps
  the Gaussian example instant + offline; sub-MB, no Git LFS, plain `state_dict`); a larger model
  hosted on HF and lazy-downloaded (natural home for the multi-MB GP model); or both
  (two-tier). Principles to preserve for any option: core stays **torch-only** (any
  HF fetch is an optional extra, never required to read/run/train); the model ships
  config + seed so it's regenerable, not a mystery binary; and it's an *example
  artifact*, not the packaged "runtime product" non-goal.
