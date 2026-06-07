# Plan: 1D Bayesian optimization example (`bo1d.py`)

Created: 2026-06-07
Updated: 2026-06-07 (revised after a two-reviewer pass; then built, run, enlarged)
Status: COMPLETE (code working & validated end-to-end; quality is a GPU-training job)

## Status

Implemented: `ace_prior.sample_contaminated`; `bo1d.py` (schema, DGP, training,
fixed three-prior diagnostic, plot, checkpoint helpers, `--scale-check`). Docs in
`README.md` and `AGENTS.md`. Default network: `d_model=192`, **6 transformer
blocks**, 16 heads, mlp 384, `K=12` (~3.9M params); default contamination
`ε=0.05`.

Validated by running (CPU, torch 2.12.0 in a local venv):

- `--scale-check`: data token values sit in `[-1, 1]` with ~0.5% stochastic-tail
  spill; the ε floor leaves ~ε/2 mass in the wrong half of a confident prior.
  Matches the pure-Python Monte-Carlo done first.
- The full train → fixed three-prior diagnostic → plot path runs end to end and
  the structural behavior is in the right direction (correct prior tightens and
  shifts `p(x_opt | D)` toward truth; wrong prior is resisted; `y_opt` recovered).

**Training budget is the real limitation, not architecture or ε.** The paper
trained 1D-BO for `5e5` steps; on CPU we can only afford ~1e4. At that budget the
small earlier model (~1.2M, 12k steps) actually showed the prior pull *better*
(correct prior: mean 0.70→0.60 toward truth 0.40) than the larger 6-block model
(~3.9M, 15k steps: 0.75→0.66, priors weakly used) -- the bigger model is simply
more undertrained at equal steps. So the committed CPU artifact is a smoke test,
not a quality result, and the ε=0.05-vs-0.10 question is deferred until a proper
run can judge it.

**Real training is a GPU job** (decision: keep the faithful 6-block default and
run it on the workstation GPU):

```
python bo1d.py --steps 500000 --batch-size 64 --save-checkpoint artifacts/bo1d.pt
```

Consider cosine-annealed LR (the paper used `5e-4` with annealing) for a long run;
not wired in (constant `--lr 3e-4`), an easy follow-up if needed.

## Summary

A fourth executable example: 1D Bayesian optimization. The two latents are the
**location** `x_opt` and **value** `y_opt` of the global optimum (minimum) of a
black-box function. ACE amortizes `p(x_opt | D)` and `p(y_opt | D)` directly --
the distributions that normally make BO need bespoke acquisition machinery -- and
accepts a **runtime Beta prior over the optimum location** (the paper's πBO /
ACEP-TS story).

This is a deliberate mix of the existing examples:

- **from `gp1d.py`**: online GP function sampling (CPU float64 Cholesky), `(x, y)`
  data tokens, `x_dim = 1`, sampled kernel/hyperparameters;
- **from `gaussian_toy.py` / `sbi_sir.py`**: ACEP Beta prior tokens over bounded
  continuous latents, the gaussian/sir reveal mechanism, observation noise, and
  the fixed diagnostic plot pattern;
- **new**: latents that are properties of the *specific sampled function* (the
  optimum) rather than of the function class; the optimum-planting DGP; an
  **ε-contamination** ("robust prior") mechanism; and a **no-grid-oracle**
  verification recipe with a structural falsifiability check.

Paper reference: Appendix C.3.1 (BO dataset generation), C.1 (GP sampling), B.1
(prior injection). The DGP is **adapted from**, not faithful to, C.3.1 (the fold
operand and the role of the min-value distribution differ -- see "DGP"). The
paper is inspiration, not a constraint (see DEVLOG).

## Scope

- **In scope**:
  - `bo1d.py`: schema, online DGP, training loop, fixed diagnostic, checkpoint
    helpers, plot.
  - `sample_contaminated` helper in `ace_prior.py` (genuinely shared); the
    plot-only `mixture_logprior_on_grid` stays local to `bo1d.py` until a second
    consumer exists.
  - Three-column prior diagnostic (uniform / correct-informative /
    wrong-informative) plus a posterior-predictive panel and a conditional panel.
  - Small Gaussian observation noise on data `y` (`--sigma-obs`).
  - Docs: this plan, a DEVLOG entry, `README.md` run section, `AGENTS.md`
    "currently implemented" list, and the no-oracle verification recipe.
  - Short verification runs and a token-scale histogram sanity check (which also
    checks the ε-contamination marginal).
- **Out of scope**:
  - **No oracle.** The other three examples already carry numerical grid oracles;
    this one deliberately demonstrates the case where no closed-form posterior
    exists (the `|·|` fold destroys Gaussianity). Verification is structural +
    qualitative (see "Verification"). A Monte-Carlo simulator posterior was
    considered and declined: the three-column structural check covers the
    prior-handling behavior, and we accept that absolute marginal shape is not
    independently validated here.
  - **No BO loop.** No iterative acquisition / Thompson-sampling rollout. We show
    the conditional `p(x_opt | y_opt = v, D)` to gesture at TS, nothing more.
  - **No new core (`ace.py`) machinery.** Both latents are bounded continuous and
    reuse the existing PRIOR-token path unchanged.
  - **No discrete latent.** Kernel/hyperparameters are sampled nuisance, not
    predicted (`gp1d.py` already exercises the discrete path).
  - Heavy convergence tuning beyond a working example artifact.

## Schema

```python
[
    Variable("y", "data", "continuous"),
    Variable("x_opt", "latent", "continuous", bounds=(-1.0, 1.0)),
    Variable("y_opt", "latent", "continuous", bounds=Y_RANGE),
]
```

- `x_opt`: identity transform, bounds `(-1, 1)`; `encode_value` is then exactly
  identity. This is the headline Beta-prior target (πBO). `x_opt` is a latent
  token only (PRIOR or QUERY) -- never a data point, so the answer is never leaked
  into the data tokens.
- `y_opt`: identity transform, bounds `Y_RANGE` covering the **full** native y
  working range (see "Scaling"), so `encode_value(y_opt)` and the data-`y` scaling
  are the *same* affine.
- `y`: continuous data variable, scaled at the token boundary by exactly
  `scale_y(y) = encode_value(y_opt_var, y)` over `Y_RANGE` (written explicitly so
  a function value equal to `y_opt` lands at the same token coordinate under both
  paths, and `y_opt ≤ all y` is legible to the model).

`p(x_opt | D)` *can* be multimodal under sparse data (competing basins from the
`|·|`-folded function), so this example can exercise the MDN's multi-component
capacity in a way the unimodal Gaussian/SIR posteriors do not. This is a "can",
not a guarantee, and depends on `mdn_components` (default 8) and convergence;
since heavy tuning is out of scope, the claim is not load-bearing.

## Data-generating process (per sample)

Adapted from Appendix C.3.1, 1D, with the optimum **value** leveled by our prior
instead of the paper's `U[-5, 5]` offset.

1. **Nuisance hyperparameters** (not latents): kernel from
   {RBF, Matérn-1/2, Matérn-3/2, Matérn-5/2} with weights `[0.35, 0.1, 0.2, 0.35]`;
   lengthscale `ℓ ~ N(1/3, 0.75)` truncated `[0.05, 2]`; output scale
   `σ_f ~ U(0.1, σ_f_max)`. `σ_f_max` is tamed (provisionally `0.5`) so the bump
   above the optimum stays near the `[-1, 1]` token convention.
2. **Latents** (truth drawn from the ε-contaminated prior; see below):
   - `x_opt` on `[-1, 1]`.
   - Natural optimum depth `d`: minimum of `N` draws from `N(0, σ_f²)` with
     `N = ceil(2 / ℓ)` (≈ uncorrelated samples across the width-2 domain); with
     `p = 0.1` subtract `Exp(1)` (the paper's "unexpectedly low optimum" kick,
     **adapted** -- the paper adds to the mean, we deepen `d`). Then **clamp
     `d = min(d, 0)`** and cap `|d|` (provisionally `≤ 2.0`) for a consistent dip
     and bounded function height (`N = 1` at large `ℓ` otherwise leaves `d`'s sign
     unconstrained, and the `Exp` tail otherwise inflates height -- see Scaling).
   - `y_opt` on `Y_RANGE` (the leveling shift that sets the optimum value).
3. **GP draw conditioned on the optimum geometry** (**Matheron's rule** -- a true
   posterior sample, not a mean-shift): sample a joint GP prior draw `g` over
   `{x_opt} ∪ x_data`, then
   `g_c(x) = g(x) − (k(x, x_opt) / k(x_opt, x_opt)) · (g(x_opt) − d)`,
   which gives `g_c(x_opt) = d` exactly. The correction adds a kernel-shaped bump
   of magnitude `d − g(x_opt)` centered at `x_opt` with **width set by the
   lengthscale** (depth set by `d`). One Cholesky over the joint set; context and
   targets come from the same draw (simpler than the paper's independent-target
   speedup, cheap at our `N`).
4. **Fold + envelope + level**:
   `f(x) = |g_c(x) − d| + (1/5) · (x − x_opt)² + y_opt`.
   Both added terms are `≥ 0`; their sum is `0` iff `g_c(x) = d` **and**
   `x = x_opt`. Since `g_c(x_opt) = d` and the envelope is strictly positive for
   every `x ≠ x_opt`, `x_opt` is the **exact, unique global minimum** with value
   `y_opt`. (Level-`d` re-crossings zero only the fold term; the envelope lifts
   them strictly above `y_opt`, producing the kinked Fig.-S12 geometry and the
   multi-basin structure -- not spurious global minima.) The `|·|` fold is exactly
   why there is no closed-form oracle.
5. **Observe + tokenize**: `y_i = f(x_i) + N(0, σ_obs²)` (small obs noise,
   `--sigma-obs`, matches the continuous MDN and is more BO-realistic); scale by
   the shared affine; emit data tokens `(x_i, scale_y(y_i))`, one Beta PRIOR token
   per latent, QUERY targets for held-out data points and unrevealed latents.

Envelope constant `1/5` is hardcoded (paper value); revisit only if functions
look too bowl-dominated.

Context/target split: observe at a set of `x` locations, then a random
permutation splits into context vs data targets (as in `sbi_sir.py`), with
`min_context`/`max_context` bounds. `min_context = 0` is safe: the two always-on
PRIOR tokens keep the context non-empty (`ACE.forward` requires `≥ 1` active
context token).

## ε-contamination (robust prior)

The effective generative prior over each latent is
`(1 − ε) · Beta(α, β) + ε · Uniform`, a classic ε-contamination / robust-Bayes
prior. ε is a fixed hyperparameter (provisionally `0.1`), exposed as
`--prior-uniform-mix`. Applied to **both** latents.

**Why it is needed, and why it is not redundant with `sample_prior_params`.**
`ace_prior.sample_prior_params` already mixes uniform/broad/concentrated Betas,
but it always draws truth from the *same* Beta that the token encodes -- the token
never lies. A model trained on that alone learns to trust a concentrated token
essentially fully, so the **wrong-informative-prior diagnostic column would
fail** (the model would follow the wrong prior). ε-contamination's entire job is
to **decouple truth from the token** a fraction `ε` of the time -- the one thing
`sample_prior_params` cannot do, and exactly what the wrong-prior column needs.

**Where it lives.** Entirely in the DGP truth-draw (and plot overlays), not in the
token or model. The token still encodes only the user's raw `Beta(α, β)` (a single
Beta cannot represent a mixture, and need not). `bo1d` reuses
`sample_prior_params` for the `(μ, ν)` hyperprior / token, then applies
contamination *only at the truth-draw* via `sample_contaminated` (flip an ε coin:
uniform over the native range, else `draw_from_beta`). Contaminating the cases
where the token is already `Beta(1, 1)` is harmless (uniform contaminated with
uniform is uniform), so no special-casing.

**What the model actually learns (corrected framing).** Not "a global discount
knob." The contaminated prior is the true generative prior; NLL training learns
the Bayes-optimal posterior under it. The uniform floor means that posterior never
fully commits to a confident-but-wrong location; the *strength* of the
data-override is **not** a constant -- it depends on the token's Beta
concentration and how informative `D` is. ε must be **fixed** (and not in the
token): a per-sample ε that the model cannot observe would be marginalized into an
average response, which is a broader hyperprior, not robust-Bayes, and would not
support the wrong-prior column.

New helper `ace_prior.sample_contaminated(mu_unit, nu, lo, hi, eps)` -> native
draw from the mixture. Plot-only `mixture_logprior_on_grid` lives in `bo1d.py`
(single consumer); restrict its grid to `[lo, hi]` so the uniform term integrates
correctly.

## Scaling (expected to need tuning; the first thing to check)

`y_opt` and data `y` are the **same physical quantity**, so they share one affine:
`scale_y(y) = encode_value(y_opt_var, y)` over a **frozen** module constant
`Y_RANGE` (frozen so checkpoints stay valid -- `Y_RANGE` is baked into
`variables()`; do not make it a CLI arg). `y_opt` posterior then lives in the
lower part of `[-1, 1]`; data fills the rest.

Honest spread budget (corrected from the first draft): away from `x_opt`,
`|g_c(x) − d| ≈ |g(x)| + |d|`, so the **natural depth `|d|` inflates the whole
function height**, not just the dip. With `σ_f = 0.5`, `|g|` tails to ~1, `|d|`
(min of up to ~40 draws, plus the `Exp` kick) can reach ~2-3 before clamping, and
the envelope adds ≤ 0.8. So `f − y_opt` can reach ~3 in the tail -- the earlier
"bump ≲ 1.5" was wrong. Mitigations: tamed `σ_f_max`, the `|d| ≤ 2` cap, and
accepting stochastic tails outside `[-1, 1]` (a soft convention per AGENTS.md, not
clipping). There is a real tension: widening `Y_RANGE` to absorb the height also
compresses `y_opt`'s resolution within `[-1, 1]`.

As-built constants (validated by the scale check, see "Status"): `σ_f ~ U(0.1,
0.5)`, `|d| ≤ 2`, native `y_opt ∈ [-1, 0]` (`Y_OPT_RANGE`), `Y_RANGE = (-1.0,
2.0)`. `Y_RANGE` lower bound is exactly the global function floor (`f ≥ y_opt ≥
-1`); the upper bound absorbs the bump. If `--sigma-f-max` is raised, `Y_RANGE`
may need regenerating.

The MDN predicts unconstrained mass in `[-1, 1]` token space (a known deferred
caveat), so the displayed `x_opt`/`y_opt` marginals are the grid-renormalized
densities over the latent bounds, and boundary mass leakage is an accepted caveat
(as in `gp1d.py`).

**First implementation step is a scale check**: sample a batch, confirm the
token-value histogram sits ~`[-1, 1]`, and confirm the drawn-`x_opt` marginal for
a fixed token matches `(1 − ε) Beta + ε Uniform` (the only place the contamination
is observable). Adjust `σ_f_max`, the `|d|` cap, and `Y_RANGE` as needed before
training.

## Diagnostic and verification (no oracle)

Fixed evaluation case (seeded), evaluated under three prior settings on `x_opt`:

1. **uniform** prior,
2. **correct informative** prior (should tighten toward true `x_opt`),
3. **wrong informative** prior (mass on the wrong basin) -- ACE should still
   recover the true optimum thanks to the ε floor (data-dependent, not
   guaranteed; ensure the fixed case has enough context near the true basin).

**Structural falsifiability check** (the substitute for an oracle): the three
columns jointly constrain behavior. uniform→correct must **tighten** (proves the
model uses priors -- a prior-ignoring model fails here); correct→wrong must
**recover** (proves the model does not blindly follow -- a prior-slaved model
fails here). ε must be small enough that correct priors still help yet large
enough that wrong priors are recoverable; `ε = 0.1` is the provisional balance.

Plot (provisional layout, mirrors `sbi_sir.plot_diagnostic`'s per-column loop):

- top: the true function, context points, true `(x_opt, y_opt)` marked, ACE
  posterior-predictive band;
- middle: `p(x_opt | D)` marginal per prior column, effective (contaminated) prior
  overlaid, true `x_opt` marked;
- bottom: `p(y_opt | D)` marginal and the conditional `p(x_opt | y_opt = v, D)`
  (its own panel, e.g. a small line/heatmap); state which prior column the
  conditional uses.

The conditional needs only `diagnostics.conditional_log_density` (which builds a
zero-spread PRIOR token for the known latent) -- **no `sample_ar`**. Requirement:
the model must have *seen* zero-spread `y_opt` tokens, so `latent_context_prob`
must reveal `y_opt` sometimes.

**Reveal mechanism**: the gaussian/sir pattern -- with probability
`latent_context_prob`, *replace* a latent's finite-spread Beta PRIOR token with a
**zero-spread known** PRIOR token (`known_latent_features`), keeping the token
**active** and dropping the matching target. This is required for
`append_or_replace_context_token` / `conditional_log_density` to find an active
PRIOR slot; it is **not** gp1d's absent-token pattern.

**Verification recipe** (the grid-oracle convention does not apply): short run
that completes; the token-scale + contamination-marginal histogram check; and the
fixed diagnostic, read for (a) marginals concentrating near marked truth, (b) the
uniform→correct tightening, and (c) the correct→wrong recovery. Recorded in
`AGENTS.md` and `DEVLOG.md` as a deliberate departure from the oracle convention.

## CLI / artifacts

Mirror `gp1d.py` / `sbi_sir.py`: `--steps`, `--batch-size`, `--device`, `--seed`,
context/target sizes, model size, `--lr`, `--latent-weight`,
`--latent-context-prob`, `--prior-uniform-mix`, `--sigma-f-max`, `--sigma-obs`,
`--jitter`, `--eval-points`, plot/checkpoint/eval-only flags. `Y_RANGE`, the
envelope constant, kernel weights, and the `|d|` cap are frozen module constants,
**not** CLI args. Artifacts: `artifacts/bo1d.pt`, `artifacts/bo1d.png`.

## Implementation order

1. Schema + `make_tokens` + scaling constants; **scale check** + contamination
   marginal check on a sampled batch.
2. `sample_contaminated` in `ace_prior.py`.
3. DGP (`sample_bo_batch`): hyperparameters, contaminated latent draws + `d`
   clamp/cap, Matheron GP draw + optimum conditioning, fold/envelope/level, obs
   noise, permutation split, tokenization, gaussian/sir reveal.
4. Training loop + checkpoint helpers (copy the established pattern).
5. Fixed eval case + three-prior diagnostic + plot (+ conditional panel).
6. Short verification run; tune scales; docs (DEVLOG entry, README, AGENTS).

## Open questions / risks

- **Spurious near-ties** off `x_opt` are expected and benign (kinks/local basins,
  not global minima) -- they drive the wanted multimodality. Optionally assert in
  the fixed diagnostic that the labeled `x_opt` is the dense-grid argmin (it is, by
  construction; cheap belt-and-braces).
- **Scale numbers are provisional** and expected to change after the histogram
  check (flagged by the user); `|d|` inflating function height is the main risk.
- **Wrong-prior recovery is data-dependent**, not guaranteed; the fixed case must
  carry enough context near the true basin for the demo to land, and the
  correct-column tightening must remain visible at `ε = 0.1`.
- **Multimodality is a "can", not a must**, and is harder to train than the
  unimodal examples; not load-bearing given tuning is out of scope.
- **Scope**: this is example #4 with several new sub-mechanisms (ε-contamination,
  the optimum-planting DGP, shared-affine scaling, the no-oracle break). DEVLOG's
  "nano ships exactly two" is already stretched by SIR. BO earns its place
  (instance-level latents + the optimum-posterior headline + robust prior
  injection), recorded as a deliberate decision in the DEVLOG entry.

## Review notes (2026-06-07)

Two Opus reviewers checked this plan. Outcomes folded in above:

- **Rejected (verified false):** the claim that the `|g_c − d|` fold fails to make
  `x_opt` the unique global minimum. The envelope is strictly positive off
  `x_opt`, so re-crossings of level `d` are lifted above `y_opt`; uniqueness +
  exactness hold (see step 4). The plan keeps the construction.
- **Adopted:** the ε-contamination rationale rewrite (decoupling truth from token;
  the corrected "what the model learns" framing; ε fixed) and its reuse of
  `sample_prior_params`; Matheron's-rule labeling; the corrected scaling budget
  (`|d|` inflates height) + explicit shared affine + frozen `Y_RANGE`; the
  `d ≤ 0` clamp and `|d|` cap; the gaussian/sir reveal spec; observation noise;
  dropping "AR" from the conditional story; softening the multimodality claim;
  `mixture_logprior_on_grid` kept local; the structural three-column
  falsifiability check; and the completeness notes (split, no-leak, `min_context`
  safety, MDN out-of-bounds renormalization, checkpoint/`Y_RANGE` compat).
- **Declined:** adding a Monte-Carlo simulator-posterior oracle (user decision:
  the other three examples carry oracles; this one demonstrates the no-oracle
  case, with the structural check as the gate) and simplifying the
  min-value/`Exp(1)` depth machinery (user wants the full DGP).
