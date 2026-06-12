# aline DEVLOG

A running log of design decisions and results for the ALINE extension. The *why*
matters as much as the *what*; read this before changing the extension's
architecture or scope. Repo-wide context lives in the root `DEVLOG.md`; the
implementation plan and verification log in `docs/plans/PLAN-aline.md`.

Reference: Huang, Wen, Bharti, Kaski & Acerbi (2025), *ALINE: Joint Amortization
for Bayesian Inference and Active Data Acquisition* (NeurIPS 2025). Paper
markdown in the (gitignored) `temp/aline.md`; the released reference
implementation was consulted from a clone in `temp/aline-repo/`.

---

## 2026-06-12 — 5k validation run: the policy learns; US gap and null ℓ-contrast

First validation fine-tune from the retained GP-1D 200k checkpoint: 5k episode
batches (≈2.5k policy updates under 1:1 alternation), defaults throughout
(B=64, pool 128, M=32, T=16, `--random-frac 0.5`, batch-mean baseline,
immediate rewards), avg 1.02 s/step on the RTX 4060. Full log:
`artifacts/gp1d_aline_5k.log`; artifact `artifacts/gp1d_aline.pt` + figure.

- **The policy learns.** Held-out predictive RMSE 0.220 vs random 0.248 (below
  along the whole curve); θ log q −0.124 vs random −0.147 (above from ~step 4).
  Per-step rewards climbed throughout training (+0.074 → ~+0.10) and were still
  rising at step 5000.
- **`q_φ` calibration survived the joint fine-tune.** On policy-acquired
  contexts, hyperparameter marginals track the grid oracle (kernel KL
  0.002–0.025; means close; stds somewhat conservative — widest on the fixed
  periodic demo case, where the ℓ-goal policy clusters queries). This is the
  50/50 policy/random rollout mix in prediction steps doing its job; the
  `--freeze-base` escape hatch was not needed.
- **Uncertainty sampling still wins on prediction** (US 0.194): ALINE tracks US
  until ~step 8, then a gap opens. Read as **undertraining**, not a capacity
  verdict — 2.5k policy updates is <1% of the paper's episode count, training
  NLL was still improving (−0.4 by the end), and rewards hadn't plateaued. The
  pre-registered fallback order stands: longer run first (same recipe, one
  change at a time), then `--reward-to-go`, then per-layer cache reads.
- **Targeting contrast is null** (log q(ℓ_true | D_T): matched −0.043 vs
  mismatched −0.042). Beyond undertraining, an honest task-structure hypothesis:
  in GP-1D, coverage queries that help prediction also pin down the
  lengthscale, so the matched/mismatched gap may be intrinsically small for ℓ —
  the paper demonstrated the contrast on the psychometric task (threshold-region
  vs extreme stimuli), not on its GP experiment. Query *placement* does differ
  by goal in the demo panel; it just doesn't yet buy measurable log q. A
  **kernel-goal contrast** would be a fairer instrument (kernel identification
  wants tight local pairs to read roughness; prediction wants coverage) — an
  eval-only change, queued as fallback #4.
- Reward tails bounded as predicted under the noiseless DGP (pred: −0.9 to
  +1.8, mean +0.14); `--sigma-obs` stays at 0.

---

## 2026-06-12 — Initial implementation: ALINE as an ACE-native extension

Implemented per `docs/plans/PLAN-aline.md` (design discussion resolved
2026-06-12; three-reviewer doublechecks on both the plan and the code).
`aline.py` is task-agnostic (knows `ace.py`, not `gp1d.py`); the GP-1D episodes,
RL loop, and diagnostics live in `gp1d_aline.py`.

### The thesis: the paper's apparatus collapses into token composition

- **The inference network IS the core `ACE`, unchanged.** Parameter targets are
  latent QUERY tokens; predictive targets are data QUERY tokens; the paper's
  target specifier ξ is *which target tokens are active* — a per-row mask over
  a fixed superset `[ℓ_ell, ℓ_scale, ℓ_kernel, x*_1..M]`. The target tokens do
  double duty: they are the queries `q_φ` answers (NLL and reward are computed
  on them) and, through their final states, the goal representation the policy
  reads — including how uncertain the model still is about each goal.
- **Query candidates are "hypothetical targets":** data QUERY tokens at
  candidate locations, embedded by the core embedder verbatim. ξ-switching at
  runtime (and the paper's D.4 demos) reduce to mask flips.
- `p(ξ)`: 50% predictive / 50% parameter rows, the parameter subset drawn by
  `sample_reveal_mask(3, b, q=0.0)` — the tested non-empty subset/count
  mixture, so singletons, pairs, and all-three are in-distribution. The
  discrete `kernel` as a parameter target is a small genuine extension over the
  paper's continuous-only GP experiment (house rule: exercise the categorical
  path).

### Architecture: read-only policy decoder, structural φ/ψ separation

- Two `PolicyBlock`s (pre-LN: query→context cross-attn, query→target
  cross-attn, MLP; ~0.4M params on the 1.2M base) read the **detached** final
  trunk states; a linear head scores candidates pointwise; masked softmax over
  the remaining pool. No query–query self-attention and no write-back — both
  match the released reference implementation (`model/encoder.py:create_mask`
  unmasks only `[:, :context]` plus query→ξ-selected-target columns; its
  Fig. A1(a) rendering contradicts its own code). The reference is literally a
  read-only query/target stream over per-layer context states with shared
  weights, which confirms the hoisting equivalence: per-layer interleaving
  without write-back ≡ this decoder reading per-layer caches. The two
  deliberate differences: **separate policy weights** (what makes the φ/ψ
  gradient firewall structural — the paper's "PG does not update q_φ" made
  exact: no-grad query embedding, detached states) and **final-state reads**
  (the per-layer-cache upgrade path is a config change, pre-registered as the
  first fallback if the policy plateaus below US).
- **Permanent bitwise parity invariant:** `forward_with_states` re-runs the
  core block loop verbatim, and `check_step0` asserts predictions equal a fresh
  base `ACE` bitwise — at warm start and in tests. The inference path of a
  trained ALINE artifact is exactly an ACE, so `gp1d.evaluate`/`gp_oracle`
  diagnostics apply unchanged. If `ace.py`'s forward changes, the warm start
  fails loudly (arbuffer's coupling guard, strengthened from step-0-only to
  always).

### Training: alternating phases, episode-batch updates

- **Alternating schedule (deliberate variant of Algorithm 1).** The paper
  applies `L_NLL + L_PG` together at every inner update, with warm-up (random
  actions, NLL-only) its only pure phase. Here, steps alternate *prediction*
  (NLL → φ; policy frozen; rollouts 50% current-policy / 50% random) and
  *policy* (REINFORCE → ψ; φ frozen) — each phase optimizes against a
  stationary partner. The 50/50 rollout mix is the calibration-retention
  device: on-policy rows adapt `q_φ` to policy-shaped contexts (Prop. 2's KL
  gap = reward-bound tightness), random rows preserve the random-context
  calibration that keeps the oracle diagnostics meaningful. The paper-literal
  scheme is kept as `--update-mode simultaneous`. Warm-up is just a
  prediction-only prefix (`--warmup-pred-steps`: 0 with a warm start — the
  fresh policy head is near-uniform — 2000 from scratch).
- **One optimizer step per episode batch (deviation from Algorithm 1's in-loop
  updates).** Within a rollout the parameters are frozen, so the reward
  `R_t = mean_active(Δ log q)` measures information from the new point, not
  optimizer movement (reward purity); it also keeps the repo's training spine —
  each episode batch is a pure function of `(seed, step)` via the `mix_seed`
  reseed, policy sampling included, giving max|dW|=0 reproducibility and
  resume-exactness. NLL (Eq. 12) backwards per step so trunk graphs free; the
  PG loss (Eq. 11, γ=1, immediate rewards, batch-mean baseline; rewards
  telescope to the total improvement, `--reward-to-go` is the unbiased-credit
  variant, off by default) backwards once at episode end — only the small
  policy-side graphs are retained. Two Adam optimizers, per-group cosine
  schedules with `T_max` = each group's expected optimizer-step count.
- Episodes: one `gp1d.draw_instances` call at pool+M points per batch (pool
  observations are lookups), 1 random seed point, T=16 default (context ≤ 17,
  inside the warm start's trained `n_context ≤ 20`; warn past 19), noiseless
  (`--sigma-obs` at lookup is the knob; MDN `min_scale` bounds the densities,
  and the 5k run confirmed unremarkable reward tails). Argmax actions at
  evaluation (deterministic fixed diagnostics), sampling during training.

### Recorded hazards (found during implementation/review)

- **Autograd version counter vs deferred PG backward:** `observe` must replace
  `query.mask` *functionally* — the pending policy graphs hold references to
  earlier masks; in-place `scatter_` corrupts them.
- **Cosine LR multiplier is exactly 0 on a run's final step** (core
  `_build_scheduler` convention): harmless for real runs, but 1-step test runs
  train at LR 0.
- **Resume restores the phase schedule from the checkpoint's `config`**
  (`warmup_pred_steps`; warns on `update_mode`/`freeze_base`/`random_frac`
  mismatch) — re-deriving the CLI default would silently switch a resumed
  from-scratch run to alternation.
- **PG normalization uses the effective on-policy episode count**, not the
  batch size — a plain batch mean halves the policy gradient under
  `simultaneous` + `random_frac`.

### Deferred / future

- ALINE + ACEP runtime Beta priors (the paper's own stated future work; the
  co-location makes `ace_prior_beta.py` available nearly for free).
- Per-layer cache reads for the policy (the reference's read pattern).
- Kernel-goal targeting contrast in `evaluate` (fairer instrument for GP-1D).
- Second experiment family (psychometric / BED) = the pre-agreed graduation
  trigger to a separate nanoALINE repo, not this extension.
