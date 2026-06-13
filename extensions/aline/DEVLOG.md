# aline DEVLOG

A running log of design decisions and results for the ALINE extension. The *why*
matters as much as the *what*; read this before changing the extension's
architecture or scope. Repo-wide context lives in the root `DEVLOG.md`; the
implementation plan and verification log in `docs/plans/PLAN-aline.md`.

Reference: Huang, Wen, Bharti, Kaski & Acerbi (2025), *ALINE: Joint Amortization
for Bayesian Inference and Active Data Acquisition* (NeurIPS 2025). Paper
markdown in `paper/`; the released reference implementation was consulted from
a local clone.

---

## 2026-06-13 — 35k run: undertraining confirmed; US gap closed; calibration improved; 128-ep eval was under-powered

Longer fine-tune from the retained GP-1D 200k checkpoint, the pre-registered
fallback #1 (longer run first, same recipe, one change). 35k episode batches
(≈17.5k policy updates under 1:1 alternation), defaults throughout (B=64, pool
128, M=32, T=16, `--random-frac 0.5`, batch-mean baseline, immediate rewards),
avg 0.90 s/step on the RTX 4060 (~8.8 h). Evaluated at `--eval-episodes 512`
(see the methodological note below for why this matters). Log:
`artifacts/gp1d_aline_35k.log`; artifact `artifacts/gp1d_aline_35k.pt` + figure.
The 5k checkpoint was re-evaluated at 512 for a matched comparison
(`artifacts/gp1d_aline_5k_eval512.log`).

Matched comparison (both eval=512, same episode draws):

| metric | 5k | 35k |
| --- | --- | --- |
| RMSE@T — ALINE | 0.209 | 0.187 |
| RMSE@T — US | 0.181 | 0.182 |
| gap to US | 0.028 | 0.005 |
| kernel contrast δ | +0.085 | +0.141 |
| ℓ contrast δ | −0.049 | +0.013 |
| θ log q — ALINE / random | −0.220 / −0.187 | −0.157 / −0.152 |
| mean kernel KL (calibration, 4 acquired contexts) | 0.415 | 0.073 |

- **Undertraining was the dominant factor, as pre-registered.** The US gap
  closed from 0.028 to 0.005 (matched) — ALINE now sits within noise of
  uncertainty sampling on predictive RMSE. Per-step rewards climbed +0.074 →
  ~+0.10 and plateaued around step ~28k (peak +0.108); training NLL still
  drifting down slowly (~0 → −0.2). Reads as approaching convergence on the
  policy side, so the credit-assignment ladder (previous entry) is now an
  *enhancement* path, not a rescue.
- **Kernel targeting contrast is real and grew** (+0.085 → +0.141). The 35k
  policy places queries by goal *and* buys measurable log q(kernel|D_T). The ℓ
  contrast stays weak (−0.049 → +0.013), exactly the recorded task-structure
  hypothesis: coverage queries pin ℓ regardless, so the matched/mismatched gap
  is intrinsically small for ℓ in GP-1D. Kernel was the fairer instrument and
  it delivered.
- **Calibration IMPROVED through the joint fine-tune — reverses an
  in-progress concern.** On the same four acquired contexts, mean kernel KL
  dropped 0.415 → 0.073, driven by ep0 going from a catastrophic 1.523 to
  0.128. The two 35k episodes that looked elevated mid-analysis (0.128, 0.142)
  are far better than the 5k model on those same functions. The 50/50
  policy/random rollout mix held calibration; `--freeze-base` again not needed.
  **The pre-registered 0.002–0.03 KL band was set from undersampled 5k=128
  readings and is not a meaningful floor; future runs should judge calibration
  at eval=512.**
- **Methodological correction: the 5k entry's 128-episode eval was
  under-powered, and two of its headline numbers do not survive at 512.** That
  entry recorded a *null* kernel contrast (−0.006) and θ log q *above* random
  (−0.124 vs −0.147). Re-evaluating the *same 5k checkpoint* at 512 gives
  kernel contrast **+0.085** (positive all along, masked by 128-ep noise) and
  θ log q **below** random (−0.220 vs −0.187 — the claimed advantage was
  sampling noise). The ±0.0x effects this extension measures need ≥512
  episodes; treat the 128-ep numbers in the 5k entry as superseded. **Action:
  bump the `--eval-episodes` default (currently 128) to 512.**
- **θ log q (aggregate) shows no clear ALINE advantage at either budget** (35k:
  −0.157 vs random −0.152, tied). Consistent with task structure — parameter
  inference here is coverage-driven, and random gives coverage too. The signal
  that *does* separate by goal is the kernel contrast, not aggregate θ log q.
- Reward tails bounded as before (pred mean +0.14, −1.76 to +2.44); noiseless
  DGP, `--sigma-obs` stayed 0.
- **Lost: the per-budget trajectory.** The intended 10k/20k/30k checkpoint
  snapshots were not captured (the helper copier failed silently; `--ckpt-every`
  overwrites one file and the final save is model-only), so RMSE-gap-vs-budget
  and contrast-vs-budget are unavailable. The reward/NLL trajectory in the log
  (every 100 steps) carries the convergence read instead, which is what the
  fallback decision actually needed. If trajectory data is wanted later, re-run
  with explicit per-snapshot `--save-checkpoint` names.

**Done same day** (2026-06-13): the 35k artifact is now the canonical
`artifacts/gp1d_aline.pt` the local-only playground tab serves; the 5k model is
preserved as `artifacts/gp1d_aline_5k.pt`. Swap followed the staleness
discipline — re-exported the fp16 blob and regenerated the parity/env/demo
fixtures *together*, then `npm test` green (37/37, incl. ALINE env + policy +
teacher-forced chain parity). Deploy workflow unaffected (the ALINE tab is
local-only; the blob is gitignored). Also done: `--eval-episodes` default raised
128 → 512 (the under-power finding above).

---

## 2026-06-12 — PG credit assignment: reward-to-go analyzed, n-step credit queued

Discussion record, no code change yet (the 35k budget run is in flight; this
entry pre-registers what to try if its endpoint says "go non-myopic": rewards
plateaued, US still ahead on prediction). The default PG weighting pairs
`log pi_t` with the *immediate* reward `R_t` (Eq. 11 as implemented) — myopic
credit: a query that buys little now but sets up an informative follow-up gets
no credit for the downstream gain. Non-myopic acquisition is much of ALINE's
point, so this is the natural next axis after budget.

- **Telescoping makes reward-to-go gentler than generic REINFORCE.** Because
  `R_t = log q_t − log q_{t−1}`, the reward-to-go collapses to
  `G_t = log q_T − log q_{t−1}` — i.e. terminal-reward REINFORCE with the
  model's own pre-action log q as a built-in, action-independent
  potential-style baseline. The extra variance comes from `log q_T`'s
  dependence on the downstream sampled actions, not from summing T noisy
  increments.
- **Remaining issues with full reward-to-go.** (1) Within-episode credit
  diffusion: all T actions share `log q_T`; what discriminates them is only the
  `log q_{t−1}` subtraction plus cross-episode averaging — weakest exactly at
  the early, most non-myopic steps. (2) Advantage scale grows from per-step
  ~0.1 to remaining-episode ~1–2 at early steps; `--policy-lr`/grad-clip were
  implicitly validated at immediate-reward scale, so watch the pg trace and be
  ready to lower the policy LR — otherwise "reward-to-go is unstable" and
  "wrong LR" are confounded. (3) Every step's credit routes through the
  terminal self-estimate at context size 17, the edge of the warm start's
  trained `n_context ≤ 20` range — the oracle calibration guard becomes more
  load-bearing. (4) It departs from Eq. 11's immediate-reward form; record as
  a deliberate variant if flipped on.
- **n-step credit is the middle ground to try first**:
  `G_t^(n) = log q_{min(t−1+n, T)} − log q_{t−1}`, with n=1 the current
  immediate scheme and n=T full reward-to-go. Small n (2–4) captures two-query
  coordination — the tight-local-pairs strategy that lengthscale/kernel
  identification wants — at a fraction of the variance. The 5k eyeball notes
  already showed beside-acquired-point clustering under xi=ell, consistent
  with myopic credit learning the *reactive* half of pairing (the second
  query's payoff is immediate); what n >= 2 adds is *anticipatory* placement.
  Implementation: a small generalization of the `weights` computation in
  `rollout` (a `--credit-n` knob subsuming `--reward-to-go`).
- **Fallback ladder refined** (supersedes the 5k entry's order): longer run
  (in flight) → n-step credit (n=2–4) → full reward-to-go → per-layer cache
  reads. Footnote for completeness: the batch-mean baseline includes each
  episode's own reward — a standard O(1/B) bias, negligible at B=64;
  unaffected by any of this.

---

## 2026-06-12 — Playground tab (TS port; local-only against the 5k artifact)

The playground gained a sixth tab running this extension in-browser (plan +
verification: `docs/plans/PLAN-aline-playground.md`). Interaction model: a
hidden GP function — sampled client-side by an exact float64 port of the gp1d
DGP — answers every query; the user only chooses *where* to sample, with the
policy's π(x | D, ξ) rendered as advice next to an uncertainty-sampling
marker; "Follow policy" unrolls the episode; the goal selector is live,
including mid-episode switches. Local-only: not in the deploy workflow; the
5k validation checkpoint serves it until the longer fine-tune lands (swap =
re-run export + parity together).

Recorded TS deviations / equivalences (all parity-pinned):

- **Final-state reads come from the inherited TS forward**, not a re-run
  block loop: the base port already returns per-layer stacks, so
  `forwardWithStates` is `ctxLayers[L-1]` plus `final_norm` re-applied to
  `tgtLayers[L-1]` — the exact values Python's `forward_with_states` returns
  (the policy fixture asserts the states directly).
- **Omission replaces masking, exactly.** Target tokens never attend to each
  other and candidates are scored pointwise, so the TS side builds only the
  active goal tokens and only the available candidates; ξ is "which target
  rows exist", and the policy reads a goal-row *slice* of one shared forward
  that also carries band, all-three-latent, and US-candidate rows (row
  independence makes the extra rows invisible; the all-three-latent rows are
  a tab refinement so marginals and a ξ-independent log q(θ_true) metric
  survive goal switches).
- **The chain fixture is teacher-forced**: the TS test replays Python's
  recorded argmax actions and asserts logits/log-probs, so an fp near-tie
  cannot flip an action and fail the build (agreement is reported instead).
- **The env's RNG streams are not parity-matched** (mulberry32 vs torch);
  the fixtures pin the deterministic math (K, Cholesky) instead, and the
  teacher-forced chain carries Python-drawn values.

Eyeball notes on the 5k policy (honest reading, not a gate): the loop works
end-to-end at ~150–200 ms/step (Edge, this workstation); log q(θ_true) and
RMSE improve over an episode; and goal-dependent placement is *visible* —
under ξ = ℓ the policy concentrates its mass immediately beside already-
acquired points (the tight-local-pairs strategy lengthscale information
wants), clearly separated from the uncertainty-sampling pick, while under
ξ = predictive it spreads toward coverage. Consistent with the 5k evaluation:
placement differs by goal even though the measured log-q contrasts are still
null at this budget. The tab's didactics were not tuned to this checkpoint.

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
  eval-only change, queued as fallback #4. *Implemented same day*: `evaluate`
  now reports both instruments (acquire under ξ={kernel} vs ξ=pred on identical
  episodes, score `log q(kernel_true | D_T)` through the categorical head). On
  the 5k model the kernel contrast is **also null** (matched −0.272 vs
  mismatched −0.266, δ = −0.006; all other metrics reproduce digit-for-digit),
  so at this budget the missing contrast is not instrument-specific —
  consistent with undertraining being the dominant factor, with the
  task-structure hypothesis still open for ℓ specifically.
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
- Second experiment family (psychometric / BED) = the pre-agreed graduation
  trigger to a separate nanoALINE repo, not this extension.
