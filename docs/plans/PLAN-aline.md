# Plan: ALINE extension (`extensions/aline/`)

Created: 2026-06-12
Status: IMPLEMENTED (Phases 1–4 + Phase 6 docs done; Phase 5 validation iterating —
5k run complete, results in `extensions/aline/DEVLOG.md`, longer fine-tune pending).
A playground tab is planned separately:
[PLAN-aline-playground.md](PLAN-aline-playground.md) (its Phase 1 closed this plan's
Phase 6 doc items, 2026-06-12).

Reference: Huang, Wen, Bharti, Kaski & Acerbi (2025), *ALINE: Joint Amortization for
Bayesian Inference and Active Data Acquisition* (NeurIPS 2025). Paper markdown in
`extensions/aline/paper/` (§3.1–3.4 objectives/architecture, Algorithm 1 + B.2 training/embedding
details, Fig. A1 attention masks, §4.1/C.1 the GP active-learning experiment this
extension reproduces). Design discussion resolved 2026-06-12 (recorded below and in the
extension's local DEVLOG once created).

## Progress tracker

Phase 1 — model (`extensions/aline/aline.py`):  [DONE 2026-06-12]
- [x] `extensions/aline/` skeleton + repo-root `sys.path` bootstrap (arbuffer pattern)
- [x] `PolicyBlock` (pre-LN residual: query→context cross-attn, query→target cross-attn,
      MLP; own kv LayerNorms; `need_weights=False`; no query–query self-attention)
- [x] `ALINE(ACE)`: `policy_blocks` ModuleList + `policy_norm` + `policy_head`;
      `forward_with_states` (core block loop kept verbatim, returns final ctx/tgt states
      alongside `Predictions`); `policy_logits(query, C, G, masks)` with the gradient
      firewall (no-grad query embedding, detached C/G); `base_parameters()` /
      `policy_parameters()` + partition assert (note: `POLICY_PREFIXES` also covers the
      added `policy_norm.`)
- [x] `load_warm_start(path, device, variables, check_batch)` with the arbuffer-style
      strict guard (`unexpected == []`; missing keys all under `POLICY_PREFIXES`)
      + step-0 bitwise parity assert vs a freshly loaded base `ACE`
- [x] `load_aline_checkpoint(path, device, variables)` (policy depth inferred from
      state-dict keys, arbuffer's read-mode-inference pattern)
- [x] verify (`artifacts/scratch_aline_phase1.py`, CPU + CUDA, all pass): bitwise parity
      of plain forward and `forward_with_states` vs base `ACE` (fixed-eval + random
      online-style batches); missing-key guard trips; checkpoint round-trip with depth
      inference; gradient isolation both directions (PG → all-and-only `policy_*`;
      NLL → base only); masked candidates get `-inf` logits, zero probability, never
      argmax-selected

Phase 2 — episode environment (`extensions/aline/gp1d_aline.py`, env half):  [DONE 2026-06-12]
- [x] episode physics: `gp1d.draw_instances(b, n_points=pool+M)` (one joint CPU-float64
      draw; pool observations become lookups), gp1d hyperprior unchanged
- [x] `sample_xi(b)`: 50% predictive / 50% parameter rows; parameter subsets via
      `sample_reveal_mask(3, b, q=0.0)` (the non-empty subset/count mixture)
- [x] `assemble_episode(...)`: superset target Tokens `[ℓ_ell, ℓ_scale, ℓ_kernel,
      x*_1..M]` with truth in `value`/`value_index` and ξ as the per-row mask; query
      Tokens = data-`y` QUERY tokens at pool locations; preallocated context block of
      width `1 + T` (seed point active, columns flipped on as queries land); pool
      availability mask (note: `observe` replaces `query.mask` *functionally* — the
      deferred PG graph holds references to earlier masks, in-place scatter would
      corrupt it)
- [x] `rollout(...)`: T+1 forwards (states D_0..D_T); per-step NLL backward (prediction
      phases; trunk graphs freed per step), rewards from consecutive detached
      log-probs, ONE deferred PG backward at episode end (policy-side graphs only;
      covers both immediate and reward-to-go weighting + on-policy row masking for
      `--random-frac` rows), drivers {policy, argmax, random, us}, optional `on_step`
      hook (ξ-switch demo), `track_predictions` for eval curves
- [x] verify (`artifacts/scratch_aline_phase23.py`, CPU + CUDA, all pass): telescoping
      identity exact (max |diff| 0.0, per row, raw rewards); ξ-mask inertness (corrupted
      masked targets: bitwise-equal predictions, policy logits, gated log-probs); pool
      bookkeeping (distinct actions, exact availability mask, full context); mid-rollout
      ξ switch applies; goal-sensitivity probe: policy logits respond to ξ (~0.17 logit
      shift at random init); random-policy NLL at warm-started gp1d level (~0.0–0.25 on
      the 10-step CUDA smoke, vs ~0.9–1.1 from scratch)

Phase 2.5 — found during implementation (recorded):
- [x] autograd version-counter hazard: deferred PG backward + in-place `query.mask`
      mutation ⇒ functional mask replacement in `observe` (comment in code)
- [x] cosine LR multiplier is exactly 0 on a run's final step (core `_build_scheduler`
      convention) — harmless for real runs, but 1-step test runs train at LR 0; the
      scratch's phase-isolation checks use `--lr-schedule constant`

Phase 3 — training loop + CLI (`gp1d_aline.py`, training half):  [DONE 2026-06-12]
- [x] `fit_episodes(model, args)`: alternating phase schedule (prediction-only prefix of
      `--warmup-pred-steps`, then step-parity alternation, policy first), `mix_seed`
      reseed per step, two Adam optimizers with per-group cosine schedules
      (`phase_counts` gives each group its own `T_max`; reuses `train._build_scheduler`)
      and per-group grad clip, one optimizer step per episode batch, batch-mean reward
      baseline, `--reward-to-go`, `--update-mode {alternate,simultaneous}`,
      `--freeze-base`, phase-appropriate no-grad, logging incl. s/step, resumable
      checkpoints (`save_resumable`: house payload + `optimizer_policy` /
      `scheduler_policy` additive keys; base pair restored via `train.load_train_state`)
- [x] CLI on `train.common_parser()` + `set_defaults` (gp1d model defaults) + task
      flags as planned; `main`: warm start (strict guard + step-0 parity vs
      `gp1d.fixed_eval_batch`) / from-scratch / resume / eval-only; inherited no-op
      flags documented in `parse_args` (and `--warmup` honored as per-group LR warmup)
- [x] verify: smoke runs all pass — CPU from-scratch 20-step (full main: fit + eval +
      save + plot), CUDA warm-started 10-step at full defaults (step-0 parity vs
      `artifacts/gp1d.pt` OK; positive rewards ~+0.07; NLL at gp1d level), CUDA
      from-scratch 6-step; same-seed reproducibility max|dW| = 0; resume-exactness
      (6 straight == 3 + resume-to-6) max|dW| = 0 with both optimizers restored;
      `--eval-only --load-checkpoint` reproduces the training run's eval metrics
      digit-for-digit

Phase 4 — evaluation, diagnostics, plot:  [DONE 2026-06-12]
- [x] `evaluate(model, args)`: held-out sweep at `EVAL_SEED` (identical physics across
      drivers per sweep) — (a) predictive RMSE-vs-t for aline/random/us, (b) parameter
      `log q(θ_true)`-vs-t for aline/random, (c) targeting contrast (acquire under
      ξ=ell vs ξ=pred on identical episodes, score `log q(ell_true|D_T)`),
      (d) oracle calibration via `acquired_gpbatch` adapter → `gp1d.gp_oracle` +
      `diagnostics.query_log_density` + `gp1d.kernel_posterior`, (e) reward stats by
      target type; compact printed metrics (house style)
- [x] US baseline: one extra forward with the pool as target tokens, scored by
      `Predictions.continuous_var()`
- [x] fixed diagnostic figure: gp1d EVAL_* demo function (pool + linspace targets),
      six panels — predictive band before/after + query order, query placement by goal,
      ell marginal vs oracle, RMSE curves, log-q curves, contrast bars; argmax actions
      so a checkpoint regenerates the same figure
- [x] runtime ξ-switch demo (mask flip at T/2 via the rollout `on_step` hook; plotted
      in the placement panel — demo, not a gate)
- [x] verify: `--eval-only` rerun reproduces all printed metrics digit-for-digit and
      regenerates the figure; six panels render correctly on the smoke model
      (untrained flatness expected); goal-sensitivity of placement is a Phase-5
      training outcome (logits respond to ξ at init, argmax ties until trained)

Phase 4.5 — implementation `/doublecheck` (2026-06-12; three Opus reviewers: RL
machinery / model + plan fidelity + tracker accuracy / eval + core APIs):
- [x] no blockers. Verified correct: rollout indexing (no off-by-one; final forward
      action-free), autograd safety in all three phase modes (per-step NLL backward
      precedes mutation; deferred PG graphs reference only no-grad embeds, detached
      states, and functionally-replaced masks), exact gradient firewall, phase /
      per-group-scheduler / zero-grad semantics, (seed, step) RNG purity, token
      layouts vs `gp1d.assemble`, `GPBatch`/oracle adapter field order, eval
      cross-driver episode identity, param surface (~0.4M policy on 1.2M base,
      base bit-identical), tracker claims substantiated, module docstring's
      reference-implementation claims re-corroborated against `temp/aline-repo/`
- [x] fixed from findings: resume restores `warmup_pred_steps` from the checkpoint
      `config` (re-deriving the default silently changed a resumed from-scratch
      run's phase schedule) + warns on `update_mode`/`freeze_base`/`random_frac`
      mismatch; PG loss normalized by the effective on-policy episode count (a
      plain batch mean halved the policy gradient under `simultaneous` +
      `random_frac`); printed note when `--steps <= --warmup-pred-steps` (no policy
      steps); `--episode-steps < --pool-size` guard; `policy_logits` non-empty-row
      precondition documented; ξ-switch demo placements-only comment; oracle CLI
      knobs added to this plan's flag list
- [x] re-verified after fixes: phase-2/3 scratch all-pass (reproducibility and
      resume-exactness still max|dW| = 0); CLI resume restores the from-scratch
      schedule (note fires, no spurious warning)

Phase 5 — validation run + gates:
- [~] validation fine-tune from the retained GP-1D 200k checkpoint (CUDA; ~10–20k
      episode-batch steps ≈ a few hours, see Risks — budget trimmed by the smoke
      measurement). MEASURED 2026-06-12: ~1.06 s/step at full defaults (B=64, pool 128,
      T=16, RTX 4060 laptop) ⇒ 5k ≈ 1.5 h, 10k ≈ 3 h, 20k ≈ 6 h. First run: 5k
      steps, completed 2026-06-12 — **results, gate readings, and interpretation
      live in `extensions/aline/DEVLOG.md`** (plan files track plan and progress
      only; results belong to DEVLOGs, a correction of the arbuffer-plan habit).
      Outcome in one line: policy learns (beats random both ways, calibration
      kept); US gap and ℓ-contrast still open; next per fallback order = longer
      run, same recipe. Gate criteria:
      learned policy dominates random on predictive RMSE; competitive with US;
      parameter-ξ `log q(θ_true)` learned > random; targeting contrast positive;
      oracle calibration on acquired contexts comparable to the fixed-case gp1d
      calibration; reward histograms unremarkable
- [ ] iterate via pre-registered fallbacks only (see Risks); record outcomes here
- [ ] retained-artifact run deferred (run separately once the recipe validates,
      arbuffer pattern)

Phase 6 — docs:
- [x] `extensions/aline/README.md` (what/why, the ACE-native thesis, reuse story,
      recipe, commands, honest performance notes) [DONE 2026-06-12]
- [x] `extensions/aline/DEVLOG.md` (created 2026-06-12, ahead of the rest of Phase 6:
      design decisions + deviations + dated run entries; all results and
      interpretation live there, not here)
- [x] root `DEVLOG.md`: short dated pointer entry [DONE 2026-06-12]
- [x] root `README.md`: one Current Status bullet; `AGENTS.md`: extensions list mention
      [DONE 2026-06-12]
- [x] update this plan's Status + tracker [DONE 2026-06-12]; `/doublecheck`: the
      implementation had the Phase 4.5 three-reviewer pass; the doc surface gets a
      docs-focused pass under PLAN-aline-playground.md Phase 1

## Summary

Add ALINE — joint amortized Bayesian inference and active data acquisition — as a
**non-core extension** in `extensions/aline/`, on the GP-1D active-learning task
(the paper's §4.1), warm-started from the retained GP-1D checkpoint. ALINE's inference
network *is* the core `ACE` unchanged: parameter targets are latent QUERY tokens,
predictive targets are data QUERY tokens, and the paper's target specifier ξ collapses
into **which target tokens are active** (a per-row mask over a fixed superset layout).
Query candidates are "hypothetical targets" — data QUERY tokens at candidate locations,
embedded by the core embedder verbatim. The only new model parts are a small read-only
**policy decoder** (cross-attention from candidates to the trunk's final context and
target states, detached) and a per-candidate scoring head; the only new training code is
an episode-rollout RL loop (REINFORCE on self-estimated information gain, the paper's
Eq. 10–12) that `train.fit` cannot express.

The thesis worth recording: the target tokens do double duty — they are simultaneously
the queries `q_φ` answers (NLL + reward are computed on them) and the goal
representation the policy reads (their final states encode both *which* goal is active
and *how uncertain the model still is about it*). That is the ACE-native re-expression
of ALINE's separate target-specifier/selective-mask apparatus, and it is why the
extension is thin.

## Scope

- **In scope**
  - `extensions/aline/aline.py`: `PolicyBlock`, `ALINE(ACE)` (states-forward, policy
    logits, gradient firewall), warm-start loader + strict guard + step-0 parity,
    checkpoint loader.
  - `extensions/aline/gp1d_aline.py`: episode environment (GP physics via `gp1d`),
    ξ sampler, rollout, alternating RL training loop, evaluation with random/US
    baselines and oracle calibration, fixed diagnostic plot, CLI.
  - Local `README.md` + `DEVLOG.md`; **pointer entries only** in root docs.
- **Out of scope** (recorded as decisions, not gaps)
  - Any change to `ace.py`, `train.py`, `gp1d.py`, `data.py`, `diagnostics.py` (zero
    core edits; the extension imports them).
  - The paper's other experiment families: BED benchmarks (Location Finding, CES) and
    the psychometric task — adding a second family is the pre-agreed graduation
    trigger to a separate nanoALINE repo.
  - Continuous design spaces (pool-based only, as the paper), autoregressive joint
    posteriors, sPCE/EIG-bound evaluation machinery, OOD benchmark functions
    (Higdon/Branin/...), offline `data.py` pool support (online Cholesky at 160
    points is the arbuffer cost class), playground tab, runtime prior (ACEP) tokens —
    the "ALINE + Beta prior tokens" synthesis goes in the local DEVLOG as the natural
    follow-up this co-location enables, not v1.
  - Time/budget embedding for the policy (stationary policy, as the paper: the
    released configs set `time_token: False`; an `EncoderWithTime` variant exists
    unused) and an entropy bonus (pre-registered fallback only).

## Design decisions (seed of the local DEVLOG)

1. **Inference network = core ACE, bit-identical.** `ALINE(ACE).forward_with_states`
   re-runs the core block loop verbatim (same modules, same order, `need_weights=False`)
   and additionally returns the final context/target states. Parity is guarded
   *permanently*: predictions must equal `ACE.forward` bitwise — so `gp1d.evaluate`
   and `gp_oracle` diagnostics stay valid for ALINE artifacts, and core drift fails
   loudly (arbuffer's coupling guard, strengthened from step-0-only to always).
2. **Read-only policy decoder, no write-back, no query self-attention.** Candidates are
   embedded by the core `_embed` (data-`y`, mode QUERY, `x` = candidate location), then
   `P = 2` `PolicyBlock`s cross-attend to the **detached** final states `(C, G)` and a
   linear head scores each candidate; softmax over the remaining pool. Pointwise
   scoring matches acquisition-function semantics (the softmax supplies the
   competition) — and the reference implementation: in the released code
   (`model/encoder.py:create_mask`; clone in gitignored `temp/aline-repo/`, checked
   2026-06-12) query–query entries stay masked at `-inf`, so omitting query
   self-attention is **not** a deviation (the paper's Fig. A1 rendering is misleading
   on this). **Equivalence note:**
   per-layer interleaving without write-back (the paper's wiring, read-only) is exactly
   this decoder reading per-layer caches instead of final states — the query stream
   hoists out of the loop because the trunk never depends on it. So the upgrade path
   (below) is a config change, not an architecture change. Write-back is ruled out: it
   would break the parity invariant and the gradient firewall — and the reference
   implementation confirms there is none. Its `create_mask` unmasks exactly
   `mask[:, :num_context]` plus query→ξ-selected-target columns: context rows attend
   only to context, target rows only to context (no target–target either — nanoACE's
   invariant), and nothing attends to queries. The paper's Fig. A1(a) rendering that
   shades context rows over all columns contradicts its own code; we follow the
   code/text. The reference is therefore literally a read-only query/target stream
   over per-layer context states with shared weights — the hoisting equivalence,
   confirmed concretely. Our two remaining real differences, both deliberate: separate
   policy weights instead of their single shared-weight masked stream (what makes the
   φ/ψ separation structural), and final-state reads by default (the per-layer-cache
   upgrade path below is exactly the reference's read pattern, modulo weight sharing).
3. **The gradient firewall is exact.** Query embeddings are computed under `no_grad`
   and `(C, G)` are detached, so the PG loss can only update `policy_blocks` +
   `policy_head` (ψ) and the NLL can only update the base (φ) — the paper's "policy
   gradients are not propagated back to the inference network" made structural.
4. **ξ = target-set composition.** Fixed superset target layout
   `[ℓ_ell, ℓ_scale, ℓ_kernel, x*_1..M]`; ξ is the per-row target mask. Masked goal
   tokens are invisible everywhere by existing machinery (embedding zeroed, excluded
   from the loss/reward average, `key_padding_mask` in the policy's query→target
   read). `p(ξ)`: 50% predictive (`x*` drawn `2·rand−1`, exactly the
   `draw_instances` data-x distribution, fresh per episode), 50% parameter
   with the subset drawn by `sample_reveal_mask(3, B, q=0.0)` — the tested non-empty
   subset/count mixture, so singletons, pairs, and all-three are all in-distribution
   (what the targeting-contrast gate needs). Either-or per row, as the paper; novel
   combinations and mid-episode switches are mask flips at runtime. The discrete
   `kernel` as a parameter target is a small genuine extension over the paper's
   continuous-only GP experiment and exercises the categorical path (house rule).
   The reference implementation realizes ξ the same way — a fixed target superset
   plus a boolean `target_mask` restricting which target positions queries may attend
   (`utils/target_mask.py`; `mask_type: none` is even labeled "ACE case") — but with
   one ξ per batch; our per-row ξ generalizes it.
5. **Episode environment.** One `gp1d.draw_gp` call per episode batch on the union of
   `N_pool = 128` candidate and `M = 32` predictive-target locations (one CPU float64
   Cholesky at 160 points per instance — ~25% above arbuffer's 128-point draws, same
   cost class); gp1d's kernel/hyperparameter priors unchanged; pool/`x*` locations
   drawn `2·rand−1`. **Noiseless** observations (gp1d physics purity; pool-based
   selection without replacement makes noise non-essential; the MDN `min_scale` floor
   bounds predictive log-densities, so rewards stay bounded — `--sigma-obs` at
   observation-lookup time is the pre-registered one-line knob if predictive-target
   rewards turn heavy-tailed). Episodes start from **1 random seed point** (paper
   protocol; also satisfies ACE's ≥1-active-context requirement), `T = 16` steps
   (default; context ≤ 17 stays inside the warm-start's trained `n_context ≤ 20`
   range — keep `T ≤ 19`), `B = 64` episodes per step.
6. **Warm start is the default; joint fine-tune.** `--base-checkpoint` loads the
   retained GP-1D model under the arbuffer strict guard (`unexpected == []`, missing
   keys all under the policy module) with a step-0 bitwise parity assert. Training is
   **joint** (not frozen-base): policy-selected contexts are informativeness-shaped,
   not uniform-random, so `q_φ` must adapt (Prop. 2's KL gap is the reward-bound
   tightness). From-scratch mode (no `--base-checkpoint`) recovers the paper's
   schedule via a large warm-up prefix. Provenance: the base checkpoint is a local
   gitignored artifact — use the retained GP-1D 200k run where present, else
   regenerate (`python gp1d.py --save-checkpoint artifacts/gp1d.pt`); the validation
   gates assume it exists. `main()` passes `gp1d.fixed_eval_batch(...).batch` as the
   warm-start `check_batch` (arbuffer's pattern).
7. **Alternating phase schedule (deliberate variant of Algorithm 1).** Steps alternate:
   *prediction steps* (NLL → φ; policy frozen; rollouts driven 50% by the current
   policy, 50% random — the on-policy half adapts `q_φ` to policy-shaped contexts, the
   random half preserves the random-context calibration that keeps `gp1d.evaluate` /
   oracle checks meaningful) and *policy steps* (PG → ψ; φ frozen — stationary reward
   within the phase). Warm-up = a prediction-only prefix (`--warmup-pred-steps`, counted
   in training steps = episode batches; default 0 with warm start — the freshly
   initialized policy head gives near-uniform logits, so early policy rollouts are
   *approximately* (not exactly) the paper's random warm-up actions, with
   `--random-frac` covering the rest — large default for from-scratch). The paper's
   Algorithm 1 actually applies both losses together at every inner update with
   gradient-level separation; that paper-literal scheme is kept as
   `--update-mode simultaneous` (same code path, both backwards enabled) and is the
   fallback if alternation's ~2× episodes-per-update-pair cost bites.
8. **One optimizer step per episode batch (deviation from Algorithm 1's in-loop
   updates).** Per-step `backward()` during the rollout keeps memory flat; gradients
   accumulate; each of the two Adam optimizers steps once per batch in its phase.
   Rationale: (a) *reward purity* — with parameters frozen across the episode,
   `R_t = log q_t − log q_{t−1}` measures information gained from the new point, not
   optimizer movement conflated in; (b) lower-variance averaged updates; (c) the repo's
   training-spine semantics survive (step = batch; `mix_seed(seed, step)` makes each
   episode batch — policy sampling included — a pure function of `(seed, step)`;
   cosine/resume coherent). Schedulers tick only when their optimizer steps, and each
    group's cosine `T_max` is **that group's expected optimizer-step count**
    (≈ `steps/2` per group under alternation, `steps` under `simultaneous`), so the
    same `--steps` reaches the same end-of-run LR floor in both modes; the inherited
    `--warmup` keeps its core meaning as per-group linear LR warmup (not a no-op).
9. **Rewards and losses.** `R_t` = per-row mean over active targets of Δ`log q`
   (paper Eq. 10), token coordinates (Jacobians are constants that cancel in
   differences; NLL likewise token-space, plain mean over active targets per Eq. 12 —
   the inherited `--latent-weight`/`--data-weight` are no-ops here, documented).
   T+1 forwards per episode (D_0..D_T): NLL over t = 1..T, rewards from consecutive
   detached log-probs, PG term for action `a_t` applied one step delayed (when `R_t`
   becomes available; only the small policy-side graph is retained across that step).
   REINFORCE with immediate per-step reward and γ = 1 (paper); **batch-mean reward
   baseline** subtracted by default (`--no-baseline` to disable; standard same-batch
   mean, O(1/B) bias, negligible at B = 64); `--reward-to-go`
   (weight `log π_t` by `Σ_{t'≥t} R_{t'}`; the rewards telescope to the total
   improvement, so this is the unbiased credit assignment) off by default,
   paper-faithful. The telescoping identity used for verification is **per row** and
   on **raw rewards** (before baseline or reward-to-go transforms).
10. **Action selection.** Stochastic sampling from the masked softmax during training
    (the exploration); **argmax at evaluation** so fixed diagnostics are deterministic
    and a checkpoint regenerates its figure (house comparability convention).
11. **Baselines from the same model.** Random policy, and uncertainty sampling scored
    by ALINE's own `continuous_var()` at the remaining candidates (one extra forward
    with the pool as target tokens) — the paper's ACE-US baseline, nearly free here.
12. **Checkpoints stay in the house format.** `train.save_checkpoint` payloads
    (`{cfg, seed, state_dict}` + `config` provenance); policy depth inferred from
    state-dict keys on load (arbuffer's pattern); resume adds the second
    optimizer/scheduler under **new** additive keys (`optimizer_policy`,
    `scheduler_policy`) via local save/load helpers — `train.load_train_state`
    handles only the base pair and must not be pointed at both.

## Architecture spec

### `aline.py`

- `PolicyBlock(nn.Module)`: pre-LN residual ops mirroring `ACEBlock` conventions —
  `q_ln`/`kv_ln` LayerNorms, `nn.MultiheadAttention(d, heads, batch_first=True)` ×2
  (query→C, query→G; `key_padding_mask` from context/target masks;
  `need_weights=False`), then LN + `_mlp`-style MLP. Width = `d_model` (128 at gp1d
  defaults; 2 blocks ≈ 0.4M params on the 1.2M base).
- `ALINE(ACE)`:
  - `__init__(variables, cfg, n_policy_blocks=2)`; `policy_head = nn.Linear(d, 1)`.
  - `forward_with_states(batch) -> (Predictions, ctx_states, tgt_states)` — the core
    loop, returning final-layer states (raw, pre-head; policy blocks own their kv LNs).
    `forward()` stays inherited (used by parity checks and the US baseline).
  - `policy_logits(query: Tokens, C, G, ctx_mask, tgt_mask) -> [B, N_q]` — no-grad
    `_embed(query)`, detached C/G, policy blocks, head; `-inf` on unavailable
    candidates (pool mask).
  - `base_parameters()` / `policy_parameters()` (prefix split; asserts the two sets
    partition `named_parameters()`).
- `load_warm_start(path, device, variables, *, n_policy_blocks, check_batch=None)` —
  strict guard + step-0 bitwise assert (vs a freshly loaded base `ACE`), mirroring
  `arbuffer.load_warm_start` / `check_step0`.
- `load_aline_checkpoint(path, device, variables)` — strict load, policy depth
  inferred from `policy_blocks.{i}.` keys.

### `gp1d_aline.py`

- Repo-root `sys.path` bootstrap (`parents[2]`), arbuffer header style.
- `Episode` dataclass: `variables`, context-block tensors (width `1 + T`), `target:
  Tokens` (superset + ξ masks; truths filled), `query: Tokens` (pool), pool
  availability mask, truths (`log_ell`, `log_scale`, `kernel`, `y` at pool and `x*`).
- Environment: `draw_episode_physics`, `sample_xi`, `assemble_episode` (token layout
  per gp1d `assemble` conventions: `encode_value` for latent truths, `make_tokens`).
  Observation = lookup of the pre-simulated pool `y` (noiseless; `--sigma-obs` adds
  noise at lookup if ever enabled).
- `rollout(model, episode, *, driver, train_nll, train_pg, baseline, ...)` — the
  T+1-forward loop described in Design 9; drivers: `policy`(sample), `argmax`,
  `random`, `us`. Trunk runs `no_grad` when `train_nll=False`; policy module `no_grad`
  when `train_pg=False`.
- `fit_episodes(model, args)` — Design 7/8 schedule; logging
  `step / phase / nll / mean_reward`; periodic resumable checkpoints.
- `evaluate(model, args)` + `plot_diagnostic(...)` — Phase 4 items; fixed eval seed
  (module constant, `EVAL_SEED` style); oracle calibration reuses `gp1d.gp_oracle`
  on acquired contexts via a small acquired-context → `GPBatch` adapter.
- `parse_args()` on `train.common_parser()`; `set_defaults(batch_size=64, d_model=128,
  heads=4, layers=4, hidden=256, components=8, plot_path="artifacts/gp1d_aline.png")`;
  task flags: `--base-checkpoint` (default `""` → from-scratch), `--episode-steps 16`,
  `--pool-size 128`, `--pred-targets 32`, `--policy-blocks 2`, `--policy-lr`
  (default = `--lr`), `--update-mode {alternate,simultaneous}`, `--warmup-pred-steps`
  (training steps = episode batches; default 0 warm-start / 2000 from-scratch),
  `--random-frac 0.5`, `--freeze-base` (the Risks escape hatch: skip prediction
  phases), `--no-baseline`, `--reward-to-go`, `--sigma-obs 0.0`,
  `--eval-episodes 128`, `--jitter GEN_JITTER`.
  Inherited no-ops documented (`--max-context`, `--min-context`, `--data-targets`,
  `--latent-context-prob`, `--latent-weight`, `--data-weight`); inherited `--warmup`
  is NOT a no-op (per-group linear LR warmup, Design 8). Rollout drivers: `policy`
  (sampled) and `random` during training (Design 7); `argmax`, `random`, and `us`
  at evaluation only (Designs 10–11). Oracle-calibration knobs mirror gp1d:
  `--oracle-episodes 4`, `--oracle-bins 64`, `--oracle-chunk 512`.
- Artifacts: `artifacts/gp1d_aline.pt`, `artifacts/gp1d_aline.png`.

## Verification

Mechanical (phase-gated, see tracker): bitwise trunk parity; warm-start key guard;
gradient isolation (including: no base/embedder grads in policy phases); reward
telescoping identity (per row, raw pre-baseline rewards); ξ-mask inertness;
pool-removal exactness; same-seed reproducibility (max|dW| = 0); smoke runs CPU + CUDA (warm-start
and from-scratch); `--eval-only` round-trip; resume restores both optimizers.

Structural gates (validation run, all falsifiable):
- predictive ξ: learned policy RMSE-vs-t **dominates random** beyond the first few
  steps and is **competitive with US** (the paper's Fig. 3 in-distribution analogue);
- parameter ξ: `log q(θ_true)`-vs-t learned **> random** (Fig. 4 analogue);
- **targeting contrast** (the headline): improvement of `log q(θ_S)` is larger under
  matched ξ than under mismatched ξ, and query placement visibly differs by goal
  (Fig. 5 analogue); mid-episode ξ switch shifts placement (D.4 analogue);
- **oracle calibration**: on policy-acquired contexts, ALINE hyperparameter/kernel
  marginals track `gp1d.gp_oracle` comparably to the fixed-case gp1d diagnostic —
  this is the exact-up-to-grid reference the paper's setup lacked, and the arbiter
  of the joint-fine-tune-vs-drift question;
- reward histograms by target type show no pathological tails (the noiseless check).

Treat the gates as the loose house convention: structural reading, not strict
thresholds (DEVLOG "Open questions" spirit).

## Risks / fallbacks (pre-registered)

- **Policy can't beat US** → switch the decoder to per-layer cache reads (Design 2's
  equivalence makes this a config change: block `i` reads cache `i`, `P = n_layers`),
  then deeper/wider policy. Only then revisit topology.
- **High PG variance / early collapse** → `--reward-to-go` on; entropy bonus flag
  (add only if needed); larger `B`.
- **`q_φ` calibration drift under joint fine-tune** → raise `--random-frac`; a
  `--freeze-base` escape hatch (skip prediction phases) restores the frozen-base
  regime and the eternal-parity property if the oracle check demands it.
- **Predictive-reward heavy tails (noiseless)** → `--sigma-obs 0.01` at lookup
  (doesn't touch the GP draw).
- **Alternation too slow** (2× episodes per update pair) → `--update-mode simultaneous`
  (paper-literal).
- **Wall-clock**: a step costs (T+1) = 17 *sequential* forwards (~180 active tokens
  each at B = 64) plus the backward work — roughly **17× arbuffer's one-forward
  steps**, so 10–20k steps lands in the **few-hours** class on the 4060, not
  arbuffer's ~35 min. Measure at smoke time before committing a budget; the levers
  are `--steps`, `T`, `B`, and `--update-mode simultaneous` (halves the episodes per
  effective update pair).

## Documentation

- `extensions/aline/README.md`: what/why, the double-duty-targets thesis, reuse list,
  warm-start recipe, run commands, honest notes (alternating variant, immediate-reward
  credit, noiseless choice).
- `extensions/aline/DEVLOG.md`: Design decisions above + dated deviations from the
  paper (alternating schedule vs Algorithm 1; accumulated update; no query
  self-attention; read-only decoupled policy + the hoisting equivalence; separate
  policy weights vs shared stream; noiseless; stationary policy; token-space NLL)
  and the follow-up ideas (ALINE + ACEP priors; per-layer reads; psychometric task =
  graduation trigger).
- Root `DEVLOG.md` dated pointer entry; root `README.md` Current Status bullet;
  `AGENTS.md` extensions mention. This plan: status + tracker updates as phases land.

## Open questions

None blocking — remaining knobs are explicitly deferred to Phase 5 with defaults set
here: `T = 16` (bound `≤ 19`), `--warmup-pred-steps` 0/2000 (warm/scratch),
`--policy-lr = --lr`, validation budget 10–20k steps (recheck against the measured
per-step wall-clock, see Risks). Tune by gate evidence, not preference; record
outcomes in the tracker.
