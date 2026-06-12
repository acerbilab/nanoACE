# Plan: ALINE active-learning playground tab

Created: 2026-06-12
Status: COMPLETE (2026-06-12; all six phases done — plan doublecheck with fixes
pre-approval, docs-focused pass after Phase 1, full verification in Phase 6.
Remaining follow-up lives outside this plan: the longer fine-tune, then the
mechanical weights swap + deploy recorded under Out-of-scope below).

Amended 2026-06-12 (post-completion): user-facing copy rewritten (one-line hint;
the explainer grounds the demo setup and drops internal project-status talk).
Goal-selector semantics changed: plain click SWITCHES the goal, shift-click
COMBINES parameter goals, and the untrained predictive+parameter mix is no longer
selectable — superseding Design 9's "mixed allowed but flagged" (the novel-combo
OOD flag and `goalIsNovelCombo` were removed with it; trained latent subsets,
including pairs and all-three, remain combinable).

Reference: Huang, Wen, Bharti, Kaski & Acerbi (2025), *ALINE: Joint Amortization for
Bayesian Inference and Active Data Acquisition* (NeurIPS 2025). Extension under
`extensions/aline/` (see `extensions/aline/DEVLOG.md` and `docs/plans/PLAN-aline.md`);
playground precedent: `docs/plans/PLAN-arbuffer-playground.md` (the AR-buffer tab is the
template for porting an extension model + animated stepping + local-only weights).

Interaction design resolved in discussion 2026-06-12 (recorded in Design decisions
below): two modalities — **ground-truth episode mode (default)**, where a hidden GP
draw answers queries and the user only chooses *where* to sample (following or ignoring
ALINE's advice; a "Follow policy" button auto-unrolls), and **user-as-oracle mode**,
where the user supplies both x and y. The policy distribution π(x | D, ξ) is always
rendered as advice; the goal selector ξ is live, including mid-episode switches.

**Checkpoint status (honesty note).** The tab is built **local-only** against the 5k
validation artifact `artifacts/gp1d_aline.pt` (policy beats random; US gap and
targeting contrasts still null at this budget — see the extension DEVLOG). The tab's
UX is designed for the eventual longer-trained checkpoint; we do **not** tune or judge
the tab's didactics against the 5k policy's behavior. Weight swap + fixture regen +
deploy happen later, after the user's longer fine-tune (arbuffer's swap pattern).

## Progress tracker

Phase 1 — extension docs (closes PLAN-aline Phase 6):  [DONE 2026-06-12; docs-focused
review pass run — two minor findings (parity-phrasing overstatement in the README
Boundary; these tracker ticks) fixed]
- [x] `extensions/aline/README.md` (arbuffer README shape: intro + the
      tokens-do-double-duty thesis + reuse story; the warm-start/joint-fine-tune
      recipe; Run commands incl. smoke/resume/eval-only; what the diagnostic shows;
      honest 5k performance notes; Boundary section)
- [x] root `README.md`: Current Status bullet for `extensions/aline/` (arbuffer bullet
      style; mention the planned tab once it lands, not before); `AGENTS.md`: extensions
      paragraph mention (inline, after arbuffer)
- [x] root `DEVLOG.md`: dated pointer entry for the extension (what/why in a few
      sentences; pointers to local DEVLOG + PLAN-aline; no results duplication)
- [x] `docs/plans/PLAN-aline.md`: status header updated (phases 1–4 + 6-docs state),
      Phase 6 items ticked as they land. Its trailing "`/doublecheck` pass" sub-item
      is owned here as a **docs-focused** pass over the Phase 1 surface (the
      implementation itself already had the Phase 4.5 three-reviewer pass)

Phase 2 — Python side: export task + parity/env fixtures:  [DONE 2026-06-12; all
in-script asserts passed (forward_with_states bit-equal to forward; manual policy
replication bit-equal to policy_logits); 178 tensors / 48 policy tensors exported,
byte-size check exact; pre-existing tracked fixtures byte-identical after the rerun]
- [x] `playground/export_weights.py`: add `"gp1d_aline": "extensions.aline.gp1d_aline"`
      to `TASK_MODULES` (the generic state-dict walk already serializes `policy_*`
      tensors; no manifest-field changes — model identity is key-presence, arbuffer
      precedent)
- [x] `playground/parity.py`: `ALINE_CKPT = artifacts/gp1d_aline.pt`; import
      `load_aline_checkpoint`; gated fixture block writing
      `test/fixtures/gp1d_aline.parity.json` with three sections —
      `plain` (reuse `gp_cases(aline_model)`: the inherited-forward invariant on the
      fine-tuned weights), `policy` (one compact all-active case: context + goal target
      set + candidate set; dumps final ctx states, final-normed tgt states, per-policy-
      block candidate states, logits; manual replication asserted against
      `model.policy_logits`), `chain` (compact teacher-forced episode — pool 16,
      grid 0, T = 6, mid-episode ξ switch at step 3, all-active tokens, Python-drawn
      pool/truths; per step: available candidates, logits, the recorded argmax action
      — the TS test *replays* recorded actions rather than re-deriving argmax — and
      per-goal-token log-probs taken from `Predictions.log_prob` — NOT `rollout()`'s
      ξ-averaged `log_q`, which is a different quantity)
- [x] `playground/parity.py`: environment fixture `test/fixtures/gp1d_aline.env.json`,
      written on every `parity.py` run — placed **before** the `ALINE_CKPT.exists()`
      gate (independent of the ALINE checkpoint; `parity.py` as a whole still loads
      the four core checkpoints first, as today): for fixed input x-vectors (one
      moderate, one adversarial: clustered x + small ℓ + periodic), each kernel's K
      matrix and Cholesky factor in float64, plus the DGP constants (ranges, jitter,
      period) so the TS env is pinned to `gp1d.draw_gp` exactly
- [x] **both new fixtures are committed to git** (`gp1d_aline.parity.json`,
      `gp1d_aline.env.json`; the blob dir stays gitignored) — the CI self-skip story
      and the always-running env suite depend on the fixtures being tracked, exactly
      as `gp1d_arbuffer.parity.json` is
- [x] export run: `python playground/export_weights.py --task gp1d_aline
      --checkpoint artifacts/gp1d_aline.pt --out playground/public/models/gp1d_aline`;
      sanity: manifest lists `policy_blocks.*`/`policy_norm.*`/`policy_head.*` tensors,
      `total_floats` × 2 == weights.bin bytes

Phase 3 — TS model port (`src/ace/aline.ts`):  [DONE 2026-06-12; all 4 suite tests
pass at the arbuffer tolerances with no tie-flip warnings; tsc --noEmit clean]
- [x] `class ALINEModel extends ACEModel`: constructor probes the sentinel tensor
      `policy_blocks.0.q_ln1.weight` (clear error otherwise: "not an ALINE export");
      `nPolicyBlocks` inferred from manifest tensor names
- [x] `forwardWithStates(context, target)`: calls the **inherited** `forward()`;
      `ctxStates = out.ctxLayers[L-1]`; `tgtStates = finalNorm(out.tgtLayers[L-1])`
      (re-applies `final_norm` with the blob's weights — exactly the states Python's
      `forward_with_states` returns; no base-port modification, no block-loop re-run)
- [x] `policyLogits(query, ctxStates, tgtStates)`: inherited `embed(query)`; per
      `PolicyBlock`: `kv = layerNorm(ctx, ctx_kv_ln)`, `multiHeadAttention(q_ln1(qry),
      kv, kv, …)`, residual; same for the target read via `tgt_kv_ln`/`q_ln2`; MLP via
      `q_ln3`; finally `linear(layerNorm(qry_i, policy_norm), policy_head)` per
      candidate. No masks anywhere (omission semantics, Design 4)
- [x] `src/aline/parity.test.ts`: `describe.skipIf(!HAVE)` (manifest + weights.bin +
      fixture, arbuffer pattern); asserts plain / policy (per-block) / chain sections;
      tolerances RAW `{atol: 3e-4, rtol: 1e-3}`, DERIVED `{atol: 1e-3, rtol: 1e-3}`
      (joint fine-tune ⇒ arbuffer's loosened-RAW precedent)

Phase 4 — TS environment (`src/aline/env.ts`):  [DONE 2026-06-12; K/L match torch
float64 at atol 1e-10 / rtol 1e-8 incl. the clustered cases; episode sampler
seed-deterministic]
- [x] kernel functions (RBF, Matern-1/2, Matern-3/2, Periodic with period = 1.0;
      replicate `_kernel_covariance`'s `ell.clamp_min(1e-6)` guard), covariance
      build + `1e-5` diagonal jitter, float64 Cholesky (net-new, ~20 lines),
      hyperprior sampling (`log_ell ~ U(ln 0.12, ln 0.80)`,
      `log_scale ~ U(ln 0.25, ln 1.00)`, kernel uniform in {0..3}), joint zero-mean
      draw `L @ z` at pool + grid locations via `rng.ts` (`mulberry32` + `randn`);
      noiseless (matches training)
- [x] `src/aline/env.test.ts` vs `gp1d_aline.env.json`: K and L match to ~1e-9
      relative (float64 both sides); **not** gated on the weights blob (fixture is
      tracked and checkpoint-independent)

Phase 5 — tab (`src/aline/infer.ts` + `demo.ts` + registration):  [DONE 2026-06-12;
full npm test 14 files / 37 tests green; ~170 ms per interaction step in the node
smoke run. One refinement over the plan: infer.ts always appends all three latent
rows (extra non-ξ rows are free by row independence), so latent marginals and a
ξ-independent log q(θ_true) metric stay available across goal switches — the policy
still reads only the ξ rows]
- [x] `infer.ts` (DOM-free): episode state (pool x/y, observed set, goal ξ, t, T,
      hidden truths); token builders (context VALUE tokens = observed points only;
      target = active goal tokens [selected latent QUERYs and/or 32 x\* QUERYs] +
      appended plot-grid band QUERYs + the available candidates appended again as
      data QUERY target rows for the US variance read; query = the same available
      candidates as the policy's query token set — one set of locations, dual-pathed:
      target rows give `continuousVar` for US, query tokens feed `policyLogits`);
      `alineStep(model, state)` → one `forwardWithStates` + `policyLogits`
      over goal-row slices → band, marginals, policy pmf + argmax, US pick (Design 7),
      metrics (Design 8); `applyObservation(state, idx)`; seeded episode determinism
- [x] `demo.ts`: plot via `plot.ts`; policy pmf along the bottom axis at candidate
      locations (BO `drawBottomDensity` precedent, peak-normalized) + argmax marker +
      US marker; goal chips (predictive ⊻ latent subset; mixed selection allowed,
      flagged "novel combination (untrained)"); controls: New function / Step /
      Follow policy (epoch-guarded rAF unroll, arbuffer animation pattern) / Reset /
      mode toggle (ground truth ↔ your own data) / Reveal truth toggle; step counter
      t/T; OOD warnings (Design 9); metric readout + mini-curve; "?" explainer modal
      (task / what ALINE does / vs classical active learning; Huang et al. 2025 +
      `aceFooter`); graceful missing-blob notice (arbuffer `mountArbuf` pattern)
- [x] registration: `index.html` sixth tab button + panel with DOM id `aline`
      (`data-tab="aline"`, `<section id="aline">` — short id like `arbuf`, while
      task/blob/fixture names stay `gp1d_aline`, mirroring the arbuf/`gp1d_arbuffer`
      split); `main.ts` mount block (`getElementById("aline")` → `mountAline`);
      `config.ts` `ALINE` block (pool 128, T 16, grid 64, M_pred 32, context hints
      17/20, Y_OOD, overlay amplitude)
- [x] tests: `demo.smoke.test.ts` (jsdom, fs-backed fetch stub, sync rAF; the smoke
      clicks Step rather than Follow policy — a synchronous 16-forward drain is too
      slow for a smoke test, and Step covers the same applyAction path; skip-guarded)
      + the missing-model notice test (NOT skip-guarded, arbuffer precedent)

Phase 6 — verification + docs wrap:  [DONE 2026-06-12]
- [x] full `npm test` green twice: with the local blob (14 files / 37 tests pass) and
      with it temporarily renamed away (33 pass / 4 skip — the env suite, the loader
      rejection, and the notice test all still run)
- [x] deploy safety: `.github/workflows/pages.yml` untouched (`git diff` clean on it);
      `gp1d_aline` absent from both hardcoded task lists; production `vite build` clean
- [x] eyeball the tab against the 5k weights (headless Edge via playwright-core,
      screenshots in `artifacts/aline_tab_*.png`): episode loop end-to-end at
      ~150–200 ms/step; goal-dependent placement VISIBLE (ξ=ℓ concentrates mass
      beside acquired points — tight local pairs — clearly separated from the US
      pick; ξ=pred spreads to coverage); metrics climb; the one console error is the
      pre-existing favicon 404. Honest notes recorded in the extension DEVLOG
- [x] docs: `playground/README.md` (sixth demo bullet + export command + local-only/
      self-skip note + parity/how-it-works sections); root `DEVLOG.md` dated entry
      (tab, local-only, pointer here); `extensions/aline/DEVLOG.md` dated entry
      (TS port deviations, Design 4/5 equivalences, eyeball notes); root
      `README.md`/`AGENTS.md` playground + extension mentions updated; this plan's
      tracker + status

## Scope

- **In scope**
  - Phase 1 extension docs (closing PLAN-aline Phase 6's remaining items).
  - `playground/export_weights.py` + `playground/parity.py` additions; one new model
    dir `playground/public/models/gp1d_aline/` (gitignored, local-only).
  - `playground/src/ace/aline.ts` (model port), `playground/src/aline/` (env, infer,
    demo, tests), registration in `index.html`/`main.ts`/`config.ts`.
  - Docs listed in Phase 6.
- **Out of scope** (decisions, not gaps)
  - **Deployment**: no `pages.yml` change, no weights-repo push, no public tab. The
    deploy + retained-weights swap is a follow-up after the longer fine-tune validates
    (procedure: re-export + regenerate fixtures together, push blob, add `gp1d_aline`
    to pages.yml's two lists, flip the README local-only note — recorded here so the
    swap is mechanical).
  - Any change to core files or to the base TS port (`src/ace/model.ts` etc. stay
    untouched; `aline.ts` is additive, arbuffer's `buffered.ts` precedent).
  - Browser-side grid oracle for hyperparameters (the GP tab has none either; the
    Python diagnostic carries oracle calibration).
  - Random/US **rollout baselines** as full alternate drivers in the tab (the US
    *marker* is in scope, Design 7; running whole US-driven episodes in-browser is
    not — the Python `evaluate` carries baseline curves).
  - Multi-episode statistics, reward displays, or any RL-training story in the tab
    (inference + acquisition only; training happened offline).
  - Continuous (non-pool) action spaces; ACEP prior tokens; episode lengths past the
    trained range (T is capped, not extended).
  - User-as-oracle metrics (no truth ⇒ no RMSE/log-q story in that mode).

## Design decisions

1. **Local-only against the 5k artifact; design for the future checkpoint.** The tab
   ships behind the same self-skip discipline as the early arbuffer tab: blob absent ⇒
   graceful notice + skipped suites, CI untouched. The 5k policy demonstrably learns
   (beats random) but its goal contrasts are null; the tab's explainer says "early
   validation checkpoint" honestly. No UX decisions are tuned to 5k behavior.
2. **Two modalities; the user always picks *where*, never *what*.** Default
   ground-truth mode: a hidden GP draw (sampled in-browser per episode) answers every
   query; clicking chooses the next x only. ALINE's policy is rendered as advice the
   user can follow or ignore; "Follow policy" automates the choice (argmax, animated);
   "Step" takes a single policy action. Secondary user-as-oracle mode: free point
   editing (GP-tab interaction), policy overlay still live, no truth machinery. This
   makes the headline experiential: race the policy on the same hidden function.
3. **ξ = token composition; TS uses omission, not masks.** Two exact equivalences,
   both worth recording: (a) a masked target token ≡ an absent one (targets never
   attend to each other; the loss/reward average and the policy's `key_padding_mask`
   both exclude it) — so the TS target set just *contains* the active goal tokens;
   (b) a masked pool candidate ≡ an absent one (policy scoring is pointwise; the
   softmax over remaining candidates supplies the competition) — so the TS query set
   contains only available candidates. B=1 throughout; `TokenList.add` semantics
   unchanged.
4. **One forward per step; the policy reads goal-row slices.** The band needs
   plot-grid QUERY tokens; the policy must read *only* the goal tokens' states.
   Because target rows are mutually independent (no target–target attention),
   appending band tokens to the same forward changes nothing per-row; `infer.ts`
   slices the goal rows out of `tgtStates` for `policyLogits` and reads band rows for
   the plot. The US marker's candidate rows (Design 7) ride the same forward as a
   third class of target rows, equally invisible to the goal rows. One
   `forwardWithStates` + one `policyLogits` per step.
5. **`ALINEModel` reads the inherited forward's outputs; no loop re-run.** Python's
   `forward_with_states` re-runs the block loop because PyTorch doesn't expose
   intermediates; the TS base port already returns `ctxLayers`/`tgtLayers` per block.
   So `forwardWithStates` = inherited `forward()` + `ctxLayers[L-1]` + re-applied
   `final_norm` on `tgtLayers[L-1]` (bit-equal math to Python's returned states; the
   `policy` fixture asserts it). Simpler than `buffered.ts` — ALINE needs no KV cache.
6. **Environment = exact `gp1d` DGP in float64 JS.** Pool of 128 candidate x drawn
   `U(-1, 1)` per episode (train-matched; not a fixed grid), plot grid of 64 linspace
   points for band/truth/metrics, 32 predictive-target x\* drawn `U(-1, 1)` per
   episode (train-matched; truth not needed — QUERY tokens ignore value). One joint
   zero-mean draw at pool + grid (192 points) via Cholesky, jitter 1e-5, noiseless.
   Clicks snap to the nearest *available* candidate so the user and the policy share
   one action space (fair comparison; 128 points reads as continuous). Seeded
   (`mulberry32`) so an episode is reproducible.
7. **US comparison marker.** Uncertainty sampling's pick (argmax predictive variance
   over available candidates) is computable from the same forward by appending the
   candidates as extra target rows — nearly free, and it shows *when the learned
   policy deviates from the classical heuristic*, which is the tab's didactic point
   (especially under parameter goals). Rendered as a second, visually distinct marker.
8. **Metrics in ground-truth mode.** Per step: predictive goal → RMSE of the band
   mean vs hidden truth on the plot grid; parameter goals → log q(θ_true) (MDN
   log-prob at the encoded true latents; categorical log-prob of the true kernel).
   Shown as a readout + small step-curve. Truth dashed-line hidden by default,
   auto-revealed at episode end ("Reveal truth" toggles anytime).
9. **OOD guardrails** (config + `pointOodReasons` style): context size soft-warn past
   17 (training: 1 seed + T=16) and strong-warn past 20 (warm start's trained
   `n_context ≤ 20`); Follow-policy stops at T; mixed parameter+predictive ξ allowed
   but flagged "novel combination (untrained — paper D.4 suggests it generalizes)";
   empty ξ disallowed (policy precondition: ≥1 active target); y-range flag in oracle
   mode as in the GP tab.
10. **Fixtures pin three layers separately.** `plain` pins the inherited base path on
    the fine-tuned weights (the cheap regression net); `policy` pins states +
    per-policy-block candidate streams + logits (divergence localizes to a block);
    `chain` pins episode orchestration **teacher-forced**: the TS test follows the
    fixture's recorded actions (immune to argmax tie-flips under fp arithmetic drift)
    while asserting logits/log-probs within tolerance. The env fixture pins the DGP
    math (K, Cholesky) checkpoint-independently, including one ill-conditioned case.
    No `.demo.json` ships: the `chain` section plays the orchestration-fixture role,
    exactly as `gp1d_arbuffer` (which also ships only a parity fixture).
11. **Blob identity by key presence.** No new manifest fields: `ALINEModel` probes
    `policy_blocks.0.q_ln1.weight` and infers depth from tensor names — exactly how
    `BufferedACEModel` keys on `buf_blocks.0.buf_bias`.

## Architecture spec

### Python: exporter + fixtures

- `export_weights.py`: one `TASK_MODULES` entry. `gp1d_aline.load_checkpoint(path,
  device)` already satisfies the 2-arg contract and infers policy depth from the
  state dict. fp16 rounding via the existing `quantize_fp16_inplace`.
- `parity.py` additions (gated on `ALINE_CKPT.exists()`, prints a skip note
  otherwise; the env fixture write sits **before** that gate, so it runs on every
  `parity.py` invocation regardless of the ALINE checkpoint):
  - `aline_policy_case(model)`: B=1; compact all-active tokens (row order =
    TS row order); context = a handful of data VALUE tokens; target = the three
    latent QUERYs + a few x\* QUERYs (a "both kinds" goal state for coverage; the
    tab's actual ξs are subsets — covered via the chain); query = ~8 candidates.
    Dumps token JSON + `ctx_states`, `tgt_states` (final-normed), per-block
    candidate streams, `logits`. Manually replicates `policy_logits` and asserts
    bitwise against the model before dumping (the `run_case` discipline; the
    `no_grad`/`detach` in `policy_logits` are gradient-only and numerically inert).
  - `aline_chain_case(model)`: seeded compact episode — pool 16, grid 0, T = 6,
    ξ = {ℓ} for steps 1–3 then ξ = predictive (M = 4 fixed x\*) for steps 4–6 (a
    mid-episode switch in the fixture exercises exactly what the tab does); per
    step: available-candidate list, logits, argmax action, observed y, per-goal-token
    log-probs. Built with its own tiny loop over `forward_with_states` +
    `policy_logits` (not `rollout()`) so token composition matches TS omission
    semantics row-for-row.
  - env fixture: two x-vectors (8 moderate points; 8 clustered points), all four
    kernels at fixed (log_ell, log_scale) — K and L (float64 nested lists), plus
    constants `{ranges, jitter, period: 1.0}`.

### TS: `src/ace/aline.ts`

```ts
export class ALINEModel extends ACEModel {
  readonly nPolicyBlocks: number;            // inferred from manifest names
  constructor(weights: Weights)              // throws if policy sentinel absent
  forwardWithStates(context: TokenSet, target: TokenSet):
    { out: ForwardOut; ctxStates: number[][]; tgtStates: number[][] }
  policyLogits(query: TokenSet, ctxStates: number[][], tgtStates: number[][]): number[]
}
```

Reuses `multiHeadAttention`, `layerNorm`, `mlp`, `linear` from `nn.ts` and inherited
`embed`. Tensor names: `policy_blocks.{i}.{q_ln1,ctx_kv_ln,q_ln2,tgt_kv_ln,q_ln3}.
{weight,bias}`, `policy_blocks.{i}.{ctx_attn,tgt_attn}.{in_proj_weight,in_proj_bias,
out_proj.weight,out_proj.bias}`, `policy_blocks.{i}.mlp.{0,2}.{weight,bias}`,
`policy_norm.{weight,bias}`, `policy_head.{weight,bias}`.

### TS: `src/aline/` (env, infer, demo)

- `env.ts`: `sampleEpisode(rng, cfg) -> { poolX, poolY, gridX, gridY, xStar, logEll,
  logScale, kernel, seedIdx }` (one joint draw at pool+grid; x\* needs no y);
  `kernelMatrix(...)`, `cholesky(K)` exported for tests.
- `infer.ts`: `EpisodeState`; `buildStep(model, state)` returning `{ band, latentMarginals,
  policy: {x, probs, argmaxIdx, usIdx}, metrics }`; `applyObservation`; pure and
  deterministic given (weights, seed, actions).
- `demo.ts`: `mountAline(el)`; rendering via `plot.ts` (`drawBottomDensity`-style
  overlay for the pmf; `vline` markers for ALINE/US picks); controls and warnings per
  Designs 2/7/8/9; epoch-guarded rAF for Follow policy; explainer via `addInfoButton`.
- `config.ts`: `ALINE = { POOL: 128, T: 16, GRID: 64, M_PRED: 32, CONTEXT_SOFT: 17,
  CONTEXT_HARD: 20, Y_OOD: 2.0, ... }`.

## Verification

1. **Export sanity** — manifest tensor table contains all `policy_*` tensors; byte
   size check; loader round-trip in a node script or the parity test itself.
2. **Plain parity** — inherited forward on the ALINE blob matches `plain` fixtures
   (RAW tolerances). Guards the frozen *path* (weights are fine-tuned; path is base).
3. **Policy parity** — ctx/tgt states and per-block candidate streams match; logits
   match (the states check is what pins Design 5's re-applied `final_norm`).
4. **Chain parity** — teacher-forced episode: logits + log-probs per step within
   DERIVED tolerance; recorded argmax actions reported (not asserted) so a tie-flip
   is visible without failing the build.
5. **Env exactness** — K and L vs float64 fixtures at ~1e-9 relative, including the
   ill-conditioned case; encode/decode constants asserted against manifest variables.
6. **Determinism** — same seed ⇒ identical episode (draw, seed point, policy pmf) in
   two `infer.ts` runs.
7. **Self-skip discipline** — `npm test` green with the blob absent (aline suites
   skip; env suite and missing-notice test still run) and with it present.
8. **Deploy safety** — `pages.yml` diff-clean; no `gp1d_aline` in its task lists.
9. **Eyeball** — run the tab on the 5k weights: episode loop, goal switch, Follow
   policy animation, oracle-mode editing, OOD flags; record observations in the
   extension DEVLOG (honest reading, not a gate on policy quality).

## Risks / fallbacks

- **5k policy demos weakly** (null contrasts): expected; the tab ships local-only
  with honest labeling, and the checkpoint swap is mechanical (Out-of-scope records
  the procedure). Do not tune UX against it.
- **Argmax tie-flips between PyTorch fp32 and JS fp64**: chain fixture is
  teacher-forced (Design 10); the demo itself is unaffected (any argmax is a valid
  demo).
- **Cholesky conditioning in JS** (periodic kernel, clustered x, small ℓ): jitter
  1e-5 matches Python; the env fixture's adversarial case catches divergence; if a
  draw still fails numerically at runtime, resample the episode (user-invisible
  fallback, logged to console).
- **Fixture size**: chain case kept compact (pool 16, T 6, no per-layer dumps);
  policy case is the only state-heavy dump (one case, d=128).
- **Step latency**: one forward (~17 ctx, ~100 tgt rows incl. band + US candidates)
  plus the policy pass (≤128 candidates × 2 blocks) per step — same cost class as
  existing tabs' interactions; Follow-policy animates one step per frame (arbuffer
  pacing). If sluggish, drop the US marker's extra target rows first.
- **Goal-chip semantics drift** (predictive ⊻ parameters trained, mixed untrained):
  enforced in `infer.ts` (flag, not prohibition) so the OOD story stays consistent
  with `sample_xi`'s training distribution.

## Documentation

- `extensions/aline/README.md` — Phase 1 (the extension's own doc debt; arbuffer
  README structure).
- `extensions/aline/DEVLOG.md` — dated entry for the tab port (TS deviations:
  inherited-forward state reads + re-applied final_norm; omission-vs-mask
  equivalences; teacher-forced chain; eyeball notes on the 5k policy).
- `playground/README.md` — sixth demo bullet, export command, local-only + self-skip
  note (flip to deployed wording only at swap time).
- Root `DEVLOG.md` — two dated entries total: extension pointer (Phase 1), tab
  pointer (Phase 6). Root `README.md`/`AGENTS.md` — extension mention in Phase 1;
  tab mention updated in Phase 6.
- `docs/plans/PLAN-aline.md` — Phase 6 ticks + status header; this plan's tracker.

## Open questions (defaults set; flag disagreement)

1. **Pool = 128 random points per episode (seeded), not a fixed grid** — matches the
   training distribution exactly; snapping makes it feel continuous. (Default: random.)
2. **Truth hidden by default, auto-reveal at episode end** — keeps the game feel;
   toggle available throughout. (Default: hidden.)
3. **US marker in v1** — one extra marker, strong didactic value, droppable for
   performance. (Default: in.)
4. **Metrics panel in v1** — readout + mini-curve only (no multi-run aggregation).
   (Default: in.)
