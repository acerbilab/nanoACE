# Plan: Offline data generation (`data.py`) + uniform per-step reseed

Created: 2026-06-08
Status: COMPLETE (2026-06-08) — implemented + verified; checkpoint retrain/re-export deferred per plan.

## Progress tracker

Phase 1 — reseed + signature:
- [x] `mix_seed` in `ace.py`
- [x] `fit`: per-step reseed + `(step)->Batch` type + docstrings (`train.py`)
- [x] four example thunks → `lambda step:`
- [x] verify: 4× short CPU runs OK; same-seed determinism max|dW|=0; resume-exact max|dW|=0

Phase 2a — draw/assemble split (bit-identical):
- [x] `ace.py`: `mix_int64`, `reveal_mask_from_index`
- [x] `gp1d`: `N_TOTAL`, `draw_instances`, `assemble` (data_targets form), online thunk
- [x] `bo1d`: `draw_instances`, `assemble`, `scale_check` update
- [x] verify: GP max|dW|=0; BO max|dW|=0; reveal dist matches DEVLOG; `--scale-check` OK; imports OK

Phase 2b — complement-targets:
- [x] `assemble`: drop `data_targets`, targets=`N_TOTAL-n_context`, tensorization
- [x] `gp1d`/`bo1d`: `set_defaults(min=1,max=20)`, `n_points=N_TOTAL`, `--data-targets` no-op note
- [x] verify: GP/BO run; `n_target∈[44,63]` exact; ctx≥1 (GP min1/BO min3); draw determinism max|dx|=0

Phase 3 — data.py + --pool:
- [x] `data.py`: `write_pool`, `PoolReader`, manifest+config_hash, "both" shuffle, splits, `__main__`
- [x] `gp1d`/`bo1d`: `gen_config()`, `draw_pool()`, `--pool`/`--pool-force`, source wiring; CLI DGP defaults wired to `GEN_*`
- [x] verify: build + interrupted-resume skip; pooled GP/BO train OK; B-independence (nctx/reveal/physical); pool resume-exact max|dW|=0; guards config/N_TOTAL/variables(hard-even-w/force); 4× online exit 0

Docs:
- [x] `train.py` (module + `fit`) + `ace.py` (`mix_seed`/`mix_int64`/`reveal_mask_from_index`) docstrings
- [x] README offline subsection + train.py/data.py bullets + resume-note fix
- [x] DEVLOG dated entry (supersedes Layout/Data-layer + "RNG not checkpointed" framing)
- [x] AGENTS.md: training-spine, currently-implemented, reveal + frozen-pool conventions

- [x] `/doublecheck` — two Opus passes: code correct, consistent, regression-free, docs accurate; playground contract + imports clean. Hardening added beyond plan: `assemble` raises if `1 <= max_context < n_points` is violated.
- [x] external (GPT) review follow-ups: (1) `--pool` now hard-errors if a frozen DGP flag (`--jitter`; BO `--sigma-obs/--sigma-f-max/--prior-uniform-mix`) is overridden — it would only affect diagnostics, not cached data; (2) `PoolReader` validates `batch_size >= 1` and `1 <= min_context <= max_context`; (3) plan wording narrowed (complete rerun refuses; interrupted rerun skips). Declined: rewriting dated DEVLOG history (superseded via dated entry, per convention; current truth in AGENTS.md) and a shard field-name check (field-set changes bump `SCHEMA`; corruption already throws on load).

## Summary

Add a stateless, reproducible offline data path (`data.py`: generate → save →
train) for the two expensive examples (GP-1D, BO), built on a `draw`/`assemble`
refactor; and make training reproducibility uniform by reseeding the global RNG
once per step inside `fit`, so every example's training stream becomes a pure
function of `(seed, step)` — reproducible, resume-exact, and independent of
model-init RNG consumption. The sampler thunk changes from `() -> Batch` to
`(step) -> Batch`.

This is the long-deferred `data.py` from the DEVLOG "Layout" section, scoped to
exactly the smallest sharded-pool reader that honors the DEVLOG invariants, plus
the reproducibility uniformization we agreed supersedes the "RNG not
checkpointed" caveat and the earlier "keep online bit-identical" intent.

## Scope

- **In scope**
  - `fit` per-step global-RNG reseed + `sample_batch` signature `() -> Batch`
    → `(step) -> Batch`; update all four example thunks.
  - `draw_instances` / `assemble` split for **GP-1D and BO only**, then switch them to
    complement-targets (`n_target = N_TOTAL - n_context`; drop fixed `data_targets`).
  - New `data.py`: `write_pool`, `PoolReader`, manifest + a single DGP
    config-hash check, "both" shuffle keyed by `(seed, pass)`, stateless
    counter-hash split decisions, atomic resumable build, a `__main__` build CLI.
  - `--pool PATH` flag on GP-1D and BO.
  - Docs: `train.py`/`ace.py` docstrings, README offline-data subsection, one
    dated DEVLOG entry, AGENTS.md updates.

- **Out of scope**
  - Gaussian/SIR `draw`/`assemble` split and pools — they are cheap to generate
    online and never need a pool. They **still** get the reseed + signature
    change (free reproducibility/resume-exactness).
  - **Retraining / re-exporting** the retained local checkpoints, playground fp16 blobs
    (hosted in `lacerbi/nanoACE-playground-weights`), or parity fixtures under the new
    stream. Deferred; the user batches this separately. The retained checkpoints stay valid
    and loadable (only *training* changes); they merely stop being
    seed-reproducible-under-new-code until regenerated.
  - RNG-state checkpointing; prefetch; the multi-axis resume-guard matrix;
    HPC/Slurm; reeval; a shuffle-mode enum (all DEVLOG cut-list items). The
    single DGP config-hash is the one provenance check we keep.
  - SIR-style permutation splits (GP/BO points are iid uniform → only
    `n_context` + reveal vary).
  - The training-state doc correction (the user fixes that separately).

## Background decisions (settled in design discussion)

- **Stateless over RNG-state checkpointing.** Reproducible resume comes from the
  stream being a pure function of `(seed, step)` (and, offline, of the absolute
  stream index), not from snapshotting mutable generator state. `torch.manual_seed`
  reseeds CPU + all CUDA in one call, sidestepping the dual-RNG-state fragility.
- **Cache only the expensive physics draws.** Token features and reveal/`n_context`
  are recomputed at assemble time (the reveal coin is assemble-time and the prior
  token is reveal-conditional). The prior *hyperparameters* `(mu_unit, nu)` and the
  truths are cached and frozen — the prior itself is generative and cannot change
  post-build.
- **Keep a DGP config-hash, reject the resume-guard matrix.** A `sha256` of the
  DGP-only config + `variables()` catches stale-pool / changed-constant footguns
  cheaply; the multi-axis guard matrix is experiment-management machinery nanoACE
  doesn't need. (The `variables()`/schema portion is enforced as a *separate hard gate*;
  only the non-schema config-hash is `--pool-force`-able — see Phase 3.)
- **No generator threading.** Determinism is via global-RNG reseed at loop/shard
  boundaries; `ace_prior_beta.py` (which uses `Beta(...).sample()`, no generator)
  is untouched.

## Phases

### Phase 1 — Per-step reseed + step-driven thunk (all four; online only)

**Goal**: every example's training stream becomes a pure function of `(seed, step)`
with the minimum change.

**Work**:
- `ace.py`: add `mix_seed(seed: int, step: int) -> int` — a splitmix64-style scalar
  hash returning an int in `[0, 2**63)` for `manual_seed` (CPU+CUDA-safe across torch
  versions; decorrelates consecutive step-seeds).
- `train.py` `fit`: at the top of each step, `torch.manual_seed(mix_seed(seed, step))`
  *before* `batch = sample_batch(step)`. Change the `sample_batch` type to
  `Callable[[int], Batch]`. Update the module + `fit` docstrings: drop the "draws
  no RNG before the first `sample_batch()` / from-scratch RNG timing matches"
  claim; document the reseed and its reproducible + resume-exact consequence
  (this supersedes the DEVLOG "RNG not checkpointed" caveat).
- Four examples (`gaussian_toy.py`, `gp1d.py`, `sbi_sir.py`, `bo1d.py`): change the
  `train.fit(model, lambda: sample_X(...).batch, ...)` to `lambda step: sample_X(...).batch`
  (the `step` arg is ignored; the reseed governs determinism).

**Steps**:
1. Add `mix_seed` to `ace.py`.
2. Edit `fit` (reseed line, type hint, docstrings) in `train.py`.
3. Edit the four `main()` thunks.

**Verification** (CPU, to avoid CUDA kernel nondeterminism):
- [ ] All four run `--device cpu --steps 20` (with each example's small batch) to
      completion with a sane decreasing loss.
- [ ] Same-seed determinism: two `--steps 20` runs print identical loss (regression
      of an existing property).
- [ ] Resume-exact (data stream): true *by construction* — `torch.manual_seed(mix_seed(seed, step))`
      makes `batch(step)` independent of how training reached `step`. The end-to-end
      (model+optimizer+data) test needs a **surviving resumable** checkpoint, but the final
      `--save-checkpoint` overwrites the periodic resumable one with a model-only file
      (train.py:365 writes only while `step < cfg.steps`, then `main()` overwrites). So verify
      via a scratch two-call `train.fit` (0→10 saving a resumable ckpt to a *temp* path, then
      resume 10→20) or an interrupted run — not a completed `--save-checkpoint` run. Assert
      identical step-20 loss (CPU).

### Phase 2 — `draw`/`assemble` split, then complement-targets, for GP-1D and BO (online; no pool yet)

**Goal**: separate expensive physics (`draw_instances`) from RNG-free tokenization
(`assemble`), then switch GP/BO to the standard "targets = all non-context points" split.
Two sub-steps so the refactor (2a) and the behavioral change (2b) are verified separately.

**Work — `ace.py` (shared, lands once)**:
- add `_mix_int64(x: Tensor) -> Tensor` (vectorized splitmix64 mixer) and
  `reveal_mask_from_index(idx, n_latents, q)` (stateless sibling of `sample_reveal_mask`
  reproducing the same mixture *distribution*, keyed on an int64 index tensor). Consumed
  by `data.py` in Phase 3.

**Phase 2a — split, bit-identical**:
- `gp1d.py`: add `N_TOTAL = 64`; `draw_instances(n_instances, *, n_points, jitter)` →
  **CPU float64 native** struct-of-arrays dict (physics stays on CPU per the WSL2-Cholesky
  rule — no training-`device` arg), drawing RNG in the **original order**
  `x → log_ell → log_scale → kernel → y` (dict *key* order is irrelevant; the *draw* order
  preserves bit-identity); `assemble(inst, *, variables, n_context, reveal_mask, max_context,
  data_targets, device) -> Batch` (RNG-free: slice `[0:max_context]` context candidates +
  `[max_context:max_context+data_targets]` targets, apply reveal — zero-spread PRIOR for
  continuous latents, VALUE label for the kernel — encode, build `Tokens`, **and move the
  final float32 token tensors to `device`**). Online thunk =
  `assemble(draw_instances(B, n_points=max_context+data_targets, ...),
  n_context=torch.randint(min_context, max_context+1, ...),
  reveal_mask=sample_reveal_mask(3, B, q=1-latent_context_prob, ...), device=device)`,
  preserving draw order so it is **bit-identical to Phase 1**.
- `bo1d.py`: same split (draw order: prior params x/y → contaminated x/y → kernel → ell →
  sigma_f → depth → x_data → planted f → noise; keep prior-param draws on `device="cpu"`).
  Update `scale_check` to consume `draw_instances` output (needs only native `y`/`x_opt`/`y_opt`).
  Keep `BOBatch` for `fixed_eval_batch`.
- 2a keeps the **current** per-example defaults (GP `max_context=14/min_context=4/data_targets=32`;
  BO `12/1/24`), so it stays bit-identical to Phase 1; the `min_context=1/max_context=20` +
  drop-`data_targets` change is **2b only**. (`N_TOTAL=64` is defined in 2a but unused until
  2b — 2a still draws `max_context+data_targets` points.)
- Leave `load_checkpoint`, `variables`, `fixed_eval_batch`, `evaluate`, oracles unchanged.

**Phase 2b — complement-targets (behavioral; applied to the online path now, reused by the pool in Phase 3 — same `assemble`)**:
- `assemble` becomes `assemble(inst, *, variables, n_context, reveal_mask, max_context, device) -> Batch`
  (**drop `data_targets`**): **targets = all non-context points**, `n_target = N_TOTAL - n_context`.
  Tensorize as **context tensor width = `max_context`** (mask `arange < n_context`) and
  **target tensor width = `N_TOTAL`** (mask `arange >= n_context`), so context self-attention
  stays O(`max_context`²) and only the target cross-attention grows linearly.
- `gp1d.py` / `bo1d.py`: `set_defaults(min_context=1, max_context=20)` for both; **drop the
  `data_targets` default** (unused by GP/BO now; Gaussian/SIR keep theirs). GP/BO still
  **inherit** `--data-targets` from `common_parser`; leave the flag but document it as a
  **no-op for GP/BO** (complement-targets supersedes it). Online thunk now draws
  `n_points=N_TOTAL` so online and the future pool train the same distribution.
- This **changes training** (the rerun you pre-accepted); 2b is intentionally *not*
  bit-identical to 2a.

**Verification** (CPU):
- [ ] 2a: GP and BO `--steps 20` loss is **bit-identical to Phase 1** (same seed) — confirms
      the split introduced no bug. (If exact order can't be cleanly preserved, downgrade to
      "sane + same-seed reproducible" and note it.)
- [ ] `reveal_mask_from_index` reproduces `sample_reveal_mask`'s count distribution
      empirically over a large index range (at q=0.5, per DEVLOG: L=2 → ~{0:.50, 1:.29,
      2:.21}; L=3 → ~{0:.50, 1:.19, 2:.19, 3:.12}).
- [ ] 2b: GP and BO `--steps 20` run to completion with sane decreasing loss and same-seed
      reproducibility; target counts vary as `N_TOTAL - n_context` (`n_context ∈ [1, 20]` →
      `n_target ∈ [44, 63]`).
- [ ] `bo1d.py --scale-check` still prints token-scale + contamination marginal.
- [ ] `python -c "import gp1d, bo1d"` clean; the playground's
      `load_checkpoint(path, device)` / `variables()` contract is untouched.

### Phase 3 — `data.py` + `--pool` for GP-1D and BO

**Goal**: the generate → save → train offline path. The pool caches only physics;
splits are recomputed statelessly at read time.

**Work**:
- `data.py` (new, task-agnostic IO + batching + stateless shuffle/splits):
  - `write_pool(draw_fn, out, *, pool_size, shard_size, gen_config, variables, seed, force=False)`:
    shard `i` produced after `torch.manual_seed(mix_seed(seed, i))` (independent, resumable
    build) from `draw_fn` (CPU native tensors); store each field in its **natural dtype** —
    continuous as float32, GP's categorical `kernel` latent as **int64** (not a rounded
    float; BO stores no categorical — its kernel/ell/etc. are nuisance baked into `y`);
    atomic temp→rename; skip valid existing shards (or rebuild all if `force`); write manifest
    **last**. Manifest = `{schema, variables() repr, gen_config,
    config_hash = sha256(canonical_json(gen_config + variables)), fields [{name, shape, dtype}],
    shards (file/start/count), pool_size, shard_size, seed}`.
  - `PoolReader(path, *, assemble, variables, batch_size, seed, max_context,
    min_context, latent_context_prob, device, force=False)`: on load, validate **schema and
    `variables()` (always hard — a wrong token schema silently misreads the cached arrays)**;
    validate the **config-hash** (the non-schema DGP constants) — refuse, or warn if `force`;
    validate `max_context < N_TOTAL` (always hard — need ≥1 target). Then `__call__(step)` →
    - logical position of item `j` in step `s` (1-based): `p = (s - 1) * B + j`. This is the
      **B- and steps-independent** enumeration of the run's datasets (position `p` is the same
      dataset under any `B`), so every key derives from `(seed, p)`.
    - fetch the "both"-shuffled physical row at `p`: `pass = p // pool_size`,
      `pass_pos = p % pool_size`; shard-order + within-shard perm keyed by
      `_mix_int64(seed, pass, salt_row)`.
    - compute `(n_context, reveal_mask)` from `_mix_int64(seed, p, salt_split)` /
      `reveal_mask_from_index` — a **separate namespace** (`salt_split ≠ salt_row`) so row and
      split streams are decorrelated. Seed enters as a **salt, not a `B`/`steps`-dependent
      offset** (the earlier `split_offset = seed*steps*B` is dropped: it broke
      batch-size/steps independence and was off-by-one against `fit`'s 1-based loop).
    - return `assemble(rows, n_context=..., reveal_mask=..., max_context=max_context, device=device)`
      (targets = `N_TOTAL - n_context`).
  - `__main__`: `python data.py <example> --out DIR --pool-size N [--shard-size M
    --seed S --force]`, dispatching to the example's `draw_instances` + `gen_config()`.
- `gp1d.py` / `bo1d.py`:
  - `gen_config()` returning **every** constant that affects stored values, so the hash
    actually catches drift:
    - GP: `KERNELS`, `LOG_LENGTHSCALE_RANGE`, `LOG_OUTPUTSCALE_RANGE`, `N_TOTAL`, `jitter`.
    - BO: `KERNELS`, `KERNEL_WEIGHTS`, `ELL_MEAN`, `ELL_STD`, `ELL_RANGE`, `X_OPT_RANGE`,
      `Y_OPT_RANGE`, `Y_RANGE`, `ENVELOPE`, `D_CAP`, `sigma_f_max`, `sigma_obs`, `eps`,
      `N_TOTAL`, `jitter`.
    Hashed via **canonical JSON** (`sort_keys=True`, tuples→lists, fixed separators), not
    `repr`. The Beta hyperprior in `ace_prior_beta.sample_prior_params` is *behavior in a
    shared helper*, not a constant, so it is covered by bumping the manifest **`SCHEMA`
    version** if that helper changes — it can't be hashed as a `gen_config` value.
  - `--pool PATH` and `--pool-force` (added **per-example**, deliberately not in
    `common_parser` since Gaussian/SIR have no pool); in `main`,
    `source = PoolReader(..., force=args.pool_force) if args.pool else online_thunk`,
    then `fit(model, source, ...)`.

**Verification** (CPU):
- [ ] `python data.py gp1d --out artifacts/pool_gp --pool-size 2048 --shard-size 512`
      builds and writes a valid manifest; an *interrupted* rerun (manifest absent, shards
      present) skips the valid shards; a *complete* rerun (manifest present) refuses without
      `--force`.
- [ ] `gp1d.py --pool artifacts/pool_gp --steps 20` trains to a sane diagnostic.
- [ ] Batch-size/steps independence: the `(n_context, reveal)` for logical position `p`
      is identical whether `p` is reached as `(step, j)` under `B=16` or `B=32`, and
      regardless of `steps` (the split keys on `(seed, p)`, not on `B`/`steps`).
- [ ] Resume-exact from pool: continuous vs resumed run identical at the same step.
- [ ] Config-hash guard: a pool built under a different `gen_config` (or after editing
      a DGP constant) makes `PoolReader` refuse with a clear "regenerate" message;
      `--pool-force` downgrades it to a warning and proceeds.
- [ ] N_TOTAL guard: `--pool P --max-context M` with `M >= N_TOTAL` refuses with a clear
      message (need ≥1 target).
- [ ] `variables()` guard: a pool built under a different `variables()` (changed bounds,
      added/reordered latent) **hard-fails even with `--pool-force`**.
- [ ] `--pool P --resume` fast-forwards correctly (PoolReader is a pure function of `step`,
      which `fit` restores) — resumed pooled run matches a continuous pooled run at the same step.
- [ ] Same build + pooled-train + guard checks for BO.

## Documentation (deliverable)

- `train.py` module + `fit` docstrings — Phase 1.
- `ace.py` docstrings for `mix_seed`, `mix_int64`, `reveal_mask_from_index` — Phases 1–2.
- README: new "Offline data generation" subsection (GP/BO only, build command,
  `--pool`, why GP/BO only, the `(seed, step)` reproducibility note); also update the
  existing "future sharded `data.py` reader drops in unchanged" line (Implemented-modules /
  train.py bullet) to present tense — Phase 3.
- DEVLOG: one dated entry covering (a) per-step reseed + signature change (supersedes
  "RNG not checkpointed" and the point-#1 "online bit-identical" intent), (b) the
  `draw`/`assemble` split, (c) the `data.py` design + explicit decisions (stateless >
  RNG-checkpoint; keep DGP config-hash, reject resume-guard matrix; GP/BO only;
  frozen-in-pool vs free-at-assemble). The entry also explicitly supersedes the "future
  sharded saved-pool path" framing in the DEVLOG "Layout"/"Data layer" sections
  (`data.py`/`train.py` now built) — Phase 3.
- AGENTS.md: update "Currently implemented" (data.py built; per-step reseed;
  `sample_batch(step)`), the training-spine bullet, a conventions note on
  frozen-in-pool vs free-at-assemble; drop the "data.py planned but not built" framing.

## Risks / Notes

- Per-step reseed **deliberately changes the training stream** → retained local
  checkpoints stop being seed-reproducible under new code until regenerated. Not a
  breakage (`artifacts/` is gitignored; weights live in the separate repo); deferred per the user.
- Bit-identity / resume-exact verifications must run on **CPU** (CUDA kernels are
  nondeterministic; nanoACE doesn't enable deterministic mode).
- Phase 2a bit-identity to Phase 1 depends on preserving RNG draw order in the online
  wrapper; fallback is the looser "sane + reproducible" check. Phase 2b (complement-targets)
  is intentionally *not* bit-identical — it changes the target composition.
- Pool size is dominated by the float32 data fields (`x`, `y`), ≈ `pool_size * N_TOTAL *
  n_data_fields * 4` bytes; the per-instance latent scalars (truths, `(mu_unit, nu)`, int64
  `kernel`) are negligible. Document the `passes ≈ steps*B / pool_size` relationship so pools
  are sized to avoid over-reuse.
- "Resume-exact data stream" ≠ "bit-identical weights" on CUDA without deterministic
  kernels — out of scope, stated for honesty.
- Rollback: each phase is an independent commit — revert P2b to restore the
  fixed-`data_targets` split, P1 to restore the pre-reseed stream; retained checkpoints stay
  loadable throughout (only training changes).
- Pools land under gitignored `artifacts/` (e.g. `artifacts/pool_gp`), so they are not
  committed — consistent with the `artifacts/` convention.
- Naming decided: the cross-module helpers are **public** (`mix_seed`, `mix_int64`,
  `reveal_mask_from_index`) — no leading underscore, since `data.py` imports them from
  `ace.py`, matching ace.py's `encode_value`-style public-helper convention.

## Resolved decisions

- **N_TOTAL = 64** for both GP and BO — now the single point budget (context + targets),
  not headroom over a separate `data_targets`. It counts **data observation points only**;
  the latent tokens (GP ≤3, BO 2) are separate columns on top (total token count =
  `N_TOTAL + n_latents`). There is no architectural cap at 64, the extra latent tokens are
  negligible for compute, and the pool's `[n, N_TOTAL]` arrays hold only data — latent
  truths/`(mu_unit, nu)` are separate per-instance scalars. ✓
- **Context split**: `min_context = 1`, `max_context = 20` for both GP and BO; **targets =
  all non-context points** (`n_target = N_TOTAL - n_context ∈ [44, 63]`). The fixed
  `data_targets` knob is dropped for GP/BO (Gaussian/SIR keep theirs). ✓
- **Build CLI**: `python data.py <example> --out DIR --pool-size N ...` (nanoGPT
  `prepare` style). ✓
- **`--force` in two distinct places** ✓:
  - Build (`data.py --force`): overwrite an existing complete pool.
  - Train (`--pool-force` on the example): downgrade a *provenance* mismatch — the
    **config-hash only** (non-schema DGP constants) — from refuse to a warning, for
    knowingly reusing a pool. **Not overridable**: a `variables()`/schema mismatch (a wrong
    token schema silently misreads the cached arrays — correctness, not provenance) or the
    `N_TOTAL` guard.
- **Gaussian/SIR stay out** of the `draw`/`assemble` split (reseed + signature only). ✓
  Rationale: their draws are cheap — Gaussian is `mu + sigma*randn`, SIR is a small CPU
  RK4 over `T_OBS=25` — nothing Cholesky/Matheron-heavy to amortize, so a pool buys no
  speedup. Splitting their samplers would be churn without function, and the monolithic
  sampler reads more clearly as one piece for the simpler examples. They still get
  Phase-1 reproducibility/resume-exactness for free. (If uniformity is later preferred
  over locality, the split is mechanical and can be applied then.)

---
**Please review. Edit directly if needed, then confirm to proceed.**
