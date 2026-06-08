# Plan: Offline data generation (`data.py`) + uniform per-step reseed

Created: 2026-06-08
Status: PENDING APPROVAL

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
  - `draw_instances` / `assemble` split for **GP-1D and BO only**.
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
  - **Retraining / re-exporting** committed checkpoints, playground fp16 blobs,
    or parity fixtures under the new stream. Deferred; the user batches this
    separately. Committed checkpoints stay valid and loadable (only *training*
    changes); they merely stop being seed-reproducible-under-new-code until
    regenerated.
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
  doesn't need.
- **No generator threading.** Determinism is via global-RNG reseed at loop/shard
  boundaries; `ace_prior_beta.py` (which uses `Beta(...).sample()`, no generator)
  is untouched.

## Phases

### Phase 1 — Per-step reseed + step-driven thunk (all four; online only)

**Goal**: every example's training stream becomes a pure function of `(seed, step)`
with the minimum change.

**Work**:
- `ace.py`: add `mix_seed(seed: int, step: int) -> int` — a splitmix64-style scalar
  hash returning a non-negative int for `manual_seed` (decorrelates consecutive
  step-seeds).
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

### Phase 2 — `draw`/`assemble` split for GP-1D and BO (online; no pool yet)

**Goal**: separate expensive physics (`draw_instances`) from RNG-free tokenization
(`assemble`); the online path routes through `assemble(draw_instances(...))`.

**Work**:
- `ace.py`: add `_mix_int64(x: Tensor) -> Tensor` (vectorized splitmix64 mixer) and
  `reveal_mask_from_index(idx, n_latents, q)` (stateless sibling of
  `sample_reveal_mask` reproducing the same mixture *distribution*, keyed on an
  int64 index tensor). Added here so `ace.py` changes land in one pass; consumed by
  `data.py` in Phase 3.
- `gp1d.py`:
  - `N_TOTAL` constant (proposed 64).
  - `draw_instances(n_instances, *, n_points, jitter, device)` → struct-of-arrays
    dict, drawing RNG in the **original order** `x → log_ell → log_scale → kernel → y`
    (dict *key* order is irrelevant; the *draw* order is what preserves bit-identity).
  - `assemble(inst, *, variables, n_context, reveal_mask, max_context, data_targets) -> Batch`
    — RNG-free tokenization: slice `[0:max_context]` context candidates and
    `[max_context:max_context+data_targets]` targets, apply reveal (zero-spread
    PRIOR for continuous latents, VALUE label for the kernel), encode to internal
    coords, build `Tokens`.
  - Online thunk: `assemble(draw_instances(B, n_points=max_context+data_targets, ...),
    n_context=torch.randint(min_context, max_context+1, ...),
    reveal_mask=sample_reveal_mask(3, B, q=1-latent_context_prob, ...))`,
    preserving the original RNG draw order so the batch is bit-identical to Phase 1.
- `bo1d.py`: same split (draw order: prior params x/y → contaminated x/y → kernel →
  ell → sigma_f → depth → x_data → planted f → noise; keep BO prior-param draws on
  `device="cpu"` as today). Update `scale_check` to consume `draw_instances` output
  (it needs only native `y`/`x_opt`/`y_opt`, all present there). Keep `BOBatch` for
  `fixed_eval_batch`.
- Leave `load_checkpoint`, `variables`, `fixed_eval_batch`, `evaluate`, oracles
  unchanged.

**Verification** (CPU):
- [ ] GP and BO `--steps 20` loss is **bit-identical to Phase 1** (same seed) —
      confirms the split preserved RNG draw order. (If exact order can't be cleanly
      preserved, downgrade to "sane decreasing loss + same-seed reproducible" and
      note it.)
- [ ] `reveal_mask_from_index` reproduces `sample_reveal_mask`'s count distribution
      empirically over a large index range (at q=0.5, per DEVLOG: L=2 → ~{0:.50, 1:.29,
      2:.21}; L=3 → ~{0:.50, 1:.19, 2:.19, 3:.12}).
- [ ] `bo1d.py --scale-check` still prints token-scale + contamination marginal.
- [ ] `python -c "import gp1d, bo1d"` clean; the playground's
      `load_checkpoint(path, device)` / `variables()` contract is untouched.

### Phase 3 — `data.py` + `--pool` for GP-1D and BO

**Goal**: the generate → save → train offline path. The pool caches only physics;
splits are recomputed statelessly at read time.

**Work**:
- `data.py` (new, task-agnostic IO + batching + stateless shuffle/splits):
  - `write_pool(draw_fn, out, *, pool_size, shard_size, gen_config, variables, seed, force=False)`:
    shard `i` produced after `torch.manual_seed(mix_seed(seed, i))` (independent,
    resumable build); store float32 struct-of-arrays; atomic temp→rename; skip valid
    existing shards (or rebuild all if `force`); write manifest **last**. Manifest = `{schema, variables() repr,
    gen_config, config_hash = sha256(gen_config + variables), shards (file/start/count),
    pool_size, shard_size, seed, dtype}`.
  - `PoolReader(path, *, assemble, variables, batch_size, seed, steps, max_context,
    min_context, data_targets, latent_context_prob, device, force=False)`: on load,
    validate schema (always hard); validate config_hash/`variables()` (refuse, or warn if
    `force`); validate `max_context + data_targets <= N_TOTAL` (always hard — correctness,
    not provenance); then `__call__(step)` →
    - fetch `step*B` "both"-shuffled physical rows (shard-order + within-shard perm,
      both keyed by `(seed, pass)` via `_mix_int64`),
    - compute `(n_context, reveal_mask)` from `_mix_int64` / `reveal_mask_from_index`
      keyed on absolute index `split_offset + step*B + j`, with
      `split_offset = seed * steps * B` (disjoint split stream per seed),
    - return `assemble(rows, n_context=..., reveal_mask=...)`.
  - `__main__`: `python data.py <example> --out DIR --pool-size N [--shard-size M
    --seed S --force]`, dispatching to the example's `draw_instances` + `gen_config()`.
- `gp1d.py` / `bo1d.py`:
  - `gen_config()` returning the frozen DGP constants (ranges, kernel set/weights,
    `N_TOTAL`, jitter; for BO also `sigma_f_max`, `sigma_obs`, `eps`).
  - `--pool PATH` and `--pool-force`; in `main`,
    `source = PoolReader(..., force=args.pool_force) if args.pool else online_thunk`,
    then `fit(model, source, ...)`.

**Verification** (CPU):
- [ ] `python data.py gp1d --out artifacts/pool_gp --pool-size 2048 --shard-size 512`
      builds; a rerun skips existing shards; manifest present and valid.
- [ ] `gp1d.py --pool artifacts/pool_gp --steps 20` trains to a sane diagnostic.
- [ ] Batch-size-independence: the split for a given absolute stream index is
      identical for `B=16` vs `B=32` (unit check on the split index hash).
- [ ] Resume-exact from pool: continuous vs resumed run identical at the same step.
- [ ] Config-hash guard: a pool built under a different `gen_config` (or after editing
      a DGP constant) makes `PoolReader` refuse with a clear "regenerate" message;
      `--pool-force` downgrades it to a warning and proceeds.
- [ ] N_TOTAL guard: `--pool P --max-context M --data-targets D` with `M + D > N_TOTAL`
      refuses with a clear message.
- [ ] `--pool P --resume` fast-forwards correctly (PoolReader is a pure function of `step`,
      which `fit` restores) — resumed pooled run matches a continuous pooled run at the same step.
- [ ] Same build + pooled-train + guard checks for BO.

## Documentation (deliverable)

- `train.py` module + `fit` docstrings — Phase 1.
- `ace.py` docstrings for `mix_seed`, `_mix_int64`, `reveal_mask_from_index` — Phases 1–2.
- README: new "Offline data generation" subsection (GP/BO only, build command,
  `--pool`, why GP/BO only, the `(seed, step)` reproducibility note) — Phase 3.
- DEVLOG: one dated entry covering (a) per-step reseed + signature change (supersedes
  "RNG not checkpointed" and the point-#1 "online bit-identical" intent), (b) the
  `draw`/`assemble` split, (c) the `data.py` design + explicit decisions (stateless >
  RNG-checkpoint; keep DGP config-hash, reject resume-guard matrix; GP/BO only;
  frozen-in-pool vs free-at-assemble) — Phase 3.
- AGENTS.md: update "Currently implemented" (data.py built; per-step reseed;
  `sample_batch(step)`), the training-spine bullet, a conventions note on
  frozen-in-pool vs free-at-assemble; drop the "data.py planned but not built" framing.

## Risks / Notes

- Per-step reseed **deliberately changes the training stream** → committed
  checkpoints stop being seed-reproducible under new code until regenerated. Not a
  breakage; deferred per the user.
- Bit-identity / resume-exact verifications must run on **CPU** (CUDA kernels are
  nondeterministic; nanoACE doesn't enable deterministic mode).
- Phase 2 bit-identity to Phase 1 depends on preserving RNG draw order in the online
  wrapper; fallback is the looser "sane + reproducible" check.
- Pool storage is float32, ≈ `pool_size * N_TOTAL * n_fields * 4` bytes; document the
  `passes ≈ steps*B / pool_size` relationship so pools are sized to avoid over-reuse.
- "Resume-exact data stream" ≠ "bit-identical weights" on CUDA without deterministic
  kernels — out of scope, stated for honesty.
- Pools land under gitignored `artifacts/` (e.g. `artifacts/pool_gp`), so they are not
  committed — consistent with the `artifacts/` convention.
- `_mix_int64` / `reveal_mask_from_index` are cross-module (ace.py → data.py); prefer
  non-underscore public names if that reads better, matching ace.py's `encode_value`-style
  public helpers.

## Resolved decisions

- **N_TOTAL = 64** for both GP and BO (headroom over current totals 46 / 36). ✓
- **Build CLI**: `python data.py <example> --out DIR --pool-size N ...` (nanoGPT
  `prepare` style). ✓
- **`--force` in two distinct places** ✓:
  - Build (`data.py --force`): overwrite an existing complete pool.
  - Train (`--pool-force` on the example): downgrade a *provenance* mismatch
    (config-hash / `variables()`) from refuse to a warning, for knowingly reusing a
    pool. The **N_TOTAL correctness guard is NOT overridable** (you genuinely lack the
    points), and neither is a schema-version mismatch.
- **Gaussian/SIR stay out** of the `draw`/`assemble` split (reseed + signature only). ✓
  Rationale: their draws are cheap — Gaussian is `mu + sigma*randn`, SIR is a small CPU
  RK4 over `T_OBS=25` — nothing Cholesky/Matheron-heavy to amortize, so a pool buys no
  speedup. Splitting their samplers would be churn without function, and the monolithic
  sampler reads more clearly as one piece for the simpler examples. They still get
  Phase-1 reproducibility/resume-exactness for free. (If uniformity is later preferred
  over locality, the split is mechanical and can be applied then.)

---
**Please review. Edit directly if needed, then confirm to proceed.**
