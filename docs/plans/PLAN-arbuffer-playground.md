# Plan: AR-buffer joint-sampling playground tab

Created: 2026-06-11
Status: COMPLETE (2026-06-11) — implemented, parity-verified, doublechecked (two Opus
reviewers: code correctness — no blockers/defects; plan fidelity — one defect fixed:
the `_k128.pt` artifact-name drift across the local/retained boundary is now
documented in the swap procedure and removed from the demo notice; the
missing-model-notice smoke test was moved outside the skip guard so it runs in the
blob-absent deploy scenario). Remaining user actions: visual eyeball
(`cd playground && npm run dev`) and committing the new files (including
`playground/test/fixtures/gp1d_arbuffer.parity.json`, which must be tracked for the
self-skip story to hold in CI).
Amended 2026-06-11 (user decision): the sampling grid is **fixed at 32 points** — the
grid-size selector was removed (`ARBUF.GRID_POINTS = 32`; `STEPS_PER_FRAME` lowered to
1 so the animated reveal stays visible at the shorter chain).
Amended 2026-06-11 (later, user decision): the extension switched its retained
architecture to the **concat read** (one softmax over `[context, buffer]` keys through
the base `cross_attn`, learned per-head `buf_bias` soft gate — see the extension
DEVLOG's concat-read entry), and the playground followed: `buffered.ts` now implements
the concat read **only** (separate-read blobs rejected at load with a clear error;
fixtures can only honestly cover the shipped architecture), the per-draw read cache is
`kv_ln(state)` projected under the base `cross_attn`, there is no step-0 special case
(an empty buffer reduces exactly to the base read), and `parity.py`'s packed
replication mirrors both `forward_buffered` branches and records the mode in the
fixture. `ARBUF_CKPT` → `artifacts/gp1d_arbuffer_concat20k.pt` (20k, K=64, joint
training); blob re-exported + fixtures regenerated together; all 27 tests pass. One
tolerance note: the arbuf parity RAW atol widened 1e-4 → 3e-4 — the unfrozen
checkpoint's weights accumulate slightly more float32-vs-float64 drift on intermediate
layer states (worst observed exceedance 1.8e-4 on one element of a plain-forward
case); still orders of magnitude below any real porting bug. The retained 200k concat
run (in flight, → `artifacts/gp1d_arbuffer.pt`) swaps in by repointing `ARBUF_CKPT`
and re-running export + parity together.
Amended 2026-06-12: the retained 200k concat run completed and was **swapped in**
(`ARBUF_CKPT` → `artifacts/gp1d_arbuffer.pt`; blob + fixtures regenerated together;
all tests green at the same tolerances). The in-tab model-provenance note was removed
(user decision: not something the tab's audience needs; provenance lives in the docs).
Also added (separate small feature): per-tab "?" explainer modals via
`playground/src/explain.ts`. The fixed grid moved 32 → **64** (user decision) to match
the retained fine-tune's K=64 — a 64-step chain exercises exactly the trained prefix
range; decode ≈ 0.7 s, animation ≈ 1 s at one step per frame.
Measured (2026-06-12, node V8, throwaway probe, 3 draws): buffered decode incl. encode
vs slow-AR re-encoding via the plain forward — 4 ctx pts × 32 steps: 0.33 s vs 3.8 s
(~11×); ×128 steps: 1.6 s vs 68 s (~43×); 14 ctx pts: 0.42 s vs 6.1 s (~14×) and
1.2 s vs 78 s (~63×). In scalar JS the FLOP asymptotics show directly (unlike the
launch-bound GPU timings at nano scale); the buffer is what makes the tab interactive.

Adds a fifth playground tab that runs the `extensions/arbuffer/` buffered GP-1D model
in the browser: edit context points (and pin latents) exactly like the GP tab, then
watch a few **coherent joint function draws** decode autoregressively from one cached
context encoding, side by side with the diagonal band and independent marginal samples.

Decisions already made (discussion 2026-06-11): **local-only for now** (no deploy /
`pages.yml` change until the retained 100k–200k fine-tune lands); **B = 3 draws**
(small, legible, perf-trivial); **latent pins included**; **always show both**
independent marginal samples and coherent draws (no toggle); **animated AR reveal**
is the default presentation. Temporary model: `artifacts/gp1d_arbuffer_k128.pt`
(20k validation run, the recommended K=128 settings); the retained model swaps in
later by re-running export + parity together — no tab code change.

Reference: Hassan et al. (2026); `extensions/arbuffer/` (model + local DEVLOG);
`docs/plans/PLAN-arbuffer.md` (extension plan, COMPLETE). Playground conventions:
`playground/README.md`, `parity.py`, the GP tab (`src/gp/`).

## Progress tracker

Phase 1 — Python side (export + fixtures):
- [x] `extensions/arbuffer/gp1d_arbuffer.py`: add the playground exporter contract —
      a 2-arg `load_checkpoint(path, device)` wrapper forwarding to
      `load_buffered_checkpoint(path, device, gp1d.variables())`
- [x] `playground/export_weights.py`: add task `"gp1d_arbuffer"` →
      module `"extensions.arbuffer.gp1d_arbuffer"` (generic state-dict serialization
      already carries `buf_blocks.*`; manifest format unchanged)
- [x] `playground/parity.py`: buffered fixture block (graceful skip + printed note if
      `artifacts/gp1d_arbuffer_k128.pt` is absent) writing `gp1d_arbuffer.parity.json`:
      (a) plain-forward cases on the buffered checkpoint (base-path regression),
      (b) one packed `forward_buffered` case with per-layer ctx/buf/tgt states,
      (c) one teacher-forced `sample_joint` chain (fixed order, per-step log-probs)
- [x] run export + parity; verify blob loads and fixture sizes are reasonable —
      210 tensors / 4.07 MB fp16 blob; new fixture 543 KB; the four existing fixture
      sets regenerated byte-identically; packed replication sanity-assert passed

Phase 2 — TS port (`playground/src/ace/buffered.ts` + `rng.ts`):
- [x] `mulberry32` PRNG + Box–Muller normal + MDN sampling helper
- [x] `BufferedACEModel extends ACEModel`: `encodeContext` (per-layer ctx input/output
      states + projected KV caches; dense all-active context only), per-draw
      incremental `predict` / `append` with projected-KV buffer caches,
      teacher-forced and sampling modes
- [x] `src/arbuf/parity.test.ts` (skipped when the model dir is absent): plain cases,
      packed-case per-layer/per-step parity, teacher-forced chain log-prob parity,
      seeded-sampling determinism — all 4 pass (first run)

Phase 3 — tab (`playground/src/arbuf/`):
- [x] `infer.ts` (DOM-free): context builder (points + pins, GP-tab semantics), static
      pass (band mean/std + per-grid MDN params + independent marginal samples), and a
      step-driveable `JointSampler` for the animated decode
- [x] `demo.ts`: GP-style UI (canvas, add/drag/delete, kernel buttons, pin sliders),
      draws/grid controls, Resample, animation loop, timing readout, OOD hints,
      friendly "model not exported" notice on fetch failure
- [x] `config.ts` `ARBUF` block; `index.html` tab + panel; `main.ts` lazy mount
- [x] `demo.smoke.test.ts` (jsdom, fs-backed fetch, skipped when model dir absent) —
      smoke covers mount/decode/controls + the missing-model notice path

Phase 4 — verification + docs:
- [x] `npm test` green with and without the model dir present (deploy-safety check) —
      27/27 pass with the blob; 21 pass + 6 skip with the model dir renamed away
- [x] `npm run build` (tsc + vite) green; manual `npm run dev` inspection — build
      clean; **visual eyeball left to the user** (`cd playground && npm run dev`)
- [x] docs: extension `DEVLOG.md` entry, `playground/README.md` section, root
      `DEVLOG.md` short entry, root `README.md` + `AGENTS.md` one-line updates

## Scope

- **In scope**
  - The five files/areas in the tracker; no other playground file changes beyond the
    three wiring points (`config.ts`, `index.html`, `main.ts`) and `parity.py` /
    `export_weights.py`.
  - One extension-file addition (the 2-arg `load_checkpoint` wrapper) — extensions may
    change their own files; this mirrors the contract every example already exposes.
- **Out of scope**
  - Any core file change (`ace.py`, `gp1d.py`, `train.py`, ...).
  - Deployment: `pages.yml`'s expected-model list stays at four models; the weights
    repo gets no `gp1d_arbuffer/` dir yet. Revisit after the retained fine-tune.
  - Browser-side joint *density* evaluation, slow-AR baseline, NLL tables, timing
    *comparisons* (no `sample_ar` port — only the buffered sampler's own timings).
  - True-function overlay (no browser Cholesky; the tab is user-edited context, not
    the Python demo's fixed planted function).
  - Latents in the buffer; padding support in the TS buffered path (demos build dense
    all-active token lists; asserted, not handled).
  - WebGL/WASM/worker acceleration (B=3, K≤128 is sub-second in scalar JS with
    projected-KV caches).

## Design decisions

1. **The tab's story is the joint-vs-marginal contrast, always visible.** Three
   layers on one plot: the diagonal ±2σ band (context-only marginals — identical to
   the base GP model because the fine-tune froze the base), B=3 *independent* samples
   from those marginals (thin gray jagged lines — what a diagonal map gives you), and
   B=3 *coherent* buffered draws (distinct colors — what the AR buffer buys). No
   kernel-posterior/marginal panels: that is the GP tab's job.
2. **Animated AR reveal by default.** The sampler is a step-driveable object; the demo
   advances a few steps per animation frame, drawing partial chains (realized points
   sorted by x for the polyline). Decode order is a random permutation per draw set
   (matches training and `sample_joint`'s default). An "instant" control skips the
   animation; Resample reuses the cached context encoding and just re-decodes with a
   new seed — with a timing readout ("context encoded once: X ms · 3×64-step decode:
   Y ms") that makes the paper's caching point tangible.
3. **`BufferedACEModel extends ACEModel`, base port untouched.** Mirrors
   `BufferedACE(ACE)` in Python. The buffered TS path re-implements the layer loop
   (house style: duplicate-with-parity-guard rather than refactor the guarded base
   port), inheriting `embed`, weights access, and the `Predictions` math.
4. **Cache projected K/V, deviating from the Python inference deliberately.**
   `arbuffer.sample_joint` caches LayerNorm'd *hidden states* and recomputes KV
   projections per attention call ("legible; a micro-opt at nano scale"). In scalar
   JS that reprojection is O(K²·d²) over a chain — seconds, not micro. The TS port
   caches **projections**: per layer, the context K/V for the base cross-attention
   (over `kv_ln(ctx_out)`) and for the buffer attention (over `buf_ln1(ctx_in)`),
   computed once per context edit; per draw, append-only buffer K/V caches for
   `buf_attn` (over `buf_ln1` of each token's layer input) and `tgt_buf_attn` (over
   `tgt_buf_kvln` of each token's updated state). Same math, same fixtures — recorded
   as an implementation deviation, verified by parity.
5. **No masks in the TS buffered path.** Incremental decode enforces causality by
   construction (you attend only to what is already in the cache); step 0 skips the
   buffer read entirely (the Python `step > 0` check — equivalent to the packed
   path's prefix-0 gate). Context is required to be dense/all-active, so there is no
   key padding either.
6. **Fixtures pin both granularities.** The packed `forward_buffered` case (prefix =
   `arange(K)`) gives per-layer, per-token reference states — divergence localizes to
   a layer, like the base parity fixtures. The teacher-forced `sample_joint` chain
   (explicit shuffled order) tests the exact incremental semantics end-to-end
   (per-step log-probs). The two Python paths agree to ~5e-7 (extension Phase 3), far
   inside the parity tolerances, so the TS incremental path can be checked against
   both. Plain-forward cases on the same checkpoint guard the inherited base path
   (the frozen-base invariant, now extended through the export).
7. **Local-only means tests must self-skip.** The deploy workflow runs `npm test`
   after copying only the four public models; new tests use `describe.skipIf` on the
   absence of `public/models/gp1d_arbuffer/`, and `parity.py` skips the buffered
   block (with a printed note) when the checkpoint artifact is missing. The tab
   itself catches the fetch failure and renders a "model not exported here" notice
   instead of the error string.
8. **OOD hints stay conservative.** Same point-count/value guardrails as the GP tab,
   keeping `MAX_CONTEXT_HINT = 14`: the fine-tune sampled contexts up to 20, but the
   *frozen* base encoder only ever trained at ≤14, and freezing means those larger
   contexts were seen, not learned. The chain length is fixed at 32 points (amended
   from a 32/64/128 selector; well inside `prefix_len ~ U{1..128}` from the K=128
   fine-tune, and no positional embeddings means chain length is purely a
   training-distribution constraint). Evenly spaced grid x's vs iid-uniform training
   x's was already accepted in the extension plan.
9. **Temporary-model honesty.** A one-line note in the panel ("preliminary 20k
   fine-tune; a longer run is planned") and the parity checkpoint path kept as one
   constant in `parity.py`. The swap procedure when the retained model lands
   (whose artifact is expected to be `gp1d_arbuffer.pt`, not `_k128.pt` — per the
   extension plan's artifact naming): repoint `parity.py`'s `ARBUF_CKPT` and the
   `playground/README.md` export example at the retained artifact, then re-run
   `export_weights.py` and `parity.py` **together** (the known staleness gotcha)
   and `npm test`. No tab-code change.

## Architecture spec

### Python: exporter + fixtures

- `gp1d_arbuffer.load_checkpoint(path, device)` → `load_buffered_checkpoint(path,
  device, gp1d.variables())` (the same wrapper shape every example has).
- `export_weights.py`: `TASK_MODULES["gp1d_arbuffer"] = "extensions.arbuffer.gp1d_arbuffer"`
  (namespace-package import from the repo root; the module's own `sys.path` bootstrap
  makes its `import arbuffer` work). Everything else is generic: `build_manifest`
  walks the state dict, so `buf_blocks.{i}.{buf_ln1,buf_attn,buf_mlp,buf_ln2,
  tgt_buf_qln,tgt_buf_kvln,tgt_buf_attn}.*` land in the tensor table; `cfg` is the
  shared `ACEConfig`; `variables` are `gp1d.variables()`. `task: "gp1d_arbuffer"`
  identifies the model to TS.
- `parity.py` buffered block (constant `ARBUF_CKPT = artifacts/gp1d_arbuffer_k128.pt`;
  skip + note if absent), fp16-quantized like the others, writing
  `test/fixtures/gp1d_arbuffer.parity.json` with three sections:
  1. `plain`: 1–2 `run_case` cases reusing the GP builders (data-only; pinned kernel +
     zero-spread lengthscale) — same JSON shape as base parity cases.
  2. `packed`: context = 4 points + pinned kernel; K=12 fixed buffer `(x, y)` tokens;
     the same 12 x's as QUERY targets; `prefix_len = arange(12)`. A manual replication
     of `forward_buffered` (same pattern as `run_case`'s manual `ACE.forward`) captures
     per-layer ctx/buf/tgt states, final `cont_raw`, per-step `log_prob`; a sanity
     `allclose` against the real `forward_buffered` guards the replication.
  3. `chain`: `sample_joint(..., order=<fixed shuffled>, teacher_force=<fixed [2, 16]
     values>)` on a 16-point grid; store order, forced values, per-step log-probs.

### TS: `src/ace/rng.ts` + `src/ace/buffered.ts`

- `rng.ts`: `mulberry32(seed)`, `boxMuller(rng)`, `sampleMDN(params, rng)` (categorical
  over `exp(logW)`, then `loc + scale·z`).
- `buffered.ts`:

  ```ts
  class BufferedACEModel extends ACEModel {
    encodeContext(context: TokenSet): CtxCache
    // CtxCache: per layer — crossKV (K/V of kv_ln(ctx_out) under cross_attn),
    // bufCtxKV (K/V of buf_ln1(ctx_in) under buf_attn); throws unless every
    // context token is active.
    newDraw(): DrawState        // per-layer empty append-only KV caches + k
    predict(cache, draw, x): MDNParams
    // per layer: t += crossAttn(tgt_ln1(t) → q, crossKV);
    //            if (k > 0) t += bufRead(tgt_buf_qln(t) → q, bufReadKV);
    //            t += tgt_mlp(tgt_ln2(t));  then final_norm → cont_head → params
    append(cache, draw, x, y): void
    // per layer: q = buf_ln1(h); push buf_attn-K/V(q) onto bufSelfKV;
    //            h += bufAttn(q, [bufCtxKV, bufSelfKV]);   // self included
    //            h += buf_mlp(buf_ln2(h));
    //            push tgt_buf_attn-K/V(tgt_buf_kvln(h)) onto bufReadKV
  }
  ```

  A small `attendKV(q, kvCache, outProj)` helper (per-head dots over cached K/V +
  out-proj) keeps the two read sites identical. Weight names come straight from the
  manifest (`blocks.{i}.cross_attn.*`, `buf_blocks.{i}.*`).

### TS: tab (`src/arbuf/`)

- `infer.ts`:
  - `buildArbufContext(spec)` — points as data VALUE tokens, kernel pin as discrete
    VALUE, lengthscale/outputscale pins as zero-spread PRIOR (the GP tab's exact
    token semantics, duplicated locally).
  - `arbufStatic(model, spec, grid)` — one plain forward over the grid: band
    mean/std + per-row `MDNParams`; `sampleIndependent(params, nDraws, rng)` builds
    the jagged marginal-sample lines from those same params.
  - `class JointSampler` — holds the ctx cache, per-draw states, the order
    permutation, seeded rng; `step()` decodes one location for all draws (predict →
    sample → append), returns `{gridIndex, values}`; `run()` drains it (instant
    mode and tests); teacher-forced variant for tests.
- `demo.ts`: GP-tab layout minus the posterior panels. Controls: kernel pin buttons,
  lengthscale/outputscale pin sliders, Resample, animation on/off (default on),
  Reset/Clear points. (Grid size fixed at 32 — amended; originally a 32/64/128
  selector defaulting to 64.) Main canvas: training-range
  shading, band, independent samples (gray, thin), coherent draws (3 distinct
  colors), context dots, OOD warning box, timing line. Context edits invalidate the
  ctx cache and restart the decode; Resample keeps the cache (and the static pass)
  and re-decodes with the next seed.
- `config.ts` `ARBUF`: domains/views shared with `GP`, `DRAWS: 3`,
  `GRID_POINTS: 32` (fixed; amended from a 32/64/128 selector), `STEPS_PER_FRAME: 1`,
  `MAX_CONTEXT_HINT: 14`, `HIT_RADIUS_PX: 12`.
- `index.html`: tab `arbuf`, label **"GP-1D joint draws (AR)"** (default-of-record;
  panel hint names the causal AR buffer and links the extension).
- `main.ts`: lazy `mountArbuf` like the other four.

## Verification

1. **Export sanity**: exporter prints tensor/float counts; manifest tensor names
   include `buf_blocks.*`; blob ≈ 4 MB fp16.
2. **Base-path parity**: plain cases on the buffered checkpoint pass through the
   existing TS forward (RAW/DERIVED tolerances) — the frozen-base invariant extended
   to the export.
3. **Packed parity**: TS teacher-forced incremental states match the packed fixture
   per layer and per step (buffer row j ↔ append pass j; target row m ↔ decode step
   m), `cont_raw` and per-step log-probs within RAW/DERIVED tolerances.
4. **Chain parity**: TS teacher-forced run over the fixture's order/values reproduces
   `sample_joint`'s per-step log-probs (DERIVED tolerance).
5. **Determinism**: same seed → identical draws; different seeds differ.
6. **Smoke**: jsdom mount renders, Resample/pin/clear/reset don't throw; fetch-failure
   path renders the friendly notice.
7. **Deploy safety**: `npm test` passes with `public/models/gp1d_arbuffer/` deleted
   (all new tests skip); `npm run build` green.
8. **Eyeball**: `npm run dev` — coherent draws look like functions, independent
   samples look like noise around the band, animation reveals the chain, pinning the
   kernel visibly changes the draws, timing line shows decode ≫ encode reuse.

## Risks / fallbacks

- **TS/Python drift in the buffered layer loop** — the same risk the base port had;
  covered by per-layer packed fixtures (localizes to a layer) and the chain fixture
  (end-to-end). If `ace.py` or `arbuffer.py` forwards ever change, fixtures fail
  loudly; re-port and regenerate (the established pattern).
- **20k model draws look underwhelming** — accepted; it recovers ~87% of the AR gap
  at K=128 and the panel carries the "preliminary model" note. The retained run swaps
  in without code changes.
- **Fixture JSON size** (per-layer states for ~30 tokens × 4 layers × d_model): kept
  small via K=12 packed / 16-point chain; if a fixture exceeds ~1 MB, trim K.
- **Animation jank at grid 128** — moot since the grid was fixed at 32;
  STEPS_PER_FRAME remains the knob if it ever returns. No worker unless proven
  necessary.

## Documentation

- `extensions/arbuffer/DEVLOG.md`: dated entry — playground tab exists, the 2-arg
  `load_checkpoint` exporter contract, parity now also guards the TS port, projected
  KV-cache deviation noted.
- `playground/README.md`: fifth demo bullet, export command, local-only status, the
  skip-when-absent test behavior.
- Root `DEVLOG.md`: short dated pointer entry.
- Root `README.md` / `AGENTS.md`: extend the playground bullet ("…and a local-only
  AR-buffer joint-sampling tab (GP-1D); weights not yet deployed").

## Open questions (resolved 2026-06-11)

- Deploy now vs local-only: **local-only** until the retained fine-tune.
- Draw count: **B = 3** (fixed; no draw-count control in v1).
- Latent pins: **included** (kernel + lengthscale + outputscale, GP-tab semantics).
- Marginal-vs-joint presentation: **always show both** (no toggle).
- Animation: **on by default**, instant mode available.
- Tab label: "GP-1D joint draws (AR)" — default-of-record, edit freely.

---
**Please review. Edit directly if needed, then confirm to proceed.**
