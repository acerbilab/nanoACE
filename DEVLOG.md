# nanoACE DEVLOG

A running log of design decisions and their rationale. The *why* matters as much as
the *what* — this file exists so a human or coding agent can reconstruct the reasoning
behind the code without spelunking git history.

Reference: Chang et al. (2025), *Amortized Probabilistic Conditioning for Optimization,
Simulation and Inference* (AISTATS 2025). Paper markdown lives in `paper/`.

---

## 2026-06-14 — Source checkpoints published to Hugging Face (`lacerbi/nanoACE`)

This resolves the **"How to distribute optional checkpoints"** open question from the
Initial-design entry. The full-precision PyTorch checkpoints (the `.pt` files from
`--save-checkpoint`, `{cfg, seed, state_dict}` + optional `config` provenance) now live on
the Hugging Face Hub at
[`lacerbi/nanoACE`](https://huggingface.co/lacerbi/nanoACE), a public model repo. Six
canonical files: the four core examples (`gaussian_toy.pt` 80k, `gp1d.pt` 200k,
`sbi_sir.pt` 100k, `bo1d.pt` 200k) and the two extension fine-tunes
(`gp1d_arbuffer.pt`, the retained 200k concat-read; `gp1d_aline.pt`, the served 35k).

- **Two kinds of weights, kept distinct.** HF holds the **source checkpoints** —
  full-precision, loadable straight back into the example scripts via
  `load_checkpoint(path, device)`. The **fp16 browser blobs** (manifest + `weights.bin`)
  stay in the separate `acerbilab/nanoACE-playground-weights` repo, copied in by the Pages
  workflow; those are derived *from* the source checkpoints. Don't conflate the two homes.
- **Why HF, and the constraints it satisfies.** The Initial-design "Open questions" already
  earmarked HF ("a larger model hosted on HF and lazy-downloaded"). It keeps the three
  rules that entry set for any distribution scheme: (1) the core stays **torch-only** — the
  fetch is an optional extra (`huggingface_hub`), never required to read/run/train; (2) the
  checkpoint ships **config + seed**, so it is regenerable, not a mystery binary; (3) it is
  an **example artifact**, not the packaged "runtime product" non-goal. HF's transparent LFS
  also avoids the playground route's "still a Git LFS pointer" failure mode. Namespace
  `lacerbi` is personal-now (can transfer to an `acerbilab` org later, losslessly with
  redirects).
- **Provenance caveat.** The four core checkpoints carry only `{cfg, seed}` (their `config`
  provenance is absent — they predate that save behavior), so their step counts come from
  this DEVLOG, not the file. The two extension checkpoints carry full `config` (so steps,
  base checkpoint, and run flags are self-documenting). The HF model card records each
  model's architecture (`cfg`) and seed, and step counts from here.
- **Fetch path is a one-liner; no committed helper.** Documented in `README.md` and the
  model card: `hf_hub_download("lacerbi/nanoACE", "gp1d.pt")` → `gp1d.load_checkpoint(...)`.
  Deliberately **no** `fetch_checkpoint()` wrapper and **no** `requirements-hf.txt` — a
  task→filename lookup and a one-package install are below the bar for committed machinery
  (the repo avoids experiment-management apparatus). `huggingface_hub` is a documented
  optional `pip install`, kept out of core `requirements.txt`.
- **Linking.** Root `README.md` gains a "Trained weights" subsection with a direct-link
  table (`…/resolve/main/<file>.pt`) to all six; each extension `README.md` links its own
  checkpoint; `playground/README.md` cross-references the source-checkpoint home so the two
  weight homes are mutually discoverable.
- **Aline = the deployed 35k (`gp1d_aline.pt`), verified SHA256-identical to
  `gp1d_aline_35k.pt`.** Uploaded now for consistency with what the playground serves;
  newer `n2` aline runs are in progress and will replace it on HF (re-upload) once one is
  promoted to deployed — same swap discipline as the playground (re-export + re-parity
  together). Scratch/experimental/superseded checkpoints (the arbuffer `k128`/`k32`/
  `concat20k` variants, the aline `5k`/`n2`/`n2_frombase`/`smoke` variants, the
  `_review_pool_*` data shards) were deliberately excluded.

---

## 2026-06-12 — ALINE playground tab (local-only)

- **The playground gained a sixth tab** running `extensions/aline/` in-browser:
  a hidden GP function (an exact float64 TS port of the gp1d DGP, fixture-pinned
  against `gp1d.draw_gp`) answers queries; the user only chooses *where* to
  sample, with the learned acquisition policy rendered as advice — π(x | data,
  goal) along the x-axis next to the classical uncertainty-sampling pick — a
  live goal selector (click switches, shift-click combines parameter goals;
  the untrained predictive+parameter mix is not selectable; mid-episode
  switches), "Follow policy" auto-unrolling, and RMSE /
  log q(θ_true) tracked against the hidden truth. A secondary "your own data"
  mode keeps the advice live on free-edited points. The TS port adds a policy
  decoder on top of the untouched base port (final-state reads from the
  inherited forward; omission ≡ masking equivalences), parity-pinned per
  policy block plus a teacher-forced episode chain. **Local-only**: the deploy
  workflow is untouched; the 5k validation checkpoint serves the tab until the
  longer fine-tune lands (then re-export + regenerate fixtures together, push
  the blob, add `gp1d_aline` to pages.yml's two lists). Plan + verification:
  [docs/plans/PLAN-aline-playground.md](docs/plans/PLAN-aline-playground.md);
  TS deviations + eyeball notes in `extensions/aline/DEVLOG.md`.

---

## 2026-06-12 — ALINE extension (`extensions/aline/`): joint amortized inference + active acquisition

- **Second `extensions/` entry: `extensions/aline/`** — ALINE (Huang et al., 2025,
  NeurIPS), joint amortized Bayesian inference and active data acquisition, on the
  GP-1D active-learning task, warm-started from the retained GP-1D checkpoint. The
  ACE-native re-expression is the point: the inference network is the **unchanged**
  core `ACE` (parameter/predictive targets are QUERY tokens; the paper's target
  specifier ξ collapses into *which target tokens are active*, and query candidates
  are data QUERY tokens), so the only new model surface is a small read-only policy
  decoder trained with REINFORCE on self-estimated information gain, behind a
  structural φ/ψ gradient firewall and a permanent bitwise parity guard on the
  inference path. First validation run (5k): the policy learns (beats random on
  both goal families, calibration kept); the US gap and the targeting contrasts
  remain open at that budget. Design decisions, paper deviations, and run results
  in `extensions/aline/DEVLOG.md`; plan + verification log in
  [docs/plans/PLAN-aline.md](docs/plans/PLAN-aline.md).

---

## 2026-06-12 — Per-tab playground explainers; retained AR-buffer weights swapped in

- **Every playground tab gained a "?" explainer.** A small inline button at the end of
  each tab's hint line opens a modal with three short didactic sections — the task,
  what ACE is doing, and how it compares with the classical approach — plus paper
  attribution (Chang et al. 2025; Hassan et al. 2026 on the AR tab). Deliberately
  matter-of-fact: each modal names the exact reference where one exists (the green
  oracle curves) and states the amortization trade-off plainly. Shared chrome in
  `playground/src/explain.ts` (reusing the global ACE-modal styles); content lives in
  each tab's `demo.ts`; smoke tests assert open/close.
- **The AR-buffer tab now serves the retained 200k concat-read weights**
  (`artifacts/gp1d_arbuffer.pt`, K=64, joint training, ~97% of the slow-AR
  joint-density gap — see `extensions/arbuffer/DEVLOG.md`): `ARBUF_CKPT` repointed,
  blob + fixtures regenerated together, all tests green.
- **The tab is deployed** (same day, after the repos moved to the `acerbilab` org):
  the `gp1d_arbuffer` blob was published to `acerbilab/nanoACE-playground-weights`
  and `pages.yml` now copies/validates five models. The tests' self-skip behavior
  stays (clones without local exports skip the arbuf suites), and in CI the blob is
  present so they run.
- **Slow-AR toggle in the AR-buffer tab.** A sampler radio reruns the chain via full
  context re-encoding (`sample_ar`'s recipe through the plain forward) with a single
  draw; the timing line reports cost *per draw* so the two modes compare directly
  (~20× slower per draw at 4 context points and K=64, growing with context size and
  chain length). Deliberately didactic: same model, same factorization, no cache.
- **The global "What is ACE?" modal now teaches amortization.** One-line ACE
  definition first, then four titled sections (per-tab explainer style); the new
  "Amortized inference" section defines the term the tab modals use in passing,
  trade-off included. Attribution moved up under the intro.
- **Slim sticky page footer.** One muted line (built on nanoACE, ACE project page,
  work by the MHI group and collaborators), `position: sticky; bottom: 0` so it
  stays visible; fuller references live in the modals. Hassan et al. (2026) deliberately stays in the AR
  tab's "?" modal (contextual, not bibliography). Inside the "What is ACE?" modal the
  attribution line moved from the bottom to right under the intro paragraph so it is
  visible without scrolling; it stays in the modal too, so the dialog stands alone.

---

## 2026-06-11 — AR-buffer playground tab (local-only)

- **The playground gained a fifth tab** running the `extensions/arbuffer/` buffered
  GP-1D model in-browser: context encoded once and cached, a few coherent joint
  function draws decoded autoregressively against the cache (animated), always shown
  next to the diagonal band and independent per-point marginal samples. TS port of
  the incremental sampler in `playground/src/ace/buffered.ts` (base port untouched),
  parity-guarded by new buffered fixtures (packed per-layer + teacher-forced chain).
  Same day, the port was switched to the extension's retained **concat-read**
  architecture (one softmax over `[context, buffer]` keys with the learned per-head
  `buf_bias` soft gate; separate-read blobs rejected at load), with fixtures
  regenerated from `gp1d_arbuffer_concat20k.pt`.
  **Local-only**: the temporary checkpoint is exported locally, not deployed — tab
  and tests self-skip when the blob is absent, so the four-model deploy flow is
  unchanged. Plan + verification:
  [docs/plans/PLAN-arbuffer-playground.md](docs/plans/PLAN-arbuffer-playground.md);
  design notes in `extensions/arbuffer/DEVLOG.md`.

---

## 2026-06-11 — `extensions/` taxonomy + AR-buffer extension (`extensions/arbuffer/`)

- **New root folder `extensions/` for non-core model extensions.** Each child is
  self-contained with its own local `README.md` and `DEVLOG.md`; the root docs (this
  file, `README.md`, `AGENTS.md`) carry only short pointers. Extensions are torch-only,
  change no core file, and import the core from the repo root (the playground's
  `sys.path` bootstrap pattern). `playground/` stays where it is — it is a port/demo,
  not a model extension.
- **First entry: `extensions/arbuffer/`** — the causal autoregressive buffer of Hassan
  et al. (2026, ICLR), attached to the retained GP-1D checkpoint as a warm-started,
  frozen-base fine-tune: encode the context once, cache it, and draw coherent joint
  function samples through a causal buffer stream instead of `sample_ar`'s
  re-encode-everything loop. Doubles as the extensibility demo: schema, embedder, heads,
  `train.fit`, checkpoints, GP physics, and the oracle diagnostics are all reused;
  only the buffer stream and the cached sampler are new. Design decisions, deviations
  from the paper (zero-init gated buffer read for bit-exact warm starts, all-buffered
  curriculum under a frozen base, no positional/role embeddings), and verification live
  in `extensions/arbuffer/DEVLOG.md`; the plan + full verification log in
  `docs/plans/PLAN-arbuffer.md`.

---

## 2026-06-08 — Offline data pool (`data.py`) + per-step reseed (stateless reproducibility)

Full plan + verification log: [docs/plans/PLAN-offline-data-and-reseed.md](docs/plans/PLAN-offline-data-and-reseed.md).

Adds `data.py` as a sharded-pool reader that honors the layout invariants, and makes
training reproducibility uniform. This updates the "Layout" / "Data layer" framing
(the saved-pool path now exists) and the "data RNG is not checkpointed" caveat from the
train-loop extraction entry.

- **Per-step reseed; `(step) -> Batch`.** `fit` now calls `torch.manual_seed(mix_seed(seed,
  step))` at the top of every step, so each batch is a pure function of `(seed, step)`:
  from-scratch runs reproduce, resumed runs replay the *exact* stream of an uninterrupted
  run, and the stream no longer depends on how much RNG model construction consumed.
  `mix_seed` is a splitmix64 hash (decorrelates consecutive step seeds); one `manual_seed`
  covers CPU + CUDA, sidestepping the dual-RNG-state fragility that made RNG-*state*
  checkpointing unattractive. The sampler thunk changed `() -> Batch` → `(step) -> Batch`
  (online thunks ignore `step`; the pool reader uses it).
  - **Reverses the earlier "keep online bit-identical" intent, deliberately.** Step-keyed
    reseeding changes the online stream vs the old continuous draw, so artifacts trained
    under the old stream are no longer seed-reproducible under the new code — fine
    (regenerable; the retrain/re-export is deferred). Verified on CPU: two same-seed runs
    give identical weights (max|dW|=0), and 0→20 vs 0→10-then-resume→20 give identical
    weights (max|dW|=0).
  - **Why stateless, not RNG-state checkpointing.** Reproducible resume comes from the
    stream being a pure function of an index (the modern PRNG-key approach) — robust across
    CPU/CUDA and device counts, where snapshotting `get_rng_state()` is fragile and only
    approximate on nondeterministic CUDA kernels anyway.

- **`draw`/`assemble` split (GP-1D, BO).** Each sampler splits into `draw_instances` (the
  expensive CPU-float64 physics — GP Cholesky / Matheron planting — the only part worth
  caching) and `assemble` (RNG-free tokenization, shared verbatim by the online and pooled
  paths). Done in two steps: first preserving RNG draw order so it was **bit-identical** to
  the pre-split online stream (proving the refactor introduced no bug; verified max|dW|=0),
  then the behavioral change below. Gaussian and SIR stay monolithic / online-only — their
  draws are cheap, so a pool buys nothing and the split would be churn.

- **Complement-targets.** GP-1D and BO moved from a fixed `data_targets` count to the
  standard "targets = all non-context points" (`n_target = N_TOTAL - n_context`,
  `N_TOTAL = 64` data points, `min_context=1`/`max_context=20`): no drawn point is wasted,
  and it is the natural pool layout. Tensorized as a width-`max_context` context block and a
  width-`N_TOTAL` target block (masked `>= n_context`), so context self-attention stays
  O(`max_context`²) and only the target cross-attention grows. `--data-targets` is now a
  no-op for GP/BO (still used by Gaussian/SIR). This is a training-distribution change (a
  retrain is expected eventually). Verified `n_target ∈ [44, 63]` and ≥1 context always
  (GP min 1; BO min 3, since its two optimum PRIOR tokens are always in context).

- **`data.py` = `write_pool` + `PoolReader`, one provenance check.** A pool stores only
  `draw_instances`'s struct-of-arrays (continuous float32; GP's categorical `kernel` int64)
  plus a manifest. The reader returns `Batch`es through the example's own `assemble`,
  recomputing the split (`n_context`) and reveal (`reveal_mask_from_index`) from a stateless
  `mix_int64`-keyed hash of the **absolute logical position** `p = (step-1)*B + j` — which is
  batch-size- and steps-independent (position `p` is the same dataset under any `B`), so the
  split stream is reproducible and resume-exact (verified max|dW|=0; B-independence verified
  for `n_context`, reveal, and physical row). The "both" shuffle (shard order + within-shard)
  is keyed on `(seed, pass)`. Build is resumable (per-shard `mix_seed`, skip valid shards,
  atomic write, manifest last).
  - **Kept: a single DGP config-hash.** The manifest carries the `variables()` schema (a hard
    gate — a wrong schema silently misreads the arrays; not overridable) and a `sha256` of
    the DGP `gen_config` (the non-schema constants; refused on mismatch, overridable with
    `--pool-force`). **Rejected: the multi-axis resume-guard matrix** — that is
    experiment-management machinery for a comparative study, which nanoACE is not.
  - **Frozen-in-pool vs free-at-assemble.** Frozen (regenerate to change): the DGP physics +
    `gen_config`. Free (no regenerate): batch size, `--steps`, and the reveal strategy /
    `latent_context_prob`. The CLI DGP-constant flags (`--jitter`, BO `--sigma-obs`/
    `--sigma-f-max`/`--prior-uniform-mix`) are wired to the same `GEN_*` constants
    `gen_config()` reports, so online and pool share one source of truth.
  - **Read-side memory.** `PoolReader` keeps only the manifest in memory at construction,
    maps each logical batch to touched `(shard, row)` pairs, lazy-loads shards through a
    bounded LRU cache, and uses a one-thread prefetcher for upcoming batch shards. Sharded
    pools therefore scale by shard size plus the cache and in-flight prefetch windows, not
    by full `pool_size`.

---

## 2026-06-08 - Current retained playground model state

- **Current retained runs.** The public playground weights are exported from
  retained runs under the shared multi-latent reveal DGP: Gaussian 80k steps,
  GP-1D 200k, SIR 100k, and BO-1D 200k. The corresponding playground parity/demo
  fixtures were regenerated together with those exports.
- **Repository boundary unchanged.** `artifacts/` and `playground/public/models/`
  remain gitignored in nanoACE; public deployment weights are versioned in the
  separate `lacerbi/nanoACE-playground-weights` repository and copied in by the
  Pages workflow.

---

## 2026-06-08 - Playground UI polish and separate weights repo

- **UI polish stayed inside the non-core playground.** The header now keeps the
  title, short description, fullscreen control, and "What is ACE?" modal trigger
  on one row; example controls expose reset/clear actions consistently, plus a
  uniform-priors action where runtime priors exist. Out-of-distribution warnings
  now render inside the plotting area so they do not move the surrounding layout.
- **The ACE explainer modal is deliberately generalist.** It describes ACE as a
  transformer probabilistic model in the same broad family as PFNs and TNPs, then
  emphasizes the token-level extension: conditioning and prediction can apply to
  data, interpretable latents, and runtime prior information, not only observed
  input-output pairs.
- **Playground weights live in a separate public repo.** The fp16 browser blobs
  remain gitignored under `playground/public/models/` in nanoACE. GitHub Pages now
  checks out `lacerbi/nanoACE-playground-weights` beside the app, copies only the
  model directories into that path before building, then fails fast if any
  expected `manifest.json`/`weights.bin` pair is missing or still a Git LFS
  pointer. It validates each manifest against its blob size, records the resolved
  weights commit and artifact hashes in the run summary, runs `npm test`, and
  then builds. The manual workflow accepts a `weights_ref` branch/tag/SHA, so a
  deploy can be pinned without changing nanoACE. This keeps ordinary nanoACE
  clones small while preserving a same-build checkout path for deployment.

---

## 2026-06-08 — Shared `train.py` (loop + checkpoint + light-YAML config + resume)

Full plan and verification log: [docs/plans/PLAN-train-loop-extraction.md](docs/plans/PLAN-train-loop-extraction.md).

- **Why.** The four examples carried byte-identical training loops, `build_model`,
  `save_checkpoint`/`load_checkpoint`, and ~21 overlapping CLI args. That boilerplate now
  lives in `train.py`; each example keeps only its task-specific science (`variables()`,
  the batch sampler, `evaluate()`, `plot_diagnostic()`) and a thin `main()`. This also adds
  two shared training features: **cosine LR** and **simple resume**.
- **`main()` is deliberately NOT centralized.** Keeping each example runnable/readable
  end-to-end is the guardrail (initial design: "no separate generic command-line wrapper").
  `train.py` exposes plain functions — `common_parser()`, `build_model(args, variables,
  device)`, `TrainConfig`, `fit(model, sample_batch, cfg, ...)`, the checkpoint helpers —
  not a `Task`/registry/config framework.
- **`fit` takes a sampler thunk.** This entry originally introduced a `() -> Batch`
  extraction; the current interface is `(step) -> Batch` after the offline-pool work above.
  The important retained decision is still one training path: examples pass online samplers,
  and `data.py`'s `PoolReader` satisfies the same interface.
- **No generic training prefetcher.** `fit` reads one batch per step synchronously. The
  optional offline path owns its shard-level lazy loading and prefetching inside
  `PoolReader`, where the deterministic pool schedule is available.
- **Cosine LR is the new default** (`--lr-schedule {cosine,constant}`, `--warmup`). This
  changes future *from-scratch retrains* vs the retained constant-LR artifacts — fine,
  artifacts are regenerable, and `--lr-schedule constant` exactly reproduces the old loop
  (used to prove the extraction is behaviour-preserving: all four examples reproduce their
  pre-refactor loss trajectories bit-for-bit under `constant`).
- **Simple resume** (`--resume`, `--ckpt-every`). A resumable checkpoint adds
  `{optimizer, scheduler, step}`; `load_train_state` restores them and continues from the
  saved step. Resume = **same total `--steps` budget** (cosine `T_max` is the run total, so
  a different `--steps` would mis-align the curve — `fit` warns on mismatch). The torch
  **RNG state is not checkpointed**, so a resumed run's data stream differs from an
  uninterrupted one — this is the intended "simple resume", not exact-stream replay.
- **Checkpoint format is backward compatible (additive keys).** `save_checkpoint` always
  writes `{cfg, seed, state_dict}`, adds `config` (resolved-run provenance, `vars(args)`)
  on the final save, and `{optimizer, scheduler, step}` only for resumable checkpoints.
  `load_checkpoint` reads only `cfg`/`state_dict`; legacy three-key checkpoint files load
  unchanged, and `load_train_state` tolerates the resume keys being absent. The **final
  `--save-checkpoint` stays model-only** (+`config`), so retained artifacts and the
  playground stay lean.
- **The one playground constraint.** `playground/export_weights.py` and
  `playground/parity.py` call `<module>.load_checkpoint(path, device)` with two args and
  use `<module>.variables()`. So each example keeps a 2-arg `load_checkpoint(path, device)`
  wrapper forwarding to `train.load_checkpoint(path, device, variables())`. No playground
  file changed; `parity.py` regenerated its 8 tracked fixtures byte-identically after the
  refactor (load+forward path unchanged).
- **Seed ordering — from-scratch is exactly preserved.** `main()` seeds once, then
  build-or-load, then `fit` (which draws no RNG before the first batch). The
  load-then-train path's RNG timing shifts slightly (today's `load_checkpoint` ran before
  the seed; now it runs after) — immaterial: it was never a reproducibility guarantee and
  `--eval-only` never trains.
- **Config: dataclass + light YAML, no framework.** `--config run.yaml` is layered under
  explicit CLI flags via `set_defaults` (precedence: parent/per-example defaults < YAML <
  CLI); YAML keys are arg dest names (underscored) and unknown keys are rejected so typos
  fail loudly. `TrainConfig.from_args` feeds `fit`. **Hydra/omegaconf were considered and
  declined**: Hydra is the "config framework" this repo excludes, adds heavy deps, and its
  `@hydra.main` working-directory change would relocate the examples' relative `artifacts/`
  outputs. Adds one small dep (PyYAML, pinned in `requirements.txt`).

---

## 2026-06-07 - BO-1D playground tab

- **BO added to the non-core web playground.** `playground/` now has a fourth tab for
  `bo1d.py`: editable black-box observations, finite Beta prior tokens for `x_opt` and
  `y_opt`, optional zero-spread fixed values for either optimum latent, and live ACE
  predictive bands. The optimum-location marginal is overlaid along the plot's x-axis
  and the optimum-value marginal along the y-axis, so the BO quantities stay in the same
  visual frame as the regression surface.
- **No oracle, consistent with `bo1d.py`.** The tab does not add a browser-side simulator
  posterior or a BO rollout. It only exposes ACE's amortized predictions and conditioning
  interface. Verification is parity plus Python-orchestration fixtures, not truth-quality.

---

## 2026-06-07 — Single shared multi-latent reveal strategy (mixture DGP)

All four examples now decide *how many* latents are revealed as context through one
helper, `ace.sample_reveal_mask`, under a single mixture. Per task:

- with probability `q`, reveal **nothing** (pure inference / pure-prior — the headline);
- otherwise split the revealing mass 50/50 between **uniform over subsets** (a uniform
  random non-empty subset) and **uniform over count** (count `k` uniform in `1..L`, then
  a uniform random size-`k` subset).

Resulting count distribution (`q = 1 - latent_context_prob`, default `q = 0.5`):
L=2 → `{0:.50, 1:.29, 2:.21}`; L=3 → `{0:.50, 1:.19, 2:.19, 3:.12}` (verified empirically).

- **Why a mixture, not one scheme.** Uniform-over-subsets is fair to every *specific*
  pin pattern but starves the extremes (revealing *all* L latents is `1/(2^L-1)` of the
  reveal mass — only ~0.07 at L=3). Uniform-over-count keeps every *count* (incl.
  all-revealed) well represented but over-weights the lone all-revealed subset. The
  50/50 blend keeps the per-subset floor while lifting "pin everything" (GP all-3 goes
  from ~0.07 to ~0.12 of total mass), which matters for the playground's "pin an
  arbitrary subset and predict" interaction. The `q` knob keeps the 0-reveal headline
  explicit and L-independent.
- **All four examples unified.** `gaussian_toy.py` and `gp1d.py` already used the helper
  (they inherit the new internals). `sbi_sir.py` and `bo1d.py` were migrated off their
  private `xor` single-reveal logic (which revealed *exactly one* latent, never both) to
  `sample_reveal_mask(2, …)`. Two-pin SIR / BO contexts are now in-distribution. The
  `--latent-context-prob` default is standardized to **0.5** across all four (was 0.20
  for SIR/bo1d), matching the agreed "½ reveal nothing".
- **Reveal-all is safe.** SIR and bo1d build targets as `[latent, latent, data…]` with
  the data-`y` columns always active, so a both-revealed row still predicts data (the
  forward/simulate direction) — no empty-target row. Gaussian/GP already allowed
  reveal-all under the old scheme.

---

## 2026-06-07 — 1D Bayesian optimization example (`bo1d.py`)

Full design in [docs/plans/PLAN-bo1d.md](docs/plans/PLAN-bo1d.md). Status: **built and run**. The plan
was checked by two reviewers, revised, then implemented and validated (CPU run,
torch 2.12.0). The DGP, training, and three-prior diagnostic work end to end;
`--scale-check` confirms data token values sit in `[-1, 1]` (~0.5% tail spill).
The structural checks pass: uniform→correct tightens/shifts `p(x_opt | D)` toward
truth, and the wrong prior is resisted (posterior stays near the data, not the
wrong prior, thanks to the ε floor). The effect is directionally correct but
modest (the chosen fixed case is deliberately hard -- the true optimum sits
unobserved between context points -- and ε=0.05 caps prior influence); sharpening
it is optional loose tuning, recorded in the plan's Status. This entry reflects
the revised plan (see `docs/plans/PLAN-bo1d.md` "Review notes").

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
  `(1−ε)·Beta + ε·Uniform` (default ε=0.05, a classic robust-Bayes / ε-contamination
  prior), applied to both latents. **Why it is not redundant with the existing
  `sample_prior_params` mixture:** that helper always draws truth from the *same*
  Beta the token encodes -- the token never lies -- so a model trained on it learns
  to trust a concentrated token fully and the wrong-prior column would fail.
  ε-contamination's job is to **decouple truth from the token** a fraction ε of the
  time, the one thing `sample_prior_params` cannot do. So `bo1d` reuses
  `sample_prior_params` for the token and applies contamination *only at the
  truth-draw* (new `ace_prior_beta.sample_contaminated`). It lives entirely in the DGP
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
  a tamed `σ_f`. The scale check passed with the current constants: token values sit
  near `[-1, 1]` and the drawn-`x_opt` marginal matches the contaminated prior.
  Stochastic tails outside `[-1, 1]` are accepted (soft convention).
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
  from 8 to 12 for the multimodal location posterior. Small CPU validation runs
  showed that equal-step comparisons were dominated by training budget rather
  than capacity; judge retained playground behavior from the later exported run
  recorded in the current-state entry, not from the wiring-time CPU checks.
  Defaults kept faithful at 6 blocks and `ε=0.05`.
- **Scope note.** This is example #4; "nano ships exactly two" (initial design) is
  already stretched by SIR. BO earns its place: it adds instance-level latents, the
  optimum-posterior headline, and the robust prior-injection mechanism, none of
  which the other examples cover. Recorded as a deliberate decision.

---

## 2026-06-07 — SIR playground tab

- **SIR added to the non-core web playground.** `playground/` now has a third tab for
  `sbi_sir.py`: editable infected-fraction observations, runtime Beta prior controls for
  `beta` and `gamma`, ACE predictive curves/marginals, and a live browser-side numerical
  SIR grid oracle. This stays entirely in the playground toolchain; `ace.py` and the
  Python SIR example are unchanged.
- **Oracle is practical in-browser.** Unlike the GP oracle, SIR only needs deterministic
  RK4 trajectories over a `(beta, gamma)` grid plus Gaussian likelihood scoring. The TS
  implementation caches the fine-grid trajectories and reuses them as observations and
  priors change, so the live oracle is cheap enough for the UI.
- **Export/parity extended.** `playground/export_weights.py` accepts `--task sbi_sir`;
  `playground/parity.py` now writes SIR parity and demo fixtures; `npm test` covers the
  SIR forward parity, ACE orchestration, TS oracle, and UI smoke path.

---

## 2026-06-07 — Web playground (in-browser TS port)

- **A non-core interactive demo lives in `playground/`.** It is an *example*, not
  part of nanoACE: the core stays torch-only and legible, while `playground/`
  carries a Vite + TypeScript toolchain. It reimplements `ace.py`'s forward pass
  in TS so trained models run fully client-side (GitHub Pages, no server). The
  playground started with GP-1D (add/drag points, infer the kernel, **pin a latent
  and predict**) and Gaussian (Beta-prior sliders + observed `y`, with the analytic
  oracle overlaid), and now also includes SIR (see the SIR playground entry above).
  The headline is that amortized conditioning is instant — a forward
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
- **Weight hosting is now resolved by a separate public repo.** The exported fp16
  blobs (`playground/public/models/`) stay gitignored in nanoACE, and GitHub
  Pages checks out `lacerbi/nanoACE-playground-weights` beside the app and copies
  only the model directories into that path before the Vite build. This avoids
  committing binary churn to nanoACE while keeping the deployment flow same-build
  and explicit; update blobs and parity fixtures together after retraining.

---

## 2026-06-07 — SIR simulation-based-inference example + `ace_prior_beta.py`

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
- **Shared prior helpers extracted to `ace_prior_beta.py`.** The Beta prior-token helpers
  (`sample_prior_params`, `beta_alpha_beta`, `prior_features`, `known_latent_features`,
  `draw_from_beta`, `beta_logprior_on_grid`) were a pure move out of `gaussian_toy.py`;
  both prior-conditioning examples now import them. The `(mean, spread)` prior token is a
  core ACEP representation, so a shared module is the right home. Audit: the model-side
  prior code (`PRIOR_FEATURES`, `spread_embed`, the `_embed` payload) and the general
  `encode_value`/`decode_value` coordinate maps stay in `ace.py`; `ace_prior_beta.py`
  imports `PRIOR_FEATURES` from the core and asserts the two-feature layout. The
  helper module is Beta-specific; model-side PRIOR token semantics stay in `ace.py`.
- **Deferred.** No discrete-latent runtime prior token (still deferred from the prior
  redesign). The SIR AR two-latent joint heatmap is computed-capable but not plotted, to
  keep the uniform-vs-informative contrast legible; the marginals carry the story.
- **Multi-reveal migration (DONE 2026-06-07).** `sbi_sir.py` (and `bo1d.py`) were migrated
  off the single-reveal `xor` onto the shared `sample_reveal_mask`, so two-pin contexts are
  now in-distribution and all four examples share one reveal helper. See the "Single shared
  multi-latent reveal strategy" entry above.

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
- **Oracle grid sanity check.** The fixed GP diagnostic was rerun with
  `--oracle-bins 32`, `64`, and `96`. Kernel posterior probabilities,
  latent posterior moments, and predictive RMSE were stable to the printed precision,
  so the default 64-bin oracle is adequate for this diagnostic. The RBF integrated
  marginal likelihood delta moves by about 0.1 log units across those grids, but RBF has
  posterior mass near 0.002 in this case, so that movement is not practically relevant.
  This is a one-case numerical check, not a benchmark harness.

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
- **No generic training-loop prefetcher.** Generate→save moved the expensive Cholesky
  *off* the training loop, so `train.py` stays synchronous. The concrete `PoolReader`
  may still lazy-load and prefetch shard files because the deterministic pool schedule
  tells it exactly which shards upcoming batches will need.
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
- `data.py`        — offline sharded data pools (`write_pool` + `PoolReader`)
- `train.py`       — shared training loop, checkpointing, config, and resume
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
