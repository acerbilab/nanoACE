# Plan: Causal AR-buffer extension (`extensions/arbuffer/`)

Created: 2026-06-11
Status: COMPLETE (2026-06-11) — implemented + validated (20k fine-tune); the retained
100k–200k artifact run is deferred per plan (run separately, fresh `--steps` budget).

Reference: Hassan et al. (2026), *Efficient Autoregressive Inference for Transformer
Probabilistic Models* (ICLR 2026). Paper markdown in `extensions/arbuffer/paper/`
(main + appendix; appendix A has the module/mask/algorithm details, H.2 the positional
embedding ablation, H.5 the K=64 GP buffer-size ablation).

## Progress tracker

Phase 1 — model (`extensions/arbuffer/arbuffer.py`):
- [x] `extensions/arbuffer/` skeleton + repo-root `sys.path` bootstrap (playground pattern)
- [x] `BufferedBatch` dataclass (`context`, `buffer`, `target`, `prefix_len`, `.to`)
- [x] `BufferBlock` (buffer-stream attention + MLP; gated target→buffer read)
- [x] `BufferedACE(ACE)`: `buf_blocks` ModuleList, `init_from_base`, `freeze_base`,
      `forward_buffered`, `loss` override
- [x] `load_warm_start(path, device)` with the strict missing/unexpected-keys guard
- [x] automatic step-0 exactness self-check at warm start
- [x] verify: key guard, `torch.equal` parity (plain forward; zero-gate with non-empty buffer)
      — `artifacts/scratch_arbuffer_phase1.py`: all bitwise checks pass; grads reach only
      `buf_blocks.*` under freeze; prefix-0 bit-equal even with non-zero buffer weights

Phase 2 — fine-tune script (`extensions/arbuffer/gp1d_arbuffer.py`):
- [x] three-way online sampler (context / buffer / complement targets) + curriculum
- [x] CLI on `train.common_parser()`; main flow (warm start → freeze → `fit` → save)
- [x] step-0 loss readout = base context-only NLL on buffered targets (printed note)
- [x] verify: `--steps 20` smoke (CPU + CUDA) pass; post-run frozen parity bit-equal;
      `--eval-only --load-checkpoint` round-trips; `gp1d.evaluate` runs unchanged on
      the buffered model

Phase 3 — inference (`arbuffer.py`):
- [x] `encode_context` per-layer state cache
- [x] `sample_joint` (batched incremental decode, shared context cache; also
      teacher-forced scoring mode, returns per-step log-probs)
- [x] `joint_log_prob` (one-pass packed evaluation + order averaging, explicit
      `orders` for like-for-like comparisons) + prefix-0 guard
- [x] `slow_ar_log_prob` baseline (teacher-forced sequential append, base path)
- [x] verify: one-pass vs step-by-step agree (atol 1e-5); incremental KV caches vs
      one-pass max |diff| 4.8e-7; prefix-0 path NaN-free and bit-equal to plain

Phase 4 — demo + the actual experiment:
- [x] wire demo eval + plot into `gp1d_arbuffer.py` `main()` (Phase 2 ends at save)
- [x] demo plot (conditioning columns × coherent-draw spaghetti, truth overlay)
- [x] joint-NLL table (diagonal | slow-AR | buffered, identical orders) + measured timing
- [x] `--resume` validated end-to-end (snapshot at step 50 → continues at 51 with
      restored optimizer/scheduler; cosine-mismatch warning fires; parity holds)
- [x] validation fine-tune (20k) from the retained GP-1D 200k checkpoint (CUDA, ~35 min):
      loss −0.47 (= base context-only NLL, step-0 readout confirmed) → ~−1.5
- [x] K=128 variant (denser 128-point sampling grid; the K=64 grid renders choppy):
      `--buffer-size 128 --n-points 192 --sample-points 128`, 20k steps, separate
      artifact `gp1d_arbuffer_k128.pt/.png` — no code change needed (no positional
      embeddings → K is a training-distribution choice). Results: loss −0.34 → −1.86;
      joint density per point on K=128 chains: diagonal −0.22 < buffered +1.37 <
      slow-AR +1.61 (~87% of the AR gap recovered, vs 76% at K=64); parity bit-equal;
      CUDA sampling 0.7× (more launch-bound steps — recorded honestly). Figure
      regenerated with the 1.5×-lengthscale demo function: visibly smoother draws.
      K=128 settings are the recommendation for the retained long run.
- [ ] retained-artifact fine-tune at 100k–200k once the recipe is validated (deferred;
      user runs this separately — fresh run, not a resume of the 20k one; buffer size
      per the K=64 vs K=128 comparison outcome)
- [x] verify: joint log-density per point (16 held-out fns, 4 ctx, K=64, 4 shared orders):
      diagonal −0.36 < buffered +0.92 < slow-AR +1.33 (buffered recovers ~76% of the
      AR gap at 20k steps); marginals bit-equal to base after fine-tune; demo plot shows
      coherent draws tightening 4→14 points and under the kernel pin; CUDA sampling
      wall-clock ≈ 1.0× vs sample_ar (launch-bound at nano scale, as recorded)

Phase 5 — docs:
- [x] `extensions/arbuffer/README.md` (what/why, reuse story, recipe, commands)
- [x] `extensions/arbuffer/DEVLOG.md` (local design log: deviations + rationale)
- [x] root `DEVLOG.md`: short dated pointer entry (extensions/ taxonomy + arbuffer)
- [x] root `README.md`: one Current Status bullet
- [x] `AGENTS.md`: short `extensions/` boundary note + currently-implemented mention
- [x] `/doublecheck` pass — two Opus reviewers (code correctness; docs/plan fidelity).
      No blockers, no code defects. Fixed from findings: local-DEVLOG GPU timing
      corrected to the final measured ~1.0×; Fig. A17 claim explicitly attributed to
      the paper (ours recovers most-but-not-all of the AR gap at 20k); clean
      `--buffer-size >= 1` guard; sampler closure bound to a local. Reviewer-verified:
      mask semantics (inclusive causal, prefix, batch-major 3D layout), no padded-row
      leakage, device/dtype hygiene, optimizer-state consistency under freeze+resume,
      no overstated performance claims, staleable numbers confined to this tracker.

## Summary

Add the paper's causal autoregressive buffer as a **non-core extension** in
`extensions/arbuffer/` — the first entry of a new `extensions/` taxonomy (peer of
`playground/`: checked in and maintained, but not part of the core; the core stays
torch-only and unchanged). The extension warm-starts from the retained GP-1D
checkpoint, attaches a per-layer buffer stream plus a zero-init gated target→buffer
read, freezes the base, and fine-tunes only the new parameters. The demo draws many
coherent GP-style function samples (K=64-point curves, B=64 draws sharing one cached
context encoding) from as few as 4 conditioning points, with `sample_ar` and the
diagonal marginals as references.

The extension is itself the message: tokens, embedder, heads, loss, `fit`,
checkpointing, and the GP physics are all reused; only the buffer stream and the
cached sampler are new. Warm-starting a pretrained set-based model is the paper's own
stated future work (its Discussion: "could be directly applied to pretrained
NPs/PFNs"), so this is a demonstration beyond the paper's tested recipe, framed
honestly as such.

## Scope

- **In scope**
  - `extensions/arbuffer/arbuffer.py`: `BufferedBatch`, `BufferBlock`,
    `BufferedACE(ACE)`, warm-start loader + guards, batched incremental sampler,
    one-pass joint log-prob, slow-AR baseline evaluator.
  - `extensions/arbuffer/gp1d_arbuffer.py`: fine-tune CLI (online sampler at
    `n_points=128`), demo diagnostics + plot, checkpoint save/load.
  - Local `README.md` + `DEVLOG.md` inside the extension; **brief pointer entries
    only** in root `DEVLOG.md` / `README.md` / `AGENTS.md`.
- **Out of scope**
  - Any change to `ace.py`, `train.py`, `gp1d.py`, `data.py` (zero core edits; the
    extension imports them).
  - Offline-pool support for the 128-point fine-tune DGP (online Cholesky at 128
    points is cheap; a pool can come later if a long run wants it).
  - Buffers for Gaussian/SIR/BO; latent variables in the buffer (v1 buffer and
    fine-tune targets are **data tokens only** — conditioning on latents stays fully
    supported via context pins, which flow through the cached encoder unchanged).
  - Playground integration of the buffered model (a playground tab is anticipated
    as separate, later work); KV-projection caching / kernels; RoPE/ALiBi; tuning
    the `--no-freeze-base` joint-training path (mechanism present, recipe coupled
    to the flag, not tuned).

## Design decisions (seed of the local DEVLOG)

1. **`extensions/` taxonomy.** Root-level `extensions/`, each child self-contained
   with its own README/DEVLOG. Non-core like `playground/` in spirit; unlike
   `playground/` these are torch/Python and may subclass core internals. Scripts use
   the playground's `sys.path` bootstrap to import root modules (`ace`, `gp1d`,
   `train`); run from the repo root.
2. **Three token streams per layer; targets still never attend to each other.** The
   buffer is a third token set carrying realized `(x, y)` values (teacher-forced in
   training, self-generated at sampling), not "causal attention among targets" —
   paper requirement R4 is preserved, and so are the base invariants (context never
   reads buffer/targets; conditioning direction stays structural).
3. **Load-bearing deviation — split target read, zero-init gate.** The paper's target
   decoder is a *single* cross-attention over concatenated `[ctx KV, buffer-prefix KV]`
   (appendix A.1). Softmax over a concatenated KV set renormalizes attention mass: no
   initialization of the buffer-side projections makes the buffer's contribution zero
   (zero K-projections give logits 0, i.e. weight `exp(0)=1` per buffer key — at 4
   context points and a 63-token buffer, ~90%+ of the pretrained context read would be
   diluted at init). So the extension reads the buffer through a **separate residual
   cross-attention with a zero-init output projection**: at step 0 the term is exactly
   0 regardless of buffer content, and warm start is bit-exact. Expressivity
   difference (additive contributions instead of one competing softmax) is accepted —
   it is the price of an exact warm start the paper never needed (it trains from
   scratch).
4. **Buffer stream is paper-faithful, initialized as a context-weight copy.** Per
   layer, one attention with `q = buffer`, `kv = [cached ctx layer-input, buffer]`,
   causal mask on the buffer block — the paper's mask blocks (2)+(3) fused, which is
   how its single training mask works anyway. Weights initialized by **copying**
   `ctx_attn`/`ctx_mlp`/LNs: at init, buffer tokens are encoded as if appended to the
   context and run through context self-attention (minus the back-edges R3 forbids) —
   the same thing `sample_ar` does today by literally appending. Copies, not shared
   modules, so fine-tuning the buffer stream cannot drift the frozen base.
5. **Freeze base by default → all-buffered curriculum.** With the base frozen and the
   buffer read zero-gated, a `v_m = 0` (context-only) target's loss is a constant —
   zero gradient reaches any trainable parameter. So the fine-tune samples
   `v_m ~ Uniform{1..K}` for **every** target; marginal preservation is structural,
   not trained. `--no-freeze-base` restores the paper's 50/50 split (the only thing
   protecting marginals under joint training); the curriculum is derived from the
   freeze flag, never set independently. Bonus: step-0 training loss equals the base's
   context-only NLL on the buffered targets, so the loss curve is a direct readout of
   the information extracted from the buffer.
6. **No buffer positional embeddings.** Paper H.2 ablates them: no significant
   difference (GP: 2.51 vs 2.51). GP function values are exchangeable given `(x, y)`
   pairs, and dropping them removes the learned-position cap on buffer length —
   K becomes a training-distribution choice, not an architecture constant. H.5
   validates K=64 on GP regression specifically.
7. **No buffer role embedding — a recorded deviation.** Appendix A.1 *does* give
   every token a role embedding (`e_ctx`, `e_buf`, `e_tgt`); the paper needs them
   because its tokens share one masked stream. nanoACE's streams are distinguished
   structurally (which attention processes them), and the buffer-stream weights are
   independent copies that can differentiate during fine-tuning, so the buffer role
   vector is dropped. Fallback if training wants it: a single zero-init `d_model`
   vector added to buffer embeddings (preserves step-0 exactness).
8. **Inclusive causal mask.** Buffer token `j` attends to `[ctx, buffer 1..j]`
   (diagonal inclusive), not the paper's strict `< j`: it matches the
   copied-context-self-attention init story, and target `k` reads only prefix
   `1..k-1` either way, so the predictive factorization is identical.
9. **Buffer tokens reuse `ACE._embed` verbatim** as VALUE-mode data tokens (`x` +
   `value`), exactly what `Predictions.sample_as_context_tokens` builds for
   `sample_ar` today. No new embedding machinery.
10. **Fine-tune DGP = base DGP with a bigger point budget.** `gp1d.draw_instances`
    already takes `n_points`; the fine-tune draws 128 points (vs `N_TOTAL=64`),
    keeps the base context distribution (`n_context ~ U{1..20}`) and the shared
    latent-reveal DGP (`latent_context_prob=0.5`, so pinned-latent contexts stay
    in-distribution *with* a buffer), takes the next 64 points as the buffer, and
    targets = all remaining non-context, non-buffer points (complement style;
    `M = 128 − 64 − n_context ∈ [44, 63]`). Latent queries are excluded from
    fine-tune targets (their marginal path is frozen; latent-in-buffer is out of
    scope). `gp1d.py` is untouched; the three-way assemble lives in the extension.
11. **Honest performance framing.** No 20× claims at nano scale (that figure is
    N=1024). The measured demo wins: one context prefill shared across B=64 draw
    streams, long-chain conditioning that is *trained* (today's `sample_ar` runs the
    base on context sizes far beyond its `max_context=20` training range by the end
    of a 64-point chain — paper Fig. A17 shows the buffer tracking and slightly
    exceeding re-encoding AR in this small-N regime), plus whatever wall-clock
    factor we measure.

## Architecture spec

### `BufferedBatch`

```python
@dataclass
class BufferedBatch:
    variables: list[Variable]
    context: Tokens          # as base
    buffer: Tokens           # [B, K] VALUE-mode data tokens, fully active
    target: Tokens           # [B, M] QUERY tokens (data only), truth in value
    prefix_len: torch.Tensor # [B, M] long, visible buffer prefix v per target
    def to(self, device) -> "BufferedBatch": ...
```

### `BufferBlock` (one per base layer)

New modules (all under `buf_blocks.*` in the state dict):

| module          | role                                   | init                          |
|-----------------|----------------------------------------|-------------------------------|
| `buf_ln1`       | LN for buffer-attention q and kv       | copy of paired `ctx_ln1`      |
| `buf_attn`      | MHA: buffer → `[ctx, buffer]` (causal) | copy of paired `ctx_attn`     |
| `buf_ln2`       | LN before buffer MLP                   | copy of paired `ctx_ln2`      |
| `buf_mlp`       | buffer MLP                             | copy of paired `ctx_mlp`      |
| `tgt_buf_qln`   | LN for target queries (buffer read)    | copy of paired `tgt_ln1`      |
| `tgt_buf_kvln`  | LN for buffer KV (target read)         | copy of paired `kv_ln`        |
| `tgt_buf_attn`  | MHA: target → buffer prefix            | **`out_proj` zero-init**      |

### `BufferedACE(ACE)` per-layer forward (`forward_buffered`)

For layer `ℓ` with base block `blk` and buffer block `bblk`
(`ctx_in` = context state entering the layer, `ctx_out` = after the base update):

```python
# all attention calls pass need_weights=False, mirroring ACEBlock — the True
# path takes a different kernel and breaks bitwise step-0 parity
# 1. base context update — frozen, code path identical to ACEBlock
ctx_q = blk.ctx_ln1(ctx); ctx = ctx + blk.ctx_attn(ctx_q, ctx_q, ctx_q, key_padding)
ctx = ctx + blk.ctx_mlp(blk.ctx_ln2(ctx))
# 2. buffer update — attends over layer-INPUT ctx states + causal self
q = bblk.buf_ln1(buf); kv = bblk.buf_ln1(torch.cat([ctx_in, buf], dim=1))
buf = buf + bblk.buf_attn(q, kv, kv, key_padding_mask=pad_ctx_plus_buf,
                          attn_mask=causal_on_buffer_block)   # [K, N+K], inclusive
buf = buf + bblk.buf_mlp(bblk.buf_ln2(buf))
# 3. base target read of updated context — frozen, identical to ACEBlock
kv_c = blk.kv_ln(ctx); tgt = tgt + blk.cross_attn(blk.tgt_ln1(tgt), kv_c, kv_c, key_padding)
# 4. NEW gated buffer read — per-target prefix mask, exactly 0 at init
kv_b = bblk.tgt_buf_kvln(buf)
tgt = tgt + bblk.tgt_buf_attn(bblk.tgt_buf_qln(tgt), kv_b, kv_b,
                              attn_mask=prefix_mask)          # [B*heads, M, K]
# 5. base target MLP — frozen
tgt = tgt + blk.tgt_mlp(blk.tgt_ln2(tgt))
# 6. zero masked rows (as ACEBlock does)
```

This re-implements ~15 lines of `ACEBlock.forward` in the extension, calling the base
block's own submodules in the same order — deliberate (it shows the block's anatomy;
the step-0 parity test catches drift if `ace.py`'s block ever changes, the same way
the playground parity test guards the TS port). The inherited plain `forward(batch:
Batch)` is untouched, so `gp1d.evaluate`, `kernel_posterior`, `query_log_density`,
and `diagnostics.py` all run on a `BufferedACE` unchanged.

Masks: causality is a 2D `[K, N+K]` bool `attn_mask` (context columns visible,
buffer block lower-triangular inclusive) combined with the per-row context
`key_padding_mask`; the per-(row, target) prefix visibility is a 3D
`[B*heads, M, K]` mask (`visible[j] = j < v_m`). Prefix-0 rows (only in the one-pass
evaluator's first position and the unfrozen 50/50 mode) attend to buffer slot 0 and
have their output multiplied by `(v_m > 0)`. On the pinned torch 2.11 a fully-masked
attention row already returns exact zeros (verified during review), so the gate is
version-robustness rather than a NaN fix — kept because it makes the prefix-0
semantics explicit instead of relying on undocumented masked-row behavior.

`loss(batch, *, data_weight, latent_weight)` accepts a `BufferedBatch` (NLL over
targets via `forward_buffered`) and delegates plain `Batch` to `super()` — so
`train.fit` works verbatim (it only calls `sample_batch(step)` and
`model.loss(...)`; Adam/`clip_grad_norm_` skip frozen `grad=None` params).

### Warm start, freeze, checkpoints

- `load_warm_start(path, device)`: read payload, `ACEConfig(**payload["cfg"])`,
  construct `BufferedACE(gp1d.variables(), cfg)`,
  `load_state_dict(payload["state_dict"], strict=False)`; **assert**
  `unexpected_keys == []` and every missing key starts with `buf_blocks.`; then
  `init_from_base()` (the copies + zero gate) and the automatic step-0 self-check.
- `freeze_base()`: `requires_grad_(False)` on everything except `buf_blocks`.
- Step-0 self-check (runs at every warm start — not on `--load-checkpoint`/
  `--resume`, where the gate is trained and non-zero — cheap and loud): on
  `gp1d.fixed_eval_batch(variables(), device=…, points=…, jitter=GEN_JITTER)`
  (kwargs required, no defaults), (a) `BufferedACE` plain forward vs base `ACE`
  forward — `torch.equal` on `cont_raw`/`disc_logits` (identical ops, exact); (b)
  `forward_buffered` with a synthetic random non-empty buffer vs context-only —
  `torch.equal` (the gate adds exact zeros). If a kernel-dispatch difference ever breaks bitwise
  equality, fall back to `allclose` with a tight tolerance and record why.
- Saving: `train.save_checkpoint` as-is (full state dict; `BufferedACE` shares
  `ACEConfig` — K is not architectural). Provenance `config` records the base
  checkpoint path and fine-tune args. `load_buffered_checkpoint(path, device)`
  builds `BufferedACE` and strict-loads (for `--eval-only` / demo reuse).

### Inference (`arbuffer.py`)

- `encode_context(model, context)`: run the frozen context stream once; cache the
  per-layer **input** states `ctx_0..ctx_{L-1}` (for buffer attention) and per-layer
  **output** states (for the target read). Plain hidden-state caches; KV projections
  recomputed per read — legible, and a micro-opt at nano scale.
- `sample_joint(model, context, x_grid, n_draws, order=None)`: expand the cached
  context states across `n_draws`; maintain per-layer buffer input-state caches
  `[n_draws, k, d]`. Per step `k`: embed the query, run the target path against
  `[ctx cache, buffer cache]`, sample from `Predictions`, embed `(x, y_k)` as a
  VALUE token, run it through the buffer stream (attending to ctx cache + cached
  buffer states), append its per-layer states. Random order by default (matches
  `sample_ar` and the random-order training buffers); step 1 skips the buffer read.
- `joint_log_prob(model, context, x, y, order, n_orders=...)`: Algorithm 2 — pack K
  ordered buffer tokens + K queries with `prefix_len[k] = k−1` in one
  `forward_buffered` pass; sum per-token `log_prob`; average over `n_orders` random
  orders (paper H.3).
- `slow_ar_log_prob(model, batch, order)`: teacher-forced sequential evaluation via
  the inherited plain forward + `append_or_replace_context_token` (the `sample_ar`
  recipe with truth instead of samples) — the slow-AR baseline for the NLL table.

### Fine-tune script (`gp1d_arbuffer.py`)

- Sampler (per step, under `fit`'s per-step seed):
  `inst = gp1d.draw_instances(B, n_points=128, jitter=GEN_JITTER)`;
  `n_context ~ U{1..20}`; reveal via `sample_reveal_mask(3, ...)`; context block
  tensorized as in `gp1d.assemble` (20 candidates + 3 latent slots); buffer = points
  `[20, 84)` (width `--buffer-size`, all active); target block = points
  `[0, 20) ∪ [84, 128)` (width 64), with the candidate part masked `idx ≥ n_context`
  and the tail all-active — every non-context, non-buffer point is a target, the
  base complement convention with the buffer slice carved out; no dead points;
  `M = 64 − n_context ∈ [44, 63]`. Data-only — no latent queries.
  `prefix_len ~ U{1..buffer_size}` per (row, target) — or the 50/50 split iff
  `--no-freeze-base`.
- CLI: `argparse` with `parents=[train.common_parser()]`;
  `set_defaults(batch_size=64, max_context=20, min_context=1, steps=20000,
  plot_path="artifacts/gp1d_arbuffer.png")`; new flags `--base-checkpoint`
  (default `artifacts/gp1d.pt`), `--buffer-size` (64), `--n-points` (128),
  `--no-freeze-base`, `--draws` (64), `--sample-points` (64), `--orders` (4).
  Model-shape flags are ignored on warm start (shape comes from the base `cfg`).
- `main()` flow (example-style, end-to-end readable): warm start → freeze → print
  step-0 loss readout → `train.fit` → save. The demo eval + plot are appended to
  this flow in Phase 4 (until then `main` ends at save). Three model-construction
  branches, mirroring `gp1d.main`:
  - default: `load_warm_start(--base-checkpoint)` (runs the step-0 self-check);
  - `--load-checkpoint`: `load_buffered_checkpoint` (strict; fine-tuned weights —
    **skip** the step-0 self-check, the gate is no longer zero);
  - `--resume`: `load_buffered_checkpoint(args.resume)` (strict; **not** a base
    warm-start, which would discard the trained buffer weights and re-zero the
    gate), then `fit` with `resume_state` restoring optimizer/scheduler/step.
  Freezing is applied after whichever branch ran.

### Demo + evaluation

- Fixed case: reuse `gp1d.fixed_eval_batch` (same underlying function); columns =
  conditioning variants sharing that function: (a) 4-context subset of the fixed 14
  points, (b) all 14, (c) 4-context + kernel pinned (VALUE token, as in
  `gp1d.assemble`). Per column: B=64 buffered coherent draws over a 64-point grid
  (thin spaghetti), true function + context overlaid, diagonal ±2σ band for
  reference. Style mirrors `gp1d.plot_diagnostic` (matplotlib inside the function,
  `artifacts/` path, dpi 160).
- Printed table: joint NLL of held-out truth (the fixed case + a small batch of
  random functions) under three readings — diagonal (sum of independent marginals),
  slow-AR (base path, teacher-forced), buffered one-pass (order-averaged) — plus
  measured wall-clock for B×K sampling via `sample_ar` vs `sample_joint`, and the
  frozen-parity confirmation line.

## Verification

1. **Warm-start guard**: `unexpected == []`, `missing ⊆ buf_blocks.*` (hard assert).
2. **Step-0 exactness** (automatic): plain forward bit-equal to base; zero-gated
   buffered forward bit-equal to context-only.
3. **Frozen-base invariant after fine-tune**: `gp1d.evaluate(buffered_model, args)`
   — with the same eval args as the base run (`eval_points`, `oracle_bins`,
   `oracle_chunk`, `jitter`) — reproduces the base checkpoint's metrics exactly;
   the oracle diagnostics remain valid for the fine-tuned artifact without
   re-validation.
4. **Training readout**: printed step-0 loss equals base context-only NLL on buffered
   targets; final loss visibly below it (the buffer's measured value).
5. **Internal consistency**: one-pass `joint_log_prob` matches a step-by-step
   teacher-forced buffered evaluation (allclose); prefix-0 path NaN-free.
6. **Joint quality** (structural, not a strict gate): buffered joint NLL strictly
   better than diagonal on eval functions; in the same range as slow-AR (expected
   comparable-or-better at long horizons per Fig. A17 small-N).
7. **Smoke**: `--steps 20 --batch-size 16` completes on CPU and CUDA; `--eval-only
   --load-checkpoint` round-trips.
8. **Timing**: measured and reported; no promised factor.

## Risks / fallbacks

- **Frozen fine-tune underperforms** (buffer can't integrate info through a frozen
  read path). Trigger: buffered joint NLL not clearly below diagonal on the eval
  functions after the full budget. Fallbacks in order — longer schedule; unfreeze
  the target-side LNs; `--no-freeze-base` with the 50/50 curriculum (≈ the paper's
  tested recipe, losing the bit-exact-marginals invariant but keeping the demo).
- **Merged bool masks** (`attn_mask` 2D/3D + `key_padding_mask`): verified working
  on the pinned torch 2.11 wheel during plan review; re-checked by the Phase 1 unit
  checks; additive float masks are the fallback if a future torch changes merging.
- **Teacher-forcing exposure bias at K=64**: accepted (the paper's setting);
  random-order sampling and order-averaged evaluation mitigate.
- **Demo grid x's evenly spaced vs iid-uniform training x's**: same support, no
  positional structure in the model; not expected to matter (note in local DEVLOG).

## Documentation

- `extensions/arbuffer/README.md`: what it is, paper reference, the reuse story
  (inherited vs new — the extensibility headline), the warm-start recipe, commands,
  demo description, artifact names, the non-core boundary statement.
- `extensions/arbuffer/DEVLOG.md`: dated entry with the design decisions above
  (esp. the paper deviations: split gated read, all-buffered curriculum, no
  positional/role embeddings, inclusive causal mask, warm start as the paper's
  future work).
- Root `DEVLOG.md`: **short** dated entry — `extensions/` taxonomy decision +
  pointer to the local README/DEVLOG.
- Root `README.md`: one Current Status bullet linking the extension.
- `AGENTS.md`: brief `extensions/` boundary note (like the `playground/` note) +
  currently-implemented mention.

## Open questions (resolved 2026-06-11)

- Demo conditioning columns: **confirmed** — (4 pts | 14 pts | 4 pts + kernel
  pinned).
- Fine-tune budget: **resolved** — the 20k default is the "does the recipe work"
  budget (Phase 4 validates with it); the retained artifact run will be **100k–200k
  steps**, done once the recipe is validated (same `--steps`-sized cosine curve,
  so it is a fresh run, not a resume of the 20k one).
- Artifact naming: **confirmed** — `artifacts/gp1d_arbuffer.pt` / `.png`.
- (Note: the 20k default and artifact names are defaults-of-record — they appear in
  the tracker, CLI defaults, and verification; changing either updates all three
  together.)

---
**Please review. Edit directly if needed, then confirm to proceed.**
