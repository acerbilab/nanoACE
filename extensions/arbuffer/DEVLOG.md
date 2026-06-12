# arbuffer DEVLOG

Local design log for the AR-buffer extension. Same spirit as the root
`DEVLOG.md` (the *why* matters as much as the *what*), scoped to this folder.
The root DEVLOG only points here; the implementation plan with the full
verification log is `docs/plans/PLAN-arbuffer.md`.

Reference: Hassan et al. (2026), *Efficient Autoregressive Inference for
Transformer Probabilistic Models* (ICLR 2026) — "the paper" below.

---

## 2026-06-11 (latest) — Concat-read variant; retained run is concat / K=64 / unfrozen

- **Why a second target read.** The separate zero-init gated read (initial
  implementation below) bought bitwise warm starts at real architectural cost:
  a fourth attention op per layer, the prefix-0 gate hack (which exists only
  because the buffer read has its own softmax with nothing else in it),
  additive context+buffer contributions that cannot renormalize against each
  other, and a target read inconsistent with the `sample_ar` append semantics
  the buffer is meant to accelerate. `--concat-read` adds the paper's own
  decoder (its appendix A.1): ONE softmax over the concatenated
  `[context, visible buffer prefix]` keys through the base
  `cross_attn`/`kv_ln` — buffer tokens enter the target read literally as
  "more context". One fewer attention op; the v=0 gate machinery disappears
  (the softmax always has context keys to fall back on).
- **The soft gate (`buf_bias`).** One learned scalar per head per layer, added
  to the buffer keys' pre-softmax logits via a float `attn_mask` (float masks
  add to logits; bool masks block). Bias 0 is the paper's read; negative
  values damp each visible buffer key's weight by `exp(bias)`. Default init −5
  (`--buf-bias-init`). Gradient reaches the bias through the mask (verified),
  at a rate proportional to the buffer's current attention mass.
- **Finding: a logit bias cannot close a sharp pretrained attention.** Step-0
  drift vs the base checkpoint is ~2.7 at bias 0, ~2.6 at −5, and still ~0.27
  at −40: the pretrained GP attention logits span ~±30 (sharply peaked on
  x-proximity), so a buffer key near the query's x out-competes distant
  context keys by more than any trainable bias. Exact warm starts are
  impossible in this read *by design*; `check_step0` still asserts the plain
  forward bit-exactly and reports the buffered drift instead of asserting it.
- **Reframe: that is a feature here.** At init the concat read treats buffer
  tokens as appended context (buffer stream = context-stream copies, read =
  base projections) — approximately the teacher-forced `sample_ar`
  conditional — so the model *starts* near slow-AR behavior on buffered
  targets instead of having to learn buffer use from a closed gate. Measured:
  a 20-step smoke already scored buffered ≈ slow-AR (+0.53 vs +0.54), and the
  20k run's first logged loss was −0.83 vs the −0.47 base context-only NLL.
- **20k validation (K=64, `--no-freeze-base`, bias −5).** Joint log-density
  per point (16 held-out functions, 4 ctx, 4 shared orders): diagonal −0.345
  < buffered **+1.554** < slow-AR +1.650 — **~95% of the AR gap recovered**,
  vs 76% for the frozen separate read at the same budget. Caveat: that
  comparison changes two things at once (architecture AND unfreezing); no
  separate-read-unfrozen arm was run. Joint training did not hurt the
  marginals (fixed-case eval NLL 0.450 → 0.363; the paper's 50/50
  context-only curriculum auto-restores under `--no-freeze-base`). Slow-AR
  itself rose (+1.33 → +1.65): joint training also improved the base's
  append-mode behavior, so the 95% is against a stronger baseline.
- **K sweep (frozen separate read, 20k each; for the record).** Gap recovered:
  K=32 ~63% < K=64 ~76% < K=128 ~87%, each scored on its own chain length.
  Training wall-clock is K-insensitive at nano scale (launch-bound): ~35 min
  per 20k at K=64 vs ~40 min at K=128, so K is a quality/rendering dial, not
  a cost dial. K=64 chosen for the retained run.
- **Retained run (completed 2026-06-12 ~06:30; supersedes the K=128 plan
  below):** fresh 200k, concat read, K=64 defaults, joint training:

      python extensions/arbuffer/gp1d_arbuffer.py --steps 200000 \
          --concat-read --no-freeze-base \
          --save-checkpoint artifacts/gp1d_arbuffer.pt --ckpt-every 5000

  **Results:** joint log-density per point (16 held-out functions, 4 ctx, 4
  shared orders): diagonal −0.364 < buffered **+1.648** < slow-AR +1.716 —
  **~97% of the AR gap** (20k validation: ~95%), with slow-AR itself again
  improved by the joint training (+1.650 → +1.716). Fixed-case predictive
  eval NLL 0.363 / RMSE 0.471 (base 0.450 / 0.479); `log_lengthscale`
  marginal moved *toward* the oracle, but the fixed case's `log_outputscale`
  marginal drifted away (mean −0.882 vs oracle −0.523, base −0.424) — a
  single-case readout, noted, not investigated. Base-parity drift 2.58.
  Sampling 1.1× vs `sample_ar` (launch-bound, as always).

  Under `--no-freeze-base` the artifact's empty-buffer marginals are *not*
  bit-equal to the base checkpoint (`base parity` prints the drift instead),
  so the oracle diagnostics describe the fine-tuned marginals, not the source
  checkpoint's.
- **Playground coupling (resolved same day).** The TS port (entry below) was
  switched to the concat read: `buffered.ts` now implements ONE softmax over
  `[context, buffer]` keys with the per-head `buf_bias` soft gate, and rejects
  separate-read blobs with a clear error — the fixtures can only honestly cover
  the shipped architecture. Fixtures regenerated from
  `gp1d_arbuffer_concat20k.pt`; `parity.py`'s packed replication mirrors both
  `forward_buffered` branches and records the read mode in the fixture. The
  retained 200k swap is back to "repoint `ARBUF_CKPT`, re-run export + parity
  together" — done 2026-06-12 once the run completed: the playground tab now
  serves the retained `gp1d_arbuffer.pt` weights (still local-only).
- **Bugfix:** `plot_demo` lacked `@torch.no_grad()`; with an unfrozen model
  its forward outputs require grad and matplotlib's implicit `.numpy()`
  raised. Latent since the initial implementation (frozen-base outputs don't
  require grad) — first triggered by the first unfrozen run to reach the plot.

---

## 2026-06-11 — Playground tab (TS port of the incremental sampler)

Plan + verification log: `docs/plans/PLAN-arbuffer-playground.md`. The playground
(`playground/src/arbuf/` + `playground/src/ace/buffered.ts`) now runs this model
in the browser: context encoded once, a few coherent joint draws decoded
against the cache (animated), with the diagonal band and independent marginal
samples always shown for contrast. **Local-only for now** — the temporary
checkpoint (originally the 20k K=128 separate-read run; since the concat-read
switch in the entry above, `gp1d_arbuffer_concat20k.pt`) is exported locally,
not deployed; the retained run swaps in by repointing `parity.py`'s
`ARBUF_CKPT` (and the README export example) at the retained artifact, then
re-running `export_weights.py` + `parity.py` together.

- **Exporter contract.** This extension now exposes the same 2-arg
  `load_checkpoint(path, device)` wrapper every example has (in
  `gp1d_arbuffer.py`), which is all `playground/export_weights.py` needs — the
  manifest format is generic, so `buf_blocks.*` flows through unchanged.
- **The parity guard now extends to the TS port.** `playground/parity.py` dumps
  buffered fixtures (plain forward on the buffered checkpoint; a packed
  `forward_buffered` pass with per-layer states; a teacher-forced `sample_joint`
  chain via the existing `teacher_force` mode). If `forward_buffered` or the
  incremental cache semantics here ever change, those fixtures fail loudly —
  the same way the step-0 check guards the coupling to `ace.py`.
- **One recorded TS deviation: projected K/V are cached**, not LayerNorm'd
  hidden states. `sample_joint`'s reproject-per-read style is a micro-opt under
  torch but O(K²·d²) in scalar JS. Same math; verified by the fixtures above.

---

## 2026-06-11 — Initial implementation (warm-started buffer on GP-1D)

- **Three token streams per layer; the base invariants survive.** The buffer is
  a third token set carrying realized `(x, y)` values — *not* causal attention
  among targets. Targets still never attend to one another (paper requirement
  R4), the context never reads buffer or targets (R3), and conditioning
  direction stays structural. Per layer: the base context self-attention
  (frozen, code path identical to `ACEBlock`); the buffer attending over
  `[cached context layer-input, buffer]` with an inclusive-causal mask; the
  base target→context cross-attention; a NEW gated target→buffer read; the base
  target MLP.

- **Load-bearing deviation: a separate zero-init target→buffer read.** The
  paper's target decoder is a single cross-attention over concatenated
  `[context, buffer-prefix]` keys (its appendix A.1). A softmax over a
  concatenated KV set renormalizes the pretrained attention pattern, and no
  initialization removes the buffer's share (zeroed buffer keys still
  contribute `exp(0) = 1` each — at 4 context points and a 63-token buffer,
  most of the pretrained context read would be diluted at init). The extension
  instead adds a separate residual cross-attention whose `out_proj` is
  zero-initialized: exactly zero at step 0 regardless of buffer content, so a
  warm start is **bit-identical** to the base checkpoint (`torch.equal`,
  asserted at every warm start). The cost is that context and buffer no longer
  compete inside one softmax (contributions add; the output projections learn
  the balance) — accepted as the price of an exact warm start the paper never
  needed, since it trains from scratch.

- **Buffer stream initialized as a copy of the context stream.** One attention
  per layer with `q = buffer`, `kv = [context layer-input, buffer]` — the
  paper's mask blocks "buffer reads context" + "causal buffer self-attention"
  fused, which is how its single training mask works anyway. Weights start as
  *copies* of `ctx_attn` / `ctx_mlp` / LNs: at init, buffer tokens are encoded
  as if appended to the context and run through context self-attention minus
  the back-edges R3 forbids — which is exactly what `sample_ar` does by
  literally appending. Copies, not shared modules, so fine-tuning the buffer
  cannot drift the frozen base.

- **Frozen base by default → all-buffered curriculum.** With the base frozen
  and the buffer read zero-gated, a context-only (`v = 0`) target's loss is a
  constant — zero gradient reaches any trainable parameter — so the paper's
  50% context-only targets would waste half the training signal. Every target
  therefore draws `v ~ U{1..K}`. Marginal preservation is structural, not
  trained: empty-buffer predictions stay bit-equal to the base checkpoint
  forever (asserted after fine-tuning; `gp1d.evaluate` runs on the buffered
  model unchanged and reproduces the base metrics). `--no-freeze-base` restores
  the paper's 50/50 split — the curriculum is derived from the freeze flag,
  never set independently, because the 50% context-only share is the only thing
  protecting marginals when the base can move. Bonus readout: at step 0 the
  training loss *is* the base's context-only NLL on buffered targets, so the
  loss curve directly plots the information extracted from the buffer.

- **No buffer positional embeddings.** The paper's own ablation (appendix H.2)
  finds no significant difference, GP function values are exchangeable given
  the `(x, y)` pairs, and dropping them removes the learned-position cap on
  buffer length — `K` becomes a training-distribution choice, not an
  architecture constant. The paper's appendix H.5 validates buffers up to
  K = 64 on GP regression, exactly the demo setting.

- **No buffer role embedding — a deviation, recorded.** The paper (appendix
  A.1) gives every token a role embedding because its tokens share one masked
  stream. Here the streams are separated structurally (different attention
  modules process them), and the buffer-stream weights are independent copies
  that can differentiate during fine-tuning. Fallback if training ever wants
  it: a single zero-init `d_model` vector added to buffer embeddings (preserves
  step-0 exactness). Buffer tokens are embedded by `ACE._embed` verbatim, as
  VALUE-mode data tokens — the same thing `sample_as_context_tokens` builds for
  `sample_ar`.

- **Inclusive causal mask.** Buffer token `j` attends to `[context, buffer
  1..j]` rather than the paper's strict `< j`: it matches the
  copied-context-self-attention initialization story, and target `k` reads only
  prefix `1..k-1` either way, so the predictive factorization is identical.

- **Data-only buffer and targets (v1).** With a frozen base the latent marginal
  path cannot learn, and latent-in-buffer (joint latent sampling through the
  buffer) is deliberately out of scope. Conditioning on latents stays fully
  supported: pins are context tokens and flow through the cached encoder. The
  fine-tune keeps the base context distribution (`n_context ~ U{1..20}`, shared
  latent-reveal mixture) so pinned-latent contexts are in-distribution *with* a
  buffer; data points are drawn at `n_points = 128` (vs the base's 64) to fit
  context + a K=64 buffer + 44–63 complement targets.

- **Prefix-0 handling.** Targets with `v = 0` formally attend to buffer slot 0
  and their buffer-read output is multiplied by `(v > 0)`. On the pinned torch
  2.11 a fully-masked attention row already returns exact zeros, so the gate is
  version-robustness and explicitness, not a NaN fix.

- **Verification (all passing; see the plan's tracker for the log).** Warm-start
  key guard; bitwise step-0 parity (plain forward and zero-gated buffered
  forward); gradient routing under freeze (only `buf_blocks.*`); prefix-0
  bit-equality with non-zero buffer weights; one-pass `joint_log_prob` vs a
  step-by-step growing-prefix chain (atol 1e-5); the incremental sampler's KV
  caches vs the one-pass evaluation (teacher-forced; max |diff| ~5e-7); CPU and
  CUDA smoke runs; `--eval-only` round-trip; post-fine-tune frozen parity.

- **Honest wall-clock at nano scale.** Measured on the demo dimensions (64
  draws × 64 points): CPU ~1.4× faster than `sample_ar`; GPU roughly parity
  (~1.0×) — per-step tensors are so small that kernel-launch overhead dominates
  and the buffered path runs two small passes (target decode + buffer encode)
  per step. The paper's up-to-20× is at N=1024. What transfers to nano scale is
  the structure (one frozen context cache shared by all draw streams) and the
  quality story: `sample_ar` pushes the base model far past its
  `n_context ≤ 20` training range by the end of a 64-point chain, while buffer
  prefixes up to K=64 are trained in-distribution. The paper's Fig. A17 shows
  *its* buffer tracking and slightly exceeding re-encoding AR in this small-N
  regime; our own 20k-step validation run recovers most but not all of the
  slow-AR gap in joint density (it clearly beats the diagonal, does not yet
  beat slow-AR — see the plan tracker for numbers; the longer retained run may
  close more). The incremental sampler caches LN'd states (LayerNorm is
  per-token) but deliberately does not cache KV projections — declined as a
  micro-opt at this scale.

- **Decode order: random, empirically confirmed (2026-06-11).** `sample_joint`
  defaults to a shared random permutation, matching training (buffer prefixes
  are random subsets, so the visible prefix typically straddles the query).
  Measured on the K=128 model, teacher-forced joint density per point of
  held-out truth (16 functions, 4 ctx): random (4 orders) 0.97 vs left-to-right
  0.59 vs right-to-left 0.45. Monotone orders also *look* worse: every step
  extrapolates at a frontier with the entire prefix on one side — a prefix
  shape that is exponentially rare among random training subsets — producing
  jagged early-chain segments and vertical drift between context points.
  Random order is the default and should stay so; order-averaged density
  evaluation (`joint_log_prob`) already mitigates the same effect.

- **Retained artifact — superseded the same day (see the concat-read entry at
  the top: the retained run is 200k at K=64, `--concat-read --no-freeze-base`).
  Kept for the record.** The fine-tune
  default is 20k steps (the "recipe works" budget). Two 20k validation runs
  exist: K=64 (defaults) and K=128 (`--buffer-size 128 --n-points 192
  --sample-points 128`); K=128 won on joint density (~87% vs ~76% of the
  slow-AR gap recovered) and renders smoother demo draws, so the retained run
  is **200k steps at the K=128 settings**:

      python extensions/arbuffer/gp1d_arbuffer.py --steps 200000 \
          --buffer-size 128 --n-points 192 --sample-points 128 \
          --save-checkpoint artifacts/gp1d_arbuffer.pt

  This must be a fresh run, not a resume of a 20k one (cosine `T_max` is sized
  to the run's `--steps`, same rule as the base examples).
