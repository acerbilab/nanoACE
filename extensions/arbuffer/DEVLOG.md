# arbuffer DEVLOG

Local design log for the AR-buffer extension. Same spirit as the root
`DEVLOG.md` (the *why* matters as much as the *what*), scoped to this folder.
The root DEVLOG only points here; the implementation plan with the full
verification log is `docs/plans/PLAN-arbuffer.md`.

Reference: Hassan et al. (2026), *Efficient Autoregressive Inference for
Transformer Probabilistic Models* (ICLR 2026) — "the paper" below.

---

## 2026-06-11 — Playground tab (TS port of the incremental sampler)

Plan + verification log: `docs/plans/PLAN-arbuffer-playground.md`. The playground
(`playground/src/arbuf/` + `playground/src/ace/buffered.ts`) now runs this model
in the browser: context encoded once, a few coherent joint draws decoded
against the cache (animated), with the diagonal band and independent marginal
samples always shown for contrast. **Local-only for now** — the temporary
20k K=128 checkpoint is exported locally, not deployed; the retained run swaps
in by repointing `parity.py`'s `ARBUF_CKPT` (and the README export example) at
the retained artifact, then re-running `export_weights.py` + `parity.py`
together.

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

- **Retained artifact — still to run (as of 2026-06-11).** The fine-tune
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
