# arbuffer — causal autoregressive buffer for nanoACE

An extension that adds the causal autoregressive buffer of
[Hassan et al. (2026), *Efficient Autoregressive Inference for Transformer
Probabilistic Models*](https://openreview.net/forum?id=5bfUqlOhAH) (ICLR 2026)
on top of a **pretrained** nanoACE model. Local paper markdown is in
[paper/](paper/).

The base ACE model is a diagonal prediction map: joint samples come from
`ace.sample_ar`, which appends each sampled point to the context and re-encodes
everything, every step. The buffer decouples that: the context is encoded
**once** and cached, realized points go into a third causal token stream, and
each new prediction attends to the cached context plus the buffer prefix. One
context encoding then serves any number of parallel draw streams — the demo
draws 64 coherent GP function samples from one cached 4-point context.

This extension is also the repository's extensibility demo. Adding a different
inference architecture reuses almost everything and touches no core file.

Reused unchanged (imported from the core): the `Variable` / `Tokens` / `Batch`
schema, the embedder (`ACE._embed`), the shared MDN + categorical heads and
`Predictions`, the whole training spine (`train.fit`, checkpoints, the CLI
parent), the GP physics (`gp1d.draw_instances`), `ace.sample_ar` as the
baseline, and `gp1d.evaluate`'s oracle diagnostic (which runs verbatim on the
buffered model).

New in this folder: one `BufferBlock` per layer (buffer attention + MLP and a
target→buffer read in one of two variants — the default separate zero-init
gated cross-attention, or the paper's single concatenated softmax via
`--concat-read`), the cached incremental sampler (`sample_joint`), the
one-pass joint density evaluation (`joint_log_prob`), and a three-way
context / buffer / target fine-tune sampler.

## The warm-start recipe

1. **Attach**: `BufferedACE(ACE)` adds `buf_blocks.*` parameters; a base
   checkpoint loads with a strict guard (`unexpected == []`, missing keys all
   under `buf_blocks.`).
2. **Initialize exactly**: the buffer stream copies the context-stream weights
   ("as if appended to the context" — what `sample_ar` does by hand), and the
   new target→buffer read is a *separate residual cross-attention with a
   zero-initialized output projection*. At step 0 the model is **bit-identical**
   to the base checkpoint — checked automatically at every warm start.
3. **Freeze the base** (default): only the buffer stream trains, so empty-buffer
   predictions stay bit-equal to the base checkpoint *forever* — the existing
   oracle diagnostics remain valid for the fine-tuned artifact, unchanged.
4. **Fine-tune all-buffered**: with a frozen base, a context-only target carries
   zero gradient, so every target trains with a random visible buffer prefix
   `v ~ U{1..K}`. The first logged loss *is* the base model's context-only NLL
   on those targets; everything below it is information extracted from the
   buffer. (`--no-freeze-base` restores the paper's 50/50 curriculum.)

Warm-starting a pretrained set-based model is the paper's own stated future
work (its Discussion suggests applying the buffer to pretrained NPs/PFNs); the
paper itself trains jointly from scratch. Deviations from the paper and their
rationale are recorded in the local [DEVLOG.md](DEVLOG.md).

## The concat read (`--concat-read`, the retained variant)

`--concat-read` replaces step 2's separate read with the paper's own target
decoder: **one softmax over the concatenated `[context, visible buffer
prefix]` keys** through the base cross-attention — `sample_ar`'s
append-to-context semantics — plus a learned per-head logit bias on the buffer
keys as a soft gate (`--buf-bias-init`, default −5, injected as an additive
float attention mask). One fewer attention op per layer, and no prefix-0
special case.

Bitwise warm starts are impossible in this read: the pretrained GP attention
is sharp enough (logits ~±30 on x-proximity) that no trainable bias can shut
relevant buffer keys out, so the step-0 check reports a drift instead of
asserting equality (the plain forward stays bit-checked). That turns out to be
a feature — at init the buffer enters the read as appended context, so the
model *starts* near slow-AR joint quality instead of learning buffer use from
a closed gate. Validated at 20k (K=64, with `--no-freeze-base`): **~95% of
the slow-AR gap recovered** vs 76% for the frozen zero-init read at equal
budget, with no marginal degradation (the paper's 50/50 context-only
curriculum auto-restores when the base unfreezes). The retained fine-tune uses
this mode; the design discussion is in the local [DEVLOG.md](DEVLOG.md).

## Run

From the repo root, with a trained GP-1D checkpoint (`python gp1d.py
--save-checkpoint artifacts/gp1d.pt`):

```powershell
# fine-tune the buffer (default mode: separate read, frozen base;
# ~800k trainable of ~2.0M params)
.\.venv\Scripts\python.exe extensions\arbuffer\gp1d_arbuffer.py `
    --base-checkpoint artifacts\gp1d.pt --save-checkpoint artifacts\gp1d_arbuffer.pt

# retained fine-tune recipe (paper-style concat read, joint training, 200k)
.\.venv\Scripts\python.exe extensions\arbuffer\gp1d_arbuffer.py `
    --steps 200000 --concat-read --no-freeze-base `
    --save-checkpoint artifacts\gp1d_arbuffer.pt --ckpt-every 5000

# short smoke run
.\.venv\Scripts\python.exe extensions\arbuffer\gp1d_arbuffer.py --steps 20 --batch-size 16

# reuse a fine-tuned checkpoint (demo + diagnostics only; the read mode is
# inferred from the checkpoint)
.\.venv\Scripts\python.exe extensions\arbuffer\gp1d_arbuffer.py `
    --eval-only --load-checkpoint artifacts\gp1d_arbuffer.pt --no-freeze-base
```

Common artifacts: `artifacts/gp1d_arbuffer.pt`, `artifacts/gp1d_arbuffer.png`.

The fine-tune draws the same GP physics as `gp1d.py` at a larger point budget
(`--n-points 128`): context candidates as in the base (`n_context ~ U{1..20}`,
same latent-reveal mixture, so pinned-latent contexts stay in-distribution
*with* a buffer), a `--buffer-size 64` buffer slice, and every remaining point
as a target. Buffer and targets are data tokens only; conditioning on latents
stays fully supported through context pins.

## What the demo shows

`gp1d_arbuffer.py` ends with diagnostics + a figure on a fixed GP function —
the base `gp1d.py` diagnostic's case redrawn with a 1.5× longer lengthscale, so
sampled curves render legibly (the canonical gp1d case still backs the
warm-start check, the base-parity assert, and `gp1d.evaluate`):

- **Three conditioning columns** — 4 context points; all 14; 4 points with the
  kernel pinned as a context VALUE token. Each shows a handful of coherent
  buffered function draws (`--plot-draws`, default 8, distinct colors so
  individual curves stay followable) over the diagonal ±2σ band, the true
  function, and the context. Reading (a)→(b): more data tightens the joint. Reading (a)→(c):
  pinning a latent tightens it differently — token-level conditioning composing
  with joint sampling.
- **A joint log-density table** — diagonal (independent) vs slow-AR
  (re-encoding `sample_ar` path) vs buffered one-pass, scored on *identical*
  orderings, plus order-averaging (autoregressive densities are
  order-dependent).
- **Base parity** — with a frozen base, the empty-buffer predictions are
  asserted bit-equal to the source checkpoint; under `--no-freeze-base` the
  drift is reported instead (expected — the marginals trained). Either way
  `gp1d.evaluate`'s oracle diagnostic runs on the buffered model unchanged.
- **Measured wall-clock** for B×K coherent sampling, `sample_ar` vs
  `sample_joint`.

**Honest performance note.** The paper's up-to-20× speedups are measured at
context sizes around N=1024. At nano scale (contexts ≤ 23 tokens, `d_model`
128) per-step work is tiny and wall-clock is dominated by kernel-launch
overhead, so expect rough parity on GPU and a modest win on CPU. What the
extension demonstrates at this scale is the *structure* (one frozen context
cache shared by all draw streams; O(N²+NK+K²) attention instead of
O(K(N+K)²)) and a *quality* point: a 64-step `sample_ar` chain runs the base
model far outside its `n_context ≤ 20` training range, while buffer prefixes up
to K=64 are trained in-distribution.

## Boundary

The core is unchanged: `ace.py`, `train.py`, and the examples never import
this folder. The extension is torch-only and reaches into core internals
(it subclasses `ACE`, calls `_embed`, and re-implements `ACEBlock`'s op order
around the buffer stream) — the automatic step-0 parity check keeps that
coupling honest: if the core forward changes, the warm start fails loudly.
(The check is bitwise for the plain forward in both read modes and for the
buffered forward in the default separate-read mode; the concat read's buffered
drift is reported, since exactness is impossible there by design.)
