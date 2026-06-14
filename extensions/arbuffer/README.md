# arbuffer — causal autoregressive buffer for nanoACE

An extension that adds the causal autoregressive buffer of
[Hassan et al. (2026), *Efficient Autoregressive Inference for Transformer
Probabilistic Models*](https://www.conorhassan.com/projects/artnp/) (ICLR 2026)
on top of a **pretrained** nanoACE GP-1D model.

The base ACE model is a diagonal prediction map: joint samples come from
`ace.sample_ar`, which appends each sampled point to the context and re-encodes
**everything, every step**. The buffer decouples that — the context is encoded
**once** and cached, realized points go into a separate causal token stream, and
each new prediction attends to the cached context plus the buffer prefix. One
context encoding then serves any number of parallel draw streams: the demo draws
64 coherent GP function samples from one cached 4-point context, plus a one-pass
joint density evaluation.

It is also the repository's **extensibility demo**. Adding this inference
architecture reuses almost everything — the schema, embedder, heads, training
spine, GP physics, and oracle diagnostics — and touches no core file.

## Reference

This extension is based on:

```bibtex
@inproceedings{hassan2026efficient,
  title={Efficient Autoregressive Inference for Transformer Probabilistic Models},
  author={Conor Hassan and Nasrulloh Ratu Bagus Satrio Loka and Cen-You Li and Daolang Huang and Paul Edmund Chang and Yang Yang and Francesco Silvestrin and Samuel Kaski and Luigi Acerbi},
  year={2026},
  booktitle={The Fourteenth International Conference on Learning Representations (ICLR 2026)},
}
```

Local paper markdown is in [paper/](paper/).

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
same latent-reveal mixture, so pinned-latent contexts stay in-distribution *with*
a buffer), a `--buffer-size 64` buffer slice, and every remaining point as a
target. Buffer and targets are data tokens only; conditioning on latents stays
fully supported through context pins.

## What the demo shows

`gp1d_arbuffer.py` ends with diagnostics + a figure on a fixed GP function — the
base `gp1d.py` diagnostic's case redrawn with a 1.5× longer lengthscale, so
sampled curves render legibly:

- **Three conditioning columns** — 4 context points; all 14; 4 points with the
  kernel pinned as a context VALUE token. Each shows a handful of coherent
  buffered function draws (`--plot-draws`, default 8) over the diagonal ±2σ band,
  the true function, and the context. Reading (a)→(b): more data tightens the
  joint. Reading (a)→(c): pinning a latent tightens it differently — token-level
  conditioning composing with joint sampling.
- **A joint log-density table** — diagonal (independent) vs slow-AR (re-encoding
  `sample_ar` path) vs buffered one-pass, scored on *identical* orderings, plus
  order-averaging.
- **Base parity** — with a frozen base, the empty-buffer predictions are asserted
  bit-equal to the source checkpoint; under `--no-freeze-base` the drift is
  reported instead (expected — the marginals trained). Either way
  `gp1d.evaluate`'s oracle diagnostic runs on the buffered model unchanged.
- **Measured wall-clock** for B×K coherent sampling, `sample_ar` vs
  `sample_joint`.

**Honest performance note.** The paper's up-to-20× speedups are measured at
context sizes around N=1024. At nano scale (contexts ≤ 23 tokens, `d_model` 128)
per-step work is tiny and wall-clock is dominated by kernel-launch overhead, so
expect rough parity on GPU and a modest win on CPU. What the extension
demonstrates at this scale is the *structure* (one frozen context cache shared by
all draw streams) and a *quality* point: a 64-step `sample_ar` chain runs the
base model far outside its `n_context ≤ 20` training range, while buffer prefixes
up to K=64 are trained in-distribution.

## Design & deviations

The full design rationale lives in the local [DEVLOG.md](DEVLOG.md): the
warm-start recipe (the bit-exact zero-init gated read vs the retained paper-style
`--concat-read`, one softmax over the concatenated `[context, buffer]` keys), the
frozen-base / all-buffered curriculum, and the deviations from the paper (which
trains jointly from scratch — warm-starting a pretrained set-based model is the
paper's own stated future work). The retained fine-tune is the concat read at
K=64, recovering ~95% of the slow-AR joint-density gap at the 20k validation
budget.

The core stays unchanged: `ace.py`, `train.py`, and the examples never import
this folder. The extension is torch-only and reaches into core internals (it
subclasses `ACE`, calls `_embed`, and re-implements `ACEBlock`'s op order around
the buffer stream), so an automatic **step-0 parity check** guards the coupling:
if the core forward changes, the warm start fails loudly.
