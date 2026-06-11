"""Causal autoregressive buffer extension for nanoACE.

Implements the causal AR buffer of Hassan et al. (2026), "Efficient
Autoregressive Inference for Transformer Probabilistic Models" (ICLR 2026), as a
warm-startable extension of a pretrained `ACE` model. The base model is a
diagonal prediction map; joint samples today come from `ace.sample_ar`, which
re-encodes the whole context at every step. The buffer decouples that: the
context is encoded once and cached, realized targets go into a third token
stream (the *buffer*), and each new prediction attends to the cached context
plus the visible buffer prefix.

Three token streams per layer (the base invariants survive: context never reads
buffer or targets; targets never read each other):

- **context** — the base model's context self-attention, byte-for-byte (frozen
  weights, same code path);
- **buffer** — realized `(x, y)` tokens; one attention per layer with
  `q = buffer`, `kv = [cached context layer-input, buffer]` and an
  inclusive-causal mask on the buffer block, initialized as a *copy* of the
  context-stream weights ("as if appended to the context", which is exactly what
  `sample_ar` does by hand);
- **target** — the base cross-attention to context (frozen), plus a NEW separate
  residual cross-attention to the visible buffer prefix whose output projection
  is zero-initialized.

The zero-init gate is the load-bearing deviation from the paper: the paper's
target decoder runs one softmax over concatenated `[context, buffer]` keys,
which renormalizes the pretrained attention pattern and cannot be initialized
away (zeroed buffer keys still contribute `exp(0) = 1` per key). A separate
residual term is exactly zero at init, so a warm-started `BufferedACE` is
bit-identical to its base checkpoint until fine-tuning moves the new weights.

Other recorded deviations: no buffer positional embeddings (paper appendix H.2
ablates them: no significant difference; dropping them removes the trained
buffer-length cap), no buffer role embedding (the paper's appendix A.1 has one;
here the streams are separated structurally rather than packed into one masked
stream), and an inclusive rather than strictly-causal diagonal (target k reads
prefix 1..k-1 either way).

This module is task-agnostic: it knows `ace.py`, not `gp1d.py`. The GP-1D
fine-tune/demo lives in `gp1d_arbuffer.py` next to this file.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

# This file lives in extensions/arbuffer/; import the core from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ace import (  # noqa: E402
    ACE,
    ACEConfig,
    Batch,
    PRIOR_FEATURES,
    Predictions,
    QUERY,
    Tokens,
    VALUE,
    Variable,
    _mlp,
    append_or_replace_context_token,
)


@dataclass
class BufferedBatch:
    """One buffered task batch: context + realized buffer + targets.

    `buffer` holds realized VALUE tokens (ground truth during training, model
    samples during generation) and must be fully active — ragged buffers are not
    supported; visibility is controlled per target by `prefix_len` instead.
    `prefix_len[b, m]` is how many buffer tokens target `m` may read (0 = the
    target is context-only). Target truth sits in `target.value` for the loss,
    exactly as in the base `Batch`.
    """

    variables: list[Variable]
    context: Tokens
    buffer: Tokens
    target: Tokens
    prefix_len: torch.Tensor  # LongTensor[B, M]

    def to(self, device: torch.device | str) -> "BufferedBatch":
        return BufferedBatch(
            variables=self.variables,
            context=self.context.to(device),
            buffer=self.buffer.to(device),
            target=self.target.to(device),
            prefix_len=self.prefix_len.to(device),
        )


class BufferBlock(nn.Module):
    """Per-layer buffer modules, paired 1:1 with an `ACEBlock`.

    Initialization (`BufferedACE.init_from_base`) copies the paired base block's
    weights so the buffer stream starts as context self-attention over the
    appended tokens:

        buf_ln1  <- ctx_ln1     buf_attn <- ctx_attn
        buf_ln2  <- ctx_ln2     buf_mlp  <- ctx_mlp
        tgt_buf_qln <- tgt_ln1  tgt_buf_kvln <- kv_ln
        tgt_buf_attn.out_proj <- zeros  (the warm-start gate)
    """

    def __init__(self, cfg: ACEConfig):
        super().__init__()
        d = cfg.d_model
        self.buf_ln1 = nn.LayerNorm(d)
        self.buf_ln2 = nn.LayerNorm(d)
        self.buf_attn = nn.MultiheadAttention(d, cfg.n_heads, batch_first=True)
        self.buf_mlp = _mlp(d, cfg.mlp_hidden, d)
        self.tgt_buf_qln = nn.LayerNorm(d)
        self.tgt_buf_kvln = nn.LayerNorm(d)
        self.tgt_buf_attn = nn.MultiheadAttention(d, cfg.n_heads, batch_first=True)


@dataclass
class ContextCache:
    """Frozen per-layer context states from a one-time encoding pass.

    `inputs[l]` is the context entering layer `l` (what the buffer stream
    attends over, mirroring `ctx_ln1`'s input in the base block); `kv[l]` is the
    updated context the targets read at layer `l` (pre-`kv_ln`, pre-zeroing,
    exactly as `ACEBlock.forward` uses it). `mask` is the context token mask.
    """

    inputs: list[torch.Tensor]  # n_layers x [B, N, D]
    kv: list[torch.Tensor]  # n_layers x [B, N, D]
    mask: torch.Tensor  # [B, N]


class BufferedACE(ACE):
    """ACE plus a causal autoregressive buffer stream.

    The plain `forward(batch: Batch)` is inherited untouched, so every base
    diagnostic (`gp1d.evaluate`, `diagnostics.query_log_density`, ...) runs on
    this model unchanged; with the base frozen its empty-buffer behavior stays
    bit-identical to the source checkpoint forever. All new parameters live
    under `buf_blocks.*`, which is what makes the warm-start key guard exact.
    """

    def __init__(self, variables: Sequence[Variable], cfg: ACEConfig | None = None):
        super().__init__(variables, cfg)
        self.buf_blocks = nn.ModuleList([BufferBlock(self.cfg) for _ in range(self.cfg.n_layers)])

    def init_from_base(self) -> None:
        """Copy base weights into the buffer stream and zero the target gate."""

        for blk, bblk in zip(self.blocks, self.buf_blocks):
            bblk.buf_ln1.load_state_dict(blk.ctx_ln1.state_dict())
            bblk.buf_ln2.load_state_dict(blk.ctx_ln2.state_dict())
            bblk.buf_attn.load_state_dict(blk.ctx_attn.state_dict())
            bblk.buf_mlp.load_state_dict(blk.ctx_mlp.state_dict())
            bblk.tgt_buf_qln.load_state_dict(blk.tgt_ln1.state_dict())
            bblk.tgt_buf_kvln.load_state_dict(blk.kv_ln.state_dict())
            nn.init.zeros_(bblk.tgt_buf_attn.out_proj.weight)
            nn.init.zeros_(bblk.tgt_buf_attn.out_proj.bias)

    def freeze_base(self) -> None:
        """Freeze everything except the buffer stream (`buf_blocks.*`)."""

        for name, p in self.named_parameters():
            p.requires_grad_(name.startswith("buf_blocks."))

    def _predictions(self, tgt: torch.Tensor) -> Predictions:
        return Predictions(
            cont_raw=self.cont_head(tgt),
            disc_logits=self.disc_head(tgt),
            is_discrete=self.is_discrete,
            is_latent=self.is_latent,
            cardinality=self.cardinality,
            has_bounds=self.has_bounds,
            bound_lo=self.bound_lo,
            bound_hi=self.bound_hi,
            min_scale=self.cfg.min_scale,
        )

    def forward_buffered(self, bbatch: BufferedBatch) -> Predictions:
        """Predict targets given context plus per-target visible buffer prefixes.

        One parallel pass (training / one-pass joint density evaluation). All
        attention calls pass `need_weights=False` to mirror `ACEBlock` — the
        weights-returning path uses a different kernel and would break the
        bitwise step-0 parity with the base model.
        """

        ctx = self._embed(bbatch.context)
        buf = self._embed(bbatch.buffer)
        tgt = self._embed(bbatch.target)
        ctx_mask, tgt_mask = bbatch.context.mask, bbatch.target.mask
        if not bool(ctx_mask.any(dim=1).all()):
            raise ValueError("BufferedACE needs at least one active context token per batch row")
        if not bool(bbatch.buffer.mask.all()):
            raise ValueError("buffer tokens must be fully active; use prefix_len for visibility")

        b, n = ctx_mask.shape
        k = buf.shape[1]
        m = tgt.shape[1]
        prefix = bbatch.prefix_len
        if prefix.shape != (b, m):
            raise ValueError(f"prefix_len must be [B, M] = [{b}, {m}], got {tuple(prefix.shape)}")
        if int(prefix.max()) > k:
            raise ValueError(f"prefix_len max {int(prefix.max())} exceeds buffer size {k}")
        device = ctx.device

        kp_ctx = ~ctx_mask
        kp_ctx_buf = torch.cat([kp_ctx, torch.zeros(b, k, dtype=torch.bool, device=device)], dim=1)

        # Buffer attention mask [K, N+K]: context fully visible, buffer block
        # inclusive-causal (token j reads context + buffer 1..j). True = blocked.
        ar = torch.arange(k, device=device)
        causal = torch.zeros(k, n + k, dtype=torch.bool, device=device)
        causal[:, n:] = ar[None, :] > ar[:, None]

        # Target->buffer prefix mask [B*heads, M, K]: target m reads buffer
        # 1..prefix_len[b, m]. Prefix-0 targets formally read slot 0 but their
        # buffer-read output is multiplied by 0 below — explicit semantics
        # rather than relying on fully-masked-row behavior.
        blocked = ar[None, None, :] >= prefix[:, :, None]  # [B, M, K]
        zero_prefix = prefix == 0
        blocked[..., 0] = blocked[..., 0] & ~zero_prefix
        prefix_mask = blocked.repeat_interleave(self.cfg.n_heads, dim=0)
        gate = (~zero_prefix).to(tgt.dtype).unsqueeze(-1)  # [B, M, 1]

        for blk, bblk in zip(self.blocks, self.buf_blocks):
            ctx_in = ctx
            # 1. base context update — frozen weights, code path as ACEBlock.
            ctx_q = blk.ctx_ln1(ctx)
            ctx_att, _ = blk.ctx_attn(ctx_q, ctx_q, ctx_q, key_padding_mask=kp_ctx, need_weights=False)
            ctx = ctx + ctx_att
            ctx = ctx + blk.ctx_mlp(blk.ctx_ln2(ctx))
            # 2. buffer update — reads layer-input context + causal self.
            buf_q = bblk.buf_ln1(buf)
            buf_kv = bblk.buf_ln1(torch.cat([ctx_in, buf], dim=1))
            buf_att, _ = bblk.buf_attn(
                buf_q, buf_kv, buf_kv, key_padding_mask=kp_ctx_buf, attn_mask=causal, need_weights=False
            )
            buf = buf + buf_att
            buf = buf + bblk.buf_mlp(bblk.buf_ln2(buf))
            # 3. base target read of the updated context — frozen, as ACEBlock.
            kv_c = blk.kv_ln(ctx)
            tgt_att, _ = blk.cross_attn(blk.tgt_ln1(tgt), kv_c, kv_c, key_padding_mask=kp_ctx, need_weights=False)
            tgt = tgt + tgt_att
            # 4. NEW gated buffer read — exactly zero at warm start.
            kv_b = bblk.tgt_buf_kvln(buf)
            read, _ = bblk.tgt_buf_attn(
                bblk.tgt_buf_qln(tgt), kv_b, kv_b, attn_mask=prefix_mask, need_weights=False
            )
            tgt = tgt + read * gate
            # 5. base target MLP + 6. zero masked rows, as ACEBlock.
            tgt = tgt + blk.tgt_mlp(blk.tgt_ln2(tgt))
            ctx = ctx * ctx_mask.unsqueeze(-1)
            tgt = tgt * tgt_mask.unsqueeze(-1)

        tgt = self.final_norm(tgt)
        return self._predictions(tgt)

    def loss(
        self,
        batch: Batch | BufferedBatch,
        *,
        data_weight: float = 1.0,
        latent_weight: float = 1.0,
    ) -> torch.Tensor:
        """NLL over active targets; accepts a plain `Batch` or a `BufferedBatch`."""

        if not isinstance(batch, BufferedBatch):
            return super().loss(batch, data_weight=data_weight, latent_weight=latent_weight)
        pred = self.forward_buffered(batch)
        logp = pred.log_prob(batch.target)
        is_latent = self.is_latent[batch.target.var_id]
        weights = torch.where(
            is_latent,
            torch.as_tensor(latent_weight, dtype=logp.dtype, device=logp.device),
            torch.as_tensor(data_weight, dtype=logp.dtype, device=logp.device),
        )
        weights = weights * batch.target.mask.to(logp.dtype)
        return -(logp * weights).sum() / weights.sum().clamp_min(1.0)


# --------------------------------------------------------------------------- #
# Warm start
# --------------------------------------------------------------------------- #


def load_warm_start(
    path: str | Path,
    device: torch.device | str,
    variables: Sequence[Variable],
    *,
    check_batch: Batch | None = None,
) -> BufferedACE:
    """Build a `BufferedACE` from a base ACE checkpoint.

    Loads with `strict=False` under a hard guard: the base checkpoint must
    account for every non-buffer parameter (`unexpected == []` and all missing
    keys under `buf_blocks.`), then `init_from_base` copies the buffer-stream
    weights and zeroes the target gate. If `check_batch` is given, the step-0
    self-check runs against a freshly loaded base `ACE` (see `check_step0`).
    """

    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = ACEConfig(**payload["cfg"])
    model = BufferedACE(list(variables), cfg).to(device)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"base checkpoint has keys BufferedACE lacks: {unexpected}")
    stray = [key for key in missing if not key.startswith("buf_blocks.")]
    if stray:
        raise RuntimeError(f"base checkpoint is missing non-buffer keys: {stray}")
    model.init_from_base()
    if check_batch is not None:
        base = ACE(list(variables), cfg).to(device)
        base.load_state_dict(payload["state_dict"])
        check_step0(model, base, check_batch)
        print("arbuffer warm start: step-0 parity OK (plain forward and zero-gated buffered forward)")
    return model


def load_buffered_checkpoint(path: str | Path, device: torch.device | str, variables: Sequence[Variable]) -> BufferedACE:
    """Load a fine-tuned `BufferedACE` checkpoint (strict; no warm-start logic)."""

    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = ACEConfig(**payload["cfg"])
    model = BufferedACE(list(variables), cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    return model


def synthetic_buffer(model: ACE, batch: Batch, k: int = 8) -> Tokens:
    """Random data-variable VALUE tokens shaped like a buffer, for self-checks."""

    data_vars = [i for i, v in enumerate(model.variables) if v.kind == "data"]
    if not data_vars:
        raise ValueError("synthetic_buffer needs at least one data variable")
    b = batch.context.shape[0]
    device = batch.context.value.device
    x_dim = batch.context.x.shape[-1]
    return Tokens(
        var_id=torch.full((b, k), data_vars[0], dtype=torch.long, device=device),
        x=2.0 * torch.rand(b, k, x_dim, device=device) - 1.0,
        value=torch.randn(b, k, device=device),
        value_index=torch.zeros(b, k, dtype=torch.long, device=device),
        prior=torch.zeros(b, k, PRIOR_FEATURES, device=device),
        mode=torch.full((b, k), VALUE, dtype=torch.long, device=device),
        mask=torch.ones(b, k, dtype=torch.bool, device=device),
    )


@torch.no_grad()
def check_step0(model: BufferedACE, base: ACE, batch: Batch) -> None:
    """Assert a warm-started model is bit-identical to its base checkpoint.

    (a) The inherited plain forward must equal the base forward exactly.
    (b) `forward_buffered` with a non-empty synthetic buffer must equal the
    context-only prediction exactly (the zero-init gate adds exact zeros).
    """

    pred_base = base(batch)
    pred_plain = model(batch)
    if not torch.equal(pred_plain.cont_raw, pred_base.cont_raw) or not torch.equal(
        pred_plain.disc_logits, pred_base.disc_logits
    ):
        raise RuntimeError("step-0 check failed: plain forward differs from the base model")

    buffer = synthetic_buffer(model, batch)
    b, m = batch.target.shape
    k = buffer.shape[1]
    prefix = torch.randint(1, k + 1, (b, m), device=buffer.value.device)
    pred_buf = model.forward_buffered(BufferedBatch(batch.variables, batch.context, buffer, batch.target, prefix))
    if not torch.equal(pred_buf.cont_raw, pred_base.cont_raw) or not torch.equal(
        pred_buf.disc_logits, pred_base.disc_logits
    ):
        raise RuntimeError("step-0 check failed: zero-gated buffered forward differs from context-only")


# --------------------------------------------------------------------------- #
# Inference: cached sampling and joint density evaluation
# --------------------------------------------------------------------------- #


@torch.no_grad()
def encode_context(model: BufferedACE, context: Tokens) -> ContextCache:
    """Run the frozen context stream once and cache its per-layer states."""

    ctx = model._embed(context)
    ctx_mask = context.mask
    if not bool(ctx_mask.any(dim=1).all()):
        raise ValueError("encode_context needs at least one active context token per batch row")
    kp_ctx = ~ctx_mask
    inputs: list[torch.Tensor] = []
    kv: list[torch.Tensor] = []
    for blk in model.blocks:
        inputs.append(ctx)
        ctx_q = blk.ctx_ln1(ctx)
        ctx_att, _ = blk.ctx_attn(ctx_q, ctx_q, ctx_q, key_padding_mask=kp_ctx, need_weights=False)
        ctx = ctx + ctx_att
        ctx = ctx + blk.ctx_mlp(blk.ctx_ln2(ctx))
        kv.append(ctx)  # pre-zeroing, exactly what ACEBlock feeds kv_ln
        ctx = ctx * ctx_mask.unsqueeze(-1)
    return ContextCache(inputs=inputs, kv=kv, mask=ctx_mask)


def _data_tokens(var_index: int, x: torch.Tensor, value: torch.Tensor, mode: int) -> Tokens:
    """Build active single-variable data tokens; `x`/`value` are `[B, T]`."""

    b, t = value.shape
    device = value.device
    return Tokens(
        var_id=torch.full((b, t), var_index, dtype=torch.long, device=device),
        x=x.unsqueeze(-1),
        value=value,
        value_index=torch.zeros(b, t, dtype=torch.long, device=device),
        prior=torch.zeros(b, t, PRIOR_FEATURES, device=device),
        mode=torch.full((b, t), mode, dtype=torch.long, device=device),
        mask=torch.ones(b, t, dtype=torch.bool, device=device),
    )


@torch.no_grad()
def sample_joint(
    model: BufferedACE,
    context: Tokens,
    x: torch.Tensor,
    *,
    var_index: int = 0,
    n_draws: int = 64,
    order: Sequence[int] | None = None,
    random_order: bool = True,
    teacher_force: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw `n_draws` coherent joint samples of a data variable at locations `x`.

    The context (`[1, N]` tokens) is encoded once; its cache is shared by every
    draw stream. Each step decodes one location for all draws at once, samples,
    and pushes the realized token through the buffer stream, growing per-layer
    buffer caches. Returns `(values, log_prob)`, both `[n_draws, K]` aligned
    with the input `x` order; `log_prob[:, j]` scores the realized value at `x[j]`
    under its step's predictive. With `order=None` the decode order is a shared
    random permutation (matching `sample_ar`'s default); buffers were trained on
    random orders.

    If `teacher_force` (`[n_draws, K]`, aligned with `x`) is given, the chain
    consumes those values instead of sampling — this scores a known joint
    sequence through the exact incremental-cache path, the consistency
    counterpart of the one-pass `joint_log_prob`.
    """

    if context.shape[0] != 1:
        raise ValueError("sample_joint expects a single-row context; tile results instead")
    k_total = x.numel()
    device = next(model.parameters()).device
    x = x.to(device).reshape(-1)
    if order is None:
        order = torch.randperm(k_total, device=device).tolist() if random_order else list(range(k_total))
    else:
        order = list(order)

    cache = encode_context(model, context)
    b = n_draws
    # LayerNorm is per-token, so LN'd states can be cached once (context) or
    # appended incrementally (buffer) instead of being recomputed every step.
    ctx_in_ln = [bblk.buf_ln1(t).expand(b, -1, -1) for bblk, t in zip(model.buf_blocks, cache.inputs)]
    kv_c = [blk.kv_ln(t).expand(b, -1, -1) for blk, t in zip(model.blocks, cache.kv)]
    kp_ctx = (~cache.mask).expand(b, -1)

    # Per-layer growing caches: LN'd layer-input states (buffer attention kv)
    # and post-update states already LN'd for the target read.
    buf_in_ln: list[torch.Tensor] = [torch.empty(b, 0, model.cfg.d_model, device=device) for _ in model.blocks]
    buf_kv: list[torch.Tensor] = [torch.empty(b, 0, model.cfg.d_model, device=device) for _ in model.blocks]

    if teacher_force is not None and teacher_force.shape != (b, k_total):
        raise ValueError(f"teacher_force must be [{b}, {k_total}], got {tuple(teacher_force.shape)}")
    values = torch.empty(b, k_total, device=device)
    logps = torch.empty(b, k_total, device=device)
    for step, j in enumerate(order):
        x_step = x[j].expand(b, 1)
        query = _data_tokens(var_index, x_step, torch.zeros(b, 1, device=device), QUERY)
        tgt = model._embed(query)
        for layer, (blk, bblk) in enumerate(zip(model.blocks, model.buf_blocks)):
            tgt_att, _ = blk.cross_attn(
                blk.tgt_ln1(tgt), kv_c[layer], kv_c[layer], key_padding_mask=kp_ctx, need_weights=False
            )
            tgt = tgt + tgt_att
            if step > 0:
                kv_b = buf_kv[layer]
                read, _ = bblk.tgt_buf_attn(bblk.tgt_buf_qln(tgt), kv_b, kv_b, need_weights=False)
                tgt = tgt + read
            tgt = tgt + blk.tgt_mlp(blk.tgt_ln2(tgt))
        pred = model._predictions(model.final_norm(tgt))
        if teacher_force is None:
            value, _ = pred.sample(query)
        else:
            value = teacher_force[:, j : j + 1].to(device)
        values[:, j] = value[:, 0]

        scored = _data_tokens(var_index, x_step, value, VALUE)
        logps[:, j] = pred.log_prob(scored)[:, 0]

        # Push the realized token through the buffer stream, appending caches.
        bstate = model._embed(scored)
        kp_step = torch.cat(
            [kp_ctx, torch.zeros(b, step + 1, dtype=torch.bool, device=device)], dim=1
        )
        for layer, bblk in enumerate(model.buf_blocks):
            q = bblk.buf_ln1(bstate)  # the new token's LN'd layer input: query AND its kv slot
            buf_in_ln[layer] = torch.cat([buf_in_ln[layer], q], dim=1)
            kv = torch.cat([ctx_in_ln[layer], buf_in_ln[layer]], dim=1)
            att, _ = bblk.buf_attn(q, kv, kv, key_padding_mask=kp_step, need_weights=False)
            bstate = bstate + att
            bstate = bstate + bblk.buf_mlp(bblk.buf_ln2(bstate))
            buf_kv[layer] = torch.cat([buf_kv[layer], bblk.tgt_buf_kvln(bstate)], dim=1)
    return values, logps


@torch.no_grad()
def joint_log_prob(
    model: BufferedACE,
    context: Tokens,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    var_index: int = 0,
    n_orders: int = 4,
    orders: Sequence[Sequence[int]] | None = None,
) -> torch.Tensor:
    """Joint log density of `y` at `x` given `context`, in one pass per order.

    Packs the K realized points as buffer tokens and the same K locations as
    queries, with the query at position `k` (0-based) reading exactly the first
    `k` buffer tokens — the strictly-preceding realized points, recovering the
    autoregressive chain in a single forward pass (Algorithm 2 of the paper).
    Autoregressive densities are order-dependent; the result averages the joint
    *density* over the given `orders` (or `n_orders` random ones) via
    log-mean-exp. Pass explicit `orders` to score different methods on identical
    orderings. Returns `[B]` log densities.
    """

    b, k = y.shape
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)
    prefix = torch.arange(k, device=device).expand(b, k)
    if orders is None:
        orders = [torch.randperm(k, device=device).tolist() for _ in range(max(1, n_orders))]
    totals = []
    for order_seq in orders:
        order = torch.as_tensor(list(order_seq), dtype=torch.long, device=device)
        buffer = _data_tokens(var_index, x[:, order], y[:, order], VALUE)
        target = _data_tokens(var_index, x[:, order], y[:, order], QUERY)
        bbatch = BufferedBatch(model.variables, context, buffer, target, prefix)
        logp = model.forward_buffered(bbatch).log_prob(target)
        totals.append(logp.sum(dim=1))
    stacked = torch.stack(totals, dim=0)  # [R, B]
    return torch.logsumexp(stacked, dim=0) - torch.log(torch.tensor(float(stacked.shape[0]), device=device))


@torch.no_grad()
def slow_ar_log_prob(
    model: ACE,
    context: Tokens,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    variables: Sequence[Variable] | None = None,
    var_index: int = 0,
    n_orders: int = 4,
    orders: Sequence[Sequence[int]] | None = None,
) -> torch.Tensor:
    """Joint log density via the base slow-AR path (context re-encoded per step).

    Teacher-forced `sample_ar` recipe: score one point, append its truth to the
    context as a VALUE token, repeat. The baseline the buffer is measured
    against; averages densities over the given `orders` (or `n_orders` random
    ones). `model` may be a base `ACE` or a `BufferedACE` (the plain forward is
    the same).
    """

    b, k = y.shape
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)
    vars_ = list(variables) if variables is not None else model.variables
    if orders is None:
        orders = [torch.randperm(k, device=device).tolist() for _ in range(max(1, n_orders))]
    totals = []
    for order_seq in orders:
        order = list(order_seq)
        ctx = context
        total = torch.zeros(b, device=device)
        for j in order:
            query = _data_tokens(var_index, x[:, j : j + 1], y[:, j : j + 1], QUERY)
            pred = model(Batch(vars_, ctx, query))
            total = total + pred.log_prob(query)[:, 0]
            truth = _data_tokens(var_index, x[:, j : j + 1], y[:, j : j + 1], VALUE)
            ctx = append_or_replace_context_token(
                ctx,
                truth,
                is_latent=model.is_latent,
                is_discrete=model.is_discrete,
                has_bounds=model.has_bounds,
            )
        totals.append(total)
    stacked = torch.stack(totals, dim=0)
    return torch.logsumexp(stacked, dim=0) - torch.log(torch.tensor(float(stacked.shape[0]), device=device))
