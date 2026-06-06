"""Core ACE model and token representation.

Defines `Variable`, `Tokens`, and `Batch`; embeds context and target tokens;
runs the separated context/self-attention and target/cross-attention blocks;
returns shared continuous/discrete predictive distributions; and provides the
autoregressive sampling helper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


VALUE = 0
PRIOR = 1
QUERY = 2
"""Token modes.

VALUE tokens carry an observed scalar or class label. PRIOR tokens carry a
histogram over a latent variable. QUERY tokens ask the model for a predictive
distribution; they may still carry truth for training, but that truth is not
visible to the embedder.
"""


@dataclass(frozen=True)
class Variable:
    """A scalar data or latent variable known to ACE.

    Variables are the schema shared by every batch. `var_id` tensors index into
    this list, so variable identity is available both to the embedder and to the
    prediction object. Continuous values are expected to already live in the
    transformed space named by `transform`.
    """

    name: str
    kind: str  # "data" | "latent"
    value_type: str = "continuous"  # "continuous" | "discrete"
    cardinality: int | None = None
    transform: str = "identity"
    prior_range: tuple[float, float] | None = None
    prior_bins: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"data", "latent"}:
            raise ValueError(f"bad variable kind {self.kind!r}")
        if self.value_type not in {"continuous", "discrete"}:
            raise ValueError(f"bad value_type {self.value_type!r}")
        if self.value_type == "discrete":
            if self.cardinality is None or self.cardinality < 2:
                raise ValueError("discrete variables need cardinality >= 2")
        if self.value_type == "continuous" and self.cardinality is not None:
            raise ValueError("continuous variables should not set cardinality")
        if self.transform not in {"identity", "log", "logit"}:
            raise ValueError(f"bad transform {self.transform!r}")
        if self.prior_bins is not None and self.prior_bins <= 0:
            raise ValueError("prior_bins must be positive")


@dataclass
class Tokens:
    """Padded token set.

    `Tokens` is deliberately just tensors. This keeps examples free to construct
    batches directly while the model sees one uniform representation. Data tokens
    use `x`; latent tokens set `x` to zeros. Continuous variables use `value`;
    discrete variables use `value_index`; unused fields are dummy zeros.

    Target tokens may carry truth in `value` / `value_index` while still having
    mode QUERY. The embedder ignores truth for QUERY tokens, and
    `Predictions.log_prob` uses it for loss.
    """

    var_id: torch.Tensor
    x: torch.Tensor
    value: torch.Tensor
    value_index: torch.Tensor
    prior: torch.Tensor
    mode: torch.Tensor
    mask: torch.Tensor

    def to(self, device: torch.device | str) -> "Tokens":
        return Tokens(
            var_id=self.var_id.to(device),
            x=self.x.to(device),
            value=self.value.to(device),
            value_index=self.value_index.to(device),
            prior=self.prior.to(device),
            mode=self.mode.to(device),
            mask=self.mask.to(device),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.var_id.shape[0]), int(self.var_id.shape[1])

    def column(self, index: int) -> "Tokens":
        sl = slice(index, index + 1)
        return Tokens(
            var_id=self.var_id[:, sl],
            x=self.x[:, sl],
            value=self.value[:, sl],
            value_index=self.value_index[:, sl],
            prior=self.prior[:, sl],
            mode=self.mode[:, sl],
            mask=self.mask[:, sl],
        )

    def with_values(
        self,
        value: torch.Tensor,
        value_index: torch.Tensor,
        mode: int = VALUE,
    ) -> "Tokens":
        return replace(
            self,
            value=value,
            value_index=value_index,
            mode=torch.full_like(self.mode, mode),
        )


@dataclass
class Batch:
    """One ACE task batch: context to condition on, targets to predict."""

    variables: list[Variable]
    context: Tokens
    target: Tokens

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(
            variables=self.variables,
            context=self.context.to(device),
            target=self.target.to(device),
        )


def cat_tokens(parts: Sequence[Tokens]) -> Tokens:
    """Concatenate token sets along the token dimension.

    Used by autoregressive sampling to append newly sampled target values back
    into the context.
    """

    if not parts:
        raise ValueError("cat_tokens needs at least one token set")
    return Tokens(
        var_id=torch.cat([p.var_id for p in parts], dim=1),
        x=torch.cat([p.x for p in parts], dim=1),
        value=torch.cat([p.value for p in parts], dim=1),
        value_index=torch.cat([p.value_index for p in parts], dim=1),
        prior=torch.cat([p.prior for p in parts], dim=1),
        mode=torch.cat([p.mode for p in parts], dim=1),
        mask=torch.cat([p.mask for p in parts], dim=1),
    )


@dataclass
class ACEConfig:
    """Small model configuration.

    The first implementation uses one global `prior_bins` and one scalar
    `x_dim`. That is enough for the Gaussian toy and GP-1D, and avoids ragged
    prior tensors before an example needs them.
    """

    x_dim: int = 1
    prior_bins: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    mlp_hidden: int = 256
    mdn_components: int = 8
    head_hidden: int = 128
    min_scale: float = 1e-3


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class ACEBlock(nn.Module):
    """One ACE transformer block.

    Context tokens update by self-attention. Target tokens then read the updated
    context by cross-attention. Targets never attend to other targets inside the
    base model, preserving the diagonal prediction-map structure; joint samples
    are handled by `sample_ar`.
    """

    def __init__(self, cfg: ACEConfig):
        super().__init__()
        d = cfg.d_model
        self.ctx_ln1 = nn.LayerNorm(d)
        self.tgt_ln1 = nn.LayerNorm(d)
        self.kv_ln = nn.LayerNorm(d)
        self.ctx_attn = nn.MultiheadAttention(d, cfg.n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, cfg.n_heads, batch_first=True)
        self.ctx_ln2 = nn.LayerNorm(d)
        self.tgt_ln2 = nn.LayerNorm(d)
        self.ctx_mlp = _mlp(d, cfg.mlp_hidden, d)
        self.tgt_mlp = _mlp(d, cfg.mlp_hidden, d)

    def forward(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        ctx_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_padding = ~ctx_mask

        # Standard pre-LN residual block for the context encoder.
        ctx_q = self.ctx_ln1(ctx)
        ctx_attn, _ = self.ctx_attn(
            ctx_q,
            ctx_q,
            ctx_q,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        ctx = ctx + ctx_attn
        ctx = ctx + self.ctx_mlp(self.ctx_ln2(ctx))

        # Decoder side: every target query reads all active context keys/values.
        kv = self.kv_ln(ctx)
        tgt_attn, _ = self.cross_attn(
            self.tgt_ln1(tgt),
            kv,
            kv,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        tgt = tgt + tgt_attn
        tgt = tgt + self.tgt_mlp(self.tgt_ln2(tgt))

        ctx = ctx * ctx_mask.unsqueeze(-1)
        tgt = tgt * tgt_mask.unsqueeze(-1)
        return ctx, tgt


class Predictions:
    """Type-aware predictive distributions for a target token set.

    `ACE` emits both continuous and discrete parameters for every target token.
    This wrapper selects the right distribution by `var_id`, so callers can use
    `log_prob`, `mean`, or `sample` without branching on variable type.
    """

    def __init__(
        self,
        cont_raw: torch.Tensor,
        disc_logits: torch.Tensor,
        *,
        is_discrete: torch.Tensor,
        cardinality: torch.Tensor,
        min_scale: float,
    ):
        self.cont_raw = cont_raw
        self.disc_logits = disc_logits
        self.is_discrete = is_discrete
        self.cardinality = cardinality
        self.min_scale = min_scale

    @property
    def components(self) -> int:
        return self.cont_raw.shape[-1] // 3

    def _cont_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return mixture log-weights, locations, and positive scales."""

        k = self.components
        raw_w, loc, raw_scale = self.cont_raw.split(k, dim=-1)
        log_w = F.log_softmax(raw_w, dim=-1)
        scale = F.softplus(raw_scale) + self.min_scale
        return log_w, loc, scale

    def continuous_mean(self) -> torch.Tensor:
        log_w, loc, _ = self._cont_params()
        return (log_w.exp() * loc).sum(dim=-1)

    def continuous_var(self) -> torch.Tensor:
        log_w, loc, scale = self._cont_params()
        w = log_w.exp()
        mean = (w * loc).sum(dim=-1, keepdim=True)
        return (w * (scale.pow(2) + (loc - mean).pow(2))).sum(dim=-1)

    def _valid_logits(self, tokens: Tokens) -> torch.Tensor:
        """Mask logits outside each discrete variable's local label set.

        A single shared categorical head emits `Kmax` logits. Variables with
        smaller cardinality read only the first `k` classes.
        """

        kmax = self.disc_logits.shape[-1]
        card = self.cardinality[tokens.var_id].clamp_min(1)
        valid = torch.arange(kmax, device=tokens.var_id.device) < card.unsqueeze(-1)
        return self.disc_logits.masked_fill(~valid, torch.finfo(self.disc_logits.dtype).min)

    def log_prob(self, tokens: Tokens) -> torch.Tensor:
        """Per-token log probability under the appropriate output family."""

        logp = torch.zeros_like(tokens.value, dtype=self.cont_raw.dtype)
        discrete = self.is_discrete[tokens.var_id]
        continuous = ~discrete

        if continuous.any():
            log_w, loc, scale = self._cont_params()
            y = tokens.value.unsqueeze(-1)
            comp = -0.5 * ((y - loc) / scale).pow(2) - scale.log() - 0.5 * math.log(2.0 * math.pi)
            cont_lp = torch.logsumexp(log_w + comp, dim=-1)
            logp = torch.where(continuous, cont_lp, logp)

        if discrete.any():
            log_probs = F.log_softmax(self._valid_logits(tokens), dim=-1)
            idx = tokens.value_index.clamp_min(0).unsqueeze(-1)
            disc_lp = log_probs.gather(-1, idx).squeeze(-1)
            logp = torch.where(discrete, disc_lp, logp)

        return logp

    def mean(self, tokens: Tokens) -> torch.Tensor:
        mean = self.continuous_mean()
        discrete = self.is_discrete[tokens.var_id]
        if discrete.any():
            probs = F.softmax(self._valid_logits(tokens), dim=-1)
            ar = torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)
            disc_mean = (probs * ar).sum(dim=-1)
            mean = torch.where(discrete, disc_mean, mean)
        return mean

    def sample(self, tokens: Tokens) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample values for target tokens.

        Continuous samples are returned in `value`; discrete samples are returned
        in both `value_index` and `value` (as floats) for convenience.
        """

        value = self.continuous_mean()
        value_index = torch.zeros_like(tokens.value_index)
        discrete = self.is_discrete[tokens.var_id]

        if (~discrete).any():
            log_w, loc, scale = self._cont_params()
            cat = torch.distributions.Categorical(logits=log_w)
            comp_idx = cat.sample()
            loc_s = loc.gather(-1, comp_idx.unsqueeze(-1)).squeeze(-1)
            scale_s = scale.gather(-1, comp_idx.unsqueeze(-1)).squeeze(-1)
            cont_sample = loc_s + scale_s * torch.randn_like(loc_s)
            value = torch.where(~discrete, cont_sample, value)

        if discrete.any():
            cat = torch.distributions.Categorical(logits=self._valid_logits(tokens))
            disc_sample = cat.sample()
            value_index = torch.where(discrete, disc_sample, value_index)
            value = torch.where(discrete, disc_sample.to(value.dtype), value)

        return value, value_index

    def sample_as_tokens(self, tokens: Tokens) -> Tokens:
        value, value_index = self.sample(tokens)
        return tokens.with_values(value=value, value_index=value_index, mode=VALUE)


class ACE(nn.Module):
    """Amortized Conditioning Engine.

    The model consumes a `Batch(context, target)`, embeds both token sets into a
    shared space, updates context with self-attention, lets targets cross-attend
    to context, and returns predictive distributions for the target variables.
    """

    def __init__(self, variables: Sequence[Variable], cfg: ACEConfig | None = None):
        super().__init__()
        self.variables = list(variables)
        self.cfg = ACEConfig() if cfg is None else cfg
        cfg = self.cfg
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if any(v.prior_bins is not None and v.prior_bins != cfg.prior_bins for v in self.variables):
            raise ValueError("nanoACE currently uses one global prior_bins")

        n_vars = len(self.variables)
        is_discrete = torch.tensor([v.value_type == "discrete" for v in self.variables], dtype=torch.bool)
        is_latent = torch.tensor([v.kind == "latent" for v in self.variables], dtype=torch.bool)
        cardinality = torch.tensor([v.cardinality or 0 for v in self.variables], dtype=torch.long)
        offsets = []
        total_disc = 0
        for v in self.variables:
            offsets.append(total_disc)
            if v.value_type == "discrete":
                total_disc += int(v.cardinality or 0)
        max_cardinality = max([int(v.cardinality or 0) for v in self.variables] + [1])

        self.register_buffer("is_discrete", is_discrete, persistent=False)
        self.register_buffer("is_latent", is_latent, persistent=False)
        self.register_buffer("cardinality", cardinality, persistent=False)
        self.register_buffer("disc_offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)

        self.var_embed = nn.Embedding(n_vars, cfg.d_model)
        self.mode_embed = nn.Embedding(3, cfg.d_model)
        self.x_embed = _mlp(cfg.x_dim, cfg.mlp_hidden, cfg.d_model)
        self.value_embed = _mlp(1, cfg.mlp_hidden, cfg.d_model)
        self.prior_embed = _mlp(cfg.prior_bins, cfg.mlp_hidden, cfg.d_model)
        self.disc_value_embed = nn.Embedding(max(total_disc, 1), cfg.d_model)
        self.unknown = nn.Parameter(torch.zeros(cfg.d_model))
        nn.init.normal_(self.unknown, std=0.02)

        self.blocks = nn.ModuleList([ACEBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.cont_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_hidden),
            nn.GELU(),
            nn.Linear(cfg.head_hidden, 3 * cfg.mdn_components),
        )
        self.disc_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_hidden),
            nn.GELU(),
            nn.Linear(cfg.head_hidden, max_cardinality),
        )

    def _embed(self, tokens: Tokens) -> torch.Tensor:
        """Embed VALUE, PRIOR, and QUERY tokens with one additive recipe."""

        var = self.var_embed(tokens.var_id)
        mode = self.mode_embed(tokens.mode.clamp(0, 2))
        x = self.x_embed(tokens.x)
        x = torch.where(self.is_latent[tokens.var_id].unsqueeze(-1), torch.zeros_like(x), x)

        discrete = self.is_discrete[tokens.var_id]
        # Discrete context values use one global table with per-variable offsets.
        disc_index = self.disc_offsets[tokens.var_id] + tokens.value_index.clamp_min(0)
        disc_index = disc_index.clamp_max(self.disc_value_embed.num_embeddings - 1)
        val_disc = self.disc_value_embed(disc_index)
        val_cont = self.value_embed(tokens.value.unsqueeze(-1))
        val = torch.where(discrete.unsqueeze(-1), val_disc, val_cont)

        prior = self.prior_embed(tokens.prior)
        unknown = self.unknown.view(1, 1, -1).expand_as(var)

        # QUERY tokens use a learned unknown-value embedding, even if target truth
        # is stored in the token for loss computation.
        payload = torch.where((tokens.mode == PRIOR).unsqueeze(-1), prior, val)
        payload = torch.where((tokens.mode == QUERY).unsqueeze(-1), unknown, payload)
        out = var + mode + x + payload
        return out * tokens.mask.unsqueeze(-1)

    def forward(self, batch: Batch) -> Predictions:
        ctx = self._embed(batch.context)
        tgt = self._embed(batch.target)
        ctx_mask, tgt_mask = batch.context.mask, batch.target.mask
        if not bool(ctx_mask.any()):
            raise ValueError("ACE needs at least one active context token")
        for block in self.blocks:
            ctx, tgt = block(ctx, tgt, ctx_mask, tgt_mask)
        tgt = self.final_norm(tgt)
        return Predictions(
            cont_raw=self.cont_head(tgt),
            disc_logits=self.disc_head(tgt),
            is_discrete=self.is_discrete,
            cardinality=self.cardinality,
            min_scale=self.cfg.min_scale,
        )

    def loss(
        self,
        batch: Batch,
        *,
        data_weight: float = 1.0,
        latent_weight: float = 1.0,
    ) -> torch.Tensor:
        """Average negative log likelihood over active target tokens."""

        pred = self(batch)
        logp = pred.log_prob(batch.target)
        is_latent = self.is_latent[batch.target.var_id]
        weights = torch.where(
            is_latent,
            torch.as_tensor(latent_weight, dtype=logp.dtype, device=logp.device),
            torch.as_tensor(data_weight, dtype=logp.dtype, device=logp.device),
        )
        weights = weights * batch.target.mask.to(logp.dtype)
        return -(logp * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def sample_ar(
    model: ACE,
    batch: Batch,
    order: Sequence[int] | None = None,
    *,
    random_order: bool = True,
) -> Tokens:
    """Autoregressively sample target tokens by appending sampled values to context.

    This is the explicit joint-distribution path. The base model predicts
    conditionally independent target marginals; autoregression turns sampled
    targets into additional context so later targets can depend on earlier ones.
    With `order=None`, the target order is randomized by default; pass an
    explicit `order` or `random_order=False` for deterministic sampling.
    """

    _, t = batch.target.shape
    if order is None:
        if random_order:
            order = torch.randperm(t, device=batch.target.var_id.device).tolist()
        else:
            order = list(range(t))
    else:
        order = list(order)
    context = batch.context
    sampled: list[Tokens | None] = [None] * t
    for j in order:
        query = batch.target.column(j)
        pred = model(Batch(batch.variables, context, query))
        value_tok = pred.sample_as_tokens(query)
        sampled[j] = value_tok
        context = cat_tokens([context, value_tok])
    return cat_tokens([tok for tok in sampled if tok is not None])
