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
PRIOR_FEATURES = 2
"""Token modes.

VALUE tokens carry an observed data scalar or class label. PRIOR tokens carry
two continuous-latent information features `(mean_internal, spread_internal)`.
For bounded continuous latents, zero spread is an exact known value. QUERY
tokens ask the model for a predictive distribution; they may still carry truth
for training, but that truth is not visible to the embedder.
"""


@dataclass(frozen=True)
class Variable:
    """A scalar data or latent variable known to ACE.

    Variables are the schema shared by every batch. `var_id` tensors index into
    this list, so variable identity is available both to the embedder and to the
    prediction object. Continuous values are expected to already live in the
    transformed semantic space named by `transform`; bounded continuous latents
    are then affine-encoded to `[-1, 1]` at token boundaries.
    """

    name: str
    kind: str  # "data" | "latent"
    value_type: str = "continuous"  # "continuous" | "discrete"
    cardinality: int | None = None
    transform: str = "identity"
    bounds: tuple[float, float] | None = None

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
        if self.bounds is not None:
            lo, hi = self.bounds
            if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                raise ValueError("bounds must be finite and ordered")
            if self.value_type != "continuous":
                raise ValueError("bounds are only valid for continuous variables")
        if self.kind == "latent" and self.value_type == "continuous" and self.bounds is None:
            raise ValueError("continuous latent variables need finite bounds")


def _is_bounded_continuous_latent(variable: Variable) -> bool:
    return variable.kind == "latent" and variable.value_type == "continuous" and variable.bounds is not None


def encode_value(variable: Variable, value: torch.Tensor) -> torch.Tensor:
    """Encode one variable's native value into ACE token coordinates.

    Only bounded continuous latents are transformed. Data variables and
    discrete labels are returned unchanged.
    """

    if not _is_bounded_continuous_latent(variable):
        return value
    assert variable.bounds is not None
    lo = torch.as_tensor(variable.bounds[0], device=value.device, dtype=value.dtype)
    hi = torch.as_tensor(variable.bounds[1], device=value.device, dtype=value.dtype)
    return 2.0 * (value - lo) / (hi - lo) - 1.0


def decode_value(variable: Variable, value: torch.Tensor) -> torch.Tensor:
    """Decode one variable's ACE token value back to native coordinates."""

    if not _is_bounded_continuous_latent(variable):
        return value
    assert variable.bounds is not None
    lo = torch.as_tensor(variable.bounds[0], device=value.device, dtype=value.dtype)
    hi = torch.as_tensor(variable.bounds[1], device=value.device, dtype=value.dtype)
    return lo + 0.5 * (value + 1.0) * (hi - lo)


def encode_token_values(variables: Sequence[Variable], var_id: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Encode a mixed token-shaped value tensor into ACE token coordinates."""

    out = value
    for idx, variable in enumerate(variables):
        if _is_bounded_continuous_latent(variable):
            out = torch.where(var_id == idx, encode_value(variable, value), out)
    return out


def decode_token_values(variables: Sequence[Variable], var_id: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Decode a mixed token-shaped value tensor from ACE token coordinates."""

    out = value
    for idx, variable in enumerate(variables):
        if _is_bounded_continuous_latent(variable):
            out = torch.where(var_id == idx, decode_value(variable, value), out)
    return out


def sample_reveal_mask(
    n_latents: int,
    batch_size: int,
    q: float,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample which latents are revealed as context per task: `bool[batch, n_latents]`.

    Shared reveal/conditioning DGP for the example samplers. Per task:

    - with probability `q`, reveal *nothing* (pure inference / pure-prior);
    - otherwise split the revealing mass evenly between two schemes:
      - **uniform over subsets**: a uniform random non-empty subset of the latents
        (every specific reveal pattern equally likely);
      - **uniform over count**: a count `k` drawn uniformly from `1..n_latents`,
        then a uniform random size-`k` subset (every reveal *count* equally likely).

    The mixture keeps the headline 0-reveal case well represented (mass `q`), gives
    every specific subset a fair floor via the first scheme, and keeps the extremes
    (notably revealing *all* latents) from being starved as `n_latents` grows via
    the second. Conditioning on any subset of latents is therefore in-distribution.

    Examples interpret a revealed latent per their own convention (e.g. an exact
    zero-spread token); a non-revealed latent is queried.
    """

    if not 0.0 <= q <= 1.0:
        raise ValueError(f"reveal q must be in [0, 1], got {q}")
    if not 1 <= n_latents <= 62:
        raise ValueError("n_latents must be in [1, 62] for int64 bitmask sampling")

    reveal_any = torch.rand(batch_size, device=device) >= q  # P(reveal any) = 1 - q

    # Scheme A: uniform over the 2^L - 1 non-empty subsets, via an integer bitmask.
    n_subsets = (1 << n_latents) - 1
    codes = torch.randint(1, n_subsets + 1, (batch_size,), device=device)
    bits = 1 << torch.arange(n_latents, device=device)
    subset_mask = (codes[:, None] & bits[None, :]) > 0

    # Scheme B: count k uniform in 1..L, then a uniform random size-k subset. Random
    # scores give a uniform permutation; the k lowest-ranked latents are revealed.
    k = torch.randint(1, n_latents + 1, (batch_size,), device=device)
    ranks = torch.rand(batch_size, n_latents, device=device).argsort(dim=1).argsort(dim=1)
    count_mask = ranks < k[:, None]

    use_count = torch.rand(batch_size, device=device) < 0.5
    mask = torch.where(use_count[:, None], count_mask, subset_mask)
    return mask & reveal_any[:, None]


@dataclass
class Tokens:
    """Padded token set.

    `Tokens` is deliberately just tensors. This keeps examples free to construct
    batches directly while the model sees one uniform representation. Data tokens
    use `x`; latent tokens set `x` to zeros. Continuous variables use `value`;
    bounded continuous latent values are in internal `[-1, 1]` coordinates.
    Discrete variables use `value_index`; unused fields are dummy zeros.

    Target tokens may carry truth in `value` / `value_index` while still having
    mode QUERY. The embedder ignores truth for QUERY tokens, and
    `Predictions.log_prob` uses it for loss.

    `prior` has shape `[B, T, PRIOR_FEATURES]`. For bounded continuous latent
    PRIOR tokens, `prior[..., 0]` is the mean/location in internal coordinates
    and `prior[..., 1]` is the internal-coordinate spread. Spread zero denotes
    an exact known latent value.
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

    The implementation uses one scalar `x_dim` for data covariates. Prior
    information uses the fixed two-feature representation named by
    `PRIOR_FEATURES`.
    """

    x_dim: int = 1
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
        is_latent: torch.Tensor,
        cardinality: torch.Tensor,
        has_bounds: torch.Tensor,
        bound_lo: torch.Tensor,
        bound_hi: torch.Tensor,
        min_scale: float,
    ):
        self.cont_raw = cont_raw
        self.disc_logits = disc_logits
        self.is_discrete = is_discrete
        self.is_latent = is_latent
        self.cardinality = cardinality
        self.has_bounds = has_bounds
        self.bound_lo = bound_lo
        self.bound_hi = bound_hi
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

    def _bounded_continuous(self, tokens: Tokens) -> torch.Tensor:
        return self.is_latent[tokens.var_id] & self.has_bounds[tokens.var_id] & ~self.is_discrete[tokens.var_id]

    def log_prob_native(self, tokens: Tokens) -> torch.Tensor:
        """Per-token log probability in native coordinates.

        Bounded continuous latent token-space densities are adjusted by the
        constant affine Jacobian. Data and discrete variables are unchanged.
        """

        logp = self.log_prob(tokens)
        bounded = self._bounded_continuous(tokens)
        width = (self.bound_hi[tokens.var_id] - self.bound_lo[tokens.var_id]).clamp_min(1e-12)
        jac = torch.log(2.0 / width)
        return logp + torch.where(bounded, jac, torch.zeros_like(jac))

    def mean_native(self, tokens: Tokens) -> torch.Tensor:
        """Predictive mean in native coordinates."""

        mean = self.mean(tokens)
        bounded = self._bounded_continuous(tokens)
        lo = self.bound_lo[tokens.var_id]
        hi = self.bound_hi[tokens.var_id]
        decoded = lo + 0.5 * (mean + 1.0) * (hi - lo)
        return torch.where(bounded, decoded, mean)

    def continuous_var_native(self, tokens: Tokens) -> torch.Tensor:
        """Continuous predictive variance in native coordinates."""

        var = self.continuous_var()
        bounded = self._bounded_continuous(tokens)
        scale = 0.5 * (self.bound_hi[tokens.var_id] - self.bound_lo[tokens.var_id])
        return torch.where(bounded, var * scale.pow(2), var)

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

    def sample_native(self, tokens: Tokens) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample target values and decode bounded continuous latents."""

        value, value_index = self.sample(tokens)
        bounded = self._bounded_continuous(tokens)
        lo = self.bound_lo[tokens.var_id]
        hi = self.bound_hi[tokens.var_id]
        decoded = lo + 0.5 * (value + 1.0) * (hi - lo)
        return torch.where(bounded, decoded, value), value_index

    def sample_as_tokens(self, tokens: Tokens) -> Tokens:
        value, value_index = self.sample(tokens)
        return tokens.with_values(value=value, value_index=value_index, mode=VALUE)

    def sample_as_context_tokens(self, tokens: Tokens) -> Tokens:
        """Sample tokens suitable for appending back into context.

        Bounded continuous latent samples are emitted as zero-spread PRIOR
        information tokens. Data and discrete samples remain VALUE tokens.
        """

        value, value_index = self.sample(tokens)
        bounded = self._bounded_continuous(tokens)
        prior = torch.zeros(
            *tokens.var_id.shape,
            PRIOR_FEATURES,
            device=tokens.value.device,
            dtype=tokens.value.dtype,
        )
        prior[..., 0] = torch.where(bounded, value, torch.zeros_like(value))
        mode = torch.where(
            bounded,
            torch.full_like(tokens.mode, PRIOR),
            torch.full_like(tokens.mode, VALUE),
        )
        return replace(tokens, value=value, value_index=value_index, prior=prior, mode=mode)


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

        n_vars = len(self.variables)
        is_discrete = torch.tensor([v.value_type == "discrete" for v in self.variables], dtype=torch.bool)
        is_latent = torch.tensor([v.kind == "latent" for v in self.variables], dtype=torch.bool)
        has_bounds = torch.tensor([v.bounds is not None for v in self.variables], dtype=torch.bool)
        bound_lo = torch.tensor([v.bounds[0] if v.bounds is not None else 0.0 for v in self.variables], dtype=torch.float32)
        bound_hi = torch.tensor([v.bounds[1] if v.bounds is not None else 1.0 for v in self.variables], dtype=torch.float32)
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
        self.register_buffer("has_bounds", has_bounds, persistent=False)
        self.register_buffer("bound_lo", bound_lo, persistent=False)
        self.register_buffer("bound_hi", bound_hi, persistent=False)
        self.register_buffer("cardinality", cardinality, persistent=False)
        self.register_buffer("disc_offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)

        self.var_embed = nn.Embedding(n_vars, cfg.d_model)
        self.mode_embed = nn.Embedding(3, cfg.d_model)
        self.x_embed = _mlp(cfg.x_dim, cfg.mlp_hidden, cfg.d_model)
        self.value_embed = _mlp(1, cfg.mlp_hidden, cfg.d_model)
        self.spread_embed = _mlp(PRIOR_FEATURES, cfg.mlp_hidden, cfg.d_model)
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

        prior_input = tokens.prior[..., :PRIOR_FEATURES]
        prior = self.value_embed(prior_input[..., 0:1])
        prior = prior + prior_input[..., 1:2] * self.spread_embed(prior_input)
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
        if not bool(ctx_mask.any(dim=1).all()):
            raise ValueError("ACE needs at least one active context token per batch row")
        for block in self.blocks:
            ctx, tgt = block(ctx, tgt, ctx_mask, tgt_mask)
        tgt = self.final_norm(tgt)
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


def append_or_replace_context_token(
    context: Tokens,
    token: Tokens,
    *,
    is_latent: torch.Tensor,
    is_discrete: torch.Tensor,
    has_bounds: torch.Tensor,
) -> Tokens:
    """Append a context token, replacing continuous-latent info when present.

    `token` may contain one or more columns. For bounded continuous latent
    PRIOR tokens, an existing active PRIOR token with the same `var_id` is
    replaced row-wise. Other active tokens are appended.
    """

    if token.shape[1] != 1:
        out = context
        for idx in range(token.shape[1]):
            out = append_or_replace_context_token(
                out,
                token.column(idx),
                is_latent=is_latent,
                is_discrete=is_discrete,
                has_bounds=has_bounds,
            )
        return out

    b, _ = token.shape
    tok_var = token.var_id[:, 0]
    tok_active = token.mask[:, 0]
    tok_info = (
        tok_active
        & (token.mode[:, 0] == PRIOR)
        & is_latent[tok_var]
        & has_bounds[tok_var]
        & ~is_discrete[tok_var]
    )

    fields = {
        "var_id": context.var_id.clone(),
        "x": context.x.clone(),
        "value": context.value.clone(),
        "value_index": context.value_index.clone(),
        "prior": context.prior.clone(),
        "mode": context.mode.clone(),
        "mask": context.mask.clone(),
    }
    replaced = torch.zeros(b, device=tok_var.device, dtype=torch.bool)

    for row in range(b):
        if not bool(tok_info[row]):
            continue
        match = torch.nonzero(
            (context.var_id[row] == tok_var[row])
            & (context.mode[row] == PRIOR)
            & context.mask[row],
            as_tuple=False,
        ).flatten()
        if match.numel() == 0:
            continue
        col = int(match[0])
        fields["var_id"][row, col] = token.var_id[row, 0]
        fields["x"][row, col] = token.x[row, 0]
        fields["value"][row, col] = token.value[row, 0]
        fields["value_index"][row, col] = token.value_index[row, 0]
        fields["prior"][row, col] = token.prior[row, 0]
        fields["mode"][row, col] = token.mode[row, 0]
        fields["mask"][row, col] = token.mask[row, 0]
        replaced[row] = True

    updated = Tokens(**fields)
    append_mask = tok_active & ~replaced
    if not bool(append_mask.any()):
        return updated
    return cat_tokens([updated, replace(token, mask=append_mask[:, None])])


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
        value_tok = pred.sample_as_context_tokens(query)
        sampled[j] = value_tok
        context = append_or_replace_context_token(
            context,
            value_tok,
            is_latent=model.is_latent,
            is_discrete=model.is_discrete,
            has_bounds=model.has_bounds,
        )
    return cat_tokens([tok for tok in sampled if tok is not None])
