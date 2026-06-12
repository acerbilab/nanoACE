"""ALINE extension for nanoACE: joint amortized inference + active data acquisition.

Implements ALINE (Huang et al., 2025, "ALINE: Joint Amortization for Bayesian
Inference and Active Data Acquisition", NeurIPS 2025) on top of an unchanged
`ACE`. The re-expression is deliberately ACE-native:

- **The inference network IS the core `ACE`.** Parameter targets are latent
  QUERY tokens, predictive targets are data QUERY tokens, and the paper's
  target specifier xi collapses into *which target tokens are active* (a
  per-row mask over a fixed target superset). The target tokens do double
  duty: they are the queries `q_phi` answers (NLL + reward are computed on
  them) and, through their final states, the goal representation the policy
  reads — including how uncertain the model still is about each goal.
- **Query candidates are "hypothetical targets".** A candidate at location `x`
  is a data QUERY token embedded by the core embedder verbatim.
- **The policy is a read-only decoder.** `PolicyBlock`s cross-attend from the
  candidate tokens to the *detached* final context/target states and a linear
  head scores each candidate (masked softmax over the remaining pool). Nothing
  is written back into the trunk, so the inference path stays bit-identical to
  `ACE.forward` — `check_step0` asserts it — and the gradient firewall is
  structural: the policy-gradient loss can only touch `policy_*` parameters,
  the NLL only the base.

Checked against the released reference implementation (huangdaolang/aline,
`model/encoder.py:create_mask`): context rows attend only to context, target
rows only to context, nothing attends to queries, and queries attend to
context + the xi-selected targets only (no query-query self-attention). The
reference is therefore itself a read-only query/target stream over per-layer
context states. The two deliberate differences here: separate policy weights
instead of the reference's single shared-weight masked stream (what makes the
phi/psi separation structural), and final-state reads instead of per-layer
reads (the per-layer-cache upgrade path is exactly the reference's pattern).

This module is task-agnostic: it knows `ace.py`, not `gp1d.py`. The GP-1D
active-learning episodes, RL training loop, and diagnostics live in
`gp1d_aline.py` next to this file; design decisions and deviations are in the
local DEVLOG.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

# This file lives in extensions/aline/; import the core from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ace import (  # noqa: E402
    ACE,
    ACEConfig,
    Batch,
    Predictions,
    Tokens,
    Variable,
    _mlp,
)


POLICY_PREFIXES = ("policy_blocks.", "policy_norm.", "policy_head.")
"""Parameter-name prefixes that belong to the acquisition policy (psi).

Everything else is the inference network (phi) = the unchanged `ACE` base.
`load_warm_start`'s strict guard, the two-optimizer training loop, and the
gradient-isolation checks all key on this split.
"""


class PolicyBlock(nn.Module):
    """One read-only policy decoder block.

    Pre-LN residual ops mirroring `ACEBlock` conventions: the candidate stream
    cross-attends to the (detached) final context states, then to the
    (detached) final target states — the goal xi and its current posterior
    state — then an MLP. Candidates never attend to each other (pointwise
    scoring; the softmax over the pool supplies the competition — this matches
    the reference implementation, whose mask keeps query-query at -inf) and
    nothing here writes back into the trunk.
    """

    def __init__(self, cfg: ACEConfig):
        super().__init__()
        d = cfg.d_model
        self.q_ln1 = nn.LayerNorm(d)
        self.ctx_kv_ln = nn.LayerNorm(d)
        self.ctx_attn = nn.MultiheadAttention(d, cfg.n_heads, batch_first=True)
        self.q_ln2 = nn.LayerNorm(d)
        self.tgt_kv_ln = nn.LayerNorm(d)
        self.tgt_attn = nn.MultiheadAttention(d, cfg.n_heads, batch_first=True)
        self.q_ln3 = nn.LayerNorm(d)
        self.mlp = _mlp(d, cfg.mlp_hidden, d)

    def forward(
        self,
        qry: torch.Tensor,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        qry_mask: torch.Tensor,
        ctx_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        ctx_kv = self.ctx_kv_ln(ctx)
        read_ctx, _ = self.ctx_attn(
            self.q_ln1(qry),
            ctx_kv,
            ctx_kv,
            key_padding_mask=~ctx_mask,
            need_weights=False,
        )
        qry = qry + read_ctx
        tgt_kv = self.tgt_kv_ln(tgt)
        read_tgt, _ = self.tgt_attn(
            self.q_ln2(qry),
            tgt_kv,
            tgt_kv,
            key_padding_mask=~tgt_mask,
            need_weights=False,
        )
        qry = qry + read_tgt
        qry = qry + self.mlp(self.q_ln3(qry))
        return qry * qry_mask.unsqueeze(-1)


class ALINE(ACE):
    """ACE + a read-only acquisition policy over a candidate pool.

    The inference path is the inherited `ACE` computation, verbatim:
    `forward_with_states` re-runs the core block loop (same modules, same
    order) and additionally returns the final context and target states for
    the policy to read. `check_step0` keeps that re-implementation honest
    bitwise. The policy side (`policy_blocks` + `policy_norm` + `policy_head`)
    is the only new parameter surface; see `POLICY_PREFIXES`.
    """

    def __init__(
        self,
        variables: Sequence[Variable],
        cfg: ACEConfig | None = None,
        *,
        n_policy_blocks: int = 2,
    ):
        super().__init__(variables, cfg)
        if n_policy_blocks < 1:
            raise ValueError("ALINE needs at least one policy block")
        self.n_policy_blocks = n_policy_blocks
        self.policy_blocks = nn.ModuleList(PolicyBlock(self.cfg) for _ in range(n_policy_blocks))
        self.policy_norm = nn.LayerNorm(self.cfg.d_model)
        self.policy_head = nn.Linear(self.cfg.d_model, 1)

    def forward_with_states(self, batch: Batch) -> tuple[Predictions, torch.Tensor, torch.Tensor]:
        """The core `ACE.forward`, additionally returning the final states.

        Predictions are bit-identical to `ACE.forward` (asserted by
        `check_step0`). Returns `(predictions, ctx_states, tgt_states)` where
        `ctx_states` is the final-layer context stream and `tgt_states` the
        final-normed target stream — the exact representation the heads read,
        so each target state carries both the goal identity and its current
        posterior state.
        """

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
        pred = Predictions(
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
        return pred, ctx, tgt

    def policy_logits(
        self,
        query: Tokens,
        ctx_states: torch.Tensor,
        tgt_states: torch.Tensor,
        ctx_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score each candidate token; `-inf` where `query.mask` is False.

        The gradient firewall lives here: candidate embeddings are computed
        under `no_grad` (the shared embedder is a base/phi parameter) and the
        trunk states enter detached, so a policy-gradient backward can reach
        only `policy_*` parameters.

        Precondition: every row needs >= 1 active context and target token
        (an all-masked key row NaNs through `MultiheadAttention`). Training
        guarantees this — xi is non-empty by construction.
        """

        with torch.no_grad():
            qry = self._embed(query)
        ctx_states = ctx_states.detach()
        tgt_states = tgt_states.detach()
        for block in self.policy_blocks:
            qry = block(qry, ctx_states, tgt_states, query.mask, ctx_mask, tgt_mask)
        logits = self.policy_head(self.policy_norm(qry)).squeeze(-1)
        return logits.masked_fill(~query.mask, float("-inf"))

    def policy_parameters(self) -> list[nn.Parameter]:
        return [p for name, p in self.named_parameters() if name.startswith(POLICY_PREFIXES)]

    def base_parameters(self) -> list[nn.Parameter]:
        return [p for name, p in self.named_parameters() if not name.startswith(POLICY_PREFIXES)]


def _check_partition(model: ALINE) -> None:
    """Assert base/policy parameter sets partition `named_parameters()`."""

    total = sum(1 for _ in model.parameters())
    if len(model.policy_parameters()) + len(model.base_parameters()) != total:
        raise RuntimeError("policy/base parameter split does not partition the model")
    if not model.policy_parameters():
        raise RuntimeError("policy parameter set is empty; POLICY_PREFIXES out of date?")


@torch.no_grad()
def check_step0(model: ALINE, base: ACE, batch: Batch) -> None:
    """Assert the inference path equals the base model bitwise.

    (a) The inherited plain forward must equal the base forward exactly (same
    code path, same weights). (b) `forward_with_states` — the re-implemented
    block loop the policy taps — must also be bit-equal; this is the coupling
    guard: if `ace.py`'s forward changes, the warm start fails loudly here.
    """

    pred_base = base(batch)
    pred_plain = model(batch)
    if not torch.equal(pred_plain.cont_raw, pred_base.cont_raw) or not torch.equal(
        pred_plain.disc_logits, pred_base.disc_logits
    ):
        raise RuntimeError("step-0 check failed: plain forward differs from the base model")
    pred_states, _, _ = model.forward_with_states(batch)
    if not torch.equal(pred_states.cont_raw, pred_base.cont_raw) or not torch.equal(
        pred_states.disc_logits, pred_base.disc_logits
    ):
        raise RuntimeError("step-0 check failed: forward_with_states differs from the base forward")


def load_warm_start(
    path: str | Path,
    device: torch.device | str,
    variables: Sequence[Variable],
    *,
    n_policy_blocks: int = 2,
    check_batch: Batch | None = None,
) -> ALINE:
    """Build an `ALINE` from a base ACE checkpoint.

    Loads with `strict=False` under a hard guard: the base checkpoint must
    account for every non-policy parameter (`unexpected == []` and all missing
    keys under `POLICY_PREFIXES`). The policy modules keep their fresh
    initialization — nothing is copied; the near-uniform initial policy is the
    paper's random warm-up actions, approximately. If `check_batch` is given,
    the step-0 bitwise parity check runs against a freshly loaded base `ACE`.
    """

    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = ACEConfig(**payload["cfg"])
    model = ALINE(list(variables), cfg, n_policy_blocks=n_policy_blocks).to(device)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"base checkpoint has keys ALINE lacks: {unexpected}")
    stray = [key for key in missing if not key.startswith(POLICY_PREFIXES)]
    if stray:
        raise RuntimeError(f"base checkpoint is missing non-policy keys: {stray}")
    _check_partition(model)
    if check_batch is not None:
        base = ACE(list(variables), cfg).to(device)
        base.load_state_dict(payload["state_dict"])
        check_step0(model, base, check_batch)
        print("aline warm start: step-0 parity OK (plain forward and forward_with_states bit-equal)")
    return model


def load_aline_checkpoint(path: str | Path, device: torch.device | str, variables: Sequence[Variable]) -> ALINE:
    """Load a trained `ALINE` checkpoint (strict; no warm-start logic).

    The policy depth is inferred from the state dict (`policy_blocks.{i}.`
    keys), so checkpoints load without carrying extra config.
    """

    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = ACEConfig(**payload["cfg"])
    block_ids = {
        int(key.split(".")[1])
        for key in payload["state_dict"]
        if key.startswith("policy_blocks.")
    }
    if not block_ids:
        raise RuntimeError(f"{path} has no policy_blocks.* keys; not an ALINE checkpoint")
    model = ALINE(list(variables), cfg, n_policy_blocks=max(block_ids) + 1).to(device)
    model.load_state_dict(payload["state_dict"])
    return model
