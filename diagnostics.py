"""Reusable grid diagnostics for ACE predictions.

Provides scalar target-token construction, batched grid queries, conditional
one-variable queries, symmetrized two-variable AR joint densities, and simple
moments for densities evaluated on grids.
"""

from __future__ import annotations

import math

import torch

from ace import ACE, Batch, PRIOR, PRIOR_FEATURES, QUERY, VALUE, Tokens, append_or_replace_context_token


def make_scalar_tokens(
    *,
    var_id: torch.Tensor,
    value: torch.Tensor,
    prior: torch.Tensor,
    mode: torch.Tensor,
    mask: torch.Tensor,
    x_dim: int,
) -> Tokens:
    """Construct tokens for scalar examples with zero covariates."""

    b, t = var_id.shape
    return Tokens(
        var_id=var_id.long(),
        x=torch.zeros(b, t, x_dim, device=value.device, dtype=value.dtype),
        value=value,
        value_index=torch.zeros(b, t, device=value.device, dtype=torch.long),
        prior=prior,
        mode=mode.long(),
        mask=mask.bool(),
    )


def repeat_tokens(tokens: Tokens, repeats: int) -> Tokens:
    """Repeat one context batch so a whole grid can be queried in parallel."""

    return Tokens(
        var_id=tokens.var_id.repeat(repeats, 1),
        x=tokens.x.repeat(repeats, 1, 1),
        value=tokens.value.repeat(repeats, 1),
        value_index=tokens.value_index.repeat(repeats, 1),
        prior=tokens.prior.repeat(repeats, 1, 1),
        mode=tokens.mode.repeat(repeats, 1),
        mask=tokens.mask.repeat(repeats, 1),
    )


def known_context_token(model: ACE, *, var_id: int, values: torch.Tensor) -> Tokens:
    """Build context tokens for known scalar values.

    Bounded continuous latents are represented as zero-spread PRIOR tokens.
    Data and discrete variables are represented as VALUE tokens.
    """

    b = values.numel()
    var = torch.full((b, 1), var_id, device=values.device)
    prior = torch.zeros(b, 1, PRIOR_FEATURES, device=values.device, dtype=values.dtype)
    bounded = bool(model.is_latent[var_id].item() and model.has_bounds[var_id].item() and not model.is_discrete[var_id].item())
    mode_value = PRIOR if bounded else VALUE
    if bounded:
        prior[:, 0, 0] = values
    tokens = make_scalar_tokens(
        var_id=var,
        value=values[:, None],
        prior=prior,
        mode=torch.full((b, 1), mode_value, device=values.device),
        mask=torch.ones(b, 1, device=values.device, dtype=torch.bool),
        x_dim=model.cfg.x_dim,
    )
    if bool(model.is_discrete[var_id].item()):
        tokens.value_index[:, 0] = values.long()
    return tokens


def query_log_density(model: ACE, batch: Batch, var_id: int, values: torch.Tensor) -> torch.Tensor:
    """Evaluate ACE's token-coordinate 1D marginal log density over a grid."""

    b = values.numel()
    target = make_scalar_tokens(
        var_id=torch.full((b, 1), var_id, device=values.device),
        value=values[:, None],
        prior=torch.zeros(b, 1, PRIOR_FEATURES, device=values.device, dtype=values.dtype),
        mode=torch.full((b, 1), QUERY, device=values.device),
        mask=torch.ones(b, 1, device=values.device, dtype=torch.bool),
        x_dim=model.cfg.x_dim,
    )
    rep = Batch(batch.variables, repeat_tokens(batch.context, b), target)
    return model(rep).log_prob(target).squeeze(1)


def conditional_log_density(
    model: ACE,
    batch: Batch,
    *,
    known_var: int,
    known_values: torch.Tensor,
    query_var: int,
    query_values: torch.Tensor,
) -> torch.Tensor:
    """Grid of token-coordinate log p(query_var | context, known_var=value)."""

    rows = []
    q = query_values.numel()
    for known in known_values:
        known_tok = known_context_token(model, var_id=known_var, values=known.expand(q))
        context = append_or_replace_context_token(
            repeat_tokens(batch.context, q),
            known_tok,
            is_latent=model.is_latent,
            is_discrete=model.is_discrete,
            has_bounds=model.has_bounds,
        )
        target = make_scalar_tokens(
            var_id=torch.full((q, 1), query_var, device=query_values.device),
            value=query_values[:, None],
            prior=torch.zeros(q, 1, PRIOR_FEATURES, device=query_values.device, dtype=query_values.dtype),
            mode=torch.full((q, 1), QUERY, device=query_values.device),
            mask=torch.ones(q, 1, device=query_values.device, dtype=torch.bool),
            x_dim=model.cfg.x_dim,
        )
        rows.append(model(Batch(batch.variables, context, target)).log_prob(target).squeeze(1))
    return torch.stack(rows, dim=0)


def ar_joint_log_density(
    model: ACE,
    batch: Batch,
    first_grid: torch.Tensor,
    second_grid: torch.Tensor,
    *,
    first_var: int,
    second_var: int,
) -> torch.Tensor:
    """Symmetrized token-coordinate two-variable AR joint density on a grid."""

    log_first = query_log_density(model, batch, first_var, first_grid)
    log_second = query_log_density(model, batch, second_var, second_grid)
    log_second_given_first = conditional_log_density(
        model,
        batch,
        known_var=first_var,
        known_values=first_grid,
        query_var=second_var,
        query_values=second_grid,
    )
    log_first_given_second = conditional_log_density(
        model,
        batch,
        known_var=second_var,
        known_values=second_grid,
        query_var=first_var,
        query_values=first_grid,
    ).transpose(0, 1)
    joint_1 = log_first[:, None] + log_second_given_first
    joint_2 = log_second[None, :] + log_first_given_second
    joint = torch.logsumexp(torch.stack([joint_1, joint_2], dim=0), dim=0) - math.log(2.0)
    return joint - torch.logsumexp(joint.reshape(-1), dim=0)


def normalized_moments(grid: torch.Tensor, log_density: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean/std of a 1D density evaluated on an evenly spaced grid."""

    p = (log_density - torch.logsumexp(log_density, dim=0)).exp()
    mean = (p * grid).sum()
    std = (p * (grid - mean).pow(2)).sum().sqrt()
    return mean, std
