"""GP-1D active learning with ALINE: episodes, RL training, and diagnostics.

Trains the `aline.ALINE` model on sequential GP-1D episodes: a candidate pool
of unobserved locations, a goal xi (a per-row mask over the fixed target
superset `[log_lengthscale, log_outputscale, kernel, x*_1..M]`), and T
acquisition steps in which the policy picks the next pool point and the
inference network updates its posterior/predictive estimates. Rewards are the
self-estimated information gain of the model's own predictions (Huang et al.,
2025, Eq. 10); training alternates prediction steps (NLL -> base/phi, policy
frozen, rollouts mixed 50/50 current-policy/random) and policy steps
(REINFORCE -> policy/psi, base frozen) — a deliberate variant of the paper's
Algorithm 1, recorded in the local DEVLOG.

The GP physics is `gp1d.draw_instances` at `pool + M` points (one joint draw,
so pool observations are lookups); the default episode length keeps contexts
inside the warm-started gp1d model's trained `n_context <= 20` range.

Run from the repo root (short smoke run / warm-started fine-tune / reuse):

    .venv/Scripts/python.exe extensions/aline/gp1d_aline.py --steps 20 --batch-size 16 ^
        --eval-episodes 16 --oracle-episodes 1
    .venv/Scripts/python.exe extensions/aline/gp1d_aline.py --base-checkpoint artifacts/gp1d.pt ^
        --steps 10000 --save-checkpoint artifacts/gp1d_aline.pt
    .venv/Scripts/python.exe extensions/aline/gp1d_aline.py --eval-only ^
        --load-checkpoint artifacts/gp1d_aline.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gp1d  # noqa: E402
import train  # noqa: E402
from ace import (  # noqa: E402
    ACEConfig,
    Batch,
    QUERY,
    Tokens,
    Variable,
    encode_value,
    mix_seed,
    sample_reveal_mask,
)
from aline import ALINE, load_aline_checkpoint, load_warm_start  # noqa: E402
from diagnostics import normalized_moments, query_log_density  # noqa: E402


EVAL_SEED = 20260612
N_LATENTS = 3  # log_lengthscale, log_outputscale, kernel — gp1d.variables()[1:4]


# --------------------------------------------------------------------------- #
# Episode environment
# --------------------------------------------------------------------------- #


@dataclass
class Episode:
    """One batch of GP-1D acquisition episodes (tensors mutate as steps land).

    `context` is preallocated at width `1 + n_steps` (column 0 = the random
    seed point); `observe` activates one column per step. `target` is the
    fixed superset `[ell, scale, kernel, x*_1..M]` whose mask IS the goal xi.
    `query` holds the candidate pool; its mask is availability (functionally
    replaced on each step — deferred policy-gradient graphs hold references to
    the old mask, so in-place mutation would corrupt them).
    """

    variables: list[Variable]
    context: Tokens
    target: Tokens
    query: Tokens
    pool_x: torch.Tensor  # [B, N_pool]
    pool_y: torch.Tensor  # [B, N_pool]
    x_star: torch.Tensor  # [B, M]
    y_star: torch.Tensor  # [B, M]
    log_ell: torch.Tensor  # [B]
    log_scale: torch.Tensor  # [B]
    kernel: torch.Tensor  # [B] long
    n_steps: int


def sample_xi(b: int, m_targets: int, device: torch.device | str) -> torch.Tensor:
    """Sample the goal mask `[B, 3+M]`: 50% predictive rows, 50% parameter rows.

    Parameter rows draw a non-empty latent subset from the shared subset/count
    mixture (`sample_reveal_mask` with q=0), so singletons, pairs, and
    all-three are all in-distribution. Either-or per row, as the paper.
    """

    is_pred = torch.rand(b, device=device) < 0.5
    theta = sample_reveal_mask(N_LATENTS, b, q=0.0, device=device)
    xi = torch.zeros(b, N_LATENTS + m_targets, dtype=torch.bool, device=device)
    xi[:, :N_LATENTS] = theta & (~is_pred)[:, None]
    xi[:, N_LATENTS:] = is_pred[:, None]
    return xi


def assemble_episode(
    inst: dict[str, torch.Tensor],
    *,
    variables: list[Variable],
    xi: torch.Tensor,
    seed_idx: torch.Tensor,
    n_pool: int,
    n_steps: int,
    device: torch.device | str,
) -> Episode:
    """Tokenize drawn GP physics into an `Episode` (RNG-free given inputs).

    The first `n_pool` drawn points are the candidate pool, the rest the
    predictive targets x*. The seed point is pool candidate `seed_idx`
    (observed at step 0, removed from the pool).
    """

    device = torch.device(device)
    b = int(inst["x"].shape[0])
    x = inst["x"].float().to(device)
    y = inst["y"].float().to(device)
    pool_x, x_star = x[:, :n_pool], x[:, n_pool:]
    pool_y, y_star = y[:, :n_pool], y[:, n_pool:]
    m = int(x_star.shape[1])
    log_ell = inst["log_ell"].float().to(device)
    log_scale = inst["log_scale"].float().to(device)
    kernel = inst["kernel"].to(device)

    # Context block: data tokens, width 1 + T, seed point active in column 0.
    width = 1 + n_steps
    ctx_x = torch.zeros(b, width, 1, device=device)
    ctx_val = torch.zeros(b, width, device=device)
    ctx_x[:, 0, 0] = pool_x.gather(1, seed_idx[:, None]).squeeze(1)
    ctx_val[:, 0] = pool_y.gather(1, seed_idx[:, None]).squeeze(1)
    ctx_mask = torch.zeros(b, width, dtype=torch.bool, device=device)
    ctx_mask[:, 0] = True
    context = gp1d.make_tokens(
        var_id=torch.zeros(b, width, device=device, dtype=torch.long),
        x=ctx_x,
        value=ctx_val,
        mode=torch.full((b, width), 0, device=device),  # VALUE
        mask=ctx_mask,
    )

    # Target superset: three latent queries + M data queries at x*; mask = xi.
    tgt_t = N_LATENTS + m
    tgt_var = torch.zeros(b, tgt_t, device=device, dtype=torch.long)
    tgt_var[:, 0], tgt_var[:, 1], tgt_var[:, 2] = 1, 2, 3
    tgt_x = torch.zeros(b, tgt_t, 1, device=device)
    tgt_x[:, N_LATENTS:, 0] = x_star
    tgt_val = torch.zeros(b, tgt_t, device=device)
    tgt_val[:, 0] = encode_value(variables[1], log_ell)
    tgt_val[:, 1] = encode_value(variables[2], log_scale)
    tgt_val[:, 2] = kernel.float()
    tgt_val[:, N_LATENTS:] = y_star
    tgt_idx = torch.zeros(b, tgt_t, device=device, dtype=torch.long)
    tgt_idx[:, 2] = kernel
    target = gp1d.make_tokens(
        var_id=tgt_var,
        x=tgt_x,
        value=tgt_val,
        value_index=tgt_idx,
        mode=torch.full((b, tgt_t), QUERY, device=device),
        mask=xi.clone(),
    )

    # Candidate pool: data QUERY tokens ("hypothetical targets"); seed removed.
    q_mask = torch.ones(b, n_pool, dtype=torch.bool, device=device)
    q_mask = q_mask.scatter(1, seed_idx[:, None], False)
    query = gp1d.make_tokens(
        var_id=torch.zeros(b, n_pool, device=device, dtype=torch.long),
        x=pool_x[..., None],
        value=torch.zeros(b, n_pool, device=device),
        mode=torch.full((b, n_pool), QUERY, device=device),
        mask=q_mask,
    )
    return Episode(
        variables=variables,
        context=context,
        target=target,
        query=query,
        pool_x=pool_x,
        pool_y=pool_y,
        x_star=x_star,
        y_star=y_star,
        log_ell=log_ell,
        log_scale=log_scale,
        kernel=kernel,
        n_steps=n_steps,
    )


def make_episode(variables: list[Variable], args: argparse.Namespace, device: torch.device | str) -> Episode:
    """Draw physics + goal + seed point for one online episode batch (global RNG)."""

    b = args.batch_size
    inst = gp1d.draw_instances(b, n_points=args.pool_size + args.pred_targets, jitter=args.jitter)
    xi = sample_xi(b, args.pred_targets, device)
    seed_idx = torch.randint(0, args.pool_size, (b,), device=device)
    return assemble_episode(
        inst,
        variables=variables,
        xi=xi,
        seed_idx=seed_idx,
        n_pool=args.pool_size,
        n_steps=args.episode_steps,
        device=device,
    )


def observe(episode: Episode, action: torch.Tensor, *, col: int, sigma_obs: float) -> None:
    """Write the chosen candidates into context column `col`; remove from pool.

    `query.mask` is replaced functionally, not mutated: the deferred policy
    backward holds graph references to the mask used at earlier steps.
    """

    y_obs = episode.pool_y.gather(1, action[:, None]).squeeze(1)
    if sigma_obs > 0.0:
        y_obs = y_obs + sigma_obs * torch.randn_like(y_obs)
    episode.context.x[:, col, 0] = episode.pool_x.gather(1, action[:, None]).squeeze(1)
    episode.context.value[:, col] = y_obs
    episode.context.mask[:, col] = True
    episode.query.mask = episode.query.mask.scatter(1, action[:, None], False)


def uniform_action(avail: torch.Tensor) -> torch.Tensor:
    """Uniform random choice among available candidates (per row)."""

    return torch.multinomial(avail.float(), 1).squeeze(1)


@torch.no_grad()
def us_action(model: ALINE, episode: Episode) -> torch.Tensor:
    """Uncertainty sampling from the model's own predictive variance (baseline)."""

    pred = model(Batch(episode.variables, episode.context, episode.query))
    var = pred.continuous_var().masked_fill(~episode.query.mask, float("-inf"))
    return var.argmax(dim=1)


# --------------------------------------------------------------------------- #
# Rollout: T+1 forwards, per-step NLL backward, one deferred PG backward
# --------------------------------------------------------------------------- #


def rollout(
    model: ALINE,
    episode: Episode,
    *,
    driver: str = "argmax",
    train_nll: bool = False,
    train_pg: bool = False,
    random_frac: float = 0.0,
    use_baseline: bool = True,
    reward_to_go: bool = False,
    sigma_obs: float = 0.0,
    track_predictions: bool = False,
    on_step: Callable[[int, Episode], None] | None = None,
) -> dict:
    """Run one episode batch; optionally accumulate NLL and PG gradients.

    Forwards run on states D_0..D_T (T+1 total). Action a_{t+1} is selected
    from the forward at D_t; the reward R_t = mean_active(log q_t − log q_{t−1})
    uses consecutive detached log-probs, so within a rollout the parameters are
    frozen and rewards measure data information only (the optimizer steps once
    per episode batch, outside). NLL (Eq. 12, mean over active targets and the
    T post-seed states) backwards per step so trunk graphs are freed; the PG
    loss (Eq. 11, gamma=1, batch-mean baseline) backwards once at the end —
    only the small policy-side graphs are retained, since the trunk states
    enter the policy detached. With `random_frac > 0` the random-driven rows
    are excluded from the PG loss (off-policy).
    """

    if train_pg and driver != "policy":
        raise ValueError("policy-gradient training requires on-policy sampling (driver='policy')")
    if driver not in ("policy", "argmax", "random", "us"):
        raise ValueError(f"unknown driver {driver!r}")

    b = episode.context.shape[0]
    t_steps = episode.n_steps
    device = episode.context.value.device
    batch = Batch(episode.variables, episode.context, episode.target)

    log_q_steps: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    pending_logpi: list[torch.Tensor] = []
    pending_onpolicy: list[torch.Tensor] = []
    logp_steps: list[torch.Tensor] = []
    y_means: list[torch.Tensor] = []
    y_stds: list[torch.Tensor] = []
    nll_total = 0.0

    for t in range(t_steps + 1):
        if on_step is not None:
            on_step(t, episode)
        with torch.enable_grad() if train_nll else torch.no_grad():
            pred, ctx_states, tgt_states = model.forward_with_states(batch)
            logp = pred.log_prob(episode.target)
            active = episode.target.mask.to(logp.dtype)
            log_q = (logp * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)
        if train_nll and t >= 1:
            (-log_q.mean() / t_steps).backward()
        log_q_det = log_q.detach()
        log_q_steps.append(log_q_det)
        if track_predictions:
            logp_steps.append(logp.detach())
            y_means.append(pred.mean(episode.target)[:, N_LATENTS:].detach())
            y_stds.append(pred.continuous_var()[:, N_LATENTS:].clamp_min(0.0).sqrt().detach())
        if t >= 1:
            rewards.append(log_q_det - log_q_steps[t - 1])
        if t < t_steps:
            if driver in ("policy", "argmax"):
                with torch.enable_grad() if train_pg else torch.no_grad():
                    logits = model.policy_logits(
                        episode.query, ctx_states, tgt_states, episode.context.mask, episode.target.mask
                    )
            if driver == "argmax":
                action = logits.argmax(dim=1)
            elif driver == "random":
                action = uniform_action(episode.query.mask)
            elif driver == "us":
                action = us_action(model, episode)
            else:  # "policy"
                action = torch.distributions.Categorical(logits=logits.detach()).sample()
                onpolicy = torch.ones(b, dtype=torch.bool, device=device)
                if random_frac > 0.0:
                    rand_rows = torch.rand(b, device=device) < random_frac
                    action = torch.where(rand_rows, uniform_action(episode.query.mask), action)
                    onpolicy = ~rand_rows
                if train_pg:
                    logpi = torch.log_softmax(logits, dim=1).gather(1, action[:, None]).squeeze(1)
                    pending_logpi.append(logpi)
                    pending_onpolicy.append(onpolicy)
            actions.append(action)
            observe(episode, action, col=1 + t, sigma_obs=sigma_obs)
        if t >= 1:
            nll_total += float(-log_q_det.mean())

    pg_value = None
    reward_mat = torch.stack(rewards, dim=1) if rewards else torch.zeros(b, 0, device=device)
    if train_pg and pending_logpi:
        logpi = torch.stack(pending_logpi, dim=1)  # [B, T]
        onpolicy = torch.stack(pending_onpolicy, dim=1).to(logpi.dtype)
        weights = reward_mat.flip(1).cumsum(dim=1).flip(1) if reward_to_go else reward_mat
        if use_baseline:
            denom = onpolicy.sum(dim=0).clamp_min(1.0)
            baseline = (weights * onpolicy).sum(dim=0) / denom
            weights = weights - baseline[None, :]
        # Normalize by the effective (on-policy) episode count, not B: with
        # off-policy rows zeroed (simultaneous mode + random_frac), a plain
        # batch mean would silently halve the policy gradient's scale.
        eff_rows = (onpolicy.sum() / max(1, t_steps)).clamp_min(1.0)
        pg_loss = -(weights * onpolicy * logpi).sum() / eff_rows
        pg_loss.backward()
        pg_value = float(pg_loss.detach())

    out = {
        "log_q": torch.stack(log_q_steps, dim=1),  # [B, T+1] detached
        "rewards": reward_mat,  # [B, T] raw, pre-baseline
        "actions": torch.stack(actions, dim=1) if actions else torch.zeros(b, 0, dtype=torch.long),
        "nll": nll_total / max(1, t_steps),
        "pg": pg_value,
    }
    if track_predictions:
        out["logp_steps"] = torch.stack(logp_steps, dim=1)  # [B, T+1, 3+M]
        out["y_means"] = torch.stack(y_means, dim=1)  # [B, T+1, M]
        out["y_stds"] = torch.stack(y_stds, dim=1)
    return out


# --------------------------------------------------------------------------- #
# Training: alternating prediction/policy phases, two optimizers
# --------------------------------------------------------------------------- #


def phase_of(step: int, args: argparse.Namespace) -> str:
    """Phase schedule: prediction-only prefix, then alternate (policy first)."""

    if args.freeze_base:
        return "policy"
    if args.update_mode == "simultaneous":
        return "both" if step > args.warmup_pred_steps else "pred"
    if step <= args.warmup_pred_steps:
        return "pred"
    return "policy" if (step - args.warmup_pred_steps) % 2 == 1 else "pred"


def phase_counts(args: argparse.Namespace) -> tuple[int, int]:
    """Expected optimizer-step counts (base, policy) — the cosine T_max per group."""

    base = sum(1 for s in range(1, args.steps + 1) if phase_of(s, args) in ("pred", "both"))
    policy = sum(1 for s in range(1, args.steps + 1) if phase_of(s, args) in ("policy", "both"))
    return base, policy


def _group_scheduler(opt: torch.optim.Optimizer, args: argparse.Namespace, total: int):
    cfg = train.TrainConfig(steps=max(1, total), lr=args.lr, lr_schedule=args.lr_schedule, warmup=args.warmup)
    return train._build_scheduler(opt, cfg)


def save_resumable(
    path: str | Path,
    model: ALINE,
    *,
    seed: int,
    config: dict,
    opt_base: torch.optim.Optimizer,
    sched_base,
    opt_policy: torch.optim.Optimizer,
    sched_policy,
    step: int,
) -> None:
    """House checkpoint payload + the policy optimizer/scheduler on additive keys.

    `train.load_train_state` restores only the base pair (`optimizer` /
    `scheduler`); the policy pair rides `optimizer_policy` / `scheduler_policy`
    and is restored by `fit_episodes` locally. Legacy loaders ignore the extras.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cfg": asdict(model.cfg),
        "seed": seed,
        "state_dict": model.state_dict(),
        "config": dict(config),
        "optimizer": opt_base.state_dict(),
        "step": int(step),
        "optimizer_policy": opt_policy.state_dict(),
    }
    if sched_base is not None:
        payload["scheduler"] = sched_base.state_dict()
    if sched_policy is not None:
        payload["scheduler_policy"] = sched_policy.state_dict()
    torch.save(payload, path)
    print(f"saved resumable checkpoint: {path}")


def fit_episodes(
    model: ALINE,
    args: argparse.Namespace,
    device: torch.device | str,
    *,
    resume_payload: dict | None = None,
) -> ALINE:
    """Episode-batch training loop (the RL analogue of `train.fit`).

    One optimizer step per episode batch (gradients accumulate across the
    rollout's per-step backwards); each step is a pure function of
    `(seed, step)` via the `mix_seed` reseed, policy sampling included.
    """

    base_params = model.base_parameters()
    policy_params = model.policy_parameters()
    opt_base = torch.optim.Adam(base_params, lr=args.lr)
    opt_policy = torch.optim.Adam(policy_params, lr=args.policy_lr)
    n_base, n_policy = phase_counts(args)
    if n_policy == 0 and not args.freeze_base:
        print(
            f"note: no policy steps in this run (--steps {args.steps} <= "
            f"--warmup-pred-steps {args.warmup_pred_steps}); the acquisition policy will not train"
        )
    sched_base = _group_scheduler(opt_base, args, n_base)
    sched_policy = _group_scheduler(opt_policy, args, n_policy)

    start_step = 1
    if resume_payload is not None:
        start_step = train.load_train_state(resume_payload, opt_base, sched_base)
        if "optimizer_policy" in resume_payload:
            opt_policy.load_state_dict(resume_payload["optimizer_policy"])
        if sched_policy is not None and resume_payload.get("scheduler_policy") is not None:
            sched_policy.load_state_dict(resume_payload["scheduler_policy"])
        if start_step > 1:
            prev_total = int(resume_payload.get("config", {}).get("steps", args.steps))
            if prev_total != args.steps:
                print(
                    f"warning: resuming with --steps {args.steps} but the run was started with "
                    f"{prev_total}; the cosine LR curves will not line up. Use the original --steps."
                )

    step_time = 0.0
    timed_steps = 0
    for step in range(start_step, args.steps + 1):
        tic = time.perf_counter()
        torch.manual_seed(mix_seed(args.seed, step))
        phase = phase_of(step, args)
        train_nll = phase in ("pred", "both")
        train_pg = phase in ("policy", "both")
        episode = make_episode(model.variables, args, device)
        if train_nll:
            opt_base.zero_grad(set_to_none=True)
        if train_pg:
            opt_policy.zero_grad(set_to_none=True)
        stats = rollout(
            model,
            episode,
            driver="policy",
            train_nll=train_nll,
            train_pg=train_pg,
            random_frac=args.random_frac if train_nll else 0.0,
            use_baseline=not args.no_baseline,
            reward_to_go=args.reward_to_go,
            sigma_obs=args.sigma_obs,
        )
        if train_nll:
            torch.nn.utils.clip_grad_norm_(base_params, args.grad_clip)
            opt_base.step()
            if sched_base is not None:
                sched_base.step()
        if train_pg:
            torch.nn.utils.clip_grad_norm_(policy_params, args.grad_clip)
            opt_policy.step()
            if sched_policy is not None:
                sched_policy.step()
        step_time += time.perf_counter() - tic
        timed_steps += 1
        if step == start_step or step % args.log_every == 0:
            reward = float(stats["rewards"].mean()) if stats["rewards"].numel() else 0.0
            pg = f"{stats['pg']:+.4f}" if stats["pg"] is not None else "  -  "
            print(
                f"step {step:5d}/{args.steps}  [{phase:6s}]  nll {stats['nll']:8.4f}  "
                f"reward {reward:+.4f}  pg {pg}  ({step_time / timed_steps:.2f}s/step)"
            )
        if args.ckpt_every and args.save_checkpoint and step % args.ckpt_every == 0 and step < args.steps:
            save_resumable(
                args.save_checkpoint,
                model,
                seed=args.seed,
                config=vars(args),
                opt_base=opt_base,
                sched_base=sched_base,
                opt_policy=opt_policy,
                sched_policy=sched_policy,
                step=step,
            )
    if timed_steps:
        print(f"fit_episodes: {timed_steps} steps, avg {step_time / timed_steps:.2f}s/step")
    return model


# --------------------------------------------------------------------------- #
# Evaluation: baselines, targeting contrast, oracle calibration, fixed demo
# --------------------------------------------------------------------------- #


@dataclass
class ALDiagnostic:
    """Held-out evaluation curves + the fixed demo artifacts for plotting."""

    rmse_curves: dict[str, torch.Tensor]  # driver -> [T+1]
    logq_curves: dict[str, torch.Tensor]  # driver -> [T+1]
    contrast: dict[str, float]
    oracle_rows: list[dict]
    reward_stats: dict[str, dict[str, float]]
    demo: dict
    metrics: dict[str, float]


def _xi_fill(b: int, m: int, fill: str, device: torch.device | str) -> torch.Tensor:
    xi = torch.zeros(b, N_LATENTS + m, dtype=torch.bool, device=device)
    if fill == "pred":
        xi[:, N_LATENTS:] = True
    elif fill == "theta":
        xi[:, :N_LATENTS] = True
    elif fill == "ell":
        xi[:, 0] = True
    elif fill == "kernel":
        xi[:, 2] = True
    else:
        raise ValueError(f"unknown xi fill {fill!r}")
    return xi


def eval_episodes(model: ALINE, args: argparse.Namespace, fill: str, seed_offset: int) -> Episode:
    """Deterministic held-out episodes; identical physics for equal seed_offset."""

    device = next(model.parameters()).device
    torch.manual_seed(EVAL_SEED + seed_offset)
    b = args.eval_episodes
    inst = gp1d.draw_instances(b, n_points=args.pool_size + args.pred_targets, jitter=args.jitter)
    xi = _xi_fill(b, args.pred_targets, fill, device)
    seed_idx = torch.randint(0, args.pool_size, (b,), device=device)
    return assemble_episode(
        inst,
        variables=model.variables,
        xi=xi,
        seed_idx=seed_idx,
        n_pool=args.pool_size,
        n_steps=args.episode_steps,
        device=device,
    )


def acquired_gpbatch(episode: Episode, row: int, *, points: int = 64) -> gp1d.GPBatch:
    """Adapt one acquired episode context to `gp1d.GPBatch` for the grid oracle."""

    device = episode.context.value.device
    mask = episode.context.mask[row]
    x_ctx = episode.context.x[row, mask, 0][None, :]
    y_ctx = episode.context.value[row, mask][None, :]
    x_tgt = torch.linspace(-1.0, 1.0, points, device=device)[None, :]
    context = gp1d.make_tokens(
        var_id=torch.zeros_like(x_ctx, dtype=torch.long),
        x=x_ctx[..., None],
        value=y_ctx,
        mode=torch.zeros_like(x_ctx, dtype=torch.long),  # VALUE
        mask=torch.ones_like(x_ctx, dtype=torch.bool),
    )
    target = gp1d.make_tokens(
        var_id=torch.zeros_like(x_tgt, dtype=torch.long),
        x=x_tgt[..., None],
        value=torch.zeros_like(x_tgt),
        mode=torch.full_like(x_tgt, QUERY, dtype=torch.long),
        mask=torch.ones_like(x_tgt, dtype=torch.bool),
    )
    batch = Batch(episode.variables, context, target)
    return gp1d.GPBatch(
        batch,
        x_ctx,
        y_ctx,
        x_tgt,
        torch.zeros_like(x_tgt),
        episode.log_ell[row : row + 1],
        episode.log_scale[row : row + 1],
        episode.kernel[row : row + 1],
    )


def _marginal_row(model: ALINE, episode: Episode, row: int, args: argparse.Namespace) -> dict:
    """Oracle-vs-model hyperparameter marginals on one acquired context."""

    device = episode.context.value.device
    toy = acquired_gpbatch(episode, row)
    oracle = gp1d.gp_oracle(toy, bins=args.oracle_bins, jitter=args.jitter, chunk=args.oracle_chunk)
    ell_grid = torch.linspace(*gp1d.LOG_LENGTHSCALE_RANGE, args.oracle_bins, device=device)
    scale_grid = torch.linspace(*gp1d.LOG_OUTPUTSCALE_RANGE, args.oracle_bins, device=device)
    ell_logp = query_log_density(model, toy.batch, 1, encode_value(episode.variables[1], ell_grid))
    scale_logp = query_log_density(model, toy.batch, 2, encode_value(episode.variables[2], scale_grid))
    kernel_probs = gp1d.kernel_posterior(model, toy.batch).cpu()
    ell_mean, ell_std = normalized_moments(ell_grid, ell_logp)
    scale_mean, scale_std = normalized_moments(scale_grid, scale_logp)
    o_ell_mean, o_ell_std = gp1d._moments_from_probs(oracle.ell_grid, oracle.ell_probs)
    o_scale_mean, o_scale_std = gp1d._moments_from_probs(oracle.scale_grid, oracle.scale_probs)
    kl = (
        oracle.kernel_probs
        * (oracle.kernel_probs.clamp_min(1e-12).log() - kernel_probs.clamp_min(1e-12).log())
    ).sum()
    return {
        "ell_mean": float(ell_mean),
        "ell_std": float(ell_std),
        "oracle_ell_mean": float(o_ell_mean),
        "oracle_ell_std": float(o_ell_std),
        "scale_mean": float(scale_mean),
        "scale_std": float(scale_std),
        "oracle_scale_mean": float(o_scale_mean),
        "oracle_scale_std": float(o_scale_std),
        "kernel_kl": float(kl),
        "true_ell": float(episode.log_ell[row]),
        "true_scale": float(episode.log_scale[row]),
        "ell_grid": ell_grid.cpu(),
        "ell_logp": ell_logp.cpu(),
        "oracle_ell_probs": oracle.ell_probs.clone(),
    }


def make_demo(model: ALINE, args: argparse.Namespace) -> dict:
    """Fixed single-function demo: the gp1d EVAL_* function, three goals."""

    device = next(model.parameters()).device
    n_pool, t_steps = args.pool_size, args.episode_steps
    points = 64
    gen = torch.Generator().manual_seed(EVAL_SEED)
    x_pool = 2.0 * torch.rand(1, n_pool, generator=gen, dtype=torch.float64) - 1.0
    x_lin = torch.linspace(-1.0, 1.0, points, dtype=torch.float64)[None, :]
    x = torch.cat([x_pool, x_lin], dim=1)
    kernel = torch.tensor([gp1d.EVAL_KERNEL], dtype=torch.long)
    log_ell = torch.tensor([gp1d.EVAL_LOG_LENGTHSCALE], dtype=torch.float64)
    log_scale = torch.tensor([gp1d.EVAL_LOG_OUTPUTSCALE], dtype=torch.float64)
    y = gp1d.draw_gp(x, kernel, log_ell, log_scale, jitter=args.jitter, generator=gen)
    inst = {"x": x, "y": y, "log_ell": log_ell, "log_scale": log_scale, "kernel": kernel}
    seed_idx = torch.randint(0, n_pool, (1,), generator=gen).to(device)

    def fresh(fill: str) -> Episode:
        return assemble_episode(
            inst,
            variables=model.variables,
            xi=_xi_fill(1, points, fill, device),
            seed_idx=seed_idx,
            n_pool=n_pool,
            n_steps=t_steps,
            device=device,
        )

    demo: dict = {
        "x_lin": x_lin[0].float(),
        "y_lin": y[0, n_pool:].float(),
        "seed_x": float(x_pool[0, int(seed_idx)]),
        "seed_y": float(y[0, int(seed_idx)]),
        "true_ell": float(log_ell[0]),
    }

    ep = fresh("pred")
    stats = rollout(model, ep, driver="argmax", track_predictions=True)
    demo["pred_actions_x"] = ep.pool_x[0, stats["actions"][0]].cpu()
    demo["band0"] = (stats["y_means"][0, 0].cpu(), stats["y_stds"][0, 0].cpu())
    demo["bandT"] = (stats["y_means"][0, -1].cpu(), stats["y_stds"][0, -1].cpu())

    ep = fresh("ell")
    stats = rollout(model, ep, driver="argmax")
    demo["ell_actions_x"] = ep.pool_x[0, stats["actions"][0]].cpu()
    demo["ell_row"] = _marginal_row(model, ep, 0, args)

    # xi-switch demo: only the query placements are meaningful — the rollout's
    # log_q/rewards straddle two different active-target sets after the flip.
    switch_at = t_steps // 2

    def flip(t: int, episode: Episode) -> None:
        if t == switch_at:
            episode.target.mask[:] = False
            episode.target.mask[:, 0] = True

    ep = fresh("pred")
    stats = rollout(model, ep, driver="argmax", on_step=flip)
    demo["switch_actions_x"] = ep.pool_x[0, stats["actions"][0]].cpu()
    demo["switch_at"] = switch_at
    return demo


@torch.no_grad()
def evaluate(model: ALINE, args: argparse.Namespace) -> ALDiagnostic:
    """Held-out evaluation: baselines, targeting contrast, oracle calibration."""

    rmse_curves = {}
    for name, driver in (("aline", "argmax"), ("random", "random"), ("us", "us")):
        ep = eval_episodes(model, args, "pred", 1)
        stats = rollout(model, ep, driver=driver, track_predictions=True, sigma_obs=args.sigma_obs)
        err = stats["y_means"] - ep.y_star[:, None, :]
        rmse_curves[name] = err.pow(2).mean(dim=(0, 2)).sqrt().cpu()

    logq_curves = {}
    for name, driver in (("aline", "argmax"), ("random", "random")):
        ep = eval_episodes(model, args, "theta", 2)
        stats = rollout(model, ep, driver=driver, sigma_obs=args.sigma_obs)
        logq_curves[name] = stats["log_q"].mean(dim=0).cpu()

    # Targeting contrast: acquire under a matched single-latent goal vs the
    # mismatched predictive goal on identical episodes; score log q(theta_S | D_T).
    # Two instruments: lengthscale (weak for GP-1D — coverage queries pin it
    # anyway) and kernel (roughness identification wants tight local pairs,
    # prediction wants coverage — the goals genuinely conflict).
    acquired: dict[str, Episode] = {}
    for goal in ("ell", "kernel", "pred"):
        ep = eval_episodes(model, args, goal, 3)
        rollout(model, ep, driver="argmax", sigma_obs=args.sigma_obs)
        acquired[goal] = ep

    def score_goal(ep: Episode, col: int) -> float:
        ep.target.mask[:] = False
        ep.target.mask[:, col] = True
        pred = model(Batch(model.variables, ep.context, ep.target))
        return float(pred.log_prob(ep.target)[:, col].mean())

    contrast = {
        "ell_matched": score_goal(acquired["ell"], 0),
        "ell_mismatched": score_goal(acquired["pred"], 0),
        "kernel_matched": score_goal(acquired["kernel"], 2),
        "kernel_mismatched": score_goal(acquired["pred"], 2),
    }
    contrast["ell_delta"] = contrast["ell_matched"] - contrast["ell_mismatched"]
    contrast["kernel_delta"] = contrast["kernel_matched"] - contrast["kernel_mismatched"]

    ep = eval_episodes(model, args, "theta", 4)
    rollout(model, ep, driver="argmax", sigma_obs=args.sigma_obs)
    oracle_rows = [_marginal_row(model, ep, row, args) for row in range(min(args.oracle_episodes, args.eval_episodes))]

    reward_stats = {}
    for fill in ("pred", "theta"):
        ep = eval_episodes(model, args, fill, 5)
        stats = rollout(model, ep, driver="policy", sigma_obs=args.sigma_obs)
        r = stats["rewards"].flatten()
        reward_stats[fill] = {
            "mean": float(r.mean()),
            "std": float(r.std()),
            "min": float(r.min()),
            "max": float(r.max()),
        }

    demo = make_demo(model, args)

    metrics = {
        "rmse_final_aline": float(rmse_curves["aline"][-1]),
        "rmse_final_random": float(rmse_curves["random"][-1]),
        "rmse_final_us": float(rmse_curves["us"][-1]),
        "logq_theta_final_aline": float(logq_curves["aline"][-1]),
        "logq_theta_final_random": float(logq_curves["random"][-1]),
        "contrast_ell_matched": contrast["ell_matched"],
        "contrast_ell_mismatched": contrast["ell_mismatched"],
        "contrast_ell_delta": contrast["ell_delta"],
        "contrast_kernel_matched": contrast["kernel_matched"],
        "contrast_kernel_mismatched": contrast["kernel_mismatched"],
        "contrast_kernel_delta": contrast["kernel_delta"],
        "oracle_kernel_kl_mean": sum(r["kernel_kl"] for r in oracle_rows) / max(1, len(oracle_rows)),
    }

    print("\nALINE GP-1D active-learning diagnostic")
    print(f"predictive RMSE @T  aline {metrics['rmse_final_aline']:.4f}  "
          f"random {metrics['rmse_final_random']:.4f}  us {metrics['rmse_final_us']:.4f}")
    print(f"theta log q @T      aline {metrics['logq_theta_final_aline']:+.4f}  "
          f"random {metrics['logq_theta_final_random']:+.4f}")
    print(f"targeting contrast  log q(ell|D_T):    matched {contrast['ell_matched']:+.4f}  "
          f"mismatched {contrast['ell_mismatched']:+.4f}  delta {contrast['ell_delta']:+.4f}")
    print(f"targeting contrast  log q(kernel|D_T): matched {contrast['kernel_matched']:+.4f}  "
          f"mismatched {contrast['kernel_mismatched']:+.4f}  delta {contrast['kernel_delta']:+.4f}")
    for i, row in enumerate(oracle_rows):
        print(
            f"oracle calib ep{i}    ell mean {row['ell_mean']:+.3f}/{row['oracle_ell_mean']:+.3f} "
            f"std {row['ell_std']:.3f}/{row['oracle_ell_std']:.3f}  "
            f"scale mean {row['scale_mean']:+.3f}/{row['oracle_scale_mean']:+.3f} "
            f"std {row['scale_std']:.3f}/{row['oracle_scale_std']:.3f}  "
            f"kernel KL {row['kernel_kl']:.3f}   (model/oracle)"
        )
    for fill, stats in reward_stats.items():
        print(
            f"rewards [{fill:5s}]     mean {stats['mean']:+.4f}  std {stats['std']:.4f}  "
            f"min {stats['min']:+.4f}  max {stats['max']:+.4f}"
        )
    return ALDiagnostic(rmse_curves, logq_curves, contrast, oracle_rows, reward_stats, demo, metrics)


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #


def plot_diagnostic(diag: ALDiagnostic, path: str | Path) -> None:
    """Save the fixed-demo + held-out-curves diagnostic figure."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    demo = diag.demo
    t_grid = list(range(diag.rmse_curves["aline"].numel()))

    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    x_lin, y_lin = demo["x_lin"], demo["y_lin"]
    mean0, std0 = demo["band0"]
    mean_t, std_t = demo["bandT"]
    ax.plot(x_lin, y_lin, color="0.25", linewidth=1.4, label="true function")
    ax.fill_between(x_lin, mean0 - 2 * std0, mean0 + 2 * std0, color="tab:gray", alpha=0.15, label="band @t=0")
    ax.plot(x_lin, mean_t, color="tab:blue", linewidth=1.4, label="mean @t=T")
    ax.fill_between(x_lin, mean_t - 2 * std_t, mean_t + 2 * std_t, color="tab:blue", alpha=0.2, label="band @t=T")
    ax.scatter([demo["seed_x"]], [demo["seed_y"]], color="black", s=40, zorder=3, label="seed")
    order = torch.arange(1, demo["pred_actions_x"].numel() + 1)
    ax.scatter(
        demo["pred_actions_x"],
        torch.full_like(demo["pred_actions_x"], float(y_lin.min()) - 0.15),
        c=order,
        cmap="Oranges",
        s=26,
        zorder=3,
        label="queries (by step)",
    )
    ax.set_title("demo: predictive goal")
    ax.set_xlabel("x")
    ax.legend(loc="best", fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    for name, key, color in (
        ("goal: predictive", "pred_actions_x", "tab:blue"),
        ("goal: lengthscale", "ell_actions_x", "tab:orange"),
        ("switch pred->ell", "switch_actions_x", "tab:green"),
    ):
        xs = demo[key]
        ax.scatter(range(1, xs.numel() + 1), xs, label=name, color=color, s=22)
    ax.axvline(demo["switch_at"] + 0.5, color="tab:green", linestyle=":", alpha=0.7)
    ax.set_title("query placement by goal (demo function)")
    ax.set_xlabel("step t")
    ax.set_ylabel("chosen x")
    ax.legend(loc="best", fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    row = demo["ell_row"]
    probs = (row["ell_logp"] - torch.logsumexp(row["ell_logp"], dim=0)).exp()
    ax.plot(row["ell_grid"], probs, color="tab:blue", label="ALINE")
    ax.plot(row["ell_grid"], row["oracle_ell_probs"], color="tab:green", label="oracle")
    ax.axvline(demo["true_ell"], color="0.4", alpha=0.6, label="truth")
    ax.set_title("log_lengthscale marginal @T (goal: lengthscale)")
    ax.set_xlabel("log_lengthscale")
    ax.legend(loc="best", fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    for name, color in (("aline", "tab:orange"), ("us", "tab:green"), ("random", "0.5")):
        ax.plot(t_grid, diag.rmse_curves[name], marker="o", markersize=3, label=name, color=color)
    ax.set_title("predictive RMSE vs steps (held-out)")
    ax.set_xlabel("step t")
    ax.set_ylabel("RMSE")
    ax.legend(loc="best", fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    for name, color in (("aline", "tab:orange"), ("random", "0.5")):
        ax.plot(t_grid, diag.logq_curves[name], marker="o", markersize=3, label=name, color=color)
    ax.set_title("log q(theta_true) vs steps (held-out)")
    ax.set_xlabel("step t")
    ax.set_ylabel("log q")
    ax.legend(loc="best", fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    pos = [0.0, 1.0, 2.6, 3.6]
    values = [
        diag.contrast["ell_matched"],
        diag.contrast["ell_mismatched"],
        diag.contrast["kernel_matched"],
        diag.contrast["kernel_mismatched"],
    ]
    ax.bar(pos, values, color=["tab:orange", "0.6", "tab:orange", "0.6"], width=0.8)
    ax.set_xticks(pos, ["ell\nmatched", "ell\nxi=pred", "kernel\nmatched", "kernel\nxi=pred"])
    ax.set_title(
        "targeting contrast: log q(theta_S | D_T)\n"
        f"delta ell {diag.contrast['ell_delta']:+.3f}, kernel {diag.contrast['kernel_delta']:+.3f}"
    )

    fig.suptitle(
        f"ALINE GP-1D AL: RMSE@T aline {diag.metrics['rmse_final_aline']:.3f} "
        f"vs random {diag.metrics['rmse_final_random']:.3f} vs US {diag.metrics['rmse_final_us']:.3f}; "
        f"targeting deltas ell {diag.contrast['ell_delta']:+.3f} / kernel {diag.contrast['kernel_delta']:+.3f}"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved diagnostic plot: {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_aline(args: argparse.Namespace, device: torch.device | str) -> ALINE:
    """Construct a from-scratch ALINE from CLI hyperparameters (gp1d schema)."""

    cfg = ACEConfig(
        x_dim=1,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        mlp_hidden=args.hidden,
        head_hidden=args.hidden,
        mdn_components=args.components,
    )
    return ALINE(gp1d.variables(), cfg, n_policy_blocks=args.policy_blocks).to(device)


def load_checkpoint(path: str | Path, device: torch.device) -> ALINE:
    """Load an ALINE checkpoint using the GP-1D variable schema (2-arg contract)."""

    return load_aline_checkpoint(path, device, gp1d.variables())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        parents=[train.common_parser()],
        description="Train/evaluate the ALINE GP-1D active-learning extension.",
    )
    # Inherited no-ops here: --max-context, --min-context, --data-targets,
    # --latent-context-prob, --latent-weight, --data-weight (episode NLL/rewards
    # are plain means over active targets). --warmup IS honored: per-group
    # linear LR warmup inside each cosine schedule.
    p.set_defaults(
        batch_size=64,
        d_model=128,
        heads=4,
        layers=4,
        hidden=256,
        components=8,
        plot_path="artifacts/gp1d_aline.png",
    )
    p.add_argument("--base-checkpoint", default="", help="warm-start ACE checkpoint (e.g. artifacts/gp1d.pt); empty = from scratch")
    p.add_argument("--episode-steps", type=int, default=16, help="acquisition steps T per episode (keep <= 19 for a gp1d warm start)")
    p.add_argument("--pool-size", type=int, default=128, help="candidate pool size per episode")
    p.add_argument("--pred-targets", type=int, default=32, help="predictive target locations x* per episode")
    p.add_argument("--policy-blocks", type=int, default=2)
    p.add_argument("--policy-lr", type=float, default=None, help="policy Adam LR (default: --lr)")
    p.add_argument("--update-mode", choices=("alternate", "simultaneous"), default="alternate")
    p.add_argument(
        "--warmup-pred-steps",
        type=int,
        default=None,
        help="prediction-only prefix, in training steps = episode batches (default 0 with a warm start, 2000 from scratch)",
    )
    p.add_argument("--random-frac", type=float, default=0.5, help="random-rollout row fraction in prediction steps")
    p.add_argument("--freeze-base", action="store_true", help="train the policy only; q_phi stays the base checkpoint")
    p.add_argument("--no-baseline", action="store_true", help="disable the batch-mean reward baseline")
    p.add_argument("--reward-to-go", action="store_true", help="weight log pi_t by the reward-to-go instead of R_t")
    p.add_argument("--sigma-obs", type=float, default=0.0, help="observation noise added at lookup (0 = noiseless)")
    p.add_argument("--eval-episodes", type=int, default=512)
    p.add_argument("--oracle-episodes", type=int, default=4, help="acquired contexts to score against the grid oracle")
    p.add_argument("--oracle-bins", type=int, default=64)
    p.add_argument("--oracle-chunk", type=int, default=512)
    p.add_argument("--jitter", type=float, default=gp1d.GEN_JITTER)
    return train.apply_config_file(p)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    resume_payload = None
    if args.resume and not args.load_checkpoint:
        resume_payload = torch.load(args.resume, map_location=device, weights_only=False)
    if args.policy_lr is None:
        args.policy_lr = args.lr
    if args.warmup_pred_steps is None:
        # On resume, restore the original run's resolved value: re-deriving the
        # default here would silently change the phase schedule (and each
        # group's cosine T_max) for a resumed from-scratch run.
        prev = (resume_payload or {}).get("config", {}).get("warmup_pred_steps")
        if prev is not None:
            args.warmup_pred_steps = int(prev)
        else:
            args.warmup_pred_steps = 0 if (args.base_checkpoint or args.load_checkpoint) else 2000
    if resume_payload is not None:
        prev_cfg = resume_payload.get("config", {})
        for key in ("warmup_pred_steps", "update_mode", "freeze_base", "random_frac"):
            if key in prev_cfg and getattr(args, key) != prev_cfg[key]:
                print(
                    f"warning: resuming with --{key.replace('_', '-')} {getattr(args, key)!r} but the "
                    f"run was started with {prev_cfg[key]!r}; the phase schedule / LR curves will not line up."
                )
    if args.episode_steps > 19:
        print(
            f"warning: --episode-steps {args.episode_steps} grows contexts past the gp1d warm start's "
            "trained n_context <= 20 range (1 seed + T points)"
        )
    if args.episode_steps >= args.pool_size:
        raise SystemExit("--episode-steps must be < --pool-size (queries sample the pool without replacement)")
    if args.eval_only and not args.load_checkpoint:
        raise SystemExit("--eval-only requires --load-checkpoint")

    if args.load_checkpoint:
        model = load_checkpoint(args.load_checkpoint, device)
    elif resume_payload is not None:
        model = load_checkpoint(args.resume, device)
    elif args.base_checkpoint:
        check = gp1d.fixed_eval_batch(gp1d.variables(), device=device, points=40, jitter=args.jitter)
        model = load_warm_start(
            args.base_checkpoint,
            device,
            gp1d.variables(),
            n_policy_blocks=args.policy_blocks,
            check_batch=check.batch,
        )
    else:
        model = build_aline(args, device)

    if not args.eval_only:
        model = fit_episodes(model, args, device, resume_payload=resume_payload)

    diag = evaluate(model, args)
    if args.save_checkpoint:
        train.save_checkpoint(args.save_checkpoint, model, seed=args.seed, config=vars(args))
    if not args.no_plot and args.plot_path:
        plot_diagnostic(diag, args.plot_path)


if __name__ == "__main__":
    main()
