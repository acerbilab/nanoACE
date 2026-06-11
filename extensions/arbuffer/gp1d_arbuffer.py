"""GP-1D fine-tune + demo for the causal AR-buffer extension.

Takes a trained `gp1d.py` checkpoint, attaches the buffer stream
(`arbuffer.BufferedACE`), freezes the base, and fine-tunes only the buffer on a
three-way context / buffer / target split of the same GP data-generating
process drawn at a larger point budget (`--n-points`, default 128, vs the base's
`N_TOTAL = 64`). Because the base is frozen and the buffer read starts at an
exact zero, the first logged loss *is* the base model's context-only NLL on the
buffered targets — everything the curve drops below that is information the
model learned to extract from the buffer.

The demo draws coherent joint function samples (`arbuffer.sample_joint`: one
cached context encoding shared by all draw streams) under three conditioning
variants of the same fixed GP function, prints a joint-NLL table (diagonal /
slow-AR re-encoding / buffered one-pass, scored on identical orderings), checks
the frozen base is still bit-identical to the source checkpoint, and reports
measured sampling wall-clock against `ace.sample_ar`.

Run from the repo root (short smoke run / full fine-tune / reuse):

    .venv/Scripts/python.exe extensions/arbuffer/gp1d_arbuffer.py --steps 20 --batch-size 16
    .venv/Scripts/python.exe extensions/arbuffer/gp1d_arbuffer.py ^
        --base-checkpoint artifacts/gp1d.pt --save-checkpoint artifacts/gp1d_arbuffer.pt
    .venv/Scripts/python.exe extensions/arbuffer/gp1d_arbuffer.py ^
        --eval-only --load-checkpoint artifacts/gp1d_arbuffer.pt
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gp1d  # noqa: E402
import train  # noqa: E402
from ace import (  # noqa: E402
    ACE,
    ACEConfig,
    Batch,
    PRIOR,
    PRIOR_FEATURES,
    QUERY,
    Tokens,
    VALUE,
    Variable,
    cat_tokens,
    encode_value,
    sample_ar,
    sample_reveal_mask,
)
from arbuffer import (  # noqa: E402
    BufferedACE,
    BufferedBatch,
    joint_log_prob,
    load_buffered_checkpoint,
    load_warm_start,
    sample_joint,
    slow_ar_log_prob,
    _data_tokens,
)
from diagnostics import repeat_tokens  # noqa: E402


def assemble_buffered(
    inst: dict[str, torch.Tensor],
    *,
    variables: list[Variable],
    n_context: torch.Tensor,
    reveal_mask: torch.Tensor,
    max_context: int,
    buffer_size: int,
    all_buffered: bool,
    device: torch.device | str,
) -> BufferedBatch:
    """Tokenize drawn GP instances into a context / buffer / target split.

    The context block is `gp1d.assemble`'s, verbatim: `max_context` data-point
    candidates (the first `n_context` active) plus the three latent slots driven
    by `reveal_mask`. The next `buffer_size` points are the buffer (always fully
    active). Targets are every remaining point — the inactive context candidates
    plus the tail — i.e. the base complement convention with the buffer slice
    carved out; data-only (no latent queries: with a frozen base the latent
    marginal path cannot learn, and latent-in-buffer is out of scope).

    `prefix_len`: with the frozen base every target draws `v ~ U{1..K}` — a
    context-only target's loss is constant there (zero gradient through the
    zero-gated read), so the paper's 50% context-only split would waste half the
    signal. With `all_buffered=False` (joint training) the 50/50 split returns,
    since it is what protects the marginals when the base can move.
    """

    device = torch.device(device)
    b = int(inst["x"].shape[0])
    n_points = int(inst["x"].shape[1])
    k = buffer_size
    if not 1 <= max_context + k < n_points:
        raise ValueError(f"need max_context + buffer_size ({max_context + k}) < n_points ({n_points}) for >=1 target")
    x = inst["x"].float().to(device)
    y = inst["y"].float().to(device)
    log_ell_internal = encode_value(variables[1], inst["log_ell"].float().to(device))
    log_scale_internal = encode_value(variables[2], inst["log_scale"].float().to(device))
    kernel = inst["kernel"].to(device)
    reveal_ell, reveal_scale, reveal_kernel = reveal_mask[:, 0], reveal_mask[:, 1], reveal_mask[:, 2]

    # Context block, as gp1d.assemble: candidates [0, max_context) + 3 latent slots.
    ctx_t = max_context + 3
    ell_pos, scale_pos, kernel_pos = max_context, max_context + 1, max_context + 2
    ctx_var = torch.zeros(b, ctx_t, device=device, dtype=torch.long)
    ctx_var[:, ell_pos] = 1
    ctx_var[:, scale_pos] = 2
    ctx_var[:, kernel_pos] = 3
    ctx_x = torch.zeros(b, ctx_t, 1, device=device)
    ctx_x[:, :max_context, 0] = x[:, :max_context]
    ctx_value = torch.zeros(b, ctx_t, device=device)
    ctx_value[:, :max_context] = y[:, :max_context]
    ctx_value[:, ell_pos] = log_ell_internal
    ctx_value[:, scale_pos] = log_scale_internal
    ctx_value[:, kernel_pos] = kernel.float()
    ctx_index = torch.zeros(b, ctx_t, device=device, dtype=torch.long)
    ctx_index[:, kernel_pos] = kernel
    ctx_prior = torch.zeros(b, ctx_t, PRIOR_FEATURES, device=device)
    ctx_prior[:, ell_pos, 0] = log_ell_internal
    ctx_prior[:, scale_pos, 0] = log_scale_internal
    ctx_mode = torch.full((b, ctx_t), VALUE, device=device)
    ctx_mode[:, ell_pos] = PRIOR
    ctx_mode[:, scale_pos] = PRIOR
    ctx_mask = torch.zeros(b, ctx_t, device=device, dtype=torch.bool)
    ctx_mask[:, :max_context] = torch.arange(max_context, device=device)[None, :] < n_context[:, None]
    ctx_mask[:, ell_pos] = reveal_ell
    ctx_mask[:, scale_pos] = reveal_scale
    ctx_mask[:, kernel_pos] = reveal_kernel
    context = gp1d.make_tokens(
        var_id=ctx_var, x=ctx_x, value=ctx_value, value_index=ctx_index,
        mode=ctx_mode, mask=ctx_mask, prior=ctx_prior,
    )

    # Buffer: points [max_context, max_context + K), fully active VALUE tokens.
    buffer = _data_tokens(0, x[:, max_context : max_context + k], y[:, max_context : max_context + k], VALUE)

    # Targets: inactive candidates + tail; truth in value, mode QUERY, data-only.
    tgt_x = torch.cat([x[:, :max_context], x[:, max_context + k :]], dim=1)
    tgt_y = torch.cat([y[:, :max_context], y[:, max_context + k :]], dim=1)
    target = _data_tokens(0, tgt_x, tgt_y, QUERY)
    tgt_t = tgt_x.shape[1]
    tgt_mask = torch.ones(b, tgt_t, device=device, dtype=torch.bool)
    tgt_mask[:, :max_context] = torch.arange(max_context, device=device)[None, :] >= n_context[:, None]
    target = Tokens(
        var_id=target.var_id, x=target.x, value=target.value,
        value_index=target.value_index, prior=target.prior, mode=target.mode, mask=tgt_mask,
    )

    if all_buffered:
        prefix_len = torch.randint(1, k + 1, (b, tgt_t), device=device)
    else:
        v = torch.randint(1, k + 1, (b, tgt_t), device=device)
        context_only = torch.rand(b, tgt_t, device=device) < 0.5
        prefix_len = torch.where(context_only, torch.zeros_like(v), v)
    return BufferedBatch(variables, context, buffer, target, prefix_len)


def buffered_online_batch(model: BufferedACE, args: argparse.Namespace, device: torch.device | str) -> BufferedBatch:
    """Draw + assemble one online buffered fine-tune batch (global RNG)."""

    inst = gp1d.draw_instances(args.batch_size, n_points=args.n_points, jitter=args.jitter)
    n_context = torch.randint(args.min_context, args.max_context + 1, (args.batch_size,), device=device)
    reveal_mask = sample_reveal_mask(3, args.batch_size, q=1.0 - args.latent_context_prob, device=device)
    return assemble_buffered(
        inst,
        variables=model.variables,
        n_context=n_context,
        reveal_mask=reveal_mask,
        max_context=args.max_context,
        buffer_size=args.buffer_size,
        all_buffered=not args.no_freeze_base,
        device=device,
    )


def load_checkpoint(path: str | Path, device: torch.device | str) -> BufferedACE:
    """Playground-exporter contract: the 2-arg loader every example exposes
    (see `playground/export_weights.py`), forwarding to the strict buffered load."""

    return load_buffered_checkpoint(path, device, gp1d.variables())


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #

DEMO_CONTEXT_SUBSET = [0, 5, 9, 13]  # 4 spread-out points of the 14 fixed ones
# The demo function is gp1d's fixed diagnostic case with a 1.5x longer
# lengthscale: the base case (0.28) renders choppy on the sampling grid.
DEMO_LOG_LENGTHSCALE = gp1d.EVAL_LOG_LENGTHSCALE + math.log(1.5)


def demo_function(
    *, device: torch.device | str, points: int, jitter: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw the demo figure's fixed GP function: `(x_ctx, y_ctx, x_true, y_true)`.

    Mirrors `gp1d.fixed_eval_batch`'s draw (same context locations, kernel,
    outputscale, and seed) with `DEMO_LOG_LENGTHSCALE`. The canonical gp1d case
    stays in use for the warm-start check, base parity, and `gp1d.evaluate`.
    """

    gen = torch.Generator(device="cpu").manual_seed(gp1d.EVAL_SEED)
    x_ctx = torch.tensor(
        [[-0.94, -0.89, -0.83, -0.62, -0.56, -0.34, -0.30, -0.05, 0.00, 0.22, 0.26, 0.53, 0.58, 0.88]],
        dtype=torch.float64,
    )
    x_true = torch.linspace(-1.0, 1.0, points, dtype=torch.float64)[None, :]
    x_all = torch.cat([x_ctx, x_true], dim=1)
    kernel = torch.tensor([gp1d.EVAL_KERNEL], dtype=torch.long)
    log_ell = torch.tensor([DEMO_LOG_LENGTHSCALE], dtype=torch.float64)
    log_scale = torch.tensor([gp1d.EVAL_LOG_OUTPUTSCALE], dtype=torch.float64)
    y_all = gp1d.draw_gp(x_all, kernel, log_ell, log_scale, jitter=jitter, generator=gen)
    device = torch.device(device)
    n = x_ctx.shape[1]
    return (
        x_ctx.float().to(device),
        y_all[:, :n].float().to(device),
        x_true.float().to(device),
        y_all[:, n:].float().to(device),
    )


def _demo_contexts(x_ctx: torch.Tensor, y_ctx: torch.Tensor, device: torch.device) -> list[tuple[str, Tokens]]:
    """The three conditioning variants of the demo GP function."""
    subset = torch.tensor(DEMO_CONTEXT_SUBSET, device=device)
    ctx4 = _data_tokens(0, x_ctx[:, subset], y_ctx[:, subset], VALUE)
    ctx14 = _data_tokens(0, x_ctx, y_ctx, VALUE)
    kernel_tok = gp1d.make_tokens(
        var_id=torch.full((1, 1), 3, device=device, dtype=torch.long),
        value=torch.full((1, 1), float(gp1d.EVAL_KERNEL), device=device),
        value_index=torch.full((1, 1), gp1d.EVAL_KERNEL, device=device, dtype=torch.long),
        mode=torch.full((1, 1), VALUE, device=device),
        mask=torch.ones(1, 1, device=device, dtype=torch.bool),
    )
    ctx4k = cat_tokens([ctx4, kernel_tok])
    kernel_name = gp1d.KERNELS[gp1d.EVAL_KERNEL]
    return [
        ("4 context points", ctx4),
        ("14 context points", ctx14),
        (f"4 points + kernel pinned ({kernel_name})", ctx4k),
    ]


@torch.no_grad()
def nll_table(model: BufferedACE, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    """Joint NLL of held-out functions under the three readings, same orders."""

    n_eval = args.eval_functions
    k = args.sample_points
    n_ctx = 4
    inst = gp1d.draw_instances(n_eval, n_points=n_ctx + k, jitter=args.jitter)
    x = inst["x"].float().to(device)
    y = inst["y"].float().to(device)
    context = _data_tokens(0, x[:, :n_ctx], y[:, :n_ctx], VALUE)
    x_t, y_t = x[:, n_ctx:], y[:, n_ctx:]
    orders = [torch.randperm(k, device=device).tolist() for _ in range(max(1, args.orders))]

    target = _data_tokens(0, x_t, y_t, QUERY)
    diagonal = model(Batch(model.variables, context, target)).log_prob(target).sum(dim=1)
    buffered = joint_log_prob(model, context, x_t, y_t, orders=orders)
    slow = slow_ar_log_prob(model, context, x_t, y_t, orders=orders)
    per = float(k)
    table = {
        "diagonal": float(diagonal.mean() / per),
        "slow_ar": float(slow.mean() / per),
        "buffered": float(buffered.mean() / per),
    }
    print(f"\njoint log-density per point, higher is better ({n_eval} held-out functions, {n_ctx} ctx points, K={k}, {len(orders)} orders)")
    print(f"diagonal (independent)    {table['diagonal']: .4f}")
    print(f"slow-AR (re-encoding)     {table['slow_ar']: .4f}")
    print(f"buffered (one-pass)       {table['buffered']: .4f}")
    return table


@torch.no_grad()
def timing(model: BufferedACE, context: Tokens, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    """Measured wall-clock: `sample_ar` vs `sample_joint` for B x K coherent draws."""

    grid = torch.linspace(-1.0, 1.0, args.sample_points, device=device)
    b = args.draws

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    sample_joint(model, context, grid, n_draws=b)
    sync()
    t_buffered = time.perf_counter() - t0

    ctx_rep = repeat_tokens(context, b)
    queries = _data_tokens(0, grid[None, :].expand(b, -1), torch.zeros(b, args.sample_points, device=device), QUERY)
    t0 = time.perf_counter()
    sample_ar(model, Batch(model.variables, ctx_rep, queries))
    sync()
    t_slow = time.perf_counter() - t0

    print(f"\nsampling wall-clock, {b} coherent draws x {args.sample_points} points ({device.type})")
    print(f"sample_ar (re-encoding)   {t_slow: .2f} s")
    print(f"sample_joint (buffer)     {t_buffered: .2f} s   ({t_slow / max(t_buffered, 1e-9):.1f}x)")
    return {"slow_s": t_slow, "buffered_s": t_buffered}


@torch.no_grad()
def base_parity(model: BufferedACE, args: argparse.Namespace, toy: gp1d.GPBatch, device: torch.device) -> None:
    """Confirm the frozen base still bit-matches the source checkpoint."""

    path = Path(args.base_checkpoint)
    if not path.exists():
        print(f"base parity: skipped ({path} not found)")
        return
    payload = torch.load(path, map_location=device, weights_only=False)
    base = ACE(model.variables, ACEConfig(**payload["cfg"])).to(device)
    base.load_state_dict(payload["state_dict"])
    pred_base = base(toy.batch)
    pred_model = model(toy.batch)
    same = torch.equal(pred_model.cont_raw, pred_base.cont_raw) and torch.equal(
        pred_model.disc_logits, pred_base.disc_logits
    )
    if same:
        print("base parity: OK -- empty-buffer predictions bit-equal to the base checkpoint")
    elif args.no_freeze_base:
        drift = (pred_model.cont_raw - pred_base.cont_raw).abs().max()
        print(f"base parity: drifted (expected with --no-freeze-base); max |diff| {float(drift):.3e}")
    else:
        raise RuntimeError("base parity FAILED: frozen base no longer matches the base checkpoint")


def plot_demo(
    model: BufferedACE,
    demo: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    path: str | Path,
) -> None:
    """Three conditioning columns x coherent-draw spaghetti over the demo function."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_ctx, y_ctx, x_true_t, y_true_t = demo
    grid = torch.linspace(-1.0, 1.0, args.sample_points, device=device)
    columns = _demo_contexts(x_ctx, y_ctx, device)

    fig, axes = plt.subplots(1, len(columns), figsize=(15, 4.4), sharey=True, constrained_layout=True)
    x_true = x_true_t[0].cpu()
    y_true = y_true_t[0].cpu()
    colors = plt.get_cmap("tab10").colors
    for ax, (title, context) in zip(axes, columns):
        draws, _ = sample_joint(model, context, grid, n_draws=args.plot_draws)
        queries = _data_tokens(0, grid[None, :], torch.zeros(1, args.sample_points, device=device), QUERY)
        pred = model(Batch(model.variables, context, queries))
        mean = pred.mean(queries)[0].cpu()
        std = pred.continuous_var()[0].clamp_min(1e-8).sqrt().cpu()
        g = grid.cpu()
        ax.fill_between(g, mean - 2 * std, mean + 2 * std, color="tab:blue", alpha=0.12, label="diagonal +/-2 std")
        # A handful of draws in distinct colors so individual curves stay followable.
        for i in range(draws.shape[0]):
            ax.plot(g, draws[i].cpu(), color=colors[i % len(colors)], linewidth=1.0, alpha=0.75)
        ax.plot([], [], color="0.4", linewidth=1.0, label=f"{args.plot_draws} coherent draws")
        ax.plot(x_true, y_true, color="0.25", linewidth=1.6, label="true function")
        n_data = int((context.var_id[0] == 0).sum())
        ax.scatter(
            context.x[0, :n_data, 0].cpu(), context.value[0, :n_data].cpu(),
            color="black", s=30, zorder=5, label="context",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("AR-buffer joint function samples (one cached context per column)")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved demo plot: {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        parents=[train.common_parser()],
        description="Fine-tune a causal AR buffer onto a trained GP-1D checkpoint and demo coherent joint sampling.",
    )
    # Model-shape flags from common_parser are ignored: the architecture comes
    # from the base checkpoint's cfg. --data-targets is likewise a no-op here.
    p.set_defaults(
        batch_size=64,
        max_context=20,
        min_context=1,
        steps=20000,
        plot_path="artifacts/gp1d_arbuffer.png",
    )
    p.add_argument("--base-checkpoint", default="artifacts/gp1d.pt", help="trained gp1d.py checkpoint to warm-start from")
    p.add_argument("--buffer-size", type=int, default=64, help="buffer length K (training max = demo chain length)")
    p.add_argument("--n-points", type=int, default=128, help="GP points drawn per fine-tune instance")
    p.add_argument("--no-freeze-base", action="store_true", help="joint training (paper 50/50 curriculum) instead of frozen base")
    p.add_argument("--draws", type=int, default=64, help="coherent draw streams in the timing measurement")
    p.add_argument("--plot-draws", type=int, default=8, help="coherent draws shown in the figure (distinct colors)")
    p.add_argument("--sample-points", type=int, default=64, help="demo sampling grid size (= eval chain length)")
    p.add_argument("--orders", type=int, default=4, help="orderings averaged in joint-NLL evaluations")
    p.add_argument("--eval-functions", type=int, default=16, help="held-out functions in the joint-NLL table")
    p.add_argument("--eval-points", type=int, default=160)
    p.add_argument("--oracle-bins", type=int, default=64)
    p.add_argument("--oracle-chunk", type=int, default=512)
    p.add_argument("--jitter", type=float, default=gp1d.GEN_JITTER)
    return train.apply_config_file(p)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if args.eval_only and not args.load_checkpoint:
        raise SystemExit("--eval-only requires --load-checkpoint")
    if args.buffer_size < 1:
        raise SystemExit("--buffer-size must be >= 1")
    if not 1 <= args.max_context + args.buffer_size < args.n_points:
        raise SystemExit("need max_context + buffer_size < n_points (at least one target)")

    variables = gp1d.variables()
    toy = gp1d.fixed_eval_batch(variables, device=device, points=args.eval_points, jitter=args.jitter)

    if args.load_checkpoint:
        model = load_buffered_checkpoint(args.load_checkpoint, device, variables)
    elif args.resume:
        model = load_buffered_checkpoint(args.resume, device, variables)
    else:
        model = load_warm_start(args.base_checkpoint, device, variables, check_batch=toy.batch)

    if not args.no_freeze_base:
        model.freeze_base()
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"frozen base: training {n_train:,} of {n_total:,} parameters (buffer stream only)")

    if not args.eval_only:
        resume_state = (
            torch.load(args.resume, map_location=device, weights_only=False) if args.resume else None
        )
        if resume_state is None:
            print(
                "note: with the zero-init gate the first logged loss IS the base model's "
                "context-only NLL on the buffered targets; everything below it is "
                "information extracted from the buffer."
            )
        sampler = lambda step: buffered_online_batch(model, args, device)  # noqa: E731
        model = train.fit(
            model,
            sampler,
            train.TrainConfig.from_args(args),
            resume_state=resume_state,
            seed=args.seed,
            checkpoint_path=args.save_checkpoint or None,
            ckpt_every=args.ckpt_every,
        )

    # Demo + diagnostics: frozen-base parity, base oracle diagnostic (unchanged
    # plain-forward path), joint-NLL table, timing, and the spaghetti plot.
    torch.manual_seed(args.seed)
    base_parity(model, args, toy, device)
    gp1d.evaluate(model, args)
    nll_table(model, args, device)
    demo = demo_function(device=device, points=args.eval_points, jitter=args.jitter)
    timing(model, _demo_contexts(demo[0], demo[1], device)[0][1], args, device)
    if args.save_checkpoint:
        train.save_checkpoint(args.save_checkpoint, model, seed=args.seed, config=vars(args))
    if not args.no_plot and args.plot_path:
        plot_demo(model, demo, args, device, args.plot_path)


if __name__ == "__main__":
    main()
