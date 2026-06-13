"""Paired large-N bootstrap comparison of two ALINE checkpoints.

Built for the n=2 credit fine-tune post-mortem (see the aline DEVLOG entry
"n=2 credit fine-tune (10k)"): it replays `gp1d_aline.evaluate`'s held-out
calls but retains per-episode values, and pools N_EPI episodes over CHUNK-sized
rollouts (so 16k fits the 4060). Every chunk is keyed to a deterministic seed,
so the two checkpoints are scored on the IDENTICAL episode set -> a valid
PAIRED comparison. Episodes are i.i.d., so a single large-N bootstrap is the
correct uncertainty estimate (no multi-seed needed; that would just be the same
i.i.d. draws partitioned). Eval-only; reaches into the extension, changes no
core file.

Run from the repo root:
    .venv/Scripts/python.exe extensions/aline/scripts/analyze_n2.py
Edit the two checkpoint paths in main() to compare a different pair.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# scripts/ -> aline/ -> extensions/ -> repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: ace, gp1d, train
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # aline/: gp1d_aline, aline
import gp1d  # noqa: E402
import gp1d_aline as G  # noqa: E402
from ace import Batch  # noqa: E402
from aline import load_aline_checkpoint  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPI = 16384
CHUNK = 2048
BOOT = 10000
STRIDE = 100  # per-chunk seed stride; > the base offsets 1/2/3 so no overlap


def make_args() -> SimpleNamespace:
    return SimpleNamespace(
        eval_episodes=CHUNK,
        pool_size=128,
        pred_targets=32,
        episode_steps=16,
        sigma_obs=0.0,
        jitter=gp1d.GEN_JITTER,
        device=str(DEVICE),
    )


@torch.no_grad()
def per_episode(model, args) -> dict[str, np.ndarray]:
    acc: dict[str, list] = defaultdict(list)
    n_chunks = N_EPI // CHUNK
    for c in range(n_chunks):
        base = c * STRIDE
        for name, driver in (("aline", "argmax"), ("random", "random"), ("us", "us")):
            ep = G.eval_episodes(model, args, "pred", base + 1)
            stats = G.rollout(model, ep, driver=driver, track_predictions=True, sigma_obs=args.sigma_obs)
            err = stats["y_means"][:, -1, :] - ep.y_star  # [B, M]
            acc[f"mse_{name}"].append(err.pow(2).mean(dim=1).cpu().numpy())
        for name, driver in (("aline", "argmax"), ("random", "random")):
            ep = G.eval_episodes(model, args, "theta", base + 2)
            stats = G.rollout(model, ep, driver=driver, sigma_obs=args.sigma_obs)
            acc[f"logq_{name}"].append(stats["log_q"][:, -1].cpu().numpy())
        acq = {}
        for goal in ("ell", "kernel", "pred"):
            ep = G.eval_episodes(model, args, goal, base + 3)
            G.rollout(model, ep, driver="argmax", sigma_obs=args.sigma_obs)
            acq[goal] = ep

        def score(ep, col):
            ep.target.mask[:] = False
            ep.target.mask[:, col] = True
            pred = model(Batch(model.variables, ep.context, ep.target))
            return pred.log_prob(ep.target)[:, col].cpu().numpy()

        acc["ell_matched"].append(score(acq["ell"], 0))
        acc["ell_mismatched"].append(score(acq["pred"], 0))
        acc["kernel_matched"].append(score(acq["kernel"], 2))
        acc["kernel_mismatched"].append(score(acq["pred"], 2))
        print(f"  chunk {c + 1}/{n_chunks} done", flush=True)
    return {k: np.concatenate(v) for k, v in acc.items()}


def load(path: str) -> dict[str, np.ndarray]:
    print(f"evaluating {path} ...", flush=True)
    model = load_aline_checkpoint(path, DEVICE, gp1d.variables()).eval()
    m = per_episode(model, make_args())
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return m


def rmse(mse: np.ndarray) -> float:
    return float(np.sqrt(mse.mean()))


def ci(s: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main() -> None:
    A = load("artifacts/gp1d_aline_35k.pt")
    B = load("artifacts/gp1d_aline_n2.pt")
    N = len(A["mse_aline"])
    print(f"\npooled N = {N} episodes,  {BOOT} bootstrap resamples")
    rng = np.random.default_rng(0)
    boots = rng.integers(0, N, size=(BOOT, N), dtype=np.int32)

    def boot_rmse(mse):
        return np.sqrt(mse[boots].mean(axis=1))

    def boot_mean(v):
        return v[boots].mean(axis=1)

    def agg(label, m):
        ed = (m["ell_matched"] - m["ell_mismatched"]).mean()
        kd = (m["kernel_matched"] - m["kernel_mismatched"]).mean()
        print(f"\n[{label}] aggregate")
        print(f"  RMSE@T  aline {rmse(m['mse_aline']):.4f}  random {rmse(m['mse_random']):.4f}  us {rmse(m['mse_us']):.4f}")
        print(f"  logq@T  aline {m['logq_aline'].mean():+.4f}  random {m['logq_random'].mean():+.4f}")
        print(f"  ell delta {ed:+.4f}   kernel delta {kd:+.4f}")

    agg("35k", A)
    agg("n2", B)

    print("\n=== within-checkpoint (mean, 95% CI) ===")
    def within(label, m):
        print(f"\n[{label}]")
        for cn, mm, mis in (("ell contrast", m["ell_matched"], m["ell_mismatched"]),
                            ("kernel contrast", m["kernel_matched"], m["kernel_mismatched"])):
            d = mm - mis
            lo, hi = ci(boot_mean(d))
            tag = "" if lo <= 0 <= hi else "  *excludes 0*"
            print(f"  {cn:16s} {d.mean():+.4f}  95%CI [{lo:+.4f},{hi:+.4f}]{tag}")
        d = m["logq_aline"] - m["logq_random"]
        lo, hi = ci(boot_mean(d))
        tag = "" if lo <= 0 <= hi else "  *excludes 0*"
        print(f"  {'logq vs random':16s} {d.mean():+.4f}  95%CI [{lo:+.4f},{hi:+.4f}]{tag}")
        gap = boot_rmse(m["mse_aline"]) - boot_rmse(m["mse_us"])
        lo, hi = ci(gap)
        tag = "" if lo <= 0 <= hi else "  *excludes 0*"
        print(f"  {'RMSE gap to US':16s} {rmse(m['mse_aline'])-rmse(m['mse_us']):+.4f}  95%CI [{lo:+.4f},{hi:+.4f}]{tag}")

    within("35k", A)
    within("n2", B)

    print("\n=== PAIRED  n2 - 35k  (95% CI; excludes 0 => significant) ===")
    for cn, km, kmm in (("ell contrast", "ell_matched", "ell_mismatched"),
                        ("kernel contrast", "kernel_matched", "kernel_mismatched")):
        v = (B[km] - B[kmm]) - (A[km] - A[kmm])
        lo, hi = ci(boot_mean(v))
        tag = "  *SIGNIFICANT*" if not (lo <= 0 <= hi) else "  (n.s.)"
        print(f"  delta {cn:16s} {v.mean():+.4f}  95%CI [{lo:+.4f},{hi:+.4f}]{tag}")
    v = (B["logq_aline"] - B["logq_random"]) - (A["logq_aline"] - A["logq_random"])
    lo, hi = ci(boot_mean(v))
    tag = "  *SIGNIFICANT*" if not (lo <= 0 <= hi) else "  (n.s.)"
    print(f"  delta {'logq margin':16s} {v.mean():+.4f}  95%CI [{lo:+.4f},{hi:+.4f}]{tag}")
    rBa, rAa = boot_rmse(B["mse_aline"]), boot_rmse(A["mse_aline"])
    lo, hi = ci(rBa - rAa)
    tag = "  *SIGNIFICANT*" if not (lo <= 0 <= hi) else "  (n.s.)"
    print(f"  delta {'RMSE@T aline':16s} {rmse(B['mse_aline'])-rmse(A['mse_aline']):+.4f}  95%CI [{lo:+.4f},{hi:+.4f}]{tag}  (+ = n2 worse)")
    rBu, rAu = boot_rmse(B["mse_us"]), boot_rmse(A["mse_us"])
    lo, hi = ci((rBa - rBu) - (rAa - rAu))
    tag = "  *SIGNIFICANT*" if not (lo <= 0 <= hi) else "  (n.s.)"
    g = (rmse(B["mse_aline"]) - rmse(B["mse_us"])) - (rmse(A["mse_aline"]) - rmse(A["mse_us"]))
    print(f"  delta {'RMSE gap-to-US':16s} {g:+.4f}  95%CI [{lo:+.4f},{hi:+.4f}]{tag}  (+ = n2 worse)")


if __name__ == "__main__":
    main()
