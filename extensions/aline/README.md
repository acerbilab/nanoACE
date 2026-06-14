# aline — joint amortized inference + active data acquisition for nanoACE

An extension that implements ALINE — Huang, Wen, Bharti, Kaski &
Acerbi (2025), *ALINE: Joint Amortization for Bayesian Inference and Active
Data Acquisition* (NeurIPS 2025;
[project page](https://www.huangdaolang.com/aline/)) — on top of an
**unchanged** nanoACE model, on the paper's GP active-learning task (§4.1).

ALINE couples two things in one network: an amortized inference model `q_φ`
(posteriors over latents, predictive distributions over data) and an acquisition
policy `π_ψ` that picks which point to query next so that `q_φ` learns fastest
**about a runtime-selectable goal** ξ — a parameter subset, or predictive
accuracy over a target region. The policy is trained with REINFORCE, the reward
being the model's *own* per-step improvement in the log-probability of the
current goal (self-estimated information gain).

The re-expression here is deliberately ACE-native: **the paper's
target-specifier apparatus collapses into token composition.** Parameter targets
are latent QUERY tokens, predictive targets are data QUERY tokens, and ξ is just
*which target tokens are active* — so switching goals mid-experiment is a mask
flip. The inference network is the **unchanged** core `ACE`; the only new model
surface is a small read-only policy decoder. That thesis — and why it is worth
reading the code for — is developed in the local [DEVLOG.md](DEVLOG.md).

## Reference

This extension is based on:

```bibtex
@inproceedings{huang2025aline,
  title={ALINE: Joint Amortization for Bayesian Inference and Active Data Acquisition},
  author={Daolang Huang and Xinyi Wen and Ayush Bharti and Samuel Kaski and Luigi Acerbi},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025)},
  year={2025},
}
```

Local paper markdown is in [paper/](paper/).

## Run

From the repo root, with a trained GP-1D checkpoint (`python gp1d.py
--save-checkpoint artifacts/gp1d.pt`):

```powershell
# warm-started joint fine-tune from a trained GP-1D checkpoint
# (the served model is 35k steps, ~8-9 h on an RTX 4060; use fewer --steps for a quick look)
.\.venv\Scripts\python.exe extensions\aline\gp1d_aline.py `
    --base-checkpoint artifacts\gp1d.pt --steps 35000 `
    --save-checkpoint artifacts\gp1d_aline.pt --ckpt-every 1000

# short smoke run (from scratch, CPU-friendly)
.\.venv\Scripts\python.exe extensions\aline\gp1d_aline.py `
    --steps 20 --batch-size 16 --eval-episodes 16 --oracle-episodes 1

# reuse a fine-tuned checkpoint (diagnostics + figure only)
.\.venv\Scripts\python.exe extensions\aline\gp1d_aline.py `
    --eval-only --load-checkpoint artifacts\gp1d_aline.pt
```

Common artifacts: `artifacts/gp1d_aline.pt`, `artifacts/gp1d_aline.png`.

Episodes use the gp1d physics at a larger point budget: a 128-point candidate
pool plus 32 predictive-target locations per episode (one joint CPU float64 draw;
pool observations are lookups, noiseless by default), one random seed point, and
`T = 16` acquisition steps — context stays ≤ 17 points, inside the warm start's
trained `n_context ≤ 20` range (keep `--episode-steps ≤ 19`). Goals ξ are 50%
predictive / 50% a non-empty parameter subset of {lengthscale, outputscale,
kernel}; the discrete kernel as an acquisition target is a small genuine
extension over the paper's continuous-only GP experiment.

## What the diagnostic shows

`gp1d_aline.py` ends with a held-out evaluation and a six-panel figure:

- **Predictive-goal demo** on the fixed gp1d evaluation function: the predictive
  band before/after the episode with the query order marked.
- **Query placement by goal** on the same function: predictive vs lengthscale
  goals, plus a mid-episode ξ switch (predictive → lengthscale at T/2) — the
  runtime-flexibility demo.
- **Lengthscale marginal vs the grid oracle** at T under the matched goal.
- **Held-out curves**: predictive RMSE vs steps (ALINE / uncertainty sampling /
  random) and parameter log q(θ_true) vs steps (ALINE / random).
- **Targeting contrasts**: log q(ℓ_true | D_T) and log q(kernel_true | D_T) after
  acquiring under the matched vs a mismatched goal on identical episodes.

**What to expect.** The served model (a 35k-step joint fine-tune) learns a
working policy: it beats a random-query baseline on parameter inference
(θ log q), keeps `q_φ` oracle-calibrated, and targets by goal — acquiring under a
parameter goal measurably improves that parameter's posterior, most clearly for
the discrete **kernel** goal. The honest trade-off: classical **uncertainty
sampling still edges ALINE on pure predictive RMSE** by a small margin. Effects
on this task are small (≈0.0x in log q / RMSE) and need thousands of held-out
episodes to measure reliably, so the diagnostic pools a large episode set by
default. Full numbers, error bars, and the credit-assignment (`--credit-n`)
experiments are in [DEVLOG.md](DEVLOG.md).

## Design & deviations

The full design rationale lives in the local [DEVLOG.md](DEVLOG.md): the
warm-start recipe (fresh policy, exact base), the alternating-phase REINFORCE
training, the structural φ/ψ gradient firewall (candidate embeddings under
`no_grad`, trunk states detached, so the policy gradient cannot touch φ), and the
deviations from the paper's Algorithm 1. **Not** reused from the core is
`train.fit` — the episode-rollout RL loop is the one thing the shared training
spine cannot express, so this folder carries its own `fit_episodes`.

The core stays unchanged: `ace.py`, `train.py`, `gp1d.py`, and the examples never
import this folder. A **permanent parity invariant** keeps the coupling honest —
the inference path must equal the base `ACE` forward **bitwise**, asserted at
every warm start and preserved afterwards by construction (the policy is
read-only and writes nothing back into the trunk), so `gp1d`'s oracle diagnostics
apply to the fine-tuned artifact unchanged.

A playground tab runs this model's full acquisition loop in-browser (hidden GP
function, user-chosen or policy-driven queries, live goal switching) — see
[playground/README.md](../../playground/README.md).
