# aline — joint amortized inference + active data acquisition for nanoACE

An extension that implements ALINE — Huang, Wen, Bharti, Kaski &
Acerbi (2025), *ALINE: Joint Amortization for Bayesian Inference and Active
Data Acquisition* (NeurIPS 2025;
[project page](https://www.huangdaolang.com/aline/)) — on top of an
unchanged nanoACE model, on the paper's GP active-learning task (§4.1).
Local paper markdown is in [paper/](paper/).

ALINE couples two things in one network: an amortized inference model `q_φ`
(posteriors over latents, predictive distributions over data) and an
acquisition policy `π_ψ` that picks which point to query next so that `q_φ`
learns fastest **about a runtime-selectable goal** ξ — a parameter subset, or
predictive accuracy over a target region. The policy is trained with
REINFORCE, the reward being the model's *own* per-step improvement in the
log-probability of the current goal (self-estimated information gain).

The re-expression here is deliberately ACE-native, and that is the thesis
worth reading the code for: **the paper's target-specifier/selective-mask
apparatus collapses into token composition.** Parameter targets are latent
QUERY tokens, predictive targets are data QUERY tokens, and ξ is just *which
target tokens are active*. The target tokens do double duty — they are the
queries `q_φ` answers (NLL and reward are computed on them) and, through their
final transformer states, the goal representation the policy reads, including
how uncertain the model still is about each goal. Query candidates are
"hypothetical targets": data QUERY tokens at candidate locations, embedded by
the core embedder verbatim. Switching goals mid-experiment is a mask flip.

Reused unchanged (imported from the core): the `Variable` / `Tokens` / `Batch`
schema, the embedder (`ACE._embed`), the full inference forward (`ACE.forward`
— it **is** `q_φ`, bit-identical), the shared MDN + categorical heads and
`Predictions`, `sample_reveal_mask` (the ξ-subset sampler), the GP physics
(`gp1d.draw_instances` / `draw_gp`), the grid oracle (`gp1d.gp_oracle`,
`kernel_posterior`, `diagnostics.query_log_density`), and `train.py`'s CLI
parent, scheduler, and checkpoint helpers. **Not** reused: `train.fit` — the
episode-rollout RL loop is the one thing the shared training spine cannot
express, so this folder carries its own `fit_episodes`.

New in this folder: a small read-only **policy decoder** (`PolicyBlock` ×2,
~0.4M params on the 1.2M-param gp1d base: candidates cross-attend to the
detached final context and target states, a linear head scores each candidate,
masked softmax over the remaining pool), the episode environment (candidate
pool, ξ sampler, rollout with per-step rewards), the alternating-phase
REINFORCE training loop, and an evaluation suite with random / uncertainty-
sampling baselines, matched-vs-mismatched targeting contrasts, and oracle
calibration on policy-acquired contexts.

## The warm-start recipe

1. **Attach**: `ALINE(ACE)` adds `policy_blocks.* / policy_norm.* /
   policy_head.*` parameters; a base GP-1D checkpoint loads under a strict
   guard (`unexpected == []`, missing keys all under the policy prefixes).
2. **Fresh policy, exact base**: the policy modules keep their fresh
   initialization — a near-uniform initial policy, approximately the paper's
   random warm-up actions — while the inference path is **bit-identical** to
   the base checkpoint, asserted at every warm start (`check_step0`).
3. **Permanent parity invariant**: unlike arbuffer's step-0-only check, the
   inference path of an ALINE artifact *stays* exactly an `ACE` forward — so
   `gp1d`'s oracle diagnostics apply to the fine-tuned artifact unchanged, and
   any change to the core forward fails the warm start loudly.
4. **Joint fine-tune, alternating phases**: *prediction* steps (NLL → base/φ;
   policy frozen; rollouts driven 50% by the current policy, 50% random — the
   on-policy half adapts `q_φ` to policy-shaped contexts, the random half
   preserves the calibration the oracle checks need) alternate with *policy*
   steps (REINFORCE → policy/ψ; base frozen). The gradient firewall is
   structural, not a convention: candidate embeddings are computed under
   `no_grad` and the trunk states enter the policy detached, so the PG loss
   *cannot* touch φ. The paper-literal both-losses-every-step scheme is kept
   as `--update-mode simultaneous`.

Within a rollout the parameters are frozen and the optimizers step once per
episode batch, so the reward `R_t = mean_active(log q_t − log q_{t−1})`
measures information gained from the new point, not optimizer movement — and
each episode batch stays a pure function of `(seed, step)` (reproducible,
resume-exact, policy sampling included). Deviations from the paper's
Algorithm 1 and their rationale are recorded in the local
[DEVLOG.md](DEVLOG.md).

## Run

From the repo root, with a trained GP-1D checkpoint (`python gp1d.py
--save-checkpoint artifacts/gp1d.pt`):

```powershell
# warm-started joint fine-tune (the validated 5k recipe; ~1.5 h on an RTX 4060)
.\.venv\Scripts\python.exe extensions\aline\gp1d_aline.py `
    --base-checkpoint artifacts\gp1d.pt --steps 5000 `
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
pool plus 32 predictive-target locations per episode (one joint CPU float64
draw; pool observations are lookups, noiseless by default), one random seed
point, and `T = 16` acquisition steps — context stays ≤ 17 points, inside the
warm start's trained `n_context ≤ 20` range (keep `--episode-steps ≤ 19`).
Goals ξ are 50% predictive / 50% a non-empty parameter subset of
{lengthscale, outputscale, kernel}; the discrete kernel as an acquisition
target is a small genuine extension over the paper's continuous-only GP
experiment.

## What the diagnostic shows

`gp1d_aline.py` ends with a held-out evaluation and a six-panel figure:

- **Predictive-goal demo** on the fixed gp1d evaluation function: the
  predictive band before/after the episode with the query order marked.
- **Query placement by goal** on the same function: predictive vs lengthscale
  goals, plus a mid-episode ξ switch (predictive → lengthscale at T/2) — the
  runtime-flexibility demo.
- **Lengthscale marginal vs the grid oracle** at T under the matched goal.
- **Held-out curves**: predictive RMSE vs steps (ALINE / uncertainty sampling /
  random) and parameter log q(θ_true) vs steps (ALINE / random).
- **Targeting contrasts**: log q(ℓ_true | D_T) and log q(kernel_true | D_T)
  after acquiring under the matched vs a mismatched goal on identical episodes.

**Honest performance note (5k validation run, 2026-06-12).** At 5k episode
batches (≈2.5k policy updates, <1% of the paper's episode count) the policy
**learns**: predictive RMSE 0.220 vs random 0.248 (below along the whole
curve), parameter log q −0.124 vs random −0.147, rewards still climbing at the
end, and `q_φ` stayed oracle-calibrated through the joint fine-tune (kernel
KL 0.002–0.025 on acquired contexts). But **uncertainty sampling still wins on
prediction** (0.194), and both targeting contrasts are **null** at this budget
(ℓ: −0.043 vs −0.042; kernel: −0.272 vs −0.266) — query placement differs by
goal in the demo panel, but does not yet buy measurable log q. Read as
undertraining (the pre-registered fallback order: longer run first); the full
gate readings and interpretation live in [DEVLOG.md](DEVLOG.md).

## Boundary

The core is unchanged: `ace.py`, `train.py`, `gp1d.py`, and the examples
never import this folder. The extension is torch-only and reaches into core
internals — it subclasses `ACE`, calls `_embed`, and re-runs the core block
loop in `forward_with_states` — with the parity check keeping that coupling
honest:
the inference path must equal the base `ACE` forward **bitwise**. The
assertion fires at every warm start (and in the phase-1 checks); the property
itself persists afterwards by construction, since the policy is read-only and
writes nothing back into the trunk. `aline.py` is task-agnostic (it knows
`ace.py`, not `gp1d.py`); everything GP-specific lives in `gp1d_aline.py`.

A **local-only** playground tab runs this model's full acquisition loop
in-browser (hidden GP function, user-chosen or policy-driven queries, live
goal switching) — see `playground/README.md` and
[docs/plans/PLAN-aline-playground.md](../../docs/plans/PLAN-aline-playground.md);
TS-port design notes are in the local [DEVLOG.md](DEVLOG.md).
