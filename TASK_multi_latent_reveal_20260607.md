# TASK: implement multi-latent reveal DGP

Spec of record: DEVLOG.md → "Multi-latent reveal (DONE 2026-06-07)".
Scope here: the sampler/DGP code, smoke verification, and the follow-up
multi-reveal checkpoint/export/fixture refresh.

## Checklist
- [x] Add shared `sample_reveal_mask(n_latents, batch_size, q, device) -> bool[B,L]` to [ace.py](ace.py)
- [x] Wire GP sampler: [gp1d.py](gp1d.py) `sample_gp_batch` → use shared mask (replace `reveal_which`)
- [x] Wire Gaussian sampler: [gaussian_toy.py](gaussian_toy.py) `sample_toy_batch` → use shared mask (replace mu xor logsig)
- [x] Bump default `--latent-context-prob` (P(reveal any)) to 0.5 in both CLIs
- [x] Smoke: `gp1d.py --device cpu --steps 20 ...` completes (trains + eval, no error)
- [x] Smoke: `gaussian_toy.py --device cpu --steps 20 ...` completes (trains + eval, no error)
- [x] Quick unit check: reveal-mask uniform over non-empty subsets (L=3 ~0.143, L=2 ~0.333; q=1→none, q=0→non-empty)
- [x] Update DEVLOG status line (samplers implemented; both examples retrained/exported)
- [x] /doublecheck — no issues; semantic check confirms complementary ctx/target masks, real multi-reveal, correct representations

Note: initial impl over-weighted singletons (force-one-true); fixed to true
uniform-over-non-empty-subsets via integer bitmask sampling.

## Deferred → DONE (2026-06-07, follow-up)
- [x] Retrain gaussian_toy (30k) + gp1d (100k) with the new DGP — both track the oracle
- [x] Re-run export_weights.py + parity.py (together); fixtures regenerated; playground 10/10
- [x] Remove ≥2-pin OOD trigger (`PIN_OOD_MIN` from config.ts; pin branch from `oodReasons`)
- Note: GP kernel posterior is a bit overconfident vs the oracle (Periodic 0.81 vs 0.50) — see DEVLOG.

## Decisions / notes
- Helper lives in `ace.py` (both examples already `from ace import ...`; precedent:
  `sample_ar`, `encode_value`). Generic torch util.
- `q` = P(reveal nothing); CLI `--latent-context-prob` = P(reveal any) = `1 - q`.
- No architecture change: embedder/attention already handle multiple latent context tokens.
- Context/target construction in both samplers already keys off per-latent reveal
  booleans, so only the reveal computation changes.

## Status
COMPLETE. Sampler/DGP code is implemented and smoke-verified. Gaussian was
retrained for 30k steps and GP-1D for 100k steps under the multi-reveal DGP;
exports and parity fixtures were regenerated together; the playground ≥2-pin OOD
banner was removed. The remaining open item is the separate playground
weight-hosting decision for Pages.
