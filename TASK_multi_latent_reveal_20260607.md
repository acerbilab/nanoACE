# TASK: implement multi-latent reveal DGP

Spec of record: DEVLOG.md → "NEXT TODO — train multi-latent reveal" (resolved DGP).
Scope here: the **sampler/DGP code** + smoke verification. Retraining is the user's
to run; weight re-export / fixture regen / dropping the ≥2-pin OOD banner are
deferred until a multi-reveal checkpoint exists.

## Checklist
- [x] Add shared `sample_reveal_mask(n_latents, batch_size, q, device) -> bool[B,L]` to [ace.py](ace.py)
- [x] Wire GP sampler: [gp1d.py](gp1d.py) `sample_gp_batch` → use shared mask (replace `reveal_which`)
- [x] Wire Gaussian sampler: [gaussian_toy.py](gaussian_toy.py) `sample_toy_batch` → use shared mask (replace mu xor logsig)
- [x] Bump default `--latent-context-prob` (P(reveal any)) to 0.5 in both CLIs
- [x] Smoke: `gp1d.py --device cpu --steps 20 ...` completes (trains + eval, no error)
- [x] Smoke: `gaussian_toy.py --device cpu --steps 20 ...` completes (trains + eval, no error)
- [x] Quick unit check: reveal-mask uniform over non-empty subsets (L=3 ~0.143, L=2 ~0.333; q=1→none, q=0→non-empty)
- [x] Update DEVLOG status line (samplers implemented; training pending)
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
COMPLETE (implementation). Sampler/DGP done and verified; both examples smoke-train
and produce correct multi-reveal batches. Changed files (uncommitted): `ace.py`,
`gp1d.py`, `gaussian_toy.py`, `DEVLOG.md` (status line), + this tracker. Retraining,
re-export, fixture regen, and dropping the ≥2-pin OOD banner remain deferred to the
user (see Deferred).
