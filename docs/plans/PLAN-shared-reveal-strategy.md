# Plan: single shared multi-latent reveal strategy

Created: 2026-06-07
Status: COMPLETE for code/docs; Gaussian retrain/export done, GP-1D retrain/export open

## Completion Notes

Implemented on 2026-06-07. This plan records the migration to one shared
latent-reveal data-generating process across the four Python examples:
`gaussian_toy.py`, `gp1d.py`, `sbi_sir.py`, and `bo1d.py`.

Current source of truth is the code plus `DEVLOG.md`. The implemented code/docs
work is complete: all four examples call `ace.sample_reveal_mask`, and all four
default `--latent-context-prob` to `0.5`. The remaining work is artifact work:
Gaussian has been retrained under this DGP with playground weights re-exported and
parity fixtures regenerated together; GP-1D still needs the same
retrain/re-export/regenerate pass. Proper SIR and BO checkpoints are eventual
training follow-ups, not blockers for this migration.

The SIR playground tab was added after this reveal migration. Its local
`artifacts/sbi_sir.pt` is a short CPU smoke checkpoint for wiring/testing, not a
quality model artifact.

## Summary

Unify how examples decide how many latents are revealed as context. The shared
helper is:

```python
sample_reveal_mask(n_latents, batch_size, q, device)
```

where `q = P(reveal nothing)` and the CLI-level
`latent_context_prob = P(reveal anything) = 1 - q`.

For each row:

1. With probability `q`, reveal no latent.
2. Otherwise split the revealing mass 50/50 between:
   - a uniform random non-empty subset;
   - a uniform count `k in 1..L`, then a uniform size-`k` subset.

With the default `latent_context_prob = 0.5`, the verified count distributions
are:

```text
L=2 -> {0:.50, 1:.29, 2:.21}
L=3 -> {0:.50, 1:.19, 2:.19, 3:.12}
```

## Rationale

Uniform-over-subsets is fair to each specific pin pattern, but it starves the
extremes. At `L=3`, revealing all latents gets only about `0.07` of total mass
when `q=0.5`.

Uniform-over-count represents each count, including all-revealed, more evenly,
but over-weights the lone all-revealed subset.

The 50/50 blend keeps a per-subset floor while giving all-revealed contexts
enough mass. That matters for the playground interaction: users may pin any
subset of latents and ask the model to predict the rest.

## Current Implementation

- `ace.sample_reveal_mask` implements the mixture and keeps the same signature.
- `gaussian_toy.py` and `gp1d.py` use the shared helper.
- `sbi_sir.py` and `bo1d.py` were migrated off their former private xor logic,
  which could reveal exactly one latent but never both.
- `--latent-context-prob` defaults to `0.5` in all four examples.
- `bo1d.py --scale-check` still passes `latent_context_prob=0.0`, which maps to
  `q=1.0` and preserves the intended no-reveal scale-check behavior.
- Reveal-all rows are safe. SIR and BO target layouts always keep data targets
  active, so revealing both latents does not produce an empty-target row.

Representations are unchanged:

- Bounded continuous latent reveals are zero-spread `PRIOR` tokens.
- Discrete latent reveals, such as GP-1D's `kernel`, remain `VALUE` tokens via
  `value_index`.
- Non-revealed latents are queried.

## Phases

### Phase 1: Core Sampler

Status: complete.

Work completed:

- Rewrote `sample_reveal_mask` with the `q` none gate and 50/50 mixture.
- Kept the function generic over `n_latents`.
- Updated the docstring.

Verification:

- `python -c "import ace"` passed.
- Empirical count checks matched the expected L=2 and L=3 distributions above.

### Phase 2: Example Migration

Status: complete.

Work completed:

- `sbi_sir.py` imports and uses `sample_reveal_mask(2, ...)`.
- `bo1d.py` imports and uses `sample_reveal_mask(2, ...)`.
- Gaussian and GP comments were aligned with the new mixture.
- `sample_toy_batch` no longer carries a stale `latent_context_prob=0.0`
  signature default.
- SIR and BO CLI defaults changed from `0.20` to `0.5`.

Verification:

- `python -c "import sbi_sir, bo1d, gaussian_toy, gp1d"` passed.
- No current example code path retains xor reveal logic.

### Phase 3: Smoke Runs

Status: complete.

Short CPU runs completed under the new DGP:

```powershell
.\.venv\Scripts\python.exe gaussian_toy.py --device cpu --steps 20 --batch-size 32
.\.venv\Scripts\python.exe gp1d.py --device cpu --steps 20 --batch-size 16
.\.venv\Scripts\python.exe sbi_sir.py --device cpu --steps 20 --batch-size 16
.\.venv\Scripts\python.exe bo1d.py --device cpu --steps 20 --batch-size 16 --no-plot
.\.venv\Scripts\python.exe bo1d.py --scale-check
```

The smoke runs verified that reveal-all rows do not break context/target
construction. They were not quality-training runs.

### Phase 4: Playground Checkpoint Refresh

Status: Gaussian done (retrained + re-exported + parity regenerated); GP-1D pending follow-up.

Goal: make the retained Gaussian and GP-1D checkpoints, exported browser blobs,
and playground parity fixtures reflect this shared reveal DGP.

Required sequence:

```powershell
.\.venv\Scripts\python.exe gaussian_toy.py --steps 30000 --save-checkpoint artifacts\gaussian_toy.pt --plot-path artifacts\gaussian_toy.png
.\.venv\Scripts\python.exe gp1d.py --steps 100000 --save-checkpoint artifacts\gp1d.pt --plot-path artifacts\gp1d.png
.\.venv\Scripts\python.exe playground\export_weights.py --task gaussian --checkpoint artifacts\gaussian_toy.pt --out playground\public\models\gaussian
.\.venv\Scripts\python.exe playground\export_weights.py --task gp1d --checkpoint artifacts\gp1d.pt --out playground\public\models\gp1d
.\.venv\Scripts\python.exe playground\parity.py
cd playground
npm test
```

Run export and parity together. `npm test` only compares the TypeScript forward
pass against the fixtures and blobs that are present; it cannot detect that both
were regenerated from a stale checkpoint.

Reload checks after retraining:

```powershell
.\.venv\Scripts\python.exe gaussian_toy.py --eval-only --load-checkpoint artifacts\gaussian_toy.pt
.\.venv\Scripts\python.exe gp1d.py --eval-only --load-checkpoint artifacts\gp1d.pt
```

### Phase 5: Documentation

Status: complete.

Work completed:

- `DEVLOG.md` records the shared mixture, rationale, code migration, the completed
  Gaussian retrain/export, the pending GP-1D retrain/export, and eventual SIR/BO training.
- `AGENTS.md` describes the current reveal gotcha and notes that all four
  examples share `sample_reveal_mask`.
- Historical DEVLOG entries remain historical and are not rewritten.

### Phase 6: Proper SIR and BO Checkpoints

Status: deferred.

No committed quality `sbi_sir.pt` or `bo1d.pt` is part of this migration. SIR and
BO code run under the shared reveal DGP, but real checkpoint quality is a
separate training task. BO in particular is expected to need a long GPU run.

## Risks and Follow-Ups

- **Playground staleness:** retraining without re-exporting weights and fixtures
  leaves the browser demo silently on the old model. Always run export and parity
  together.
- **GP wall-clock:** the GP-1D 100k retrain is the long pole because data/oracle
  sampling uses CPU float64 Cholesky.
- **Smoke artifacts:** short CPU checkpoints are valid for wiring tests only.
  They should not be mistaken for quality retained models.
- **Weight hosting:** browser blobs remain gitignored pending the Pages weight
  hosting decision.

## Open Questions

None for the code/docs migration. The open work is execution of the retrain and
export follow-up.
