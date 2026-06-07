# Plan: Single shared multi-latent reveal strategy

Created: 2026-06-07
Status: APPROVED 2026-06-07 — executing lane (b): code + docs now; retrain +
playground refresh deferred to a follow-up; SIR/bo1d real training eventual.

## Summary
Unify how all four examples decide *how many* latents are conditioned on (revealed
as context) under one shared helper, `ace.sample_reveal_mask`. The agreed
distribution per task: with probability `q` reveal **nothing** (default 1/2);
otherwise split the revealing mass 50/50 between **(A)** a uniform random non-empty
subset and **(B)** a uniform count `k∈1..L` then a uniform size-`k` subset. Migrate
`sbi_sir.py` and `bo1d.py` off their `xor` single-reveal logic (which can never
reveal both latents) onto the shared helper, then retrain the two playground
checkpoints (Gaussian, GP-1D) under the new DGP and refresh the playground export.

## Background (verified current state)
- `ace.sample_reveal_mask(n_latents, batch_size, q, device)` is already shared by
  `gaussian_toy.py` (L=2) and `gp1d.py` (L=3). **Its body has already been edited**
  to implement the mixture (Phase 1) — verified empirically:
  - L=2 → `{0: .50, 1: .29, 2: .21}`
  - L=3 → `{0: .50, 1: .19, 2: .19, 3: .12}`
- `sbi_sir.py` (L=2: `beta`, `gamma`) and `bo1d.py` (L=2: `x_opt`, `y_opt`) use a
  private `xor`: reveal w.p. `latent_context_prob`, then exactly one latent. They
  **cannot** reveal both — so two-pin conditioning is out-of-distribution for them.
- Both SIR and bo1d build targets as `[latent_0, latent_1, data_0 … data_{T-1}]`
  with the data-`y` target columns **always active** (`tgt_t = 2 + data_targets`).
  So a reveal-all row still has data to predict — no empty-target row. (Gaussian/GP
  already allow reveal-all under the old DGP, so they gain no new edge case.)
- Checkpoints present locally: **only** `artifacts/gaussian_toy.pt` and
  `artifacts/gp1d.pt`. There is **no** `sbi_sir.pt` / `bo1d.pt` (gitignored, trained
  elsewhere). The playground ships **only** the Gaussian and GP-1D models.
- Default `--latent-context-prob`: **0.5** (Gaussian, GP) but **0.20** (SIR, bo1d).
- `bo1d.py:787` passes `latent_context_prob=0.0` in the `--scale-check` path; with
  the helper that maps to `q=1.0` → reveal nothing, preserving behavior.

## Scope
- **In scope**
  - The `sample_reveal_mask` mixture in `ace.py` (done; verify in this plan).
  - Migrate `sbi_sir.py` and `bo1d.py` to call `sample_reveal_mask`.
  - Standardize the `--latent-context-prob` default to **0.5** across all four
    (matches the agreed "1/2 reveal nothing"). See Open Question 1.
  - Smoke-verify all four run under the new DGP, including reveal-all rows.
  - Retrain the two **playground** checkpoints (Gaussian 30k, GP-1D 100k) under the
    new DGP; re-export weights + regenerate parity fixtures; `npm test` green.
  - Docs: new DEVLOG entry; resolve the SIR multi-reveal TODO (and note bo1d);
    update the AGENTS.md reveal gotcha; touch README if it claims single-reveal.
- **Out of scope**
  - Producing committed `sbi_sir.pt` / `bo1d.pt` checkpoints. None exist today, the
    playground does not use them, and a proper bo1d GPU run is already deferred in
    the DEVLOG. SIR/bo1d get the code migration + a short smoke run only. See OQ2.
  - Any change to the playground TS (multi-pin is already in-distribution and
    allowed; the mixture only reweights training — no OOD-logic change needed).
  - Changing per-latent reveal weighting (the strategy treats all latents
    symmetrically, by design).
  - Tuning model size / diagnostics quality beyond producing honest artifacts.

## Phases

### Phase 1: Core sampler (`ace.py`) — already implemented, verify only
**Goal**: `sample_reveal_mask` realizes the 1/2 · 1/4 · 1/4 mixture with an
unchanged signature.

**Work** (already applied):
- Body rewritten: `reveal_any` gate (`q`), then per-row 50/50 between the bitmask
  "uniform subset" scheme and the "uniform count k then uniform size-k subset"
  scheme (`rand().argsort().argsort()` ranks `< k`). Docstring updated.

**Verification**:
- [ ] `python -c "import ace"` clean.
- [ ] Count distribution matches L=2 `{.50,.29,.21}` and L=3 `{.50,.19,.19,.12}`
      (already confirmed; re-run as the regression check).

### Phase 2: Migrate SIR and bo1d; align Gaussian/GP comments + default
**Goal**: SIR and bo1d use `sample_reveal_mask`; both can now reveal 0, 1, or both
latents. The two already-migrated examples get stale-comment / default cleanup so all
four read consistently.

**Work**:
- `sbi_sir.py`
  - Import: add `sample_reveal_mask` to the `from ace import …` line (33).
  - Replace the `xor` block (292–294) with:
    ```python
    reveal_mask = sample_reveal_mask(2, batch_size, q=1.0 - latent_context_prob, device=device)
    reveal_beta = reveal_mask[:, 0]
    reveal_gamma = reveal_mask[:, 1]
    ```
  - Update the `sample_sir_batch` docstring (257–258) to describe the shared mixture
    instead of "one is revealed".
  - `--latent-context-prob` default 0.20 → 0.5 (728). (OQ1)
- `bo1d.py`
  - Import: add `sample_reveal_mask` to the `from ace import …` line (41).
  - Replace the `xor` block (399–401) with the `sample_reveal_mask(2, …)` form for
    `reveal_x`, `reveal_y`.
  - `--latent-context-prob` default 0.20 → 0.5 (836). (OQ1)
  - Leave `latent_context_prob=0.0` at the `--scale-check` call site (787) unchanged.
  - Update any module docstring text that implies single-reveal (the borrowing note
    at lines 11–12 references "the reveal mechanism" — fine to leave, but verify it
    reads correctly post-migration).
- `gaussian_toy.py` / `gp1d.py` (no behavior change — already call the helper)
  - Update the inline reveal comments that still describe the old pure-subset scheme
    (gaussian_toy.py ~95–96, gp1d.py ~218–219, both say "uniform random non-empty
    subset") to reference the shared mixture / point at `sample_reveal_mask`'s
    docstring, so they don't contradict the new DGP.
  - **`gaussian_toy.py:79` signature default** `latent_context_prob: float = 0.0`:
    remove the `= 0.0` so it matches the bare annotations on the other three samplers
    (`gp1d.py:195`, `sbi_sir.py:249`, `bo1d.py:361`). First confirm the trainer
    (`gaussian_toy.py:279`) is the only caller and passes it explicitly; if any caller
    relies on the default, set it to 0.5 instead of removing.

**Verification**:
- [ ] `python -c "import sbi_sir, bo1d, gaussian_toy, gp1d"` clean.
- [ ] `rg -n "def sample_toy_batch" -A1 gaussian_toy.py` shows no `= 0.0` default;
      grep confirms no caller of `sample_toy_batch` omits `latent_context_prob`.

### Phase 3: Smoke-verify all four under the new DGP
**Goal**: every example runs end to end under the mixture, including reveal-all rows.

**Steps** (CPU, short, no artifacts where a flag exists):
1. `gaussian_toy.py --device cpu --steps 20 --batch-size 32`
2. `gp1d.py --device cpu --steps 20 --batch-size 16`
3. `sbi_sir.py --device cpu --steps 20 --batch-size 16`
4. `bo1d.py --device cpu --steps 20 --batch-size 16 --no-plot`
   plus `bo1d.py --scale-check`

**Verification**:
- [ ] All four complete without error and print their diagnostics.
- [ ] No regenerated `artifacts/*.png` for the smoke runs (use `--no-plot` for bo1d;
      for gaussian/gp/sir direct `--plot-path` to a throwaway temp file so the
      retained `.png`s are not clobbered by an untrained model).

### Phase 4: Retrain playground checkpoints + refresh export
**Goal**: the shipped Gaussian and GP-1D checkpoints (and the playground blobs +
fixtures) reflect the new DGP. **GP-1D 100k is the long pole.** (OQ2 — timing/owner.)

**Steps**:
1. Retrain Gaussian (≈30k):
   `gaussian_toy.py --steps 30000 --save-checkpoint artifacts\gaussian_toy.pt --plot-path artifacts\gaussian_toy.png`
   - [ ] Printed model moments track the oracle (loose check).
2. Retrain GP-1D (≈100k, GPU; CPU-float64 sampling so no GPU watchdog risk):
   `gp1d.py --steps 100000 --save-checkpoint artifacts\gp1d.pt --plot-path artifacts\gp1d.png`
   - [ ] Diagnostic runs; predictive RMSE roughly tracks the oracle.
3. Re-export both blobs: `playground/export_weights.py` for `gaussian` and `gp1d`.
4. Regenerate fixtures: `playground/parity.py` (pins the *current* checkpoints).
   **Run export + parity together** (DEVLOG fixtures+blob staleness gotcha).
5. `cd playground && npm test`.

**Verification**:
- [ ] `npm test` parity green for both models, all cases.
- [ ] Reload paths: `gaussian_toy.py --eval-only --load-checkpoint artifacts\gaussian_toy.pt`
      and `gp1d.py --eval-only --load-checkpoint artifacts\gp1d.pt`.

### Phase 5: Documentation
**Goal**: docs describe the implemented shared mixture and retire the stale TODO.

**Work**:
- `DEVLOG.md`: new dated entry — the shared reveal strategy (1/2 · 1/4 · 1/4
  mixture; rationale: headline 0-reveal + per-subset floor + count-extreme coverage;
  all four examples now share `sample_reveal_mask`; SIR + bo1d migrated off `xor`;
  default `latent_context_prob` standardized to 0.5; retrain results/numbers).
- `DEVLOG.md`: resolve the "TODO — migrate SIR to the shared multi-reveal DGP"
  bullet (DEVLOG.md:269) — mark done, point at the new entry, note bo1d migrated too.
- `AGENTS.md`: rewrite the "Latent reveal uses uniform non-empty subsets" gotcha
  (AGENTS.md:118–121) to (a) rename + describe the **mixture** (½ none · ¼ uniform
  subset · ¼ uniform count), not just "uniform non-empty subset", and (b) state that
  **all four** examples now share `sample_reveal_mask` (the list at :121 currently
  says only Gaussian and GP-1D).
- `README.md`: no SIR/bo1d single-reveal claims exist to fix; the Gaussian setup text
  (README.md:101–104, 107) is already multi-reveal-framed. Optional: the example
  command `--latent-context-prob 0.25` (README.md:115/121) is just a flag demo, not a
  default — leave or bump to match the new 0.5 default. No required README change.
- **Historical DEVLOG entries are the dated record — do NOT rewrite them.** The new
  entry supersedes them. Specifically leave as-is: DEVLOG.md:171 ("at most one ...",
  in the DONE multi-reveal entry), :271–272 (the old-SIR description inside the TODO
  bullet being resolved — it's fine for that bullet to describe the pre-migration
  state it's retiring), and :498 (the 2026-06-06 Gaussian entry: single-reveal,
  default 0.25, "VALUE token" — accurate for its date).

**Verification**:
- [ ] `rg -n "uniform non-empty subset|reveals .* one|at most one"` over the
      **current-truth** surfaces only (`AGENTS.md` gotcha, `README.md` setup, the
      `sample_reveal_mask` docstring) shows no stale single-reveal / pure-subset
      claims. Pre-dated DEVLOG entries are expected to still match and are left as the
      historical record.

### Phase 6 (deferred / out of scope): proper SIR + bo1d checkpoints
Not done here. No committed checkpoint exists today and a proper bo1d GPU run is
already deferred in the DEVLOG. If wanted later: train under the new DGP and record
diagnostics. Tracked as a follow-up, not a blocker.

## Commit strategy
- Commit 1 — code: `ace.py` (verified) + `sbi_sir.py` + `bo1d.py` migration +
  default knob (Phases 1–2), after smoke (Phase 3).
- Commit 2 — retrained artifacts are gitignored, so this commit is the playground
  export/fixtures refresh (Phase 4) + docs (Phase 5), once retraining completes.
- (If OQ2 defers retraining, Commit 2 is docs-only and notes the checkpoints/blobs
  are pending a retrain.)

## Risks
- **Playground staleness** (DEVLOG gotcha): a retrain without re-exporting blobs +
  fixtures leaves the demo silently on the old model while `npm test` stays green.
  Mitigation: Phase 4 runs export + parity together; never split them.
- **GP 100k wall-clock**: long. Mitigation: run in background / let the user own it
  (OQ2). GP sampling is CPU float64, so no GPU host-watchdog concern.
- **Smoke runs clobbering retained PNGs**: an untrained 20-step model overwrites
  `artifacts/*.png` by default. Mitigation: `--no-plot` / temp `--plot-path` in
  Phase 3 (this already bit us once this session).
- **Default knob change (OQ1)**: shifts SIR/bo1d behavior, but neither has a
  committed checkpoint, so there is no staleness; only diagnostics shift.
- **Consistency if retraining is deferred (OQ2)**: ace.py/docs would describe the
  mixture while the shipped Gaussian/GP blobs reflect the old uniform-subset DGP.
  Multi-pin is still in-distribution (it already was), so nothing breaks; but the
  blobs would not yet match the documented DGP. If deferring, the docs must say so.

## Open Questions — RESOLVED
1. **Standardize `--latent-context-prob` default to 0.5 for SIR and bo1d? → YES.**
2. **Retraining execution/timing → lane (b):** land code + docs now; retraining
   (Gaussian 30k + GP 100k) and the playground export/fixtures refresh are a separate
   follow-up. Docs (Phase 5) must state the shipped Gaussian/GP checkpoints + playground
   blobs were trained under the old uniform-subset DGP and are **pending a retrain**
   under the new mixture. SIR/bo1d stay code-migration + smoke only now; real SIR/bo1d
   training is wanted **eventually** (record as a tracked follow-up, not done here).

## Execution status (lane b)
- **DONE (this commit):** Phase 1 (mixture verified — L2 `{.50,.29,.21}`, L3
  `{.50,.19,.19,.12}`) · Phase 2 (SIR + bo1d migrated off `xor`; Gaussian/GP comments,
  argparse help, and `sample_toy_batch` signature default aligned; all four defaults
  → 0.5) · Phase 3 (all four smoke-run under the new DGP, incl. reveal-all rows) ·
  Phase 5 (docs: new DEVLOG entry, SIR TODO resolved, AGENTS gotcha rewritten,
  "blobs pending retrain" + "eventual SIR/bo training" recorded).
- **Follow-up (not now):** Phase 4 (retrain Gaussian + GP, refresh playground export +
  fixtures, `npm test`) and Phase 6 (real SIR/bo1d checkpoints).

---
**Please review. Edit directly if needed, then confirm to proceed.**
