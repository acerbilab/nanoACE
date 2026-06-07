/**
 * Gaussian demo verification: TS ACE inference and the TS analytic oracle must
 * both reproduce the gaussian_toy.py reference (fixed eval batch) for the latent
 * marginals and the posterior predictive density.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { ACEModel } from "../ace/model";
import { type Manifest, weightsFromBytes } from "../ace/weights";
import { gaussInfer } from "./infer";
import { analyticPosterior, predictiveDensity } from "./oracle";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

interface DemoRef {
  y_obs: number[];
  mu_unit: number;
  mu_nu: number;
  ls_unit: number;
  ls_nu: number;
  mu_grid: number[];
  ls_grid: number[];
  y_grid: number[];
  mu_post_ace: number[];
  ls_post_ace: number[];
  pred_ace: number[];
  mu_post_oracle: number[];
  ls_post_oracle: number[];
  pred_oracle: number[];
}

function maxViolation(a: number[], b: number[], atol: number, rtol: number): number {
  expect(a.length).toBe(b.length);
  let worst = -Infinity;
  for (let i = 0; i < a.length; i++) {
    const slack = Math.abs(a[i] - b[i]) - (atol + rtol * Math.abs(b[i]));
    if (slack > worst) worst = slack;
  }
  return worst;
}

describe("gaussian demo vs gaussian_toy.py", () => {
  const manifest = JSON.parse(
    readFileSync(join(ROOT, "public", "models", "gaussian", "manifest.json"), "utf8"),
  ) as Manifest;
  const bytes = readFileSync(join(ROOT, "public", "models", "gaussian", "weights.bin"));
  const model = new ACEModel(weightsFromBytes(manifest, new Uint8Array(bytes)));
  const ref = JSON.parse(readFileSync(join(ROOT, "test", "fixtures", "gaussian.demo.json"), "utf8")) as DemoRef;

  const params = { yObs: ref.y_obs, muUnit: ref.mu_unit, muNu: ref.mu_nu, lsUnit: ref.ls_unit, lsNu: ref.ls_nu };
  const grids = { muGrid: ref.mu_grid, lsGrid: ref.ls_grid, yGrid: ref.y_grid };

  it("ACE marginals + predictive match", () => {
    const res = gaussInfer(model, params, grids);
    expect(maxViolation(res.muPost, ref.mu_post_ace, 1e-3, 1e-3)).toBeLessThanOrEqual(0);
    expect(maxViolation(res.lsPost, ref.ls_post_ace, 1e-3, 1e-3)).toBeLessThanOrEqual(0);
    expect(maxViolation(res.predDensity, ref.pred_ace, 1e-3, 1e-3)).toBeLessThanOrEqual(0);
  });

  it("analytic oracle marginals + predictive match", () => {
    const muRange: [number, number] = [model.variables[1].bound_lo, model.variables[1].bound_hi];
    const lsRange: [number, number] = [model.variables[2].bound_lo, model.variables[2].bound_hi];
    const oracle = analyticPosterior(ref.y_obs, ref.mu_grid, ref.ls_grid, muRange, lsRange, {
      muUnit: ref.mu_unit,
      muNu: ref.mu_nu,
      lsUnit: ref.ls_unit,
      lsNu: ref.ls_nu,
    });
    expect(maxViolation(oracle.muPost, ref.mu_post_oracle, 1e-4, 1e-3)).toBeLessThanOrEqual(0);
    expect(maxViolation(oracle.lsPost, ref.ls_post_oracle, 1e-4, 1e-3)).toBeLessThanOrEqual(0);
    const pred = predictiveDensity(oracle, ref.y_grid);
    expect(maxViolation(pred, ref.pred_oracle, 1e-4, 1e-3)).toBeLessThanOrEqual(0);
  });
});
