/**
 * SIR demo verification: TS ACE inference and the TS numerical oracle must
 * reproduce the sbi_sir.py reference on the fixed informative-prior diagnostic.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { ACEModel } from "../ace/model";
import { type Manifest, weightsFromBytes } from "../ace/weights";
import { sirInfer } from "./infer";
import { buildSirOracleCache, sirOracle } from "./oracle";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

interface DemoRef {
  t_context: number[];
  y_context: number[];
  beta_unit: number;
  beta_nu: number;
  gamma_unit: number;
  gamma_nu: number;
  t_grid: number[];
  beta_grid: number[];
  gamma_grid: number[];
  beta_post_ace: number[];
  gamma_post_ace: number[];
  y_mean_ace: number[];
  y_std_ace: number[];
  beta_post_oracle: number[];
  gamma_post_oracle: number[];
  y_mean_oracle: number[];
  y_std_oracle: number[];
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

describe("SIR demo vs sbi_sir.py", () => {
  const manifest = JSON.parse(
    readFileSync(join(ROOT, "public", "models", "sbi_sir", "manifest.json"), "utf8"),
  ) as Manifest;
  const bytes = readFileSync(join(ROOT, "public", "models", "sbi_sir", "weights.bin"));
  const model = new ACEModel(weightsFromBytes(manifest, new Uint8Array(bytes)));
  const ref = JSON.parse(readFileSync(join(ROOT, "test", "fixtures", "sbi_sir.demo.json"), "utf8")) as DemoRef;

  const observations = ref.t_context.map((t, i) => ({ t, y: ref.y_context[i] }));
  const params = {
    observations,
    betaUnit: ref.beta_unit,
    betaNu: ref.beta_nu,
    gammaUnit: ref.gamma_unit,
    gammaNu: ref.gamma_nu,
  };
  const grids = { tGrid: ref.t_grid, betaGrid: ref.beta_grid, gammaGrid: ref.gamma_grid };

  it("ACE marginals + predictive curve match", () => {
    const res = sirInfer(model, params, grids);
    expect(maxViolation(res.betaPost, ref.beta_post_ace, 1e-3, 1e-3)).toBeLessThanOrEqual(0);
    expect(maxViolation(res.gammaPost, ref.gamma_post_ace, 1e-3, 1e-3)).toBeLessThanOrEqual(0);
    expect(maxViolation(res.predMean, ref.y_mean_ace, 1e-3, 1e-3)).toBeLessThanOrEqual(0);
    expect(maxViolation(res.predStd, ref.y_std_ace, 1e-3, 1e-3)).toBeLessThanOrEqual(0);
  });

  it("grid oracle marginals + predictive curve match", () => {
    const cache = buildSirOracleCache(ref.beta_grid, ref.gamma_grid);
    const betaRange: [number, number] = [model.variables[1].bound_lo, model.variables[1].bound_hi];
    const gammaRange: [number, number] = [model.variables[2].bound_lo, model.variables[2].bound_hi];
    const oracle = sirOracle(params, grids, cache, { betaRange, gammaRange });
    expect(maxViolation(oracle.betaPost, ref.beta_post_oracle, 1e-4, 1e-3)).toBeLessThanOrEqual(0);
    expect(maxViolation(oracle.gammaPost, ref.gamma_post_oracle, 1e-4, 1e-3)).toBeLessThanOrEqual(0);
    expect(maxViolation(oracle.yMean, ref.y_mean_oracle, 1e-4, 1e-3)).toBeLessThanOrEqual(0);
    expect(maxViolation(oracle.yStd, ref.y_std_oracle, 1e-4, 1e-3)).toBeLessThanOrEqual(0);
  });
});
