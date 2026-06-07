/**
 * End-to-end check of the GP demo's orchestration: `gpInfer` on the fixed eval
 * context (data-only, no pins) must match the Python reference computed with
 * gp1d.py's own helpers (band, kernel posterior, latent marginals).
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { ACEModel } from "../ace/model";
import { type Manifest, weightsFromBytes } from "../ace/weights";
import { gpInfer } from "./infer";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

interface DemoRef {
  x_context: number[];
  y_context: number[];
  band_x: number[];
  band_mean: number[];
  band_std: number[];
  kernel_probs: number[];
  ell_grid: number[];
  ell_post: number[];
  scale_grid: number[];
  scale_post: number[];
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

describe("gp demo orchestration vs gp1d.py", () => {
  const manifest = JSON.parse(
    readFileSync(join(ROOT, "public", "models", "gp1d", "manifest.json"), "utf8"),
  ) as Manifest;
  const bytes = readFileSync(join(ROOT, "public", "models", "gp1d", "weights.bin"));
  const model = new ACEModel(weightsFromBytes(manifest, new Uint8Array(bytes)));
  const ref = JSON.parse(readFileSync(join(ROOT, "test", "fixtures", "gp1d.demo.json"), "utf8")) as DemoRef;

  it("matches band / kernel / latent posteriors", () => {
    const points = ref.x_context.map((x, i) => ({ x, y: ref.y_context[i] }));
    const res = gpInfer(
      model,
      { points, pinKernel: null, pinEll: null, pinScale: null },
      { bandX: ref.band_x, latentGrid: ref.ell_grid.length },
    );

    expect(res.hasContext).toBe(true);
    expect(maxViolation(res.bandMean, ref.band_mean, 1e-3, 1e-3)).toBeLessThanOrEqual(0);
    expect(maxViolation(res.bandStd, ref.band_std, 1e-3, 1e-3)).toBeLessThanOrEqual(0);
    expect(maxViolation(res.kernelProbs!, ref.kernel_probs, 1e-3, 1e-3)).toBeLessThanOrEqual(0);
    expect(maxViolation(res.ellGrid, ref.ell_grid, 1e-5, 1e-5)).toBeLessThanOrEqual(0);
    expect(maxViolation(res.ellPost!, ref.ell_post, 1e-3, 1e-3)).toBeLessThanOrEqual(0);
    expect(maxViolation(res.scalePost!, ref.scale_post, 1e-3, 1e-3)).toBeLessThanOrEqual(0);
  });
});
