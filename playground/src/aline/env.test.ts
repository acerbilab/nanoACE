/**
 * Environment parity for the ALINE tab's GP sampler (`env.ts`) against the
 * checkpoint-independent `gp1d_aline.env.json` fixture (written by `parity.py`
 * on every run, no ALINE checkpoint needed): kernel matrices and Cholesky
 * factors in float64 on a moderate and an adversarial (clustered x, short
 * lengthscale) x-set, all four kernels each. Both sides are float64, so the
 * gate is tight — well below anything a formula or indexing bug could pass.
 *
 * Note the 8-point fixture cases do not cover the 192-point runtime regime;
 * `sampleEpisode`'s resample-on-failure fallback is the defense there.
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  GEN_JITTER,
  KERNELS,
  LOG_LENGTHSCALE_RANGE,
  LOG_OUTPUTSCALE_RANGE,
  cholesky,
  kernelMatrix,
  sampleEpisode,
} from "./env";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const FIXTURE = join(ROOT, "test", "fixtures", "gp1d_aline.env.json");

interface EnvCase {
  set: string;
  kernel: string;
  kernel_index: number;
  x: number[];
  log_ell: number;
  log_scale: number;
  K: number[][];
  L: number[][];
}

interface EnvFixture {
  jitter: number;
  period: number;
  log_lengthscale_range: number[];
  log_outputscale_range: number[];
  kernels: string[];
  cases: EnvCase[];
}

// float64 on both sides; only libm ulp differences and Cholesky summation-order
// effects remain (the clustered cases are deliberately ill-conditioned).
const ATOL = 1e-10;
const RTOL = 1e-8;

function checkMatrix(label: string, ts: number[][], pt: number[][]): void {
  expect(ts.length).toBe(pt.length);
  for (let i = 0; i < ts.length; i++) {
    for (let j = 0; j < ts[i].length; j++) {
      const d = Math.abs(ts[i][j] - pt[i][j]);
      const allowed = ATOL + RTOL * Math.abs(pt[i][j]);
      if (d > allowed) {
        throw new Error(`${label}[${i}][${j}]: ts=${ts[i][j]} pt=${pt[i][j]} |Δ|=${d.toExponential(3)}`);
      }
    }
  }
}

describe.skipIf(!existsSync(FIXTURE))("aline env: DGP parity vs gp1d", () => {
  const fx = JSON.parse(readFileSync(FIXTURE, "utf8")) as EnvFixture;

  it("constants match the Python DGP", () => {
    expect(fx.jitter).toBe(GEN_JITTER);
    expect(fx.period).toBe(1.0);
    expect(fx.kernels).toEqual([...KERNELS]);
    expect(fx.log_lengthscale_range[0]).toBeCloseTo(LOG_LENGTHSCALE_RANGE[0], 12);
    expect(fx.log_lengthscale_range[1]).toBeCloseTo(LOG_LENGTHSCALE_RANGE[1], 12);
    expect(fx.log_outputscale_range[0]).toBeCloseTo(LOG_OUTPUTSCALE_RANGE[0], 12);
    expect(fx.log_outputscale_range[1]).toBeCloseTo(LOG_OUTPUTSCALE_RANGE[1], 12);
  });

  it("kernel matrices and Cholesky factors match torch float64", () => {
    for (const c of fx.cases) {
      const K = kernelMatrix(c.x, c.kernel_index, c.log_ell, c.log_scale, fx.jitter);
      checkMatrix(`${c.set}/${c.kernel}/K`, K, c.K);
      checkMatrix(`${c.set}/${c.kernel}/L`, cholesky(K), c.L);
    }
  });

  it("sampleEpisode is seed-deterministic with the right shapes", () => {
    const cfg = { pool: 128, grid: 64, mPred: 32 };
    const a = sampleEpisode(7, cfg);
    const b = sampleEpisode(7, cfg);
    expect(a).toEqual(b);
    expect(a.poolX.length).toBe(128);
    expect(a.poolY.length).toBe(128);
    expect(a.gridX.length).toBe(64);
    expect(a.gridY.length).toBe(64);
    expect(a.xStar.length).toBe(32);
    expect(a.seedIdx).toBeGreaterThanOrEqual(0);
    expect(a.seedIdx).toBeLessThan(128);
    expect(a.poolY.every((v) => Number.isFinite(v))).toBe(true);
    expect(a.gridY.every((v) => Number.isFinite(v))).toBe(true);
    expect(a.kernel).toBeGreaterThanOrEqual(0);
    expect(a.kernel).toBeLessThan(4);
    expect(a.logEll).toBeGreaterThanOrEqual(LOG_LENGTHSCALE_RANGE[0]);
    expect(a.logEll).toBeLessThanOrEqual(LOG_LENGTHSCALE_RANGE[1]);

    const c = sampleEpisode(8, cfg);
    expect(c.poolY.some((v, i) => v !== a.poolY[i])).toBe(true);
  });
});
