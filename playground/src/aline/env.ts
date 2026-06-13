/**
 * Browser-side GP-1D environment for the ALINE tab: samples the hidden
 * ground-truth function exactly the way `gp1d.py`'s data-generating process
 * does — same kernels (Periodic period = 1.0, `ell` clamped at 1e-6), same
 * hyperprior ranges, same 1e-5 jitter, zero-mean joint draw `L @ z` via
 * Cholesky, noiseless observations. JS numbers ARE IEEE-754 float64, so the
 * math matches the Python CPU-float64 physics; `env.test.ts` pins kernel
 * matrices and Cholesky factors against `gp1d_aline.env.json`.
 *
 * The RNG is the playground's seeded mulberry32: the *distributions* match the
 * DGP, the streams deliberately don't — fixtures cover the deterministic math,
 * not the draws (the parity chain carries Python-drawn values for that).
 */

import { mulberry32, randn } from "../ace/rng";
import { linspace } from "../util";

// gp1d.py DGP constants (also pinned by the env fixture).
export const KERNELS = ["RBF", "Matern12", "Matern32", "Periodic"] as const;
export const LOG_LENGTHSCALE_RANGE: [number, number] = [Math.log(0.12), Math.log(0.8)];
export const LOG_OUTPUTSCALE_RANGE: [number, number] = [Math.log(0.25), Math.log(1.0)];
export const GEN_JITTER = 1e-5;
const PERIOD = 1.0;
const SQRT3 = Math.sqrt(3.0);

/** Cross-covariance matrix [x1.length][x2.length], mirroring gp1d._kernel_covariance. */
export function kernelCovariance(
  x1: number[],
  x2: number[],
  kernel: number,
  logEll: number,
  logScale: number,
): number[][] {
  const ell = Math.max(Math.exp(logEll), 1e-6);
  const amp2 = Math.exp(logScale) ** 2;
  const out: number[][] = [];
  for (let i = 0; i < x1.length; i++) {
    const row = new Array<number>(x2.length);
    for (let j = 0; j < x2.length; j++) {
      const r = Math.abs(x1[i] - x2[j]);
      let base: number;
      if (kernel === 0) {
        const z = r / ell;
        base = Math.exp(-0.5 * z * z);
      } else if (kernel === 1) {
        base = Math.exp(-r / ell);
      } else if (kernel === 2) {
        const z = (SQRT3 * r) / ell;
        base = (1.0 + z) * Math.exp(-z);
      } else {
        const s = Math.sin((Math.PI * r) / PERIOD);
        base = Math.exp((-2.0 * s * s) / (ell * ell));
      }
      row[j] = amp2 * base;
    }
    out.push(row);
  }
  return out;
}

/** Covariance at x with diagonal jitter, mirroring gp1d._kernel_matrix. */
export function kernelMatrix(
  x: number[],
  kernel: number,
  logEll: number,
  logScale: number,
  jitter = GEN_JITTER,
): number[][] {
  const K = kernelCovariance(x, x, kernel, logEll, logScale);
  for (let i = 0; i < x.length; i++) K[i][i] += jitter;
  return K;
}

/** Lower-triangular Cholesky factor of a positive-definite matrix; throws otherwise. */
export function cholesky(K: number[][]): number[][] {
  const n = K.length;
  const L: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let j = 0; j < n; j++) {
    let s = K[j][j];
    for (let k = 0; k < j; k++) s -= L[j][k] * L[j][k];
    if (!(s > 0)) throw new Error(`cholesky: not positive definite at pivot ${j}`);
    L[j][j] = Math.sqrt(s);
    for (let i = j + 1; i < n; i++) {
      let v = K[i][j];
      for (let k = 0; k < j; k++) v -= L[i][k] * L[j][k];
      L[i][j] = v / L[j][j];
    }
  }
  return L;
}

export interface EnvConfig {
  pool: number; // candidate locations ~ U(-1, 1)  (train-matched)
  grid: number; // plot/metric grid: linspace(-1, 1) — joint-drawn with the pool
  mPred: number; // predictive-target x* ~ U(-1, 1) (no truth needed; QUERY tokens)
}

export interface EpisodeDraw {
  poolX: number[];
  poolY: number[];
  gridX: number[];
  gridY: number[];
  xStar: number[];
  kernel: number;
  logEll: number;
  logScale: number;
  seedIdx: number; // the pool index observed at episode start
}

/**
 * Sample one hidden episode function: hyperparameters from the gp1d hyperprior,
 * one joint zero-mean draw at pool + grid locations. Numerically unlucky draws
 * (clustered x + short lengthscale can defeat the jitter) are resampled under a
 * perturbed seed — user-invisible, logged to the console.
 */
export function sampleEpisode(seed: number, cfg: EnvConfig): EpisodeDraw {
  for (let attempt = 0; attempt < 5; attempt++) {
    const rng = mulberry32((seed + attempt * 0x9e3779b9) >>> 0);
    const kernel = Math.min(Math.floor(rng() * KERNELS.length), KERNELS.length - 1);
    const logEll =
      LOG_LENGTHSCALE_RANGE[0] + rng() * (LOG_LENGTHSCALE_RANGE[1] - LOG_LENGTHSCALE_RANGE[0]);
    const logScale =
      LOG_OUTPUTSCALE_RANGE[0] + rng() * (LOG_OUTPUTSCALE_RANGE[1] - LOG_OUTPUTSCALE_RANGE[0]);
    const poolX = Array.from({ length: cfg.pool }, () => 2 * rng() - 1);
    const gridX = linspace(-1, 1, cfg.grid);
    const xStar = Array.from({ length: cfg.mPred }, () => 2 * rng() - 1);
    const xAll = poolX.concat(gridX);

    let L: number[][];
    try {
      L = cholesky(kernelMatrix(xAll, kernel, logEll, logScale));
    } catch {
      console.warn(`aline env: Cholesky failed on attempt ${attempt}, resampling`);
      continue;
    }
    const n = xAll.length;
    const z = Array.from({ length: n }, () => randn(rng));
    const y = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = 0; k <= i; k++) s += L[i][k] * z[k];
      y[i] = s;
    }
    const seedIdx = Math.min(Math.floor(rng() * cfg.pool), cfg.pool - 1);
    return {
      poolX,
      poolY: y.slice(0, cfg.pool),
      gridX,
      gridY: y.slice(cfg.pool),
      xStar,
      kernel,
      logEll,
      logScale,
      seedIdx,
    };
  }
  throw new Error("aline env: GP draw failed repeatedly; try another seed");
}
