/**
 * Numerical SIR grid oracle for the browser demo. This is a TS port of
 * `sbi_sir.sir_oracle`: deterministic RK4 trajectories over a beta/gamma grid,
 * Gaussian observation likelihood, Beta runtime priors, and a posterior-weighted
 * predictive epidemic curve.
 */

import { SIR } from "../config";
import { betaLogPriorOnGrid } from "../gaussian/oracle";
import type { SIRGrids, SIRParams } from "./infer";

const I0 = 0.01;
const HALF_LOG_2PI = 0.5 * Math.log(2.0 * Math.PI);

export interface SIROracleCache {
  betaGrid: number[];
  gammaGrid: number[];
  fineSteps: number;
  dt: number;
  traj: Float64Array; // [beta * gamma][fineSteps + 1]
}

export interface SIROracleResult {
  betaGrid: number[];
  betaPost: number[];
  gammaGrid: number[];
  gammaPost: number[];
  yMean: number[];
  yStd: number[];
}

function deriv(s: number, i: number, beta: number, gamma: number): [number, number] {
  const infection = beta * s * i;
  return [-infection, infection - gamma * i];
}

export function buildSirOracleCache(
  betaGrid: number[],
  gammaGrid: number[],
  fineSteps = SIR.FINE_STEPS,
): SIROracleCache {
  const n = betaGrid.length * gammaGrid.length;
  const stride = fineSteps + 1;
  const dt = SIR.T_DOMAIN[1] / fineSteps;
  const traj = new Float64Array(n * stride);

  let idx = 0;
  for (const beta of betaGrid) {
    for (const gamma of gammaGrid) {
      let s = 1.0 - I0;
      let i = I0;
      const off = idx * stride;
      traj[off] = i;
      for (let step = 0; step < fineSteps; step++) {
        const [ds1, di1] = deriv(s, i, beta, gamma);
        const [ds2, di2] = deriv(s + 0.5 * dt * ds1, i + 0.5 * dt * di1, beta, gamma);
        const [ds3, di3] = deriv(s + 0.5 * dt * ds2, i + 0.5 * dt * di2, beta, gamma);
        const [ds4, di4] = deriv(s + dt * ds3, i + dt * di3, beta, gamma);
        s = Math.min(Math.max(s + (dt / 6.0) * (ds1 + 2 * ds2 + 2 * ds3 + ds4), 0.0), 1.0);
        i = Math.max(i + (dt / 6.0) * (di1 + 2 * di2 + 2 * di3 + di4), 0.0);
        traj[off + step + 1] = i;
      }
      idx++;
    }
  }
  return { betaGrid, gammaGrid, fineSteps, dt, traj };
}

export function cacheValue(cache: SIROracleCache, gridIndex: number, time: number): number {
  const stride = cache.fineSteps + 1;
  const t = Math.min(Math.max(time, SIR.T_DOMAIN[0]), SIR.T_DOMAIN[1]);
  const pos = t / cache.dt;
  const lo = Math.min(Math.floor(pos), cache.fineSteps - 1);
  const w = Math.min(Math.max(pos - lo, 0.0), 1.0);
  const off = gridIndex * stride + lo;
  return cache.traj[off] + (cache.traj[off + 1] - cache.traj[off]) * w;
}

export function integrateSirAtTimes(beta: number, gamma: number, times: number[]): number[] {
  const cache = buildSirOracleCache([beta], [gamma]);
  return times.map((t) => cacheValue(cache, 0, t));
}

export function sirOracle(
  params: SIRParams,
  grids: SIRGrids,
  cache: SIROracleCache,
  opts: {
    betaRange?: [number, number];
    gammaRange?: [number, number];
    sigmaObs?: number;
  } = {},
): SIROracleResult {
  const betaRange = opts.betaRange ?? [grids.betaGrid[0], grids.betaGrid[grids.betaGrid.length - 1]];
  const gammaRange = opts.gammaRange ?? [grids.gammaGrid[0], grids.gammaGrid[grids.gammaGrid.length - 1]];
  const sigmaObs = opts.sigmaObs ?? SIR.SIGMA_OBS;
  const nBeta = grids.betaGrid.length;
  const nGamma = grids.gammaGrid.length;
  const nGrid = nBeta * nGamma;

  const betaPrior = betaLogPriorOnGrid(grids.betaGrid, params.betaUnit, params.betaNu, betaRange[0], betaRange[1]);
  const gammaPrior = betaLogPriorOnGrid(
    grids.gammaGrid,
    params.gammaUnit,
    params.gammaNu,
    gammaRange[0],
    gammaRange[1],
  );

  const logPost = new Array<number>(nGrid);
  let maxLog = -Infinity;
  for (let bi = 0; bi < nBeta; bi++) {
    for (let gi = 0; gi < nGamma; gi++) {
      const k = bi * nGamma + gi;
      let logLike = 0;
      for (const obs of params.observations) {
        const pred = cacheValue(cache, k, obs.t);
        const z = (obs.y - pred) / sigmaObs;
        logLike += -0.5 * z * z - Math.log(sigmaObs) - HALF_LOG_2PI;
      }
      const lp = betaPrior[bi] + gammaPrior[gi] + logLike;
      logPost[k] = lp;
      if (lp > maxLog) maxLog = lp;
    }
  }

  let sum = 0;
  for (const lp of logPost) sum += Math.exp(lp - maxLog);
  const logNorm = maxLog + Math.log(sum);

  const betaPost = new Array<number>(nBeta).fill(0);
  const gammaPost = new Array<number>(nGamma).fill(0);
  const weights = new Array<number>(nGrid);
  for (let bi = 0; bi < nBeta; bi++) {
    for (let gi = 0; gi < nGamma; gi++) {
      const k = bi * nGamma + gi;
      const w = Math.exp(logPost[k] - logNorm);
      weights[k] = w;
      betaPost[bi] += w;
      gammaPost[gi] += w;
    }
  }

  const yMean = new Array<number>(grids.tGrid.length).fill(0);
  const ySecond = new Array<number>(grids.tGrid.length).fill(0);
  for (let k = 0; k < nGrid; k++) {
    const w = weights[k];
    for (let ti = 0; ti < grids.tGrid.length; ti++) {
      const y = cacheValue(cache, k, grids.tGrid[ti]);
      yMean[ti] += w * y;
      ySecond[ti] += w * (sigmaObs * sigmaObs + y * y);
    }
  }
  const yStd = yMean.map((m, i) => Math.sqrt(Math.max(ySecond[i] - m * m, 1e-12)));

  return { betaGrid: grids.betaGrid, betaPost, gammaGrid: grids.gammaGrid, gammaPost, yMean, yStd };
}
