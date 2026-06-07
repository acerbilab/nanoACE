/**
 * Analytic grid oracle for the Gaussian (μ, log σ) toy — a TS port of
 * `gaussian_toy.analytic_posterior` + `predictive_grid` + the Beta log-prior.
 * Cheap (a bins×bins grid, no Cholesky), so it overlays the ACE posterior live.
 */

const HALF_LOG_2PI = 0.5 * Math.log(2.0 * Math.PI);

// Lanczos approximation to log Γ(x) (g=7), accurate to ~1e-13 for x>0.
const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
  1.5056327351493116e-7,
];

export function lgamma(x: number): number {
  if (x < 0.5) {
    // Reflection: Γ(x)Γ(1-x) = π / sin(πx)
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  }
  x -= 1;
  let a = LANCZOS[0];
  const t = x + 7.5;
  for (let i = 1; i < LANCZOS.length; i++) a += LANCZOS[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

function betaLogB(alpha: number, beta: number): number {
  return lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta);
}

/** Native-coordinate Beta prior log-density on a grid (mirrors beta_logprior_on_grid). */
export function betaLogPriorOnGrid(
  gridNative: number[],
  muUnit: number,
  nu: number,
  lo: number,
  hi: number,
): number[] {
  const width = hi - lo;
  const alpha = muUnit * nu;
  const beta = (1 - muUnit) * nu;
  const logB = betaLogB(alpha, beta);
  return gridNative.map((g) => {
    let u = (g - lo) / width;
    u = Math.min(Math.max(u, 1e-6), 1 - 1e-6);
    return (alpha - 1) * Math.log(u) + (beta - 1) * Math.log(1 - u) - logB - Math.log(width);
  });
}

export interface GaussPriorParams {
  muUnit: number;
  muNu: number;
  lsUnit: number;
  lsNu: number;
}

export interface OracleResult {
  muGrid: number[];
  lsGrid: number[];
  post: number[][]; // [mu][logsig], normalized
  muPost: number[]; // marginal over mu
  lsPost: number[]; // marginal over log_sigma
}

/** Exact grid posterior over (mu, log_sigma) given observed y and Beta priors. */
export function analyticPosterior(
  yObs: number[],
  muGrid: number[],
  lsGrid: number[],
  muRange: [number, number],
  lsRange: [number, number],
  pp: GaussPriorParams,
): OracleResult {
  const nMu = muGrid.length;
  const nLs = lsGrid.length;
  const sigma = lsGrid.map((v) => Math.exp(v));
  const logPriorMu = betaLogPriorOnGrid(muGrid, pp.muUnit, pp.muNu, muRange[0], muRange[1]);
  const logPriorLs = betaLogPriorOnGrid(lsGrid, pp.lsUnit, pp.lsNu, lsRange[0], lsRange[1]);

  const logpost: number[][] = [];
  let maxLog = -Infinity;
  for (let i = 0; i < nMu; i++) {
    const rowarr: number[] = new Array(nLs);
    for (let j = 0; j < nLs; j++) {
      let loglike = 0;
      for (const y of yObs) {
        const z = (y - muGrid[i]) / sigma[j];
        loglike += -0.5 * z * z - Math.log(sigma[j]) - HALF_LOG_2PI;
      }
      const lp = logPriorMu[i] + logPriorLs[j] + loglike;
      rowarr[j] = lp;
      if (lp > maxLog) maxLog = lp;
    }
    logpost.push(rowarr);
  }

  let sum = 0;
  for (let i = 0; i < nMu; i++) for (let j = 0; j < nLs; j++) sum += Math.exp(logpost[i][j] - maxLog);
  const logNorm = maxLog + Math.log(sum);

  const post: number[][] = [];
  const muPost = new Array<number>(nMu).fill(0);
  const lsPost = new Array<number>(nLs).fill(0);
  for (let i = 0; i < nMu; i++) {
    const rowarr: number[] = new Array(nLs);
    for (let j = 0; j < nLs; j++) {
      const p = Math.exp(logpost[i][j] - logNorm);
      rowarr[j] = p;
      muPost[i] += p;
      lsPost[j] += p;
    }
    post.push(rowarr);
  }
  return { muGrid, lsGrid, post, muPost, lsPost };
}

/** Posterior predictive density at each y: Σ post[i,j] N(y | mu_i, sigma_j). */
export function predictiveDensity(oracle: OracleResult, yGrid: number[]): number[] {
  const sigma = oracle.lsGrid.map((v) => Math.exp(v));
  return yGrid.map((y) => {
    let d = 0;
    for (let i = 0; i < oracle.muGrid.length; i++) {
      for (let j = 0; j < oracle.lsGrid.length; j++) {
        const z = (y - oracle.muGrid[i]) / sigma[j];
        d += oracle.post[i][j] * (Math.exp(-0.5 * z * z) / (sigma[j] * Math.sqrt(2 * Math.PI)));
      }
    }
    return d;
  });
}
