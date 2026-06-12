/**
 * Tunable demo constants — all the "magic numbers" live here so they are easy to
 * find and adjust. OOD thresholds are deliberately generous (the models saw
 * stochastic tails outside [-1, 1]); see the plan's guardrail discussion.
 */

export const GP = {
  X_DOMAIN: [-1, 1] as [number, number], // function domain (x is hard-clamped here)
  Y_VIEW: [-2.6, 2.6] as [number, number], // visible y range (wider than training)
  Y_NORMAL: [-1, 1] as [number, number], // "in training range" band, lightly shaded
  Y_OOD: 2.0, // |y| beyond this flags a point as out-of-distribution
  MAX_CONTEXT_HINT: 14, // training max_context; more points = soft OOD hint
  BAND_POINTS: 161, // predictive-band resolution
  LATENT_GRID: 80, // resolution of the lengthscale/outputscale marginals
  HIT_RADIUS_PX: 12, // click radius for grabbing/deleting a point
};

export const GAUSSIAN = {
  Y_VIEW: [-3.5, 3.5] as [number, number],
  Y_NORMAL: [-1.5, 1.5] as [number, number],
  Y_OOD: 3.0,
  BINS: 80, // latent grid + oracle 2D grid resolution (per axis)
  Y_POINTS: 161, // predictive-density resolution over y
  NU_RANGE: [2, 1000] as [number, number], // Beta concentration slider (log scale)
};

export const SIR = {
  T_DOMAIN: [0, 40] as [number, number],
  Y_VIEW: [-0.04, 0.62] as [number, number],
  Y_NORMAL: [0, 0.45] as [number, number],
  Y_OOD: [-0.08, 0.75] as [number, number],
  MAX_CONTEXT_HINT: 12,
  MIN_CONTEXT_HINT: 4,
  BINS: 48, // beta/gamma oracle grid resolution per axis
  TIME_POINTS: 121,
  FINE_STEPS: 400,
  SIGMA_OBS: 0.02,
  DATA_LOC: 0.2,
  DATA_SCALE: 0.2,
  NU_RANGE: [2, 1000] as [number, number], // Beta concentration slider (log scale)
  HIT_RADIUS_PX: 12,
};

export const BO = {
  X_DOMAIN: [-1, 1] as [number, number],
  Y_RANGE: [-1, 2] as [number, number], // bo1d.Y_RANGE: data-y scaling and y_opt latent bounds
  Y_OPT_RANGE: [-1, 0] as [number, number], // bo1d.Y_OPT_RANGE: prior support for optimum value
  Y_VIEW: [-1.35, 2.45] as [number, number],
  Y_NORMAL: [-1, 2] as [number, number],
  Y_OOD: [-1.15, 2.25] as [number, number],
  MAX_CONTEXT_HINT: 12,
  MIN_CONTEXT_HINT: 1,
  BAND_POINTS: 161,
  LATENT_GRID: 80,
  NU_RANGE: [2, 1000] as [number, number],
  HIT_RADIUS_PX: 12,
};

export const ARBUF = {
  X_DOMAIN: [-1, 1] as [number, number],
  Y_VIEW: [-2.6, 2.6] as [number, number],
  Y_NORMAL: [-1, 1] as [number, number],
  Y_OOD: 2.0,
  // The frozen base encoder trained at <= 14 context points; the fine-tune *saw*
  // (but could not learn from) contexts up to 20, so the hint stays conservative.
  MAX_CONTEXT_HINT: 14,
  DRAWS: 3, // coherent draw streams (and independent-sample lines)
  SLOW_DRAWS: 1, // draws in slow-AR mode (re-encoding is ~50x the per-draw cost; timing line compares per draw)
  GRID_POINTS: 64, // fixed chain length = the retained fine-tune's K=64 (prefixes 0..63 all trained)
  STEPS_PER_FRAME: 1, // decode steps per animation frame (64 steps ≈ one second)
  HIT_RADIUS_PX: 12,
};

export const ALINE = {
  X_DOMAIN: [-1, 1] as [number, number],
  Y_VIEW: [-2.6, 2.6] as [number, number],
  Y_NORMAL: [-1, 1] as [number, number],
  Y_OOD: 2.0,
  POOL: 128, // candidate pool per episode (the training value)
  T: 16, // acquisition budget = the training episode length (counts ALL observations after the seed)
  GRID: 64, // band + RMSE-metric grid resolution
  M_PRED: 32, // predictive-target x* per episode (the training value)
  LATENT_GRID: 64, // latent marginal density resolution
  // Episode contexts trained at 1 seed + T=16 ≤ 17 points; the gp1d warm start's
  // own ceiling is n_context ≤ 20 — soft and hard OOD hints respectively.
  CONTEXT_SOFT: 17,
  CONTEXT_HARD: 20,
  POLICY_AMP: 0.2, // bottom policy-pmf overlay amplitude (fraction of the y view)
  SNAP_EPS: 0.02, // your-own-data mode: candidates this close to an observation are unavailable
  HIT_RADIUS_PX: 12,
};

// Display labels for the discrete kernel latent (order matches gp1d.KERNELS).
export const KERNEL_LABELS = ["RBF", "Matérn-½", "Matérn-3/2", "Periodic"];
