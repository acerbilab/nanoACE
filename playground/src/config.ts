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

// Display labels for the discrete kernel latent (order matches gp1d.KERNELS).
export const KERNEL_LABELS = ["RBF", "Matérn-½", "Matérn-3/2", "Periodic"];
