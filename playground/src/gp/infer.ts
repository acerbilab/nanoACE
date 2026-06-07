/**
 * Pure GP-1D inference: given context points and optional pinned latents, run a
 * single ACE forward over a combined target (predictive band + lengthscale grid +
 * outputscale grid + kernel) and return the pieces the demo plots.
 *
 * Kept free of DOM so it can be parity-tested against the Python reference.
 */

import { GP } from "../config";
import { ACEModel, PRIOR, QUERY, VALUE } from "../ace/model";
import { Predictions } from "../ace/predictions";
import { encodeValue } from "../ace/schema";
import { TokenList } from "../ace/tokens";
import { linspace, normalize } from "../util";

export interface GPSpec {
  points: { x: number; y: number }[];
  pinKernel: number | null;
  pinEll: number | null; // native log-lengthscale value when pinned, else null
  pinScale: number | null; // native log-outputscale value when pinned, else null
}

export interface GPResult {
  hasContext: boolean;
  bandX: number[];
  bandMean: number[];
  bandStd: number[];
  ellGrid: number[];
  ellPost: number[] | null; // normalized over the grid, or null when pinned
  scaleGrid: number[];
  scalePost: number[] | null;
  kernelProbs: number[] | null; // null when pinned
}

export function gpInfer(
  model: ACEModel,
  spec: GPSpec,
  opts: { bandPoints?: number; latentGrid?: number; bandX?: number[] } = {},
): GPResult {
  const ellMeta = model.variables[1];
  const scaleMeta = model.variables[2];
  const nKernel = model.variables[3].cardinality ?? 4;
  const bandPoints = opts.bandPoints ?? GP.BAND_POINTS;
  const latentGrid = opts.latentGrid ?? GP.LATENT_GRID;
  const bandX = opts.bandX ?? linspace(GP.X_DOMAIN[0], GP.X_DOMAIN[1], bandPoints);

  const ellGrid = linspace(ellMeta.bound_lo, ellMeta.bound_hi, latentGrid);
  const scaleGrid = linspace(scaleMeta.bound_lo, scaleMeta.bound_hi, latentGrid);

  // Context.
  const c = new TokenList();
  for (const p of spec.points) c.add(0, VALUE, { x: p.x, value: p.y });
  if (spec.pinKernel !== null) c.add(3, VALUE, { value: spec.pinKernel, valueIndex: spec.pinKernel });
  if (spec.pinEll !== null) {
    const e = encodeValue(ellMeta, spec.pinEll);
    c.add(1, PRIOR, { value: e, prior: [e, 0] });
  }
  if (spec.pinScale !== null) {
    const e = encodeValue(scaleMeta, spec.pinScale);
    c.add(2, PRIOR, { value: e, prior: [e, 0] });
  }
  const context = c.get();

  const empty: GPResult = {
    hasContext: false,
    bandX,
    bandMean: [],
    bandStd: [],
    ellGrid,
    ellPost: null,
    scaleGrid,
    scalePost: null,
    kernelProbs: null,
  };
  if (!context.mask.some((m) => m)) return empty;

  // Combined target: band + (ell grid) + (scale grid) + (kernel).
  const t = new TokenList();
  for (const x of bandX) t.add(0, QUERY, { x });
  const bandRange: [number, number] = [0, bandX.length];

  let ellRange: [number, number] | null = null;
  if (spec.pinEll === null) {
    ellRange = [t.varId.length, t.varId.length + latentGrid];
    for (const g of ellGrid) t.add(1, QUERY, { value: encodeValue(ellMeta, g) });
  }
  let scaleRange: [number, number] | null = null;
  if (spec.pinScale === null) {
    scaleRange = [t.varId.length, t.varId.length + latentGrid];
    for (const g of scaleGrid) t.add(2, QUERY, { value: encodeValue(scaleMeta, g) });
  }
  let kernelRow: number | null = null;
  if (spec.pinKernel === null) {
    kernelRow = t.varId.length;
    t.add(3, QUERY, {});
  }

  const target = t.get();
  const out = model.forward(context, target);
  const pred = new Predictions(model, out);

  const bandMean: number[] = [];
  const bandStd: number[] = [];
  for (let i = bandRange[0]; i < bandRange[1]; i++) {
    bandMean.push(pred.continuousMean(i));
    bandStd.push(Math.sqrt(Math.max(pred.continuousVar(i), 0)));
  }

  const logpAll = pred.logProb(target);
  const ellPost = ellRange ? normalize(logpAll.slice(ellRange[0], ellRange[1])) : null;
  const scalePost = scaleRange ? normalize(logpAll.slice(scaleRange[0], scaleRange[1])) : null;
  const kernelProbs = kernelRow !== null ? pred.categoricalProbs(kernelRow, nKernel) : null;

  return { hasContext: true, bandX, bandMean, bandStd, ellGrid, ellPost, scaleGrid, scalePost, kernelProbs };
}
