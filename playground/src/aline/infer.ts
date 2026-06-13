/**
 * ALINE tab inference orchestration (DOM-free, parity-backed by the chain
 * fixture's semantics).
 *
 * One `forwardWithStates` per step carries every read in a single forward —
 * target rows are mutually independent (targets never attend to each other),
 * so extra rows change nothing per-row:
 *
 *   [ goal rows (ξ) | extra latent rows | band grid rows | candidate rows ]
 *
 * - goal rows: the active ξ — selected latent QUERYs and/or the predictive
 *   x* QUERYs. ONLY these are sliced as the policy's target key set, exactly
 *   reproducing Python's `key_padding_mask = ~ξ` over the target superset.
 * - extra latent rows: any of the three latents not in ξ, appended so latent
 *   marginals and the log q(θ_true) metric are always available.
 * - band grid rows: the predictive band/metric grid.
 * - candidate rows: the available pool, read for per-candidate predictive
 *   variance (the uncertainty-sampling comparison marker) — bit-identical to
 *   running them as their own target set.
 *
 * The same candidate locations are dual-pathed: target rows give variance,
 * query tokens feed `policyLogits` (scored pointwise against the goal rows).
 */

import { ALINEModel } from "../ace/aline";
import { QUERY, VALUE } from "../ace/model";
import { softmax } from "../ace/nn";
import { Predictions } from "../ace/predictions";
import { encodeValue } from "../ace/schema";
import { TokenList } from "../ace/tokens";
import { linspace, normalize } from "../util";

export interface Goal {
  pred: boolean;
  ell: boolean;
  scale: boolean;
  kernel: boolean;
}

export function goalActive(g: Goal): boolean {
  return g.pred || g.ell || g.scale || g.kernel;
}

export interface Obs {
  x: number;
  y: number;
}

export interface AlineInputs {
  obs: Obs[]; // observed data points (>= 1; ACE needs an active context token)
  candX: number[]; // available candidate locations (the action space)
  goal: Goal; // at least one component true
  xStar: number[]; // predictive-target locations (enter ξ iff goal.pred)
  gridX: number[]; // band/metric display grid
  latentGrid?: number; // marginal density resolution (default 64)
}

export interface Truth {
  gridY: number[]; // hidden function values on gridX
  logEll: number;
  logScale: number;
  kernel: number;
}

export interface Density {
  grid: number[]; // native coordinates
  probs: number[]; // normalized over the grid
}

export interface AlineStep {
  bandMean: number[];
  bandStd: number[];
  candVar: number[]; // predictive variance per candidate (US read)
  logits: number[]; // policy logits per candidate
  policyProbs: number[]; // softmax(logits)
  argmaxIdx: number; // ALINE's pick (index into candX)
  usIdx: number; // uncertainty sampling's pick (index into candX)
  ell: Density;
  scale: Density;
  kernelProbs: number[];
  // Present iff truth given: predictive RMSE on gridX + per-latent log q(θ_true)
  // (ξ-independent instruments, so metric series stay comparable across goal switches).
  metrics?: { rmse: number; logq: { ell: number; scale: number; kernel: number } };
}

/** Index of the available candidate nearest to x (the click-snap rule). */
export function nearestCandidate(candX: number[], x: number): number {
  let best = 0;
  for (let i = 1; i < candX.length; i++) {
    if (Math.abs(candX[i] - x) < Math.abs(candX[best] - x)) best = i;
  }
  return best;
}

export function alineStep(model: ALINEModel, inp: AlineInputs, truth?: Truth): AlineStep {
  if (inp.obs.length === 0) throw new Error("alineStep needs at least one observation");
  if (!goalActive(inp.goal)) throw new Error("alineStep needs a non-empty goal");
  if (inp.candX.length === 0) throw new Error("alineStep needs at least one candidate");
  const nGrid = inp.latentGrid ?? 64;

  const ctx = new TokenList();
  for (const o of inp.obs) ctx.add(0, VALUE, { x: o.x, value: o.y });

  // Target layout: goal rows | extra latent rows | band rows | candidate rows.
  const tgt = new TokenList();
  const goalRows: number[] = [];
  let row = 0;
  let ellRow = -1;
  let scaleRow = -1;
  let kernelRow = -1;
  if (inp.goal.ell) {
    ellRow = row;
    goalRows.push(row++);
    tgt.add(1, QUERY);
  }
  if (inp.goal.scale) {
    scaleRow = row;
    goalRows.push(row++);
    tgt.add(2, QUERY);
  }
  if (inp.goal.kernel) {
    kernelRow = row;
    goalRows.push(row++);
    tgt.add(3, QUERY);
  }
  if (inp.goal.pred) {
    for (const xs of inp.xStar) {
      goalRows.push(row++);
      tgt.add(0, QUERY, { x: xs });
    }
  }
  if (ellRow < 0) {
    ellRow = row++;
    tgt.add(1, QUERY);
  }
  if (scaleRow < 0) {
    scaleRow = row++;
    tgt.add(2, QUERY);
  }
  if (kernelRow < 0) {
    kernelRow = row++;
    tgt.add(3, QUERY);
  }
  const bandStart = row;
  for (const xg of inp.gridX) {
    tgt.add(0, QUERY, { x: xg });
    row++;
  }
  const usStart = row;
  for (const xc of inp.candX) {
    tgt.add(0, QUERY, { x: xc });
    row++;
  }

  const { out, ctxStates, tgtStates } = model.forwardWithStates(ctx.get(), tgt.get());
  const pred = new Predictions(model, out);

  const bandMean = new Array<number>(inp.gridX.length);
  const bandStd = new Array<number>(inp.gridX.length);
  for (let i = 0; i < inp.gridX.length; i++) {
    bandMean[i] = pred.continuousMean(bandStart + i);
    bandStd[i] = Math.sqrt(Math.max(pred.continuousVar(bandStart + i), 0));
  }

  const candVar = inp.candX.map((_, i) => pred.continuousVar(usStart + i));
  let usIdx = 0;
  for (let i = 1; i < candVar.length; i++) if (candVar[i] > candVar[usIdx]) usIdx = i;

  // The policy reads ONLY the goal rows' states (ξ), never the helper rows.
  const goalStates = goalRows.map((r) => tgtStates[r]);
  const qry = new TokenList();
  for (const xc of inp.candX) qry.add(0, QUERY, { x: xc });
  const logits = model.policyLogits(qry.get(), ctxStates, goalStates);
  const policyProbs = softmax(logits);
  let argmaxIdx = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[argmaxIdx]) argmaxIdx = i;

  const ell = latentDensity(model, pred, ellRow, 1, nGrid);
  const scale = latentDensity(model, pred, scaleRow, 2, nGrid);
  const kernelCard = model.variables[3].cardinality ?? 4;
  const kernelProbs = pred.categoricalProbs(kernelRow, kernelCard);

  let metrics: AlineStep["metrics"];
  if (truth) {
    let se = 0;
    for (let i = 0; i < inp.gridX.length; i++) {
      const d = bandMean[i] - truth.gridY[i];
      se += d * d;
    }
    const rmse = Math.sqrt(se / inp.gridX.length);
    const lpEll = pred.logProbContinuous(ellRow, encodeValue(model.variables[1], truth.logEll));
    const lpScale = pred.logProbContinuous(scaleRow, encodeValue(model.variables[2], truth.logScale));
    const lpKernel = Math.log(Math.max(kernelProbs[truth.kernel] ?? 0, 1e-300));
    metrics = { rmse, logq: { ell: lpEll, scale: lpScale, kernel: lpKernel } };
  }

  return { bandMean, bandStd, candVar, logits, policyProbs, argmaxIdx, usIdx, ell, scale, kernelProbs, metrics };
}

function latentDensity(
  model: ALINEModel,
  pred: Predictions,
  row: number,
  varId: number,
  n: number,
): Density {
  const meta = model.variables[varId];
  const grid = linspace(meta.bound_lo, meta.bound_hi, n);
  const logp = grid.map((v) => pred.logProbContinuous(row, encodeValue(meta, v)));
  return { grid, probs: normalize(logp) };
}
