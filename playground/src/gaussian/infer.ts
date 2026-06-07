/**
 * Pure ACE inference for the Gaussian demo. Builds context (observed y + the two
 * always-present Beta prior tokens) and a combined target (mu grid + log_sigma
 * grid + predictive y grid), runs one forward, and returns the marginals and the
 * predictive density. DOM-free so it can be parity-tested.
 */

import { ACEModel, PRIOR, QUERY, VALUE } from "../ace/model";
import { Predictions } from "../ace/predictions";
import { encodeValue } from "../ace/schema";
import { TokenList } from "../ace/tokens";
import { normalize } from "../util";

/** Internal (mean, spread) features for a Beta prior — mirrors gaussian_toy.prior_features. */
export function priorFeatures(muUnit: number, nu: number): [number, number] {
  const mean = 2 * muUnit - 1;
  const spread = Math.sqrt(Math.max(1 - mean * mean, 0) / (nu + 1));
  return [mean, spread];
}

export interface GaussParams {
  yObs: number[];
  muUnit: number;
  muNu: number;
  lsUnit: number;
  lsNu: number;
}

export interface GaussGrids {
  muGrid: number[]; // native mu grid
  lsGrid: number[]; // native log_sigma grid
  yGrid: number[]; // predictive y grid
}

export interface GaussResult {
  muGrid: number[];
  muPost: number[];
  lsGrid: number[];
  lsPost: number[];
  yGrid: number[];
  predDensity: number[];
}

export function gaussInfer(model: ACEModel, params: GaussParams, grids: GaussGrids): GaussResult {
  const muMeta = model.variables[1];
  const lsMeta = model.variables[2];

  const c = new TokenList();
  for (const y of params.yObs) c.add(0, VALUE, { value: y });
  c.add(1, PRIOR, { prior: priorFeatures(params.muUnit, params.muNu) });
  c.add(2, PRIOR, { prior: priorFeatures(params.lsUnit, params.lsNu) });
  const context = c.get();

  const t = new TokenList();
  const muRange: [number, number] = [0, grids.muGrid.length];
  for (const g of grids.muGrid) t.add(1, QUERY, { value: encodeValue(muMeta, g) });
  const lsRange: [number, number] = [t.varId.length, t.varId.length + grids.lsGrid.length];
  for (const g of grids.lsGrid) t.add(2, QUERY, { value: encodeValue(lsMeta, g) });
  const yRange: [number, number] = [t.varId.length, t.varId.length + grids.yGrid.length];
  for (const y of grids.yGrid) t.add(0, QUERY, { value: y });
  const target = t.get();

  const out = model.forward(context, target);
  const pred = new Predictions(model, out);
  const logp = pred.logProb(target);

  const muPost = normalize(logp.slice(muRange[0], muRange[1]));
  const lsPost = normalize(logp.slice(lsRange[0], lsRange[1]));
  const predDensity = logp.slice(yRange[0], yRange[1]).map((v) => Math.exp(v));

  return {
    muGrid: grids.muGrid,
    muPost,
    lsGrid: grids.lsGrid,
    lsPost,
    yGrid: grids.yGrid,
    predDensity,
  };
}
