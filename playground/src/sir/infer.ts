/**
 * Pure ACE inference for the SIR demo. Builds context from observed infected
 * fractions plus always-present Beta prior tokens for beta/gamma, then queries
 * the epidemic curve and the two latent marginals in one forward pass.
 */

import { SIR } from "../config";
import { ACEModel, PRIOR, QUERY, VALUE } from "../ace/model";
import { Predictions } from "../ace/predictions";
import { encodeValue } from "../ace/schema";
import { TokenList } from "../ace/tokens";
import { linspace, normalize } from "../util";
import { priorFeatures } from "../gaussian/infer";

export interface SIRObservation {
  t: number;
  y: number;
}

export interface SIRParams {
  observations: SIRObservation[];
  betaUnit: number;
  betaNu: number;
  gammaUnit: number;
  gammaNu: number;
}

export interface SIRGrids {
  tGrid: number[];
  betaGrid: number[];
  gammaGrid: number[];
}

export interface SIRResult {
  tGrid: number[];
  predMean: number[];
  predStd: number[];
  betaGrid: number[];
  betaPost: number[];
  gammaGrid: number[];
  gammaPost: number[];
}

export function scaleTime(t: number): number {
  return (2.0 * t) / SIR.T_DOMAIN[1] - 1.0;
}

export function scaleValue(y: number): number {
  return (y - SIR.DATA_LOC) / SIR.DATA_SCALE;
}

export function unscaleValue(v: number): number {
  return v * SIR.DATA_SCALE + SIR.DATA_LOC;
}

export function defaultSIRGrids(model: ACEModel): SIRGrids {
  return {
    tGrid: linspace(SIR.T_DOMAIN[0], SIR.T_DOMAIN[1], SIR.TIME_POINTS),
    betaGrid: linspace(model.variables[1].bound_lo, model.variables[1].bound_hi, SIR.BINS),
    gammaGrid: linspace(model.variables[2].bound_lo, model.variables[2].bound_hi, SIR.BINS),
  };
}

export function sirInfer(model: ACEModel, params: SIRParams, grids: SIRGrids): SIRResult {
  const betaMeta = model.variables[1];
  const gammaMeta = model.variables[2];

  const c = new TokenList();
  for (const obs of params.observations) c.add(0, VALUE, { x: scaleTime(obs.t), value: scaleValue(obs.y) });
  c.add(1, PRIOR, { prior: priorFeatures(params.betaUnit, params.betaNu) });
  c.add(2, PRIOR, { prior: priorFeatures(params.gammaUnit, params.gammaNu) });
  const context = c.get();

  const t = new TokenList();
  const yRange: [number, number] = [0, grids.tGrid.length];
  for (const time of grids.tGrid) t.add(0, QUERY, { x: scaleTime(time) });

  const betaRange: [number, number] = [t.varId.length, t.varId.length + grids.betaGrid.length];
  for (const b of grids.betaGrid) t.add(1, QUERY, { value: encodeValue(betaMeta, b) });

  const gammaRange: [number, number] = [t.varId.length, t.varId.length + grids.gammaGrid.length];
  for (const g of grids.gammaGrid) t.add(2, QUERY, { value: encodeValue(gammaMeta, g) });
  const target = t.get();

  const out = model.forward(context, target);
  const pred = new Predictions(model, out);

  const predMean: number[] = [];
  const predStd: number[] = [];
  for (let i = yRange[0]; i < yRange[1]; i++) {
    predMean.push(unscaleValue(pred.continuousMean(i)));
    predStd.push(Math.sqrt(Math.max(pred.continuousVar(i), 0)) * SIR.DATA_SCALE);
  }

  const logp = pred.logProb(target);
  const betaPost = normalize(logp.slice(betaRange[0], betaRange[1]));
  const gammaPost = normalize(logp.slice(gammaRange[0], gammaRange[1]));

  return {
    tGrid: grids.tGrid,
    predMean,
    predStd,
    betaGrid: grids.betaGrid,
    betaPost,
    gammaGrid: grids.gammaGrid,
    gammaPost,
  };
}
