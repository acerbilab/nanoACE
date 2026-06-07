/**
 * Predictive distributions for a target token set — a port of `Predictions` in
 * `ace.py`. Continuous targets use the shared MDN; discrete targets use the shared
 * categorical head with per-variable cardinality masking. Dispatch is by `var_id`.
 *
 * All quantities are in token coordinates; callers decode bounded latents or
 * normalize over a native grid as needed (see schema.ts / the demos).
 */

import { logSoftmax, logSumExp, softmax, softplus } from "./nn";
import type { ACEModel, ForwardOut, TokenSet } from "./model";

const HALF_LOG_2PI = 0.5 * Math.log(2.0 * Math.PI);

export interface MDNParams {
  logW: number[];
  loc: number[];
  scale: number[];
}

export class Predictions {
  constructor(
    private model: ACEModel,
    private out: ForwardOut,
  ) {}

  get components(): number {
    return this.model.mdnComponents;
  }

  contParams(row: number): MDNParams {
    const k = this.components;
    const raw = this.out.contRaw[row];
    const logW = logSoftmax(raw.slice(0, k));
    const loc = raw.slice(k, 2 * k);
    const scale = raw.slice(2 * k, 3 * k).map((s) => softplus(s) + this.model.minScale);
    return { logW, loc, scale };
  }

  continuousMean(row: number): number {
    const { logW, loc } = this.contParams(row);
    let m = 0;
    for (let i = 0; i < loc.length; i++) m += Math.exp(logW[i]) * loc[i];
    return m;
  }

  continuousVar(row: number): number {
    const { logW, loc, scale } = this.contParams(row);
    let mean = 0;
    for (let i = 0; i < loc.length; i++) mean += Math.exp(logW[i]) * loc[i];
    let v = 0;
    for (let i = 0; i < loc.length; i++) {
      const w = Math.exp(logW[i]);
      v += w * (scale[i] * scale[i] + (loc[i] - mean) * (loc[i] - mean));
    }
    return v;
  }

  /** MDN log density at a continuous value y (token coordinates). */
  logProbContinuous(row: number, y: number): number {
    const { logW, loc, scale } = this.contParams(row);
    const terms = new Array<number>(loc.length);
    for (let i = 0; i < loc.length; i++) {
      const z = (y - loc[i]) / scale[i];
      terms[i] = logW[i] - 0.5 * z * z - Math.log(scale[i]) - HALF_LOG_2PI;
    }
    return logSumExp(terms);
  }

  /** Logits masked to a variable's local label set (length = max_cardinality). */
  private validLogits(row: number, card: number): number[] {
    const logits = this.out.discLogits[row].slice();
    for (let j = 0; j < logits.length; j++) if (j >= card) logits[j] = -Infinity;
    return logits;
  }

  /** Posterior probabilities over a discrete variable's classes (length = card). */
  categoricalProbs(row: number, card: number): number[] {
    return softmax(this.validLogits(row, card)).slice(0, card);
  }

  /** Per-token log probability, dispatched by variable type (mirrors log_prob). */
  logProb(target: TokenSet): number[] {
    const T = target.varId.length;
    const out = new Array<number>(T);
    for (let t = 0; t < T; t++) {
      const meta = this.model.variables[target.varId[t]];
      if (meta.is_discrete) {
        const card = meta.cardinality ?? 1;
        const lp = logSoftmax(this.validLogits(t, card));
        out[t] = lp[Math.max(target.valueIndex[t], 0)];
      } else {
        out[t] = this.logProbContinuous(t, target.value[t]);
      }
    }
    return out;
  }

  /** Per-token predictive mean (continuous mixture mean, or discrete E[label]). */
  mean(target: TokenSet): number[] {
    const T = target.varId.length;
    const out = new Array<number>(T);
    for (let t = 0; t < T; t++) {
      const meta = this.model.variables[target.varId[t]];
      if (meta.is_discrete) {
        const card = meta.cardinality ?? 1;
        const probs = this.categoricalProbs(t, card);
        let m = 0;
        for (let j = 0; j < probs.length; j++) m += probs[j] * j;
        out[t] = m;
      } else {
        out[t] = this.continuousMean(t);
      }
    }
    return out;
  }
}
