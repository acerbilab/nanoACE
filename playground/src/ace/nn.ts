/**
 * Minimal neural-net primitives, reimplemented to match PyTorch numerically.
 *
 * Conventions: vectors are `number[]`; linear weights follow PyTorch's `[out, in]`
 * row-major layout; `multiHeadAttention` replicates `nn.MultiheadAttention`
 * (packed `in_proj` split into q/k/v, `1/sqrt(head_dim)` query scaling,
 * `key_padding_mask`, then `out_proj`).
 */

import type { Tensor } from "./weights";

// Abramowitz & Stegun 7.1.26 (max abs error ~1.5e-7) — enough for our tolerance.
export function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1.0 / (1.0 + 0.3275911 * ax);
  const y =
    1.0 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-ax * ax);
  return sign * y;
}

const SQRT2 = Math.SQRT2;

/** Exact (erf-based) GELU, matching `nn.GELU()`'s default. */
export function gelu(vec: number[]): number[] {
  const out = new Array<number>(vec.length);
  for (let i = 0; i < vec.length; i++) {
    const x = vec[i];
    out[i] = 0.5 * x * (1.0 + erf(x / SQRT2));
  }
  return out;
}

export function softplus(x: number): number {
  // log(1+exp(x)), numerically stable.
  return Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
}

export function logSumExp(vec: number[]): number {
  let m = -Infinity;
  for (const v of vec) if (v > m) m = v;
  if (m === -Infinity) return -Infinity;
  let s = 0;
  for (const v of vec) s += Math.exp(v - m);
  return m + Math.log(s);
}

export function logSoftmax(vec: number[]): number[] {
  const lse = logSumExp(vec);
  return vec.map((v) => v - lse);
}

export function softmax(vec: number[]): number[] {
  let m = -Infinity;
  for (const v of vec) if (v > m) m = v;
  const ex = vec.map((v) => Math.exp(v - m));
  let s = 0;
  for (const e of ex) s += e;
  return ex.map((e) => e / s);
}

/** y = W x + b, with W of shape [out, in] (row-major). */
export function linear(vec: number[], W: Tensor, b: Tensor | null): number[] {
  const out = W.shape[0];
  const inn = W.shape[1];
  const wd = W.data;
  const res = new Array<number>(out);
  for (let o = 0; o < out; o++) {
    let s = b ? b.data[o] : 0;
    const base = o * inn;
    for (let i = 0; i < inn; i++) s += vec[i] * wd[base + i];
    res[o] = s;
  }
  return res;
}

/** Linear -> GELU -> Linear (the `_mlp` building block in ace.py). */
export function mlp(vec: number[], w0: Tensor, b0: Tensor, w2: Tensor, b2: Tensor): number[] {
  return linear(gelu(linear(vec, w0, b0)), w2, b2);
}

export function layerNorm(vec: number[], gamma: Tensor, beta: Tensor, eps = 1e-5): number[] {
  const n = vec.length;
  let mean = 0;
  for (const v of vec) mean += v;
  mean /= n;
  let varr = 0;
  for (const v of vec) {
    const d = v - mean;
    varr += d * d;
  }
  varr /= n; // PyTorch LayerNorm uses biased variance.
  const inv = 1.0 / Math.sqrt(varr + eps);
  const g = gamma.data;
  const b = beta.data;
  const res = new Array<number>(n);
  for (let i = 0; i < n; i++) res[i] = (vec[i] - mean) * inv * g[i] + b[i];
  return res;
}

export function addInto(a: number[], b: number[]): number[] {
  const out = new Array<number>(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

/**
 * Multi-head attention matching nn.MultiheadAttention (batch_first, no attn_mask).
 *
 * `query` is [Tq][d]; `key`/`value` are [Tk][d] (same tensor for self-attention).
 * `keyPadMask[j] === true` means key j is ignored (padding). Returns [Tq][d].
 */
export function multiHeadAttention(
  query: number[][],
  key: number[][],
  value: number[][],
  keyPadMask: boolean[],
  inProjW: Tensor,
  inProjB: Tensor,
  outProjW: Tensor,
  outProjB: Tensor,
  nHeads: number,
): number[][] {
  const d = inProjW.shape[1];
  const headDim = d / nHeads;
  const scale = 1.0 / Math.sqrt(headDim);
  const w = inProjW.data;
  const bdat = inProjB.data;

  // Project once. Wq = rows[0:d], Wk = rows[d:2d], Wv = rows[2d:3d].
  const proj = (vec: number[], rowBase: number, biasBase: number): number[] => {
    const res = new Array<number>(d);
    for (let o = 0; o < d; o++) {
      let s = bdat[biasBase + o];
      const base = (rowBase + o) * d;
      for (let i = 0; i < d; i++) s += vec[i] * w[base + i];
      res[o] = s;
    }
    return res;
  };

  const Q = query.map((q) => proj(q, 0, 0));
  const K = key.map((k) => proj(k, d, d));
  const V = value.map((v) => proj(v, 2 * d, 2 * d));

  const Tq = Q.length;
  const Tk = K.length;
  const ctx: number[][] = [];
  for (let i = 0; i < Tq; i++) {
    const outVec = new Array<number>(d).fill(0);
    for (let h = 0; h < nHeads; h++) {
      const off = h * headDim;
      // scores over keys for this head.
      const scores = new Array<number>(Tk);
      for (let j = 0; j < Tk; j++) {
        if (keyPadMask[j]) {
          scores[j] = -Infinity;
          continue;
        }
        let dot = 0;
        for (let c = 0; c < headDim; c++) dot += Q[i][off + c] * K[j][off + c];
        scores[j] = dot * scale;
      }
      const attn = softmax(scores);
      for (let j = 0; j < Tk; j++) {
        const a = attn[j];
        if (a === 0) continue;
        for (let c = 0; c < headDim; c++) outVec[off + c] += a * V[j][off + c];
      }
    }
    ctx.push(linear(outVec, outProjW, outProjB));
  }
  return ctx;
}
