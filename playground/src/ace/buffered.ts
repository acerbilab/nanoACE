/**
 * Causal AR-buffer inference in TypeScript — a port of `extensions/arbuffer/`'s
 * `BufferedACE` *incremental* path (`encode_context` + the per-step loop of
 * `sample_joint`) in its **concat-read** form: the target read is ONE softmax
 * through the frozen base `cross_attn` over the concatenated
 * `[context, realized buffer]` keys, with a learned per-head logit bias
 * (`buf_blocks.*.buf_bias`) added to the buffer keys as a soft gate. With an
 * empty buffer the read reduces exactly to the base cross-attention, so there
 * is no step-0 special case. The inherited plain forward stays `ACEModel`'s,
 * untouched. Separate-read checkpoints (the earlier `tgt_buf_attn` variant) are
 * rejected at load with a clear error — the port follows the retained
 * architecture only, and the parity fixtures cover exactly what ships.
 *
 * Two deliberate differences from the Python inference code, same math:
 * - **Projected K/V are cached**, not LayerNorm'd hidden states. Python recomputes
 *   KV projections per attention call ("a micro-opt at nano scale" under torch);
 *   in scalar JS that reprojection is O(K²·d²) over a chain. Caches: per layer,
 *   context K/V for the target read (over `kv_ln(ctx_out)`) and for the buffer
 *   stream (over `buf_ln1(ctx_in)`), built once per context; per draw,
 *   append-only buffer K/V for `buf_attn` (over `buf_ln1` of each token's layer
 *   input) and for the target read (over `kv_ln` of each token's updated state).
 * - **No masks.** Incremental decode is causal by construction (you only attend
 *   to what is already cached), and the context must be dense and all-active —
 *   the demos always build dense token lists, so the padding path is asserted
 *   away, not handled. The per-head buffer bias is applied directly to the
 *   buffer keys' attention logits instead of through a float attn_mask.
 */

import { addInto, layerNorm, linear, logSoftmax, logSumExp, mlp, multiHeadAttention, softmax, softplus } from "./nn";
import { ACEModel, QUERY, VALUE, type TokenSet } from "./model";
import type { MDNParams } from "./predictions";
import type { Tensor, Weights } from "./weights";

const HALF_LOG_2PI = 0.5 * Math.log(2.0 * Math.PI);

/** MDN log density at y — the standalone counterpart of Predictions.logProbContinuous. */
export function mdnLogProb(params: MDNParams, y: number): number {
  const terms = new Array<number>(params.loc.length);
  for (let i = 0; i < params.loc.length; i++) {
    const z = (y - params.loc[i]) / params.scale[i];
    terms[i] = params.logW[i] - 0.5 * z * z - Math.log(params.scale[i]) - HALF_LOG_2PI;
  }
  return logSumExp(terms);
}

/** Project `vec` through one third of a packed in_proj: part 0 = Q, 1 = K, 2 = V. */
function projPart(vec: number[], W: Tensor, b: Tensor, part: 0 | 1 | 2): number[] {
  const d = W.shape[1];
  const rowBase = part * d;
  const wd = W.data;
  const bd = b.data;
  const out = new Array<number>(d);
  for (let o = 0; o < d; o++) {
    let s = bd[rowBase + o];
    const base = (rowBase + o) * d;
    for (let i = 0; i < d; i++) s += vec[i] * wd[base + i];
    out[o] = s;
  }
  return out;
}

/**
 * Single-query multi-head attention over already-projected K/V caches. Keys from
 * index `biasFrom` onward get `bias[head]` added to their pre-softmax logits —
 * the concat read's soft gate (equivalent to Python's float attn_mask carrying
 * `buf_bias` on the buffer columns).
 */
function attendCached(
  q: number[],
  keys: number[][],
  vals: number[][],
  nHeads: number,
  outProjW: Tensor,
  outProjB: Tensor,
  biasFrom: number = Number.POSITIVE_INFINITY,
  bias: Tensor | null = null,
): number[] {
  const d = q.length;
  const headDim = d / nHeads;
  const scale = 1.0 / Math.sqrt(headDim);
  const Tk = keys.length;
  const outVec = new Array<number>(d).fill(0);
  for (let h = 0; h < nHeads; h++) {
    const off = h * headDim;
    const headBias = bias ? bias.data[h] : 0;
    const scores = new Array<number>(Tk);
    for (let j = 0; j < Tk; j++) {
      let dot = 0;
      for (let c = 0; c < headDim; c++) dot += q[off + c] * keys[j][off + c];
      scores[j] = dot * scale + (j >= biasFrom ? headBias : 0);
    }
    const attn = softmax(scores);
    for (let j = 0; j < Tk; j++) {
      const a = attn[j];
      if (a === 0) continue;
      for (let c = 0; c < headDim; c++) outVec[off + c] += a * vals[j][off + c];
    }
  }
  return linear(outVec, outProjW, outProjB);
}

interface LayerCtxCache {
  crossK: number[][]; // target read of ctx: K/V of kv_ln(ctx_out) under cross_attn
  crossV: number[][];
  bufCtxK: number[][]; // buffer read of ctx: K/V of buf_ln1(ctx_in) under buf_attn
  bufCtxV: number[][];
}

export interface CtxCache {
  layers: LayerCtxCache[];
  /** Raw per-layer context states (inputs entering / outputs leaving each layer);
   *  kept for the parity tests, mirroring Python's ContextCache. */
  inputs: number[][][];
  outputs: number[][][];
  n: number;
}

interface DrawLayerCache {
  selfK: number[][]; // buf_attn K/V of buf_ln1(token layer input)
  selfV: number[][];
  readK: number[][]; // target-read K/V of kv_ln(token updated state) under cross_attn
  readV: number[][];
}

export interface DrawState {
  layers: DrawLayerCache[];
  k: number;
}

export interface StepOut {
  params: MDNParams;
  raw: number[]; // cont_head output, for parity tests
  layers: number[][]; // per-layer target state, for parity tests
}

export class BufferedACEModel extends ACEModel {
  private bw: Weights;

  constructor(weights: Weights) {
    super(weights);
    this.bw = weights;
    const concat = weights.manifest.tensors.some((t) => t.name === "buf_blocks.0.buf_bias");
    if (!concat) {
      throw new Error(
        "this blob was exported from a separate-read AR-buffer checkpoint; " +
          "the TS port implements the concat-read architecture only — re-export a --concat-read checkpoint",
      );
    }
  }

  private bt(name: string): Tensor {
    return this.bw.get(name);
  }

  private embedOne(varId: number, mode: number, x: number, value: number): number[] {
    const tokens: TokenSet = {
      varId: [varId],
      x: [[x]],
      value: [value],
      valueIndex: [0],
      prior: [[0, 0]],
      mode: [mode],
      mask: [true],
    };
    return this.embed(tokens)[0];
  }

  /** Run the frozen context stream once; cache per-layer states and projected K/V. */
  encodeContext(context: TokenSet): CtxCache {
    if (context.mask.length === 0 || context.mask.some((m) => !m)) {
      throw new Error("buffered path needs a dense, all-active, non-empty context");
    }
    let ctx = this.embed(context);
    const keyPad = context.mask.map(() => false);
    const layers: LayerCtxCache[] = [];
    const inputs: number[][][] = [];
    const outputs: number[][][] = [];

    for (let i = 0; i < this.nLayers; i++) {
      const p = `blocks.${i}.`;
      const bp = `buf_blocks.${i}.`;
      inputs.push(ctx.map((r) => r.slice()));

      // Base context update — mirrors ACEModel.forward / ACEBlock. The per-block
      // mask zeroing is the identity here (all tokens active).
      const ln1 = ctx.map((r) => layerNorm(r, this.bt(p + "ctx_ln1.weight"), this.bt(p + "ctx_ln1.bias")));
      const att = multiHeadAttention(
        ln1, ln1, ln1, keyPad,
        this.bt(p + "ctx_attn.in_proj_weight"), this.bt(p + "ctx_attn.in_proj_bias"),
        this.bt(p + "ctx_attn.out_proj.weight"), this.bt(p + "ctx_attn.out_proj.bias"),
        this.nHeads,
      );
      ctx = ctx.map((r, k) => addInto(r, att[k]));
      ctx = ctx.map((r) =>
        addInto(r, mlp(
          layerNorm(r, this.bt(p + "ctx_ln2.weight"), this.bt(p + "ctx_ln2.bias")),
          this.bt(p + "ctx_mlp.0.weight"), this.bt(p + "ctx_mlp.0.bias"),
          this.bt(p + "ctx_mlp.2.weight"), this.bt(p + "ctx_mlp.2.bias"),
        )),
      );
      outputs.push(ctx.map((r) => r.slice()));

      const crossW = this.bt(p + "cross_attn.in_proj_weight");
      const crossB = this.bt(p + "cross_attn.in_proj_bias");
      const kvLn = ctx.map((r) => layerNorm(r, this.bt(p + "kv_ln.weight"), this.bt(p + "kv_ln.bias")));
      const bufW = this.bt(bp + "buf_attn.in_proj_weight");
      const bufB = this.bt(bp + "buf_attn.in_proj_bias");
      const bufLn = inputs[i].map((r) => layerNorm(r, this.bt(bp + "buf_ln1.weight"), this.bt(bp + "buf_ln1.bias")));
      layers.push({
        crossK: kvLn.map((r) => projPart(r, crossW, crossB, 1)),
        crossV: kvLn.map((r) => projPart(r, crossW, crossB, 2)),
        bufCtxK: bufLn.map((r) => projPart(r, bufW, bufB, 1)),
        bufCtxV: bufLn.map((r) => projPart(r, bufW, bufB, 2)),
      });
    }
    return { layers, inputs, outputs, n: context.mask.length };
  }

  /** Fresh per-draw buffer caches (one per coherent draw stream). */
  newDraw(): DrawState {
    return {
      layers: Array.from({ length: this.nLayers }, () => ({ selfK: [], selfV: [], readK: [], readV: [] })),
      k: 0,
    };
  }

  /** One decode step: predictive MDN at x given the cached context + this draw's buffer. */
  predict(cache: CtxCache, draw: DrawState, x: number, varIndex = 0): StepOut {
    let t = this.embedOne(varIndex, QUERY, x, 0);
    const perLayer: number[][] = [];

    for (let i = 0; i < this.nLayers; i++) {
      const p = `blocks.${i}.`;
      const bp = `buf_blocks.${i}.`;
      const L = cache.layers[i];
      const dl = draw.layers[i];

      // ONE softmax over [context, realized buffer] through the frozen base
      // cross-attention; the per-head bias soft-gates the buffer keys. With an
      // empty buffer this is exactly the base context read.
      const q = layerNorm(t, this.bt(p + "tgt_ln1.weight"), this.bt(p + "tgt_ln1.bias"));
      const qp = projPart(q, this.bt(p + "cross_attn.in_proj_weight"), this.bt(p + "cross_attn.in_proj_bias"), 0);
      t = addInto(t, attendCached(
        qp,
        L.crossK.concat(dl.readK),
        L.crossV.concat(dl.readV),
        this.nHeads,
        this.bt(p + "cross_attn.out_proj.weight"), this.bt(p + "cross_attn.out_proj.bias"),
        L.crossK.length,
        this.bt(bp + "buf_bias"),
      ));

      t = addInto(t, mlp(
        layerNorm(t, this.bt(p + "tgt_ln2.weight"), this.bt(p + "tgt_ln2.bias")),
        this.bt(p + "tgt_mlp.0.weight"), this.bt(p + "tgt_mlp.0.bias"),
        this.bt(p + "tgt_mlp.2.weight"), this.bt(p + "tgt_mlp.2.bias"),
      ));
      perLayer.push(t.slice());
    }

    const norm = layerNorm(t, this.bt("final_norm.weight"), this.bt("final_norm.bias"));
    const raw = mlp(norm,
      this.bt("cont_head.0.weight"), this.bt("cont_head.0.bias"),
      this.bt("cont_head.2.weight"), this.bt("cont_head.2.bias"));
    const k = this.mdnComponents;
    const params: MDNParams = {
      logW: logSoftmax(raw.slice(0, k)),
      loc: raw.slice(k, 2 * k),
      scale: raw.slice(2 * k, 3 * k).map((s) => softplus(s) + this.minScale),
    };
    return { params, raw, layers: perLayer };
  }

  /** Push a realized (x, y) through the buffer stream, growing this draw's caches. */
  append(cache: CtxCache, draw: DrawState, x: number, y: number, varIndex = 0): { layers: number[][] } {
    let h = this.embedOne(varIndex, VALUE, x, y);
    const perLayer: number[][] = [];

    for (let i = 0; i < this.nLayers; i++) {
      const p = `blocks.${i}.`;
      const bp = `buf_blocks.${i}.`;
      const L = cache.layers[i];
      const dl = draw.layers[i];
      const bufW = this.bt(bp + "buf_attn.in_proj_weight");
      const bufB = this.bt(bp + "buf_attn.in_proj_bias");

      // The new token's LN'd layer input is both the query and its own K/V slot
      // (inclusive causal: the token attends to context + buffer including itself).
      const q = layerNorm(h, this.bt(bp + "buf_ln1.weight"), this.bt(bp + "buf_ln1.bias"));
      dl.selfK.push(projPart(q, bufW, bufB, 1));
      dl.selfV.push(projPart(q, bufW, bufB, 2));
      const qp = projPart(q, bufW, bufB, 0);
      h = addInto(h, attendCached(qp, L.bufCtxK.concat(dl.selfK), L.bufCtxV.concat(dl.selfV), this.nHeads,
        this.bt(bp + "buf_attn.out_proj.weight"), this.bt(bp + "buf_attn.out_proj.bias")));
      h = addInto(h, mlp(
        layerNorm(h, this.bt(bp + "buf_ln2.weight"), this.bt(bp + "buf_ln2.bias")),
        this.bt(bp + "buf_mlp.0.weight"), this.bt(bp + "buf_mlp.0.bias"),
        this.bt(bp + "buf_mlp.2.weight"), this.bt(bp + "buf_mlp.2.bias"),
      ));

      // Concat read: the target sees this token through the BASE kv_ln/cross_attn
      // projections — buffer tokens enter the target read literally as more context.
      const kvr = layerNorm(h, this.bt(p + "kv_ln.weight"), this.bt(p + "kv_ln.bias"));
      const crossW = this.bt(p + "cross_attn.in_proj_weight");
      const crossB = this.bt(p + "cross_attn.in_proj_bias");
      dl.readK.push(projPart(kvr, crossW, crossB, 1));
      dl.readV.push(projPart(kvr, crossW, crossB, 2));
      perLayer.push(h.slice());
    }
    draw.k += 1;
    return { layers: perLayer };
  }
}
