/**
 * ACE forward pass in TypeScript — a faithful port of `ace.py`'s `_embed`,
 * `ACEBlock`, and `ACE.forward`. Parity-tested against the PyTorch model.
 *
 * A `TokenSet` is a single (unbatched) set of tokens; the demos always use one
 * context with multiple targets (B=1), which is equivalent to PyTorch's batched
 * grid queries because targets are predicted independently (diagonal map).
 */

import { addInto, layerNorm, mlp, multiHeadAttention } from "./nn";
import type { Tensor, Weights } from "./weights";
import type { VariableMeta } from "./schema";

export const VALUE = 0;
export const PRIOR = 1;
export const QUERY = 2;

export interface TokenSet {
  varId: number[]; // [T]
  x: number[][]; // [T][x_dim]
  value: number[]; // [T]
  valueIndex: number[]; // [T]
  prior: number[][]; // [T][2]
  mode: number[]; // [T]  (VALUE | PRIOR | QUERY)
  mask: boolean[]; // [T]
}

export interface ForwardOut {
  embedContext: number[][];
  contRaw: number[][]; // [Tt][3*K]
  discLogits: number[][]; // [Tt][maxCard]
  ctxLayers: number[][][]; // per-block context state
  tgtLayers: number[][][]; // per-block target state
}

function zeros(n: number): number[] {
  return new Array<number>(n).fill(0);
}

export class ACEModel {
  readonly variables: VariableMeta[];
  readonly d: number;
  readonly nHeads: number;
  readonly nLayers: number;
  readonly mdnComponents: number;
  readonly minScale: number;
  readonly maxCardinality: number;

  private w: Weights;
  private discRows: number;

  constructor(weights: Weights) {
    this.w = weights;
    const m = weights.manifest;
    this.variables = m.variables;
    this.d = m.cfg.d_model;
    this.nHeads = m.cfg.n_heads;
    this.nLayers = m.cfg.n_layers;
    this.mdnComponents = m.cfg.mdn_components;
    this.minScale = m.cfg.min_scale;
    this.maxCardinality = m.derived.max_cardinality;
    this.discRows = weights.get("disc_value_embed.weight").shape[0];
  }

  private t(name: string): Tensor {
    return this.w.get(name);
  }

  private row(W: Tensor, r: number): number[] {
    const d = this.d;
    const o = r * d;
    const out = new Array<number>(d);
    for (let i = 0; i < d; i++) out[i] = W.data[o + i];
    return out;
  }

  /** Embed a token set into [T][d_model], mirroring ACE._embed. */
  embed(tokens: TokenSet): number[][] {
    const varEmbed = this.t("var_embed.weight");
    const modeEmbed = this.t("mode_embed.weight");
    const discEmbed = this.t("disc_value_embed.weight");
    const unknown = this.t("unknown");
    const x0 = this.t("x_embed.0.weight");
    const xb0 = this.t("x_embed.0.bias");
    const x2 = this.t("x_embed.2.weight");
    const xb2 = this.t("x_embed.2.bias");
    const v0 = this.t("value_embed.0.weight");
    const vb0 = this.t("value_embed.0.bias");
    const v2 = this.t("value_embed.2.weight");
    const vb2 = this.t("value_embed.2.bias");
    const s0 = this.t("spread_embed.0.weight");
    const sb0 = this.t("spread_embed.0.bias");
    const s2 = this.t("spread_embed.2.weight");
    const sb2 = this.t("spread_embed.2.bias");
    const unknownVec = Array.from(unknown.data);

    const T = tokens.varId.length;
    const out: number[][] = [];
    for (let t = 0; t < T; t++) {
      const vid = tokens.varId[t];
      const meta = this.variables[vid];

      const varVec = this.row(varEmbed, vid);
      const modeIdx = Math.min(Math.max(tokens.mode[t], 0), 2);
      const modeVec = this.row(modeEmbed, modeIdx);

      let xVec = mlp(tokens.x[t], x0, xb0, x2, xb2);
      if (meta.is_latent) xVec = zeros(this.d);

      // VALUE payload: discrete via shared table + per-var offset; continuous via MLP.
      let val: number[];
      if (meta.is_discrete) {
        const idx = Math.min(
          Math.max(meta.disc_offset + Math.max(tokens.valueIndex[t], 0), 0),
          this.discRows - 1,
        );
        val = this.row(discEmbed, idx);
      } else {
        val = mlp([tokens.value[t]], v0, vb0, v2, vb2);
      }

      // PRIOR payload: shared value_embed on the mean + spread-gated spread_embed.
      const priorInput = tokens.prior[t];
      const priorMean = mlp([priorInput[0]], v0, vb0, v2, vb2);
      const spreadVec = mlp(priorInput, s0, sb0, s2, sb2);
      const priorEmb = new Array<number>(this.d);
      for (let i = 0; i < this.d; i++) priorEmb[i] = priorMean[i] + priorInput[1] * spreadVec[i];

      let payload = tokens.mode[t] === PRIOR ? priorEmb : val;
      if (tokens.mode[t] === QUERY) payload = unknownVec;

      const sum = new Array<number>(this.d);
      const active = tokens.mask[t] ? 1 : 0;
      for (let i = 0; i < this.d; i++) {
        sum[i] = active * (varVec[i] + modeVec[i] + xVec[i] + payload[i]);
      }
      out.push(sum);
    }
    return out;
  }

  forward(context: TokenSet, target: TokenSet): ForwardOut {
    if (!context.mask.some((m) => m)) {
      throw new Error("ACE needs at least one active context token");
    }

    const embedContext = this.embed(context);
    let ctx = embedContext.map((r) => r.slice());
    let tgt = this.embed(target);
    const ctxMask = context.mask;
    const tgtMask = target.mask;
    const keyPad = ctxMask.map((m) => !m);

    const ctxLayers: number[][][] = [];
    const tgtLayers: number[][][] = [];

    for (let i = 0; i < this.nLayers; i++) {
      const p = `blocks.${i}.`;
      const ctxLn1 = ctx.map((r) => layerNorm(r, this.t(p + "ctx_ln1.weight"), this.t(p + "ctx_ln1.bias")));
      const ctxAttn = multiHeadAttention(
        ctxLn1, ctxLn1, ctxLn1, keyPad,
        this.t(p + "ctx_attn.in_proj_weight"), this.t(p + "ctx_attn.in_proj_bias"),
        this.t(p + "ctx_attn.out_proj.weight"), this.t(p + "ctx_attn.out_proj.bias"),
        this.nHeads,
      );
      ctx = ctx.map((r, k) => addInto(r, ctxAttn[k]));
      ctx = ctx.map((r) =>
        addInto(r, mlp(
          layerNorm(r, this.t(p + "ctx_ln2.weight"), this.t(p + "ctx_ln2.bias")),
          this.t(p + "ctx_mlp.0.weight"), this.t(p + "ctx_mlp.0.bias"),
          this.t(p + "ctx_mlp.2.weight"), this.t(p + "ctx_mlp.2.bias"),
        )),
      );

      const kv = ctx.map((r) => layerNorm(r, this.t(p + "kv_ln.weight"), this.t(p + "kv_ln.bias")));
      const tgtLn1 = tgt.map((r) => layerNorm(r, this.t(p + "tgt_ln1.weight"), this.t(p + "tgt_ln1.bias")));
      const tgtAttn = multiHeadAttention(
        tgtLn1, kv, kv, keyPad,
        this.t(p + "cross_attn.in_proj_weight"), this.t(p + "cross_attn.in_proj_bias"),
        this.t(p + "cross_attn.out_proj.weight"), this.t(p + "cross_attn.out_proj.bias"),
        this.nHeads,
      );
      tgt = tgt.map((r, k) => addInto(r, tgtAttn[k]));
      tgt = tgt.map((r) =>
        addInto(r, mlp(
          layerNorm(r, this.t(p + "tgt_ln2.weight"), this.t(p + "tgt_ln2.bias")),
          this.t(p + "tgt_mlp.0.weight"), this.t(p + "tgt_mlp.0.bias"),
          this.t(p + "tgt_mlp.2.weight"), this.t(p + "tgt_mlp.2.bias"),
        )),
      );

      // Per-block mask zeroing (matches ACEBlock: ctx*=mask, tgt*=mask).
      ctx = ctx.map((r, k) => (ctxMask[k] ? r : zeros(this.d)));
      tgt = tgt.map((r, k) => (tgtMask[k] ? r : zeros(this.d)));
      ctxLayers.push(ctx.map((r) => r.slice()));
      tgtLayers.push(tgt.map((r) => r.slice()));
    }

    const fnW = this.t("final_norm.weight");
    const fnB = this.t("final_norm.bias");
    const tgtNorm = tgt.map((r) => layerNorm(r, fnW, fnB));

    const ch0 = this.t("cont_head.0.weight");
    const chb0 = this.t("cont_head.0.bias");
    const ch2 = this.t("cont_head.2.weight");
    const chb2 = this.t("cont_head.2.bias");
    const dh0 = this.t("disc_head.0.weight");
    const dhb0 = this.t("disc_head.0.bias");
    const dh2 = this.t("disc_head.2.weight");
    const dhb2 = this.t("disc_head.2.bias");

    const contRaw = tgtNorm.map((r) => mlp(r, ch0, chb0, ch2, chb2));
    const discLogits = tgtNorm.map((r) => mlp(r, dh0, dhb0, dh2, dhb2)); // disc_head is Linear-GELU-Linear

    return { embedContext, contRaw, discLogits, ctxLayers, tgtLayers };
  }
}
