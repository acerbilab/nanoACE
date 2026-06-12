/**
 * ALINE (extensions/aline/) in TypeScript: the unchanged base ACE forward plus a
 * read-only acquisition-policy decoder, for the playground's ALINE tab.
 *
 * The inference path IS the inherited `ACEModel.forward` — `forwardWithStates`
 * just re-reads its returned per-layer stacks: the final context states are
 * `ctxLayers[L-1]`, and the final target states are `final_norm` re-applied to
 * `tgtLayers[L-1]` (the same values the heads consume, and exactly what Python's
 * `ALINE.forward_with_states` returns). The policy embeds candidates with the
 * inherited embedder and runs the PolicyBlock stack: cross-attention to the
 * context states, cross-attention to the goal-target states, MLP. Candidates
 * never attend to each other (pointwise scoring; the softmax over whatever pool
 * the caller supplies provides the competition), so the playground passes only
 * the *available* candidates and only the *active* goal tokens — omission is
 * exactly equivalent to Python's masking (targets never attend to each other
 * either). Parity-tested per policy block against fixtures from `parity.py`.
 */

import { addInto, layerNorm, linear, mlp, multiHeadAttention } from "./nn";
import { ACEModel } from "./model";
import type { ForwardOut, TokenSet } from "./model";
import type { Tensor, Weights } from "./weights";

export interface StatesOut {
  out: ForwardOut;
  ctxStates: number[][]; // final-block context states (policy keys)
  tgtStates: number[][]; // final-normed target states (the goal representation)
}

export class ALINEModel extends ACEModel {
  readonly nPolicyBlocks: number;
  private aw: Weights;

  constructor(weights: Weights) {
    super(weights);
    this.aw = weights;
    const names = new Set(weights.manifest.tensors.map((t) => t.name));
    if (!names.has("policy_blocks.0.q_ln1.weight") || !names.has("policy_head.weight")) {
      throw new Error(
        "weights are not an ALINE export (no policy_blocks.* tensors); " +
          "re-export with export_weights.py --task gp1d_aline",
      );
    }
    let n = 0;
    while (names.has(`policy_blocks.${n}.q_ln1.weight`)) n++;
    this.nPolicyBlocks = n;
  }

  private at(name: string): Tensor {
    return this.aw.get(name);
  }

  /**
   * The inherited forward, plus the final trunk states the policy reads — the
   * return values of Python's `ALINE.forward_with_states`.
   */
  forwardWithStates(context: TokenSet, target: TokenSet): StatesOut {
    const out = this.forward(context, target);
    const fnW = this.at("final_norm.weight");
    const fnB = this.at("final_norm.bias");
    const ctxStates = out.ctxLayers[out.ctxLayers.length - 1];
    const tgtStates = out.tgtLayers[out.tgtLayers.length - 1].map((r) => layerNorm(r, fnW, fnB));
    return { out, ctxStates, tgtStates };
  }

  /**
   * Run the policy decoder, returning the candidate stream after each block
   * (the parity test pins these per block so divergence localizes).
   * All supplied tokens/states are treated as active — omit, don't mask.
   */
  policyBlockStates(query: TokenSet, ctxStates: number[][], tgtStates: number[][]): number[][][] {
    let qry = this.embed(query);
    const noPadCtx = ctxStates.map(() => false);
    const noPadTgt = tgtStates.map(() => false);
    const states: number[][][] = [];
    for (let i = 0; i < this.nPolicyBlocks; i++) {
      const p = `policy_blocks.${i}.`;
      const ctxKv = ctxStates.map((r) =>
        layerNorm(r, this.at(p + "ctx_kv_ln.weight"), this.at(p + "ctx_kv_ln.bias")),
      );
      const q1 = qry.map((r) => layerNorm(r, this.at(p + "q_ln1.weight"), this.at(p + "q_ln1.bias")));
      const readCtx = multiHeadAttention(
        q1, ctxKv, ctxKv, noPadCtx,
        this.at(p + "ctx_attn.in_proj_weight"), this.at(p + "ctx_attn.in_proj_bias"),
        this.at(p + "ctx_attn.out_proj.weight"), this.at(p + "ctx_attn.out_proj.bias"),
        this.nHeads,
      );
      qry = qry.map((r, k) => addInto(r, readCtx[k]));
      const tgtKv = tgtStates.map((r) =>
        layerNorm(r, this.at(p + "tgt_kv_ln.weight"), this.at(p + "tgt_kv_ln.bias")),
      );
      const q2 = qry.map((r) => layerNorm(r, this.at(p + "q_ln2.weight"), this.at(p + "q_ln2.bias")));
      const readTgt = multiHeadAttention(
        q2, tgtKv, tgtKv, noPadTgt,
        this.at(p + "tgt_attn.in_proj_weight"), this.at(p + "tgt_attn.in_proj_bias"),
        this.at(p + "tgt_attn.out_proj.weight"), this.at(p + "tgt_attn.out_proj.bias"),
        this.nHeads,
      );
      qry = qry.map((r, k) => addInto(r, readTgt[k]));
      qry = qry.map((r) =>
        addInto(
          r,
          mlp(
            layerNorm(r, this.at(p + "q_ln3.weight"), this.at(p + "q_ln3.bias")),
            this.at(p + "mlp.0.weight"), this.at(p + "mlp.0.bias"),
            this.at(p + "mlp.2.weight"), this.at(p + "mlp.2.bias"),
          ),
        ),
      );
      states.push(qry.map((r) => r.slice()));
    }
    return states;
  }

  /** Score each candidate token (pointwise); higher = more informative for ξ. */
  policyLogits(query: TokenSet, ctxStates: number[][], tgtStates: number[][]): number[] {
    const states = this.policyBlockStates(query, ctxStates, tgtStates);
    const qry = states[states.length - 1];
    const pnW = this.at("policy_norm.weight");
    const pnB = this.at("policy_norm.bias");
    const hw = this.at("policy_head.weight");
    const hb = this.at("policy_head.bias");
    return qry.map((r) => linear(layerNorm(r, pnW, pnB), hw, hb)[0]);
  }
}
