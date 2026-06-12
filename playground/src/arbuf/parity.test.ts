/**
 * Parity gate for the TS AR-buffer port (`src/ace/buffered.ts`) against the
 * Python fixtures from `parity.py`'s buffered block:
 *
 * - `plain`: the inherited base forward on the buffered checkpoint (the frozen-base
 *   invariant extended through the export);
 * - `packed`: per-layer / per-token states from a packed `forward_buffered` pass
 *   with `prefix_len = arange(K)` — buffer row j ↔ TS append pass j, target row m ↔
 *   TS decode step m, so divergence localizes to a layer;
 * - `chain`: per-step log-probs from a teacher-forced `sample_joint` chain — the
 *   exact incremental-cache semantics the sampler implements, end to end.
 *
 * Everything self-skips when the model blob or fixture is absent: the model is
 * local-only until the retained fine-tune is deployed, and the deploy workflow's
 * `npm test` must stay green with only the four public models present.
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";

import { BufferedACEModel, mdnLogProb } from "../ace/buffered";
import { Predictions } from "../ace/predictions";
import { mulberry32 } from "../ace/rng";
import type { TokenSet } from "../ace/model";
import { type Manifest, weightsFromBytes } from "../ace/weights";
import { JointSampler } from "./infer";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const MODEL_DIR = join(ROOT, "public", "models", "gp1d_arbuffer");
const FIXTURE = join(ROOT, "test", "fixtures", "gp1d_arbuffer.parity.json");
const HAVE = existsSync(join(MODEL_DIR, "manifest.json")) && existsSync(join(MODEL_DIR, "weights.bin")) && existsSync(FIXTURE);

interface CaseTokens {
  var_id: number[];
  x: number[][];
  value: number[];
  value_index: number[];
  prior: number[][];
  mode: number[];
  mask: boolean[];
}

interface PlainCase {
  name: string;
  context: CaseTokens;
  target: CaseTokens;
  embed_context: number[][];
  per_layer_ctx: number[][][];
  per_layer_tgt: number[][][];
  cont_raw: number[][];
  disc_logits: number[][];
  log_prob: number[];
  mean: number[];
}

interface PackedCase {
  context: CaseTokens;
  buffer: CaseTokens;
  target: CaseTokens;
  prefix_len: number[];
  per_layer_ctx: number[][][];
  per_layer_buf: number[][][];
  per_layer_tgt: number[][][];
  cont_raw: number[][];
  log_prob: number[];
}

interface ChainCase {
  context: CaseTokens;
  grid: number[];
  order: number[];
  values: number[][];
  log_prob: number[][];
}

interface Fixture {
  concat_read: boolean;
  plain: PlainCase[];
  packed: PackedCase;
  chain: ChainCase;
}

function toTokenSet(j: CaseTokens): TokenSet {
  return {
    varId: j.var_id,
    x: j.x,
    value: j.value,
    valueIndex: j.value_index,
    prior: j.prior,
    mode: j.mode,
    mask: j.mask,
  };
}

function flat(x: unknown): number[] {
  if (typeof x === "number") return [x];
  if (Array.isArray(x)) return x.flatMap(flat);
  throw new Error("not numeric");
}

/** Worst (|Δ| − allowed) across all elements; throws on the worst violation. */
function check(label: string, ts: unknown, pt: unknown, atol: number, rtol: number): void {
  const a = flat(ts);
  const b = flat(pt);
  expect(a.length).toBe(b.length);
  let slack = -Infinity;
  let info = "";
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    const allowed = atol + rtol * Math.abs(b[i]);
    if (d - allowed > slack) {
      slack = d - allowed;
      info = `idx ${i}: ts=${a[i]} pt=${b[i]} |Δ|=${d.toExponential(3)} allowed=${allowed.toExponential(3)}`;
    }
  }
  if (slack > 0) throw new Error(`${label}: ${info}`);
}

// Slightly wider raw atol than the base parity test (1e-4): the concat checkpoint
// was fine-tuned with the base unfrozen, and its weights happen to accumulate a
// touch more float32-vs-float64 drift on intermediate layer states. Still tight —
// a real porting bug shows up orders of magnitude above this.
const RAW = { atol: 3e-4, rtol: 1e-3 };
const DERIVED = { atol: 1e-3, rtol: 1e-3 };

describe.skipIf(!HAVE)("arbuffer parity: gp1d_arbuffer", () => {
  let model: BufferedACEModel;
  let fx: Fixture;

  beforeAll(() => {
    const manifest = JSON.parse(readFileSync(join(MODEL_DIR, "manifest.json"), "utf8")) as Manifest;
    const bytes = readFileSync(join(MODEL_DIR, "weights.bin"));
    model = new BufferedACEModel(weightsFromBytes(manifest, new Uint8Array(bytes)));
    fx = JSON.parse(readFileSync(FIXTURE, "utf8")) as Fixture;
    if (!fx.concat_read) {
      throw new Error("fixture was generated from a separate-read checkpoint; the TS port is concat-read only");
    }
  });

  it("plain forward matches (frozen-base invariant through the export)", () => {
    for (const c of fx.plain) {
      const out = model.forward(toTokenSet(c.context), toTokenSet(c.target));
      check(`${c.name}/embed_context`, out.embedContext, c.embed_context, RAW.atol, RAW.rtol);
      for (let i = 0; i < c.per_layer_ctx.length; i++) {
        check(`${c.name}/ctx_layer_${i}`, out.ctxLayers[i], c.per_layer_ctx[i], RAW.atol, RAW.rtol);
        check(`${c.name}/tgt_layer_${i}`, out.tgtLayers[i], c.per_layer_tgt[i], RAW.atol, RAW.rtol);
      }
      check(`${c.name}/cont_raw`, out.contRaw, c.cont_raw, RAW.atol, RAW.rtol);
      check(`${c.name}/disc_logits`, out.discLogits, c.disc_logits, RAW.atol, RAW.rtol);
      const pred = new Predictions(model, out);
      check(`${c.name}/log_prob`, pred.logProb(toTokenSet(c.target)), c.log_prob, DERIVED.atol, DERIVED.rtol);
      check(`${c.name}/mean`, pred.mean(toTokenSet(c.target)), c.mean, DERIVED.atol, DERIVED.rtol);
    }
  });

  it("incremental decode matches the packed forward_buffered per layer and per step", () => {
    const p = fx.packed;
    const cache = model.encodeContext(toTokenSet(p.context));
    for (let l = 0; l < p.per_layer_ctx.length; l++) {
      check(`packed/ctx_layer_${l}`, cache.outputs[l], p.per_layer_ctx[l], RAW.atol, RAW.rtol);
    }

    const draw = model.newDraw();
    const k = p.buffer.value.length;
    for (let m = 0; m < k; m++) {
      // Target row m has prefix m: predict at step m, before appending token m.
      const step = model.predict(cache, draw, p.target.x[m][0]);
      for (let l = 0; l < step.layers.length; l++) {
        check(`packed/tgt_layer_${l}/step_${m}`, step.layers[l], p.per_layer_tgt[l][m], RAW.atol, RAW.rtol);
      }
      check(`packed/cont_raw/step_${m}`, step.raw, p.cont_raw[m], RAW.atol, RAW.rtol);
      check(`packed/log_prob/step_${m}`, mdnLogProb(step.params, p.target.value[m]), p.log_prob[m], DERIVED.atol, DERIVED.rtol);

      const app = model.append(cache, draw, p.buffer.x[m][0], p.buffer.value[m]);
      for (let l = 0; l < app.layers.length; l++) {
        check(`packed/buf_layer_${l}/token_${m}`, app.layers[l], p.per_layer_buf[l][m], RAW.atol, RAW.rtol);
      }
    }
  });

  it("teacher-forced chain matches sample_joint per-step log-probs", () => {
    const c = fx.chain;
    const cache = model.encodeContext(toTokenSet(c.context));
    const sampler = new JointSampler(model, cache, c.grid, {
      nDraws: c.values.length,
      rng: mulberry32(0),
      order: c.order,
      teacher: c.values,
    });
    sampler.runAll();
    check("chain/values", sampler.values, c.values, 0, 0);
    check("chain/log_prob", sampler.logps, c.log_prob, DERIVED.atol, DERIVED.rtol);
  });

  it("sampling is seed-deterministic", () => {
    const c = fx.chain;
    const cache = model.encodeContext(toTokenSet(c.context));
    const run = (seed: number) => {
      const s = new JointSampler(model, cache, c.grid, { nDraws: 2, rng: mulberry32(seed) });
      s.runAll();
      return s.values;
    };
    const a = run(7);
    const b = run(7);
    expect(a).toEqual(b);
    const d = run(8);
    expect(flat(a).some((v, i) => v !== flat(d)[i])).toBe(true);
  });
});
