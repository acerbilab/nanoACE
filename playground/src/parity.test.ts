/**
 * Parity gate: the TS forward must reproduce the PyTorch model's embeddings,
 * per-layer states, raw head outputs, and derived quantities for every fixture
 * case. PyTorch runs float32 while JS runs float64, so we use a combined
 * relative+absolute tolerance rather than chasing bit-parity.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { ACEModel, type TokenSet } from "./ace/model";
import { Predictions } from "./ace/predictions";
import { type Manifest, weightsFromBytes } from "./ace/weights";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

interface CaseTokens {
  var_id: number[];
  x: number[][];
  value: number[];
  value_index: number[];
  prior: number[][];
  mode: number[];
  mask: boolean[];
}

interface ParityCase {
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

function loadModel(dir: string): ACEModel {
  const manifest = JSON.parse(readFileSync(join(ROOT, "public", "models", dir, "manifest.json"), "utf8")) as Manifest;
  const bytes = readFileSync(join(ROOT, "public", "models", dir, "weights.bin"));
  return new ACEModel(weightsFromBytes(manifest, new Uint8Array(bytes)));
}

function loadCases(file: string): ParityCase[] {
  return JSON.parse(readFileSync(join(ROOT, "test", "fixtures", file), "utf8")) as ParityCase[];
}

function flat(x: unknown): number[] {
  if (typeof x === "number") return [x];
  if (Array.isArray(x)) return x.flatMap(flat);
  throw new Error("not numeric");
}

/** Worst (|Δ| − allowed) across all elements; ≤ 0 means every element is in tolerance. */
function worstViolation(ts: unknown, pt: unknown, atol: number, rtol: number): { slack: number; info: string } {
  const a = flat(ts);
  const b = flat(pt);
  expect(a.length).toBe(b.length);
  let slack = -Infinity;
  let info = "ok";
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    const allowed = atol + rtol * Math.abs(b[i]);
    const s = d - allowed;
    if (s > slack) {
      slack = s;
      info = `idx ${i}: ts=${a[i]} pt=${b[i]} |Δ|=${d.toExponential(3)} allowed=${allowed.toExponential(3)}`;
    }
  }
  return { slack, info };
}

const RAW = { atol: 1e-4, rtol: 1e-3 };
const DERIVED = { atol: 1e-3, rtol: 1e-3 };

const TASKS = [
  { dir: "gp1d", file: "gp1d.parity.json" },
  { dir: "gaussian", file: "gaussian.parity.json" },
];

for (const task of TASKS) {
  describe(`parity: ${task.dir}`, () => {
    const model = loadModel(task.dir);
    const cases = loadCases(task.file);

    for (const c of cases) {
      it(c.name, () => {
        const out = model.forward(toTokenSet(c.context), toTokenSet(c.target));

        const check = (label: string, ts: unknown, pt: unknown, tol: { atol: number; rtol: number }) => {
          const { slack, info } = worstViolation(ts, pt, tol.atol, tol.rtol);
          if (slack > 0) throw new Error(`${c.name} / ${label}: ${info}`);
          expect(slack).toBeLessThanOrEqual(0);
        };

        check("embed_context", out.embedContext, c.embed_context, RAW);
        for (let i = 0; i < c.per_layer_ctx.length; i++) {
          check(`ctx_layer_${i}`, out.ctxLayers[i], c.per_layer_ctx[i], RAW);
          check(`tgt_layer_${i}`, out.tgtLayers[i], c.per_layer_tgt[i], RAW);
        }
        check("cont_raw", out.contRaw, c.cont_raw, RAW);
        check("disc_logits", out.discLogits, c.disc_logits, RAW);

        const pred = new Predictions(model, out);
        check("log_prob", pred.logProb(toTokenSet(c.target)), c.log_prob, DERIVED);
        check("mean", pred.mean(toTokenSet(c.target)), c.mean, DERIVED);
      });
    }
  });
}
