/**
 * Parity gate for the TS ALINE port (`src/ace/aline.ts`) against the Python
 * fixtures from `parity.py`'s aline block:
 *
 * - `plain`: the inherited base forward on the ALINE checkpoint (the inference
 *   path is the unchanged ACE forward — pinned through the export);
 * - `policy`: final trunk states (`forwardWithStates`) + per-policy-block
 *   candidate streams + logits, so divergence localizes to a block;
 * - `chain`: a teacher-forced compact episode with a mid-episode goal switch —
 *   the TS test REPLAYS the recorded actions (immune to argmax tie-flips under
 *   fp drift; agreement is reported, not asserted) and checks logits and
 *   per-goal-token log-probs at every step.
 *
 * Everything self-skips when the model blob or fixture is absent: the model is
 * local-only until a retained fine-tune is deployed, and CI's `npm test` must
 * stay green with only the public models present.
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";

import { ALINEModel } from "../ace/aline";
import { QUERY, VALUE } from "../ace/model";
import type { TokenSet } from "../ace/model";
import { Predictions } from "../ace/predictions";
import { TokenList } from "../ace/tokens";
import { type Manifest, weightsFromBytes } from "../ace/weights";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const MODEL_DIR = join(ROOT, "public", "models", "gp1d_aline");
const FIXTURE = join(ROOT, "test", "fixtures", "gp1d_aline.parity.json");
const HAVE =
  existsSync(join(MODEL_DIR, "manifest.json")) &&
  existsSync(join(MODEL_DIR, "weights.bin")) &&
  existsSync(FIXTURE);

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

interface PolicyCase {
  context: CaseTokens;
  target: CaseTokens;
  query: CaseTokens;
  ctx_states: number[][];
  tgt_states: number[][];
  per_policy_qry: number[][][];
  logits: number[];
  cont_raw: number[][];
  log_prob: number[];
}

interface ChainStep {
  xi: "ell" | "pred";
  available: number[];
  logits: number[];
  action: number;
  observed_y: number;
  log_prob: number[];
}

interface ChainCase {
  pool_x: number[];
  pool_y: number[];
  star_x: number[];
  star_y: number[];
  ell_value_internal: number;
  seed_index: number;
  switch_at: number;
  steps: ChainStep[];
  final_log_prob: number[];
}

interface Fixture {
  plain: PlainCase[];
  policy: PolicyCase;
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

/** Combined atol/rtol gate; throws on the worst violation. */
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

// Same tolerances as the arbuffer suite: the ALINE checkpoint is also a joint
// (base-unfrozen) fine-tune, so intermediate states carry a touch more
// float32-vs-float64 drift than the core 1e-4 gate. A porting bug shows up
// orders of magnitude above this.
const RAW = { atol: 3e-4, rtol: 1e-3 };
const DERIVED = { atol: 1e-3, rtol: 1e-3 };

/** Episode token builders shared by the chain steps (TS omission semantics). */
function chainTokens(c: ChainCase, observed: number[], xi: "ell" | "pred", available: number[]) {
  const ctx = new TokenList();
  for (const i of observed) ctx.add(0, VALUE, { x: c.pool_x[i], value: c.pool_y[i] });
  const tgt = new TokenList();
  if (xi === "ell") {
    tgt.add(1, QUERY, { value: c.ell_value_internal });
  } else {
    for (let m = 0; m < c.star_x.length; m++) tgt.add(0, QUERY, { x: c.star_x[m], value: c.star_y[m] });
  }
  const qry = new TokenList();
  for (const i of available) qry.add(0, QUERY, { x: c.pool_x[i] });
  return { ctx: ctx.get(), tgt: tgt.get(), qry: qry.get() };
}

describe.skipIf(!HAVE)("aline parity: gp1d_aline", () => {
  let model: ALINEModel;
  let fx: Fixture;

  beforeAll(() => {
    const manifest = JSON.parse(readFileSync(join(MODEL_DIR, "manifest.json"), "utf8")) as Manifest;
    const bytes = readFileSync(join(MODEL_DIR, "weights.bin"));
    model = new ALINEModel(weightsFromBytes(manifest, new Uint8Array(bytes)));
    fx = JSON.parse(readFileSync(FIXTURE, "utf8")) as Fixture;
  });

  it("plain forward matches (the inference path is the unchanged ACE forward)", () => {
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

  it("policy decoder matches per block (states, candidate streams, logits)", () => {
    const p = fx.policy;
    const { out, ctxStates, tgtStates } = model.forwardWithStates(
      toTokenSet(p.context),
      toTokenSet(p.target),
    );
    check("policy/ctx_states", ctxStates, p.ctx_states, RAW.atol, RAW.rtol);
    check("policy/tgt_states", tgtStates, p.tgt_states, RAW.atol, RAW.rtol);
    check("policy/cont_raw", out.contRaw, p.cont_raw, RAW.atol, RAW.rtol);
    const pred = new Predictions(model, out);
    check("policy/log_prob", pred.logProb(toTokenSet(p.target)), p.log_prob, DERIVED.atol, DERIVED.rtol);

    const states = model.policyBlockStates(toTokenSet(p.query), ctxStates, tgtStates);
    expect(states.length).toBe(p.per_policy_qry.length);
    for (let i = 0; i < states.length; i++) {
      check(`policy/block_${i}`, states[i], p.per_policy_qry[i], RAW.atol, RAW.rtol);
    }
    const logits = model.policyLogits(toTokenSet(p.query), ctxStates, tgtStates);
    check("policy/logits", logits, p.logits, DERIVED.atol, DERIVED.rtol);
  });

  it("teacher-forced chain matches per step (logits + per-goal-token log-probs)", () => {
    const c = fx.chain;
    const observed: number[] = [c.seed_index];
    for (let t = 0; t < c.steps.length; t++) {
      const step = c.steps[t];
      const { ctx, tgt, qry } = chainTokens(c, observed, step.xi, step.available);
      const { out, ctxStates, tgtStates } = model.forwardWithStates(ctx, tgt);
      const logits = model.policyLogits(qry, ctxStates, tgtStates);
      check(`chain/step_${t}/logits`, logits, step.logits, DERIVED.atol, DERIVED.rtol);
      const pred = new Predictions(model, out);
      check(`chain/step_${t}/log_prob`, pred.logProb(tgt), step.log_prob, DERIVED.atol, DERIVED.rtol);

      // Teacher-forced: follow the recorded action. Argmax agreement is
      // reported, not asserted, so an fp tie-flip cannot fail the build.
      let best = 0;
      for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
      if (step.available[best] !== step.action) {
        console.warn(
          `aline chain step ${t}: TS argmax picks pool idx ${step.available[best]}, ` +
            `fixture recorded ${step.action} (near-tie under fp drift)`,
        );
      }
      observed.push(step.action);
    }

    // Post-episode predictive state (no action) — the convergence-shaped check.
    const { ctx, tgt } = chainTokens(c, observed, "pred", []);
    const { out } = model.forwardWithStates(ctx, tgt);
    const pred = new Predictions(model, out);
    check("chain/final_log_prob", pred.logProb(tgt), c.final_log_prob, DERIVED.atol, DERIVED.rtol);
  });
});

// The rejection path needs only a public core blob, which CI always has — so it
// is deliberately NOT behind the aline HAVE guard (arbuffer's notice-test spirit).
const GP_DIR = join(ROOT, "public", "models", "gp1d");
const HAVE_GP = existsSync(join(GP_DIR, "manifest.json")) && existsSync(join(GP_DIR, "weights.bin"));

describe.skipIf(!HAVE_GP)("aline loader rejection", () => {
  it("rejects a non-ALINE blob with a clear error", () => {
    const manifest = JSON.parse(readFileSync(join(GP_DIR, "manifest.json"), "utf8")) as Manifest;
    const bytes = readFileSync(join(GP_DIR, "weights.bin"));
    expect(() => new ALINEModel(weightsFromBytes(manifest, new Uint8Array(bytes)))).toThrow(
      /not an ALINE export/,
    );
  });
});
