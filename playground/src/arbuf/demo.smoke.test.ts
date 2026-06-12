// @vitest-environment jsdom
/**
 * UI smoke test: mount the real AR-buffer demo against the real weights with a
 * no-op canvas, an fs-backed fetch, and a synchronous requestAnimationFrame (so
 * the animated decode drains deterministically). Self-skips when the local-only
 * model blob is absent, keeping the deploy workflow's `npm test` green.
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";

import { mountArbuf } from "./demo";

const MODELS = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "public", "models");
const HAVE = existsSync(join(MODELS, "gp1d_arbuffer", "manifest.json")) && existsSync(join(MODELS, "gp1d_arbuffer", "weights.bin"));

function stubGlobals(): void {
  vi.stubGlobal("fetch", async (url: string) => {
    if (url.endsWith("manifest.json")) {
      const text = readFileSync(join(MODELS, "gp1d_arbuffer", "manifest.json"), "utf8");
      return { json: async () => JSON.parse(text) };
    }
    if (url.endsWith("weights.bin")) {
      const buf = readFileSync(join(MODELS, "gp1d_arbuffer", "weights.bin"));
      const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
      return { arrayBuffer: async () => ab };
    }
    throw new Error(`unexpected fetch: ${url}`);
  });
  // jsdom has no 2D canvas; return a no-op context so drawing calls are harmless.
  const noop = new Proxy({}, { get: () => () => {}, set: () => true });
  HTMLCanvasElement.prototype.getContext = (() => noop) as unknown as HTMLCanvasElement["getContext"];
  // Synchronous rAF: the animated decode drains inside the triggering call
  // (recursion depth = GRID_POINTS / STEPS_PER_FRAME = 64, well within limits).
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
    cb(0);
    return 0;
  });
}

describe.skipIf(!HAVE)("AR-buffer demo UI smoke", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("mounts, decodes, and handles controls without throwing", async () => {
    stubGlobals();
    const el = document.createElement("div");
    document.body.appendChild(el);

    await mountArbuf(el);

    expect(el.querySelector(".ab-main")).not.toBeNull();

    // Status line reports the completed decode (synchronous rAF drains it).
    expect(el.querySelector<HTMLParagraphElement>(".ab-status")!.textContent).toContain("decode");

    // Resample reuses the cached encoding; pinning re-encodes; clear/reset guard paths.
    el.querySelector<HTMLButtonElement>(".ab-resample")!.click();
    const kernelBtns = el.querySelectorAll<HTMLButtonElement>(".ab-kernel-btns button");
    expect(kernelBtns.length).toBe(5); // Unknown + 4 kernels
    kernelBtns[4].click();
    el.querySelector<HTMLButtonElement>(".clear")!.click();
    el.querySelector<HTMLButtonElement>(".reset")!.click();

    // Per-tab explainer opens and closes.
    const modal = el.querySelector<HTMLElement>(".explain-modal")!;
    expect(modal.hidden).toBe(true);
    el.querySelector<HTMLButtonElement>(".info-btn")!.click();
    expect(modal.hidden).toBe(false);
    modal.querySelector<HTMLButtonElement>(".ace-modal-close")!.click();
    expect(modal.hidden).toBe(true);
  });
});

// Deliberately NOT behind the skip guard: this is exactly the path a deployment
// without the blob takes, so it must stay covered when the model dir is absent.
describe("AR-buffer demo missing-model notice", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("renders the local-only notice when the model is absent", async () => {
    vi.stubGlobal("fetch", async () => {
      throw new Error("404");
    });
    const el = document.createElement("div");
    document.body.appendChild(el);
    await mountArbuf(el);
    expect(el.textContent).toContain("local-only");
  });
});
