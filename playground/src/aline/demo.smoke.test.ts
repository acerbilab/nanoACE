// @vitest-environment jsdom
/**
 * UI smoke test: mount the real ALINE demo against the real weights with a
 * no-op canvas, an fs-backed fetch, and a synchronous requestAnimationFrame.
 * Exercises an episode (snap-click query, Step, goal switch, reveal, mode
 * switch) without judging the policy. Self-skips when the local-only model
 * blob is absent, keeping the deploy workflow's `npm test` green.
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";

import { mountAline } from "./demo";

const MODELS = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "public", "models");
const HAVE =
  existsSync(join(MODELS, "gp1d_aline", "manifest.json")) &&
  existsSync(join(MODELS, "gp1d_aline", "weights.bin"));

function stubGlobals(): void {
  vi.stubGlobal("fetch", async (url: string) => {
    if (url.endsWith("manifest.json")) {
      const text = readFileSync(join(MODELS, "gp1d_aline", "manifest.json"), "utf8");
      return { json: async () => JSON.parse(text) };
    }
    if (url.endsWith("weights.bin")) {
      const buf = readFileSync(join(MODELS, "gp1d_aline", "weights.bin"));
      const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
      return { arrayBuffer: async () => ab };
    }
    throw new Error(`unexpected fetch: ${url}`);
  });
  // jsdom has no 2D canvas; return a no-op context so drawing calls are harmless.
  const noop = new Proxy({}, { get: () => () => {}, set: () => true });
  HTMLCanvasElement.prototype.getContext = (() => noop) as unknown as HTMLCanvasElement["getContext"];
  // Synchronous rAF would drain a whole Follow-policy run in-call (16 full
  // forwards) — too slow for a smoke test, so Follow policy is NOT clicked
  // here; Step covers the same applyAction path one forward at a time.
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
    cb(0);
    return 0;
  });
}

describe.skipIf(!HAVE)("ALINE demo UI smoke", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("mounts an episode and handles controls without throwing", async () => {
    stubGlobals();
    const el = document.createElement("div");
    document.body.appendChild(el);

    await mountAline(el);

    expect(el.querySelector(".al-main")).not.toBeNull();
    // An episode auto-starts: step 0 of the budget, truth hidden.
    const counter = el.querySelector<HTMLSpanElement>(".al-counter")!;
    expect(counter.textContent).toContain("step 0/");
    expect(el.querySelector<HTMLParagraphElement>(".al-status")!.textContent).toContain("truth hidden");

    // One policy step advances the budget.
    el.querySelector<HTMLButtonElement>(".al-step")!.click();
    expect(counter.textContent).toContain("step 1/");

    // Plain click SWITCHES the goal exclusively (pred and parameters never mix).
    const sel = (cls: string) => el.querySelector<HTMLButtonElement>(cls)!.classList.contains("sel");
    const shiftClick = (cls: string) =>
      el
        .querySelector<HTMLButtonElement>(cls)!
        .dispatchEvent(new MouseEvent("click", { shiftKey: true, bubbles: true }));
    el.querySelector<HTMLButtonElement>(".g-kernel")!.click();
    expect(sel(".g-kernel")).toBe(true);
    expect(sel(".g-pred")).toBe(false);
    // Shift-click COMBINES parameter goals…
    shiftClick(".g-ell");
    expect(sel(".g-ell")).toBe(true);
    expect(sel(".g-kernel")).toBe(true);
    // …toggles them off again…
    shiftClick(".g-ell");
    expect(sel(".g-ell")).toBe(false);
    // …but cannot empty the selection.
    shiftClick(".g-kernel");
    expect(sel(".g-kernel")).toBe(true);
    // Predictive never combines: shift-click on it still switches exclusively.
    shiftClick(".g-pred");
    expect(sel(".g-pred")).toBe(true);
    expect(sel(".g-kernel")).toBe(false);

    // Reveal toggle and restart (same hidden function, back to the seed).
    el.querySelector<HTMLInputElement>(".al-reveal")!.click();
    el.querySelector<HTMLButtonElement>(".al-restart")!.click();
    expect(counter.textContent).toContain("step 0/");

    // Mode switch to your-own-data and back: oracle controls toggle, no metrics text.
    el.querySelector<HTMLInputElement>(".al-mode-oracle")!.click();
    expect(el.querySelector<HTMLParagraphElement>(".al-status")!.textContent).toContain("no ground truth");
    el.querySelector<HTMLButtonElement>(".al-clear")!.click();
    el.querySelector<HTMLButtonElement>(".al-reset")!.click();
    el.querySelector<HTMLInputElement>(".al-mode-episode")!.click();
    expect(counter.textContent).toContain("step");

    // Per-tab explainer opens and closes.
    const modal = el.querySelector<HTMLElement>(".explain-modal")!;
    expect(modal.hidden).toBe(true);
    el.querySelector<HTMLButtonElement>(".info-btn")!.click();
    expect(modal.hidden).toBe(false);
    modal.querySelector<HTMLButtonElement>(".ace-modal-close")!.click();
    expect(modal.hidden).toBe(true);
  });
});

// Deliberately NOT behind the skip guard: this is the fallback path when the
// blob is absent, so it must stay covered on clones without local exports.
describe("ALINE demo missing-model notice", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("renders the export-it-locally notice when the model is absent", async () => {
    vi.stubGlobal("fetch", async () => {
      throw new Error("404");
    });
    const el = document.createElement("div");
    document.body.appendChild(el);
    await mountAline(el);
    expect(el.textContent).toContain("export_weights.py --task gp1d_aline");
  });
});
