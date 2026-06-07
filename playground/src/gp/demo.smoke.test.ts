// @vitest-environment jsdom
/**
 * UI smoke test: mount the real GP demo against the real weights with a no-op
 * canvas and an fs-backed fetch. Verifies the whole UI code path (DOM build,
 * control wiring, render -> gpInfer -> draw) runs without throwing — catching
 * selector/null-ref bugs the numeric tests can't.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";

import { mountGP } from "./demo";

const MODELS = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "public", "models");

function stubFetch(): void {
  vi.stubGlobal("fetch", async (url: string) => {
    if (url.endsWith("manifest.json")) {
      const text = readFileSync(join(MODELS, "gp1d", "manifest.json"), "utf8");
      return { json: async () => JSON.parse(text) };
    }
    if (url.endsWith("weights.bin")) {
      const buf = readFileSync(join(MODELS, "gp1d", "weights.bin"));
      const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
      return { arrayBuffer: async () => ab };
    }
    throw new Error(`unexpected fetch: ${url}`);
  });
  // jsdom has no 2D canvas; return a no-op context so drawing calls are harmless.
  const noop = new Proxy({}, { get: () => () => {}, set: () => true });
  HTMLCanvasElement.prototype.getContext = (() => noop) as unknown as HTMLCanvasElement["getContext"];
}

describe("GP demo UI smoke", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("mounts, renders, and handles a pin without throwing", async () => {
    stubFetch();
    const el = document.createElement("div");
    document.body.appendChild(el);

    await mountGP(el);

    expect(el.querySelector(".gp-main")).not.toBeNull();
    const banner = el.querySelector<HTMLDivElement>(".gp-banner")!;
    expect(banner.hidden).toBe(true); // 5 in-range points, no pins -> no OOD

    const kernelBtns = el.querySelectorAll<HTMLButtonElement>(".gp-kernel-btns button");
    expect(kernelBtns.length).toBe(5); // Unknown + 4 kernels

    // Pin the kernel (one latent pinned is still in-distribution).
    kernelBtns[4].click();
    expect(banner.hidden).toBe(true);

    // Clearing points should not throw (empty-context guard path).
    el.querySelector<HTMLButtonElement>(".clear")!.click();
  });
});
