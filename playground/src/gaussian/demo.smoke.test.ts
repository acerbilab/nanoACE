// @vitest-environment jsdom
/**
 * UI smoke test for the Gaussian demo: mount against real weights with a no-op
 * canvas + fs-backed fetch, then exercise a prior-slider change and clear.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";

import { mountGaussian } from "./demo";

const MODELS = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "public", "models");

function stubFetch(): void {
  vi.stubGlobal("fetch", async (url: string) => {
    if (url.endsWith("manifest.json")) {
      return { json: async () => JSON.parse(readFileSync(join(MODELS, "gaussian", "manifest.json"), "utf8")) };
    }
    if (url.endsWith("weights.bin")) {
      const buf = readFileSync(join(MODELS, "gaussian", "weights.bin"));
      return { arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
    }
    throw new Error(`unexpected fetch: ${url}`);
  });
  const noop = new Proxy({}, { get: () => () => {}, set: () => true });
  HTMLCanvasElement.prototype.getContext = (() => noop) as unknown as HTMLCanvasElement["getContext"];
}

describe("Gaussian demo UI smoke", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("mounts, renders, and handles a prior change without throwing", async () => {
    stubFetch();
    const el = document.createElement("div");
    document.body.appendChild(el);

    await mountGaussian(el);

    expect(el.querySelector(".ga-main")).not.toBeNull();
    expect(el.querySelector(".ga-mu")).not.toBeNull();
    expect(el.querySelector<HTMLDivElement>(".ga-banner")!.hidden).toBe(true);

    // Crank the μ-prior concentration and re-render.
    const muNu = el.querySelector<HTMLInputElement>(".mu-nu")!;
    muNu.value = muNu.max;
    muNu.dispatchEvent(new Event("input"));

    // Clear observations (priors keep the context non-empty).
    el.querySelector<HTMLButtonElement>(".clear")!.click();
  });
});
