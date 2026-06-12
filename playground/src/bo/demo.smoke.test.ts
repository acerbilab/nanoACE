// @vitest-environment jsdom
/**
 * UI smoke test: mount the BO demo against real exported weights with a no-op
 * canvas and fs-backed fetch.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";

import { mountBO } from "./demo";

const MODELS = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "public", "models");

function stubFetch(): void {
  vi.stubGlobal("fetch", async (url: string) => {
    if (url.endsWith("manifest.json")) {
      const text = readFileSync(join(MODELS, "bo1d", "manifest.json"), "utf8");
      return { json: async () => JSON.parse(text) };
    }
    if (url.endsWith("weights.bin")) {
      const buf = readFileSync(join(MODELS, "bo1d", "weights.bin"));
      const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
      return { arrayBuffer: async () => ab };
    }
    throw new Error(`unexpected fetch: ${url}`);
  });
  const noop = new Proxy({}, { get: () => () => {}, set: () => true });
  HTMLCanvasElement.prototype.getContext = (() => noop) as unknown as HTMLCanvasElement["getContext"];
}

describe("BO demo UI smoke", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("mounts, renders, and handles latent fixing without throwing", async () => {
    stubFetch();
    const el = document.createElement("div");
    document.body.appendChild(el);

    await mountBO(el);

    expect(el.querySelector(".bo-main")).not.toBeNull();
    expect(el.querySelector(".bo-banner")).toBeNull();

    el.querySelector<HTMLInputElement>(".pin-x")!.click();
    el.querySelector<HTMLInputElement>(".pin-y")!.click();
    expect(el.querySelector(".bo-main")).not.toBeNull();

    el.querySelector<HTMLButtonElement>(".clear")!.click();
    expect(el.querySelector(".bo-main")).not.toBeNull();

    // Per-tab explainer opens and closes.
    const modal = el.querySelector<HTMLElement>(".explain-modal")!;
    expect(modal.hidden).toBe(true);
    el.querySelector<HTMLButtonElement>(".info-btn")!.click();
    expect(modal.hidden).toBe(false);
    modal.querySelector<HTMLButtonElement>(".ace-modal-close")!.click();
    expect(modal.hidden).toBe(true);
  });
});
