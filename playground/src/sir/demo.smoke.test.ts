// @vitest-environment jsdom
/**
 * UI smoke test for the SIR demo: mount against real weights with a no-op
 * canvas + fs-backed fetch, then exercise a prior-slider change and clear.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";

import { mountSIR } from "./demo";

const MODELS = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "public", "models");

function stubFetch(): void {
  vi.stubGlobal("fetch", async (url: string) => {
    if (url.endsWith("manifest.json")) {
      return { json: async () => JSON.parse(readFileSync(join(MODELS, "sbi_sir", "manifest.json"), "utf8")) };
    }
    if (url.endsWith("weights.bin")) {
      const buf = readFileSync(join(MODELS, "sbi_sir", "weights.bin"));
      return { arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
    }
    throw new Error(`unexpected fetch: ${url}`);
  });
  const noop = new Proxy({}, { get: () => () => {}, set: () => true });
  HTMLCanvasElement.prototype.getContext = (() => noop) as unknown as HTMLCanvasElement["getContext"];
}

describe("SIR demo UI smoke", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("mounts, renders, and handles prior/observation changes without throwing", async () => {
    stubFetch();
    const el = document.createElement("div");
    document.body.appendChild(el);

    await mountSIR(el);

    expect(el.querySelector(".sir-main")).not.toBeNull();
    expect(el.querySelector(".sir-beta")).not.toBeNull();

    const betaNu = el.querySelector<HTMLInputElement>(".beta-nu")!;
    betaNu.value = betaNu.max;
    betaNu.dispatchEvent(new Event("input"));

    el.querySelector<HTMLButtonElement>(".clear")!.click();
  });
});
