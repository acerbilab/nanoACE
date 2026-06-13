/**
 * AR-buffer joint-sampling demo (UI layer).
 *
 * GP-tab interaction (click to add a context point, drag to move, shift-click to
 * delete, pin any latent) — but the headline is coherent joint draws: the context
 * is encoded once and cached, then a few draw streams decode autoregressively
 * against the cache, animated step by step. The diagonal ±2σ band and independent
 * per-point marginal samples are always shown for contrast, and Resample reuses
 * the cached encoding (only the decode reruns).
 */

import { ARBUF, KERNEL_LABELS } from "../config";
import { aceFooter, addInfoButton } from "../explain";
import { clamp, hitPoint, pointOodReasons } from "../interaction";
import { makePlot, type Plot } from "../plot";
import { BufferedACEModel, type CtxCache } from "../ace/buffered";
import { loadWeights } from "../ace/weights";
import { mulberry32 } from "../ace/rng";
import {
  arbufStatic,
  buildContext,
  JointSampler,
  sampleIndependent,
  SlowARSampler,
  type ArbufSpec,
  type ArbufStatic,
  type ChainSampler,
} from "./infer";

interface Point {
  x: number;
  y: number;
}

interface PinState {
  kernel: number | null;
  ell: boolean;
  scale: boolean;
}

const DRAW_COLORS = ["#9333ea", "#ea580c", "#16a34a"];
const INDEP_COLOR = "rgba(107,114,128,0.45)";
const SEED0 = 12345;

const EXPLAINER = {
  title: "About: joint draws with an AR buffer",
  html: `
    <h3>The task</h3>
    <p>ACE returns an independent marginal distribution for each queried target — good
    one-value answers, but silent on how the values co-vary. Any question about several
    targets at once (a whole curve, a coherent scenario) needs samples from the joint
    distribution, where correlated values move together. In this tab the targets are the
    values of an unknown function on a dense grid: a joint draw is a plausible whole
    function, while the gray lines — each grid point sampled from its own marginal — are
    right pointwise but jump independently from point to point, because nothing ties
    neighboring values together.</p>
    <h3>What this tab is doing</h3>
    <p>Joint samples come from the chain rule of probability: predict one point, sample it, 
    condition on the realized value, move to the next — later points see the earlier draws, which is
    what creates the correlation. The standard implementation re-encodes the entire context
    plus all realized points at every step. Our <em>causal autoregressive (AR) buffer</em> instead encodes the
    context once, caches it, and routes each realized sample through a separate causal
    stream that later steps attend to, with large savings in memory and compute. 
    Each colored curve is one draw decoded against the same cached encoding; Resample reuses the cache.</p>
    <h3>Compared with the standard approach</h3>
    <p>Same factorization, different cost: re-encoding does O(K·(N+K)²) attention work over
    a K-point chain on N context tokens, the buffer O(N² + K·(N+K)) — in this page's
    implementation, several seconds versus a fraction of a second per resample. The sampler
    toggle under <em>sampling</em> runs the re-encoding variant live, and the line below
    the plot reports cost per draw, so the two compare directly. The buffer's read of
    realized points is learned by fine-tuning the GP-1D model (here it nearly matches the
    re-encoding chain's joint quality), and the rest of that model carries over: the blue
    band is its ordinary marginal prediction, and pinned latents condition the joint draws
    too.</p>
    ${aceFooter(
      // OpenReview link is deliberate for now; switch to the paper's project page later.
      'The buffer mechanism follows Hassan et al. (2026), <em>Efficient Autoregressive Inference for Transformer Probabilistic Models</em> (ICLR 2026) — <a href="https://openreview.net/forum?id=5bfUqlOhAH">OpenReview</a>.',
    )}`,
};

const CSS = `
.ab-root { display: flex; flex-direction: column; gap: 12px; }
.ab-hint { color: var(--muted); margin: 0; }
.ab-top { display: flex; gap: 18px; flex-wrap: wrap; align-items: flex-start; }
.ab-plot-col { display: flex; flex-direction: column; gap: 6px; }
.ab-main { border: 1px solid var(--line); border-radius: 8px; background: #fff; touch-action: none; }
.ab-status { color: var(--muted); margin: 0; font-size: 12px; font-variant-numeric: tabular-nums; }
.ab-controls { display: flex; flex-direction: column; gap: 14px; min-width: 240px; }
.ab-controls fieldset { border: 1px solid var(--line); border-radius: 8px; margin: 0; padding: 8px 10px; }
.ab-controls legend { color: var(--muted); font-size: 12px; padding: 0 4px; }
.ab-kernel-btns { display: flex; flex-wrap: wrap; gap: 6px; }
.ab-kernel-btns button { font: inherit; padding: 4px 8px; border: 1px solid var(--line);
  background: #fff; border-radius: 6px; cursor: pointer; }
.ab-kernel-btns button.sel { border-color: var(--accent); color: var(--accent); font-weight: 600; }
.ab-slider-row { display: flex; align-items: center; gap: 8px; }
.ab-slider-row input[type=range] { flex: 1; }
.ab-slider-row .val { font-variant-numeric: tabular-nums; color: var(--muted); min-width: 70px; text-align: right; }
.ab-btns { display: flex; gap: 8px; flex-wrap: wrap; }
.ab-btn { font: inherit; padding: 6px 10px; border: 1px solid var(--line); background: #fff;
  border-radius: 6px; cursor: pointer; }
`;

function injectCss(): void {
  if (document.getElementById("ab-style")) return;
  const s = document.createElement("style");
  s.id = "ab-style";
  s.textContent = CSS;
  document.head.appendChild(s);
}

const raf: (cb: () => void) => void =
  typeof requestAnimationFrame === "function"
    ? (cb) => requestAnimationFrame(() => cb())
    : (cb) => {
        setTimeout(cb, 16);
      };

export async function mountArbuf(el: HTMLElement): Promise<void> {
  injectCss();
  let model: BufferedACEModel;
  try {
    const weights = await loadWeights(
      `${import.meta.env.BASE_URL}models/gp1d_arbuffer`,
    );
    model = new BufferedACEModel(weights);
  } catch {
    el.innerHTML = `<p class="loading">The AR-buffer model weights are unavailable here.
      To run this tab locally, export the
      weights with <code>export_weights.py --task gp1d_arbuffer</code> — see the
      "Run locally" section of <code>playground/README.md</code>.</p>`;
    return;
  }

  const ellMeta = model.variables[1];
  const scaleMeta = model.variables[2];

  // --- state ---
  const defaultPoints: Point[] = [
    { x: -0.9, y: -0.4 },
    { x: -0.55, y: 0.15 },
    { x: -0.2, y: 0.55 },
    { x: 0.5, y: 0.05 },
  ];
  const points: Point[] = defaultPoints.map((p) => ({ ...p }));
  const pin: PinState = { kernel: null, ell: false, scale: false };
  let ellVal = 0.5 * (ellMeta.bound_lo + ellMeta.bound_hi);
  let scaleVal = 0.5 * (scaleMeta.bound_lo + scaleMeta.bound_hi);
  let samplerMode: "buffer" | "slow" = "buffer";
  let dragIdx: number | null = null;
  let seedCounter = 0;

  // Derived inference state (cache survives Resample; static survives Resample).
  let cache: CtxCache | null = null;
  let stat: ArbufStatic | null = null;
  let indep: number[][] = [];
  let sampler: ChainSampler | null = null;
  let encodeMs = 0;
  let decodeMs = 0;
  let epoch = 0; // bumped to cancel stale animation loops

  // --- DOM ---
  el.innerHTML = "";
  const root = document.createElement("div");
  root.className = "ab-root";
  root.innerHTML = `
    <p class="ab-hint">Coherent joint draws from a causal AR buffer
      (<a href="https://github.com/acerbilab/nanoACE/tree/main/extensions/arbuffer">extensions/arbuffer</a>):
      each colored curve is one whole-function sample — gray curves are independent
      per-point samples, for contrast.</p>
    <div class="ab-top">
      <div class="ab-plot-col">
        <canvas class="ab-main" width="660" height="380" style="width:660px;height:380px;"></canvas>
        <p class="ab-status"></p>
      </div>
      <div class="ab-controls">
        <fieldset>
          <legend>kernel</legend>
          <div class="ab-kernel-btns"></div>
        </fieldset>
        <fieldset>
          <legend>lengthscale</legend>
          <label class="ab-slider-row"><input type="checkbox" class="pin-ell"/>pin (condition on this value)</label>
          <div class="ab-slider-row"><input type="range" class="ell"/><span class="val ell-val"></span></div>
        </fieldset>
        <fieldset>
          <legend>outputscale</legend>
          <label class="ab-slider-row"><input type="checkbox" class="pin-scale"/>pin (condition on this value)</label>
          <div class="ab-slider-row"><input type="range" class="scale"/><span class="val scale-val"></span></div>
        </fieldset>
        <fieldset>
          <legend>sampling</legend>
          <label class="ab-slider-row"><input type="radio" name="ab-sampler" class="ab-mode-buffer" checked/>AR buffer (context cached)</label>
          <label class="ab-slider-row"><input type="radio" name="ab-sampler" class="ab-mode-slow"/>slow AR (re-encodes every step)</label>
          <div class="ab-btns">
            <button class="ab-btn ab-resample" title="Reuses the cached context encoding (AR buffer) — only the decode reruns">Resample</button>
          </div>
        </fieldset>
        <div class="ab-btns">
          <button class="ab-btn reset">Reset points</button>
          <button class="ab-btn clear">Clear points</button>
        </div>
      </div>
    </div>
  `;
  el.appendChild(root);
  addInfoButton(root.querySelector<HTMLElement>(".ab-hint")!, EXPLAINER);

  const mainCanvas = root.querySelector<HTMLCanvasElement>(".ab-main")!;
  const statusEl = root.querySelector<HTMLParagraphElement>(".ab-status")!;
  const kernelBtns = root.querySelector<HTMLDivElement>(".ab-kernel-btns")!;
  const ellSlider = root.querySelector<HTMLInputElement>(".ell")!;
  const scaleSlider = root.querySelector<HTMLInputElement>(".scale")!;
  const ellValEl = root.querySelector<HTMLSpanElement>(".ell-val")!;
  const scaleValEl = root.querySelector<HTMLSpanElement>(".scale-val")!;
  const pinEll = root.querySelector<HTMLInputElement>(".pin-ell")!;
  const pinScale = root.querySelector<HTMLInputElement>(".pin-scale")!;
  const modeBuffer = root.querySelector<HTMLInputElement>(".ab-mode-buffer")!;
  const modeSlow = root.querySelector<HTMLInputElement>(".ab-mode-slow")!;
  const resampleBtn = root.querySelector<HTMLButtonElement>(".ab-resample")!;

  function updateResampleTitle(): void {
    resampleBtn.title =
      samplerMode === "buffer"
        ? "Reuses the cached context encoding (AR buffer) — only the decode reruns"
        : "Slow AR: no cache — the context is re-encoded at every step";
  }

  const kernelOptions = ["Unknown", ...KERNEL_LABELS];
  const kernelButtonEls: HTMLButtonElement[] = [];
  kernelOptions.forEach((label, i) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.addEventListener("click", () => {
      pin.kernel = i === 0 ? null : i - 1;
      onContextChange();
    });
    kernelBtns.appendChild(b);
    kernelButtonEls.push(b);
  });

  ellSlider.min = String(ellMeta.bound_lo);
  ellSlider.max = String(ellMeta.bound_hi);
  ellSlider.step = String((ellMeta.bound_hi - ellMeta.bound_lo) / 200);
  ellSlider.value = String(ellVal);
  scaleSlider.min = String(scaleMeta.bound_lo);
  scaleSlider.max = String(scaleMeta.bound_hi);
  scaleSlider.step = String((scaleMeta.bound_hi - scaleMeta.bound_lo) / 200);
  scaleSlider.value = String(scaleVal);

  ellSlider.addEventListener("input", () => {
    ellVal = parseFloat(ellSlider.value);
    if (pin.ell) onContextChange();
    else updateControls();
  });
  scaleSlider.addEventListener("input", () => {
    scaleVal = parseFloat(scaleSlider.value);
    if (pin.scale) onContextChange();
    else updateControls();
  });
  pinEll.addEventListener("change", () => {
    pin.ell = pinEll.checked;
    onContextChange();
  });
  pinScale.addEventListener("change", () => {
    pin.scale = pinScale.checked;
    onContextChange();
  });
  modeBuffer.addEventListener("change", () => {
    if (modeBuffer.checked) {
      samplerMode = "buffer";
      updateResampleTitle();
      resample();
    }
  });
  modeSlow.addEventListener("change", () => {
    if (modeSlow.checked) {
      samplerMode = "slow";
      updateResampleTitle();
      resample();
    }
  });
  root
    .querySelector<HTMLButtonElement>(".ab-resample")!
    .addEventListener("click", () => {
      resample();
    });
  root
    .querySelector<HTMLButtonElement>(".reset")!
    .addEventListener("click", () => {
      points.length = 0;
      points.push(...defaultPoints.map((p) => ({ ...p })));
      onContextChange();
    });
  root
    .querySelector<HTMLButtonElement>(".clear")!
    .addEventListener("click", () => {
      points.length = 0;
      onContextChange();
    });

  // pointer interaction (GP-tab semantics)
  let mainPlot: Plot | null = null;
  const clampX = (x: number) => clamp(x, ARBUF.X_DOMAIN[0], ARBUF.X_DOMAIN[1]);
  const clampY = (y: number) => clamp(y, ARBUF.Y_VIEW[0], ARBUF.Y_VIEW[1]);

  mainCanvas.addEventListener("contextmenu", (e) => e.preventDefault());
  mainCanvas.addEventListener("pointerdown", (e) => {
    if (!mainPlot) return;
    const hit = hitPoint(
      points,
      mainPlot,
      e.offsetX,
      e.offsetY,
      ARBUF.HIT_RADIUS_PX,
    );
    if (hit !== null && (e.shiftKey || e.button === 2)) {
      points.splice(hit, 1);
      onContextChange();
      return;
    }
    if (hit !== null) {
      dragIdx = hit;
      mainCanvas.setPointerCapture(e.pointerId);
      return;
    }
    points.push({
      x: clampX(mainPlot.pxToX(e.offsetX)),
      y: clampY(mainPlot.pxToY(e.offsetY)),
    });
    onContextChange();
  });
  mainCanvas.addEventListener("pointermove", (e) => {
    if (dragIdx === null || !mainPlot) return;
    points[dragIdx] = {
      x: clampX(mainPlot.pxToX(e.offsetX)),
      y: clampY(mainPlot.pxToY(e.offsetY)),
    };
    onContextChange();
  });
  const endDrag = () => {
    dragIdx = null;
  };
  mainCanvas.addEventListener("pointerup", endDrag);
  mainCanvas.addEventListener("pointercancel", endDrag);

  // --- inference orchestration ---
  function spec(): ArbufSpec {
    return {
      points,
      pinKernel: pin.kernel,
      pinEll: pin.ell ? ellVal : null,
      pinScale: pin.scale ? scaleVal : null,
    };
  }

  function grid(): number[] {
    const n = ARBUF.GRID_POINTS;
    const out = new Array<number>(n);
    for (let i = 0; i < n; i++)
      out[i] =
        ARBUF.X_DOMAIN[0] +
        ((ARBUF.X_DOMAIN[1] - ARBUF.X_DOMAIN[0]) * i) / (n - 1);
    return out;
  }

  function hasContext(): boolean {
    return points.length > 0 || pin.kernel !== null || pin.ell || pin.scale;
  }

  function onContextChange(): void {
    updateControls();
    epoch += 1;
    if (!hasContext()) {
      cache = null;
      stat = null;
      sampler = null;
      drawEmpty();
      return;
    }
    const context = buildContext(model, spec());
    const t0 = performance.now();
    cache = model.encodeContext(context);
    encodeMs = performance.now() - t0;
    stat = arbufStatic(model, context, grid());
    resample();
  }

  function resample(): void {
    if (!cache || !stat) return;
    epoch += 1;
    const myEpoch = epoch;
    const rng = mulberry32(SEED0 + seedCounter++);
    indep = sampleIndependent(stat, ARBUF.DRAWS, rng);
    sampler =
      samplerMode === "buffer"
        ? new JointSampler(model, cache, stat.grid, {
            nDraws: ARBUF.DRAWS,
            rng,
          })
        : new SlowARSampler(model, buildContext(model, spec()), stat.grid, {
            nDraws: ARBUF.SLOW_DRAWS,
            rng,
          });
    decodeMs = 0;

    const tick = () => {
      if (myEpoch !== epoch || !sampler) return;
      const t0 = performance.now();
      for (let s = 0; s < ARBUF.STEPS_PER_FRAME && sampler.step(); s++) {
        // advance the chain
      }
      decodeMs += performance.now() - t0;
      draw();
      if (!sampler.done) raf(tick);
    };
    raf(tick);
  }

  // --- rendering ---
  function oodReasons(): string[] {
    const reasons = pointOodReasons(points, {
      yIsOod: (y) => Math.abs(y) > ARBUF.Y_OOD,
      yReason: `beyond training y-range (|y| > ${ARBUF.Y_OOD})`,
      maxPoints: ARBUF.MAX_CONTEXT_HINT,
      maxReason: (n) =>
        `${n} points (training used ≤ ${ARBUF.MAX_CONTEXT_HINT})`,
    });
    const nPinned =
      (pin.kernel !== null ? 1 : 0) + (pin.ell ? 1 : 0) + (pin.scale ? 1 : 0);
    if (points.length === 0 && nPinned > 0) {
      reasons.push(
        "latent-only context (training used at least 4 data points)",
      );
    }
    return reasons;
  }

  function updateControls(): void {
    kernelButtonEls.forEach((b, i) => {
      const sel = i === 0 ? pin.kernel === null : pin.kernel === i - 1;
      b.classList.toggle("sel", sel);
    });
    pinEll.checked = pin.ell;
    pinScale.checked = pin.scale;
    ellValEl.textContent = `ℓ=${Math.exp(ellVal).toFixed(3)}`;
    scaleValEl.textContent = `σ=${Math.exp(scaleVal).toFixed(3)}`;
  }

  function basePlot(): Plot {
    const p = makePlot(mainCanvas, {
      xDomain: ARBUF.X_DOMAIN,
      yDomain: ARBUF.Y_VIEW,
    });
    p.clear();
    p.rectData(
      ARBUF.X_DOMAIN[0],
      ARBUF.X_DOMAIN[1],
      ARBUF.Y_NORMAL[0],
      ARBUF.Y_NORMAL[1],
      "rgba(37,99,235,0.05)",
    );
    p.hline(0, "#eceef2", 1);
    return p;
  }

  function drawEmpty(): void {
    mainPlot = basePlot();
    mainPlot.axes();
    const ctx = mainPlot.ctx;
    ctx.fillStyle = "#9ca3af";
    ctx.font = "14px system-ui";
    ctx.textAlign = "center";
    ctx.fillText(
      "Add a point or pin a latent to draw joint samples.",
      mainPlot.width / 2,
      mainPlot.height / 2,
    );
    ctx.textAlign = "start";
    statusEl.textContent = "";
  }

  function draw(): void {
    if (!stat || !sampler) return;
    mainPlot = basePlot();
    const lo = stat.bandMean.map((m, i) => m - 2 * stat!.bandStd[i]);
    const hi = stat.bandMean.map((m, i) => m + 2 * stat!.bandStd[i]);
    mainPlot.band(stat.grid, lo, hi, "rgba(37,99,235,0.12)");

    // Independent marginal samples: full jagged lines, no coherence.
    for (const line of indep) mainPlot.line(stat.grid, line, INDEP_COLOR, 1);

    // Coherent draws: realized prefix of each chain, sorted by x for the polyline.
    for (let b = 0; b < sampler.values.length; b++) {
      const pts: Array<[number, number]> = [];
      for (let j = 0; j < stat.grid.length; j++) {
        const v = sampler.values[b][j];
        if (!Number.isNaN(v)) pts.push([stat.grid[j], v]);
      }
      const color = DRAW_COLORS[b % DRAW_COLORS.length];
      if (pts.length > 1) {
        mainPlot.line(
          pts.map((p) => p[0]),
          pts.map((p) => p[1]),
          color,
          1.4,
        );
      }
      mainPlot.dots(pts, color, 2);
    }

    mainPlot.dots(
      points
        .filter((p) => Math.abs(p.y) <= ARBUF.Y_OOD)
        .map((p) => [p.x, p.y] as [number, number]),
      "#111827",
      4,
    );
    mainPlot.dots(
      points
        .filter((p) => Math.abs(p.y) > ARBUF.Y_OOD)
        .map((p) => [p.x, p.y] as [number, number]),
      "#b45309",
      4,
    );
    mainPlot.axes();
    const b = sampler.values.length;
    mainPlot.label("diagonal ±2σ (context only)", 50, 16, { fill: "#2563eb" });
    mainPlot.label("independent marginal samples", 50, 30, { fill: "#6b7280" });
    mainPlot.label(
      `${b} coherent draw${b === 1 ? "" : "s"} (${samplerMode === "buffer" ? "AR buffer" : "slow AR"})`,
      50,
      44,
      { fill: DRAW_COLORS[0] },
    );

    const reasons = oodReasons();
    mainPlot.warning(
      reasons.length
        ? `Out of training distribution: ${reasons.join(" / ")}`
        : "",
    );

    const k = stat.grid.length;
    const decoding = sampler.done ? "" : ` (decoding ${sampler.steps}/${k}…)`;
    const perDraw =
      sampler.steps > 0 ? ` (≈${(decodeMs / b).toFixed(0)} ms per draw)` : "";
    statusEl.textContent =
      samplerMode === "buffer"
        ? `context encoded once: ${encodeMs.toFixed(1)} ms · ` +
          `decode ${b} × ${k} steps: ${decodeMs.toFixed(0)} ms${perDraw}${decoding}`
        : `slow AR — context re-encoded at every step · ` +
          `${b} × ${k} steps: ${decodeMs.toFixed(0)} ms${perDraw}${decoding}`;
  }

  onContextChange();
}
