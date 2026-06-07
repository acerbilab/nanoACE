/**
 * GP-1D interactive demo (UI layer).
 *
 * Click to add a context point, drag to move, shift-click to delete. Every change
 * triggers one ACE forward (via `gpInfer`) and redraws the posterior predictive
 * band, the kernel posterior, and the lengthscale/outputscale marginals. Any
 * latent can be pinned into context; pinning ≥2 at once is OOD for the current
 * checkpoint and raises the banner.
 */

import { GP, KERNEL_LABELS } from "../config";
import { makePlot, type Plot } from "../plot";
import { ACEModel } from "../ace/model";
import { loadWeights } from "../ace/weights";
import { gpInfer, type GPResult, type GPSpec } from "./infer";

interface Point {
  x: number;
  y: number;
}

interface PinState {
  kernel: number | null;
  ell: boolean;
  scale: boolean;
}

const CSS = `
.gp-root { display: flex; flex-direction: column; gap: 12px; }
.gp-hint { color: var(--muted); margin: 0; }
.gp-banner { background: var(--warn-bg); color: var(--warn); border: 1px solid #fed7aa;
  border-radius: 8px; padding: 8px 12px; font-size: 13px; }
.gp-top { display: flex; gap: 18px; flex-wrap: wrap; align-items: flex-start; }
.gp-main { border: 1px solid var(--line); border-radius: 8px; background: #fff; touch-action: none; }
.gp-controls { display: flex; flex-direction: column; gap: 14px; min-width: 240px; }
.gp-controls fieldset { border: 1px solid var(--line); border-radius: 8px; margin: 0; padding: 8px 10px; }
.gp-controls legend { color: var(--muted); font-size: 12px; padding: 0 4px; }
.gp-kernel-btns { display: flex; flex-wrap: wrap; gap: 6px; }
.gp-kernel-btns button { font: inherit; padding: 4px 8px; border: 1px solid var(--line);
  background: #fff; border-radius: 6px; cursor: pointer; }
.gp-kernel-btns button.sel { border-color: var(--accent); color: var(--accent); font-weight: 600; }
.gp-slider-row { display: flex; align-items: center; gap: 8px; }
.gp-slider-row input[type=range] { flex: 1; }
.gp-slider-row .val { font-variant-numeric: tabular-nums; color: var(--muted); min-width: 70px; text-align: right; }
.gp-panels { display: flex; gap: 18px; flex-wrap: wrap; }
.gp-panel { border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 6px; }
.gp-panel h4 { margin: 2px 6px 4px; font-size: 12px; color: var(--muted); font-weight: 600; }
.gp-btn { font: inherit; padding: 6px 10px; border: 1px solid var(--line); background: #fff;
  border-radius: 6px; cursor: pointer; align-self: flex-start; }
`;

function injectCss(): void {
  if (document.getElementById("gp-style")) return;
  const s = document.createElement("style");
  s.id = "gp-style";
  s.textContent = CSS;
  document.head.appendChild(s);
}

export async function mountGP(el: HTMLElement): Promise<void> {
  injectCss();
  const weights = await loadWeights(`${import.meta.env.BASE_URL}models/gp1d`);
  const model = new ACEModel(weights);

  const ellMeta = model.variables[1];
  const scaleMeta = model.variables[2];

  // --- state ---
  const points: Point[] = [
    { x: -0.9, y: -0.4 },
    { x: -0.55, y: 0.15 },
    { x: -0.2, y: 0.55 },
    { x: 0.15, y: 0.5 },
    { x: 0.5, y: 0.05 },
    { x: 0.85, y: -0.5 },
  ];
  const pin: PinState = { kernel: null, ell: false, scale: false };
  let ellVal = 0.5 * (ellMeta.bound_lo + ellMeta.bound_hi);
  let scaleVal = 0.5 * (scaleMeta.bound_lo + scaleMeta.bound_hi);
  let dragIdx: number | null = null;

  // --- DOM ---
  el.innerHTML = "";
  const root = document.createElement("div");
  root.className = "gp-root";
  root.innerHTML = `
    <p class="gp-hint">Click empty space to add a point · drag a point to move it · shift-click to delete.
      The model re-conditions instantly on every change.</p>
    <div class="gp-banner" hidden></div>
    <div class="gp-top">
      <canvas class="gp-main" width="660" height="380" style="width:660px;height:380px;"></canvas>
      <div class="gp-controls">
        <fieldset>
          <legend>kernel</legend>
          <div class="gp-kernel-btns"></div>
        </fieldset>
        <fieldset>
          <legend>lengthscale</legend>
          <label class="gp-slider-row"><input type="checkbox" class="pin-ell"/>pin (condition on this value)</label>
          <div class="gp-slider-row"><input type="range" class="ell"/><span class="val ell-val"></span></div>
        </fieldset>
        <fieldset>
          <legend>outputscale</legend>
          <label class="gp-slider-row"><input type="checkbox" class="pin-scale"/>pin (condition on this value)</label>
          <div class="gp-slider-row"><input type="range" class="scale"/><span class="val scale-val"></span></div>
        </fieldset>
        <button class="gp-btn clear">Clear points</button>
      </div>
    </div>
    <div class="gp-panels">
      <div class="gp-panel"><h4>kernel posterior</h4>
        <canvas class="gp-kernel" width="320" height="200" style="width:320px;height:200px;"></canvas></div>
      <div class="gp-panel"><h4>latent marginals (log scale, scaled to peak)</h4>
        <canvas class="gp-latents" width="340" height="200" style="width:340px;height:200px;"></canvas></div>
    </div>
  `;
  el.appendChild(root);

  const banner = root.querySelector<HTMLDivElement>(".gp-banner")!;
  const mainCanvas = root.querySelector<HTMLCanvasElement>(".gp-main")!;
  const kernelCanvas = root.querySelector<HTMLCanvasElement>(".gp-kernel")!;
  const latentCanvas = root.querySelector<HTMLCanvasElement>(".gp-latents")!;
  const kernelBtns = root.querySelector<HTMLDivElement>(".gp-kernel-btns")!;
  const ellSlider = root.querySelector<HTMLInputElement>(".ell")!;
  const scaleSlider = root.querySelector<HTMLInputElement>(".scale")!;
  const ellValEl = root.querySelector<HTMLSpanElement>(".ell-val")!;
  const scaleValEl = root.querySelector<HTMLSpanElement>(".scale-val")!;
  const pinEll = root.querySelector<HTMLInputElement>(".pin-ell")!;
  const pinScale = root.querySelector<HTMLInputElement>(".pin-scale")!;

  const kernelOptions = ["Unknown", ...KERNEL_LABELS];
  const kernelButtonEls: HTMLButtonElement[] = [];
  kernelOptions.forEach((label, i) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.addEventListener("click", () => {
      pin.kernel = i === 0 ? null : i - 1;
      render();
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
    render();
  });
  scaleSlider.addEventListener("input", () => {
    scaleVal = parseFloat(scaleSlider.value);
    render();
  });
  pinEll.addEventListener("change", () => {
    pin.ell = pinEll.checked;
    render();
  });
  pinScale.addEventListener("change", () => {
    pin.scale = pinScale.checked;
    render();
  });
  root.querySelector<HTMLButtonElement>(".clear")!.addEventListener("click", () => {
    points.length = 0;
    render();
  });

  // pointer interaction
  let mainPlot: Plot | null = null;
  const hitPoint = (px: number, py: number): number | null => {
    if (!mainPlot) return null;
    for (let i = 0; i < points.length; i++) {
      const dx = mainPlot.xPx(points[i].x) - px;
      const dy = mainPlot.yPx(points[i].y) - py;
      if (Math.hypot(dx, dy) <= GP.HIT_RADIUS_PX) return i;
    }
    return null;
  };
  const clampX = (x: number) => Math.min(Math.max(x, GP.X_DOMAIN[0]), GP.X_DOMAIN[1]);
  const clampY = (y: number) => Math.min(Math.max(y, GP.Y_VIEW[0]), GP.Y_VIEW[1]);

  mainCanvas.addEventListener("contextmenu", (e) => e.preventDefault());
  mainCanvas.addEventListener("pointerdown", (e) => {
    if (!mainPlot) return;
    const hit = hitPoint(e.offsetX, e.offsetY);
    if (hit !== null && (e.shiftKey || e.button === 2)) {
      points.splice(hit, 1);
      render();
      return;
    }
    if (hit !== null) {
      dragIdx = hit;
      mainCanvas.setPointerCapture(e.pointerId);
      return;
    }
    points.push({ x: clampX(mainPlot.pxToX(e.offsetX)), y: clampY(mainPlot.pxToY(e.offsetY)) });
    render();
  });
  mainCanvas.addEventListener("pointermove", (e) => {
    if (dragIdx === null || !mainPlot) return;
    points[dragIdx] = { x: clampX(mainPlot.pxToX(e.offsetX)), y: clampY(mainPlot.pxToY(e.offsetY)) };
    render();
  });
  const endDrag = () => {
    dragIdx = null;
  };
  mainCanvas.addEventListener("pointerup", endDrag);
  mainCanvas.addEventListener("pointercancel", endDrag);

  // --- rendering ---
  function oodReasons(): string[] {
    const reasons: string[] = [];
    const nFar = points.filter((p) => Math.abs(p.y) > GP.Y_OOD).length;
    if (nFar > 0) reasons.push(`${nFar} point(s) beyond training y-range (|y| > ${GP.Y_OOD})`);
    if (points.length > GP.MAX_CONTEXT_HINT)
      reasons.push(`${points.length} points (training used ≤ ${GP.MAX_CONTEXT_HINT})`);
    // Multi-pin is in-distribution since the multi-latent-reveal retrain, so pinning
    // any subset of latents no longer flags OOD.
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

  function render(): void {
    updateControls();
    const reasons = oodReasons();
    banner.hidden = reasons.length === 0;
    banner.textContent = reasons.length ? `⚠ Out of training distribution: ${reasons.join(" · ")}` : "";

    const spec: GPSpec = {
      points,
      pinKernel: pin.kernel,
      pinEll: pin.ell ? ellVal : null,
      pinScale: pin.scale ? scaleVal : null,
    };
    const res = gpInfer(model, spec);

    if (!res.hasContext) {
      drawEmptyMain();
      clearPanels();
      return;
    }
    drawMain(res);
    drawKernel(res.kernelProbs);
    drawLatents(res);
  }

  function baseMain(): Plot {
    const p = makePlot(mainCanvas, { xDomain: GP.X_DOMAIN, yDomain: GP.Y_VIEW });
    p.clear();
    p.rectData(GP.X_DOMAIN[0], GP.X_DOMAIN[1], GP.Y_NORMAL[0], GP.Y_NORMAL[1], "rgba(37,99,235,0.05)");
    p.hline(0, "#eceef2", 1);
    return p;
  }

  function drawEmptyMain(): void {
    mainPlot = baseMain();
    mainPlot.axes();
    const ctx = mainPlot.ctx;
    ctx.fillStyle = "#9ca3af";
    ctx.font = "14px system-ui";
    ctx.textAlign = "center";
    ctx.fillText("Add a point or pin a latent to see predictions.", mainPlot.width / 2, mainPlot.height / 2);
    ctx.textAlign = "start";
  }

  function drawMain(res: GPResult): void {
    mainPlot = baseMain();
    const lo = res.bandMean.map((m, i) => m - 2 * res.bandStd[i]);
    const hi = res.bandMean.map((m, i) => m + 2 * res.bandStd[i]);
    mainPlot.band(res.bandX, lo, hi, "rgba(37,99,235,0.16)");
    mainPlot.line(res.bandX, res.bandMean, "#2563eb", 1.8);
    mainPlot.dots(
      points.filter((p) => Math.abs(p.y) <= GP.Y_OOD).map((p) => [p.x, p.y] as [number, number]),
      "#111827",
      4,
    );
    mainPlot.dots(
      points.filter((p) => Math.abs(p.y) > GP.Y_OOD).map((p) => [p.x, p.y] as [number, number]),
      "#b45309",
      4,
    );
    mainPlot.axes();
  }

  function clearPanels(): void {
    makePlot(kernelCanvas, { xDomain: [0, 1], yDomain: [0, 1] }).clear();
    makePlot(latentCanvas, { xDomain: [0, 1], yDomain: [0, 1] }).clear();
  }

  function drawKernel(probs: number[] | null): void {
    const p = makePlot(kernelCanvas, { xDomain: [0, 1], yDomain: [0, 1], padding: { l: 8, r: 8, t: 8, b: 28 } });
    p.clear();
    const n = KERNEL_LABELS.length;
    const ctx = p.ctx;
    const w = (p.width - 16) / n;
    for (let i = 0; i < n; i++) {
      const x0 = 8 + i * w + 4;
      const bw = w - 8;
      const isPinned = pin.kernel === i;
      const prob = probs ? probs[i] : isPinned ? 1 : 0;
      const barH = (p.height - 36) * prob;
      ctx.fillStyle = isPinned ? "rgba(180,83,9,0.30)" : "rgba(37,99,235,0.55)";
      ctx.fillRect(x0, p.height - 28 - barH, bw, barH);
      if (isPinned) {
        ctx.strokeStyle = "#b45309";
        ctx.lineWidth = 2;
        ctx.strokeRect(x0, 8, bw, p.height - 36);
      }
      ctx.fillStyle = "#374151";
      ctx.font = "11px system-ui";
      ctx.textAlign = "center";
      ctx.fillText(KERNEL_LABELS[i], x0 + bw / 2, p.height - 12);
      if (probs) ctx.fillText(prob.toFixed(2), x0 + bw / 2, Math.max(p.height - 32 - barH, 14));
    }
    ctx.textAlign = "start";
  }

  function drawLatents(res: GPResult): void {
    const xLo = Math.min(ellMeta.bound_lo, scaleMeta.bound_lo);
    const xHi = Math.max(ellMeta.bound_hi, scaleMeta.bound_hi);
    const p = makePlot(latentCanvas, { xDomain: [xLo, xHi], yDomain: [0, 1.08] });
    p.clear();
    p.axes();
    const peak = (arr: number[]) => {
      const m = Math.max(...arr, 1e-12);
      return arr.map((v) => v / m);
    };
    if (res.ellPost) p.line(res.ellGrid, peak(res.ellPost), "#2563eb", 1.8);
    else p.vline(ellVal, "#2563eb", 3);
    if (res.scalePost) p.line(res.scaleGrid, peak(res.scalePost), "#ea580c", 1.8);
    else p.vline(scaleVal, "#ea580c", 3);
    const ctx = p.ctx;
    ctx.font = "11px system-ui";
    ctx.fillStyle = "#2563eb";
    ctx.fillText(pin.ell ? "lengthscale (pinned)" : "lengthscale", 44, 16);
    ctx.fillStyle = "#ea580c";
    ctx.fillText(pin.scale ? "outputscale (pinned)" : "outputscale", 44, 30);
  }

  render();
}
