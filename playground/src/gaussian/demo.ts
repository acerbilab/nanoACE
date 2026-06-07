/**
 * Gaussian (μ, σ) prior-conditioning demo (UI layer).
 *
 * Set Beta priors over μ and log σ with the sliders, add observations by clicking
 * the predictive panel, and watch ACE's posterior marginals + posterior
 * predictive track the analytic oracle. This dramatizes runtime prior
 * conditioning (ACEP): the two prior tokens are always present (Beta(1,1) = the
 * uninformative case at the neutral slider settings).
 */

import { GAUSSIAN } from "../config";
import { makePlot, type Plot } from "../plot";
import { ACEModel } from "../ace/model";
import { loadWeights } from "../ace/weights";
import { linspace, normalize } from "../util";
import { gaussInfer, type GaussGrids } from "./infer";
import { analyticPosterior, betaLogPriorOnGrid, predictiveDensity } from "./oracle";

const CSS = `
.ga-root { display: flex; flex-direction: column; gap: 12px; }
.ga-hint { color: var(--muted); margin: 0; }
.ga-banner { background: var(--warn-bg); color: var(--warn); border: 1px solid #fed7aa;
  border-radius: 8px; padding: 8px 12px; font-size: 13px; }
.ga-top { display: flex; gap: 18px; flex-wrap: wrap; align-items: flex-start; }
.ga-main { border: 1px solid var(--line); border-radius: 8px; background: #fff; touch-action: none; }
.ga-controls { display: flex; flex-direction: column; gap: 12px; min-width: 260px; }
.ga-controls fieldset { border: 1px solid var(--line); border-radius: 8px; margin: 0; padding: 8px 10px; }
.ga-controls legend { color: var(--muted); font-size: 12px; padding: 0 4px; }
.ga-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
.ga-row label { width: 34px; color: var(--muted); }
.ga-row input[type=range] { flex: 1; }
.ga-row .val { font-variant-numeric: tabular-nums; color: var(--muted); min-width: 58px; text-align: right; }
.ga-panels { display: flex; gap: 18px; flex-wrap: wrap; }
.ga-panel { border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 6px; }
.ga-panel h4 { margin: 2px 6px 4px; font-size: 12px; color: var(--muted); font-weight: 600; }
.ga-btn { font: inherit; padding: 6px 10px; border: 1px solid var(--line); background: #fff;
  border-radius: 6px; cursor: pointer; align-self: flex-start; }
.ga-legend { font-size: 11px; color: var(--muted); display: flex; gap: 12px; padding: 0 6px; }
.ga-legend span::before { content: "■ "; }
`;

function injectCss(): void {
  if (document.getElementById("ga-style")) return;
  const s = document.createElement("style");
  s.id = "ga-style";
  s.textContent = CSS;
  document.head.appendChild(s);
}

const COL = { ace: "#2563eb", oracle: "#16a34a", prior: "#9ca3af" };
const BETA_UNIT_EPS = 1e-4;

function clampBetaUnit(x: number): number {
  return Math.min(Math.max(x, BETA_UNIT_EPS), 1.0 - BETA_UNIT_EPS);
}

export async function mountGaussian(el: HTMLElement): Promise<void> {
  injectCss();
  const weights = await loadWeights(`${import.meta.env.BASE_URL}models/gaussian`);
  const model = new ACEModel(weights);

  const muMeta = model.variables[1];
  const lsMeta = model.variables[2];
  const muRange: [number, number] = [muMeta.bound_lo, muMeta.bound_hi];
  const lsRange: [number, number] = [lsMeta.bound_lo, lsMeta.bound_hi];
  const grids: GaussGrids = {
    muGrid: linspace(muRange[0], muRange[1], GAUSSIAN.BINS),
    lsGrid: linspace(lsRange[0], lsRange[1], GAUSSIAN.BINS),
    yGrid: linspace(GAUSSIAN.Y_VIEW[0], GAUSSIAN.Y_VIEW[1], GAUSSIAN.Y_POINTS),
  };

  // --- state (priors stored as native mean + concentration nu) ---
  const yObs: number[] = [0.6, 0.85, -0.1, 1.0];
  let muMean = 0.5 * (muRange[0] + muRange[1]);
  let lsMean = 0.5 * (lsRange[0] + lsRange[1]);
  let muNu = 2;
  let lsNu = 2;
  let dragIdx: number | null = null;

  // --- DOM ---
  el.innerHTML = "";
  const root = document.createElement("div");
  root.className = "ga-root";
  const lnLo = Math.log(GAUSSIAN.NU_RANGE[0]);
  const lnHi = Math.log(GAUSSIAN.NU_RANGE[1]);
  root.innerHTML = `
    <p class="ga-hint">Click the top panel to add an observation (its value is the x-position) ·
      drag to move · shift-click to delete. Set Beta priors below and watch ACE track the oracle.</p>
    <div class="ga-banner" hidden></div>
    <div class="ga-legend"><span style="color:${COL.ace}">ACE</span>
      <span style="color:${COL.oracle}">oracle</span>
      <span style="color:${COL.prior}">prior</span></div>
    <div class="ga-top">
      <canvas class="ga-main" width="560" height="300" style="width:560px;height:300px;"></canvas>
      <div class="ga-controls">
        <fieldset>
          <legend>prior on μ (Beta)</legend>
          <div class="ga-row"><label>mean</label><input type="range" class="mu-mean"/><span class="val mu-mean-v"></span></div>
          <div class="ga-row"><label>conc.</label><input type="range" class="mu-nu"/><span class="val mu-nu-v"></span></div>
        </fieldset>
        <fieldset>
          <legend>prior on log σ (Beta)</legend>
          <div class="ga-row"><label>mean</label><input type="range" class="ls-mean"/><span class="val ls-mean-v"></span></div>
          <div class="ga-row"><label>conc.</label><input type="range" class="ls-nu"/><span class="val ls-nu-v"></span></div>
        </fieldset>
        <button class="ga-btn clear">Clear observations</button>
      </div>
    </div>
    <div class="ga-panels">
      <div class="ga-panel"><h4>μ posterior</h4>
        <canvas class="ga-mu" width="320" height="200" style="width:320px;height:200px;"></canvas></div>
      <div class="ga-panel"><h4>log σ posterior</h4>
        <canvas class="ga-ls" width="320" height="200" style="width:320px;height:200px;"></canvas></div>
    </div>
  `;
  el.appendChild(root);

  const banner = root.querySelector<HTMLDivElement>(".ga-banner")!;
  const mainCanvas = root.querySelector<HTMLCanvasElement>(".ga-main")!;
  const muCanvas = root.querySelector<HTMLCanvasElement>(".ga-mu")!;
  const lsCanvas = root.querySelector<HTMLCanvasElement>(".ga-ls")!;

  const muMeanS = root.querySelector<HTMLInputElement>(".mu-mean")!;
  const muNuS = root.querySelector<HTMLInputElement>(".mu-nu")!;
  const lsMeanS = root.querySelector<HTMLInputElement>(".ls-mean")!;
  const lsNuS = root.querySelector<HTMLInputElement>(".ls-nu")!;
  const muMeanV = root.querySelector<HTMLSpanElement>(".mu-mean-v")!;
  const muNuV = root.querySelector<HTMLSpanElement>(".mu-nu-v")!;
  const lsMeanV = root.querySelector<HTMLSpanElement>(".ls-mean-v")!;
  const lsNuV = root.querySelector<HTMLSpanElement>(".ls-nu-v")!;

  const setupRange = (s: HTMLInputElement, lo: number, hi: number, val: number) => {
    s.min = String(lo);
    s.max = String(hi);
    s.step = String((hi - lo) / 200);
    s.value = String(val);
  };
  setupRange(muMeanS, muRange[0], muRange[1], muMean);
  setupRange(lsMeanS, lsRange[0], lsRange[1], lsMean);
  setupRange(muNuS, lnLo, lnHi, Math.log(muNu));
  setupRange(lsNuS, lnLo, lnHi, Math.log(lsNu));

  muMeanS.addEventListener("input", () => {
    muMean = parseFloat(muMeanS.value);
    render();
  });
  lsMeanS.addEventListener("input", () => {
    lsMean = parseFloat(lsMeanS.value);
    render();
  });
  muNuS.addEventListener("input", () => {
    muNu = Math.exp(parseFloat(muNuS.value));
    render();
  });
  lsNuS.addEventListener("input", () => {
    lsNu = Math.exp(parseFloat(lsNuS.value));
    render();
  });
  root.querySelector<HTMLButtonElement>(".clear")!.addEventListener("click", () => {
    yObs.length = 0;
    render();
  });

  // observation interaction on the predictive panel (x-axis = observation value)
  let mainPlot: Plot | null = null;
  const hitObs = (px: number): number | null => {
    if (!mainPlot) return null;
    for (let i = 0; i < yObs.length; i++) {
      if (Math.abs(mainPlot.xPx(yObs[i]) - px) <= 8) return i;
    }
    return null;
  };
  const clampY = (y: number) => Math.min(Math.max(y, GAUSSIAN.Y_VIEW[0]), GAUSSIAN.Y_VIEW[1]);

  mainCanvas.addEventListener("contextmenu", (e) => e.preventDefault());
  mainCanvas.addEventListener("pointerdown", (e) => {
    if (!mainPlot) return;
    const hit = hitObs(e.offsetX);
    if (hit !== null && (e.shiftKey || e.button === 2)) {
      yObs.splice(hit, 1);
      render();
      return;
    }
    if (hit !== null) {
      dragIdx = hit;
      mainCanvas.setPointerCapture(e.pointerId);
      return;
    }
    yObs.push(clampY(mainPlot.pxToX(e.offsetX)));
    render();
  });
  mainCanvas.addEventListener("pointermove", (e) => {
    if (dragIdx === null || !mainPlot) return;
    yObs[dragIdx] = clampY(mainPlot.pxToX(e.offsetX));
    render();
  });
  const endDrag = () => {
    dragIdx = null;
  };
  mainCanvas.addEventListener("pointerup", endDrag);
  mainCanvas.addEventListener("pointercancel", endDrag);

  // --- rendering ---
  function updateControls(): void {
    muMeanV.textContent = muMean.toFixed(2);
    lsMeanV.textContent = `σ=${Math.exp(lsMean).toFixed(2)}`;
    muNuV.textContent = `ν=${muNu.toFixed(0)}`;
    lsNuV.textContent = `ν=${lsNu.toFixed(0)}`;
  }

  function render(): void {
    updateControls();
    const muUnit = clampBetaUnit((muMean - muRange[0]) / (muRange[1] - muRange[0]));
    const lsUnit = clampBetaUnit((lsMean - lsRange[0]) / (lsRange[1] - lsRange[0]));

    const nFar = yObs.filter((y) => Math.abs(y) > GAUSSIAN.Y_OOD).length;
    banner.hidden = nFar === 0;
    banner.textContent = nFar ? `⚠ ${nFar} observation(s) beyond training y-range (|y| > ${GAUSSIAN.Y_OOD})` : "";

    const params = { yObs, muUnit, muNu, lsUnit, lsNu };
    const ace = gaussInfer(model, params, grids);
    const oracle = analyticPosterior(yObs, grids.muGrid, grids.lsGrid, muRange, lsRange, { muUnit, muNu, lsUnit, lsNu });
    const predOracle = predictiveDensity(oracle, grids.yGrid);

    const priorMu = normalize(betaLogPriorOnGrid(grids.muGrid, muUnit, muNu, muRange[0], muRange[1]));
    const priorLs = normalize(betaLogPriorOnGrid(grids.lsGrid, lsUnit, lsNu, lsRange[0], lsRange[1]));

    drawPredictive(ace.predDensity, predOracle);
    drawMarginal(muCanvas, muRange, grids.muGrid, oracle.muPost, ace.muPost, priorMu);
    drawMarginal(lsCanvas, lsRange, grids.lsGrid, oracle.lsPost, ace.lsPost, priorLs);
  }

  function drawPredictive(ace: number[], oracle: number[]): void {
    const top = Math.max(...ace, ...oracle, 1e-6) * 1.1;
    mainPlot = makePlot(mainCanvas, { xDomain: GAUSSIAN.Y_VIEW, yDomain: [0, top] });
    mainPlot.clear();
    mainPlot.rectData(GAUSSIAN.Y_NORMAL[0], GAUSSIAN.Y_NORMAL[1], 0, top, "rgba(37,99,235,0.04)");
    mainPlot.line(grids.yGrid, oracle, COL.oracle, 1.8);
    mainPlot.line(grids.yGrid, ace, COL.ace, 1.8);
    // observation ticks along the baseline
    const ctx = mainPlot.ctx;
    for (const y of yObs) {
      const far = Math.abs(y) > GAUSSIAN.Y_OOD;
      mainPlot.vline(y, far ? "rgba(180,83,9,0.35)" : "rgba(17,24,39,0.18)", 1);
      ctx.fillStyle = far ? "#b45309" : "#111827";
      ctx.beginPath();
      ctx.arc(mainPlot.xPx(y), mainPlot.yPx(0), 4, 0, 2 * Math.PI);
      ctx.fill();
    }
    mainPlot.axes();
    ctx.fillStyle = "#9ca3af";
    ctx.font = "11px system-ui";
    ctx.fillText("posterior predictive p(new y)", 44, 14);
  }

  function drawMarginal(
    canvas: HTMLCanvasElement,
    xRange: [number, number],
    grid: number[],
    oracle: number[],
    ace: number[],
    prior: number[],
  ): void {
    const top = Math.max(...oracle, ...ace, ...prior, 1e-6) * 1.1;
    const p = makePlot(canvas, { xDomain: xRange, yDomain: [0, top] });
    p.clear();
    p.axes();
    p.line(grid, prior, COL.prior, 1.3);
    p.line(grid, oracle, COL.oracle, 1.8);
    p.line(grid, ace, COL.ace, 1.8);
  }

  render();
}
