/**
 * SIR simulation-based-inference demo (UI layer).
 *
 * Edit noisy infected-fraction observations and runtime Beta priors over beta
 * and gamma; ACE updates instantly, while the browser-side grid oracle provides
 * a numerical reference.
 */

import { SIR } from "../config";
import { makePlot, type Plot } from "../plot";
import { ACEModel } from "../ace/model";
import { loadWeights } from "../ace/weights";
import { linspace, normalize } from "../util";
import { betaLogPriorOnGrid } from "../gaussian/oracle";
import { defaultSIRGrids, sirInfer, type SIRObservation } from "./infer";
import { buildSirOracleCache, integrateSirAtTimes, sirOracle } from "./oracle";

const CSS = `
.sir-root { display: flex; flex-direction: column; gap: 12px; }
.sir-hint { color: var(--muted); margin: 0; }
.sir-banner { background: var(--warn-bg); color: var(--warn); border: 1px solid #fed7aa;
  border-radius: 8px; padding: 8px 12px; font-size: 13px; }
.sir-legend { font-size: 11px; color: var(--muted); display: flex; gap: 12px; padding: 0 6px; }
.sir-legend span::before { content: "■ "; }
.sir-top { display: flex; gap: 18px; flex-wrap: wrap; align-items: flex-start; }
.sir-main { border: 1px solid var(--line); border-radius: 8px; background: #fff; touch-action: none; }
.sir-controls { display: flex; flex-direction: column; gap: 12px; min-width: 260px; }
.sir-controls fieldset { border: 1px solid var(--line); border-radius: 8px; margin: 0; padding: 8px 10px; }
.sir-controls legend { color: var(--muted); font-size: 12px; padding: 0 4px; }
.sir-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
.sir-row label { width: 42px; color: var(--muted); }
.sir-row input[type=range] { flex: 1; }
.sir-row .val { font-variant-numeric: tabular-nums; color: var(--muted); min-width: 66px; text-align: right; }
.sir-panels { display: flex; gap: 18px; flex-wrap: wrap; }
.sir-panel { border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 6px; }
.sir-panel h4 { margin: 2px 6px 4px; font-size: 12px; color: var(--muted); font-weight: 600; }
.sir-btns { display: flex; gap: 8px; flex-wrap: wrap; }
.sir-btn { font: inherit; padding: 6px 10px; border: 1px solid var(--line); background: #fff;
  border-radius: 6px; cursor: pointer; }
`;

const COL = { ace: "#2563eb", oracle: "#16a34a", prior: "#9ca3af" };
const BETA_UNIT_EPS = 1e-4;
const DEFAULT_BETA = 0.55;
const DEFAULT_GAMMA = 0.18;
const DEFAULT_TIMES = [3, 6, 9, 12];

function injectCss(): void {
  if (document.getElementById("sir-style")) return;
  const s = document.createElement("style");
  s.id = "sir-style";
  s.textContent = CSS;
  document.head.appendChild(s);
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(Math.max(x, lo), hi);
}

function clampBetaUnit(x: number): number {
  return clamp(x, BETA_UNIT_EPS, 1.0 - BETA_UNIT_EPS);
}

export async function mountSIR(el: HTMLElement): Promise<void> {
  injectCss();
  const weights = await loadWeights(`${import.meta.env.BASE_URL}models/sbi_sir`);
  const model = new ACEModel(weights);

  const betaMeta = model.variables[1];
  const gammaMeta = model.variables[2];
  const betaRange: [number, number] = [betaMeta.bound_lo, betaMeta.bound_hi];
  const gammaRange: [number, number] = [gammaMeta.bound_lo, gammaMeta.bound_hi];
  const grids = defaultSIRGrids(model);
  const oracleCache = buildSirOracleCache(grids.betaGrid, grids.gammaGrid);

  const observations: SIRObservation[] = [];
  function resetObservations(): void {
    observations.length = 0;
    const ys = integrateSirAtTimes(DEFAULT_BETA, DEFAULT_GAMMA, DEFAULT_TIMES);
    for (let i = 0; i < DEFAULT_TIMES.length; i++) observations.push({ t: DEFAULT_TIMES[i], y: ys[i] });
  }
  resetObservations();

  let betaMean = betaRange[0] + 0.60 * (betaRange[1] - betaRange[0]);
  let gammaMean = gammaRange[0] + 0.45 * (gammaRange[1] - gammaRange[0]);
  let betaNu = 12;
  let gammaNu = 10;
  let dragIdx: number | null = null;

  el.innerHTML = "";
  const root = document.createElement("div");
  root.className = "sir-root";
  const lnLo = Math.log(SIR.NU_RANGE[0]);
  const lnHi = Math.log(SIR.NU_RANGE[1]);
  root.innerHTML = `
    <p class="sir-hint">Click the curve panel to add an observation · drag to move · shift-click to delete.
      The oracle is a live beta/gamma grid over deterministic SIR trajectories.</p>
    <div class="sir-banner" hidden></div>
    <div class="sir-legend"><span style="color:${COL.ace}">ACE</span>
      <span style="color:${COL.oracle}">oracle</span>
      <span style="color:${COL.prior}">prior</span></div>
    <div class="sir-top">
      <canvas class="sir-main" width="660" height="380" style="width:660px;height:380px;"></canvas>
      <div class="sir-controls">
        <fieldset>
          <legend>prior on beta</legend>
          <div class="sir-row"><label>mean</label><input type="range" class="beta-mean"/><span class="val beta-mean-v"></span></div>
          <div class="sir-row"><label>conc.</label><input type="range" class="beta-nu"/><span class="val beta-nu-v"></span></div>
        </fieldset>
        <fieldset>
          <legend>prior on gamma</legend>
          <div class="sir-row"><label>mean</label><input type="range" class="gamma-mean"/><span class="val gamma-mean-v"></span></div>
          <div class="sir-row"><label>conc.</label><input type="range" class="gamma-nu"/><span class="val gamma-nu-v"></span></div>
        </fieldset>
        <div class="sir-btns">
          <button class="sir-btn reset">Reset case</button>
          <button class="sir-btn clear">Clear observations</button>
        </div>
      </div>
    </div>
    <div class="sir-panels">
      <div class="sir-panel"><h4>beta posterior</h4>
        <canvas class="sir-beta" width="320" height="200" style="width:320px;height:200px;"></canvas></div>
      <div class="sir-panel"><h4>gamma posterior</h4>
        <canvas class="sir-gamma" width="320" height="200" style="width:320px;height:200px;"></canvas></div>
    </div>
  `;
  el.appendChild(root);

  const banner = root.querySelector<HTMLDivElement>(".sir-banner")!;
  const mainCanvas = root.querySelector<HTMLCanvasElement>(".sir-main")!;
  const betaCanvas = root.querySelector<HTMLCanvasElement>(".sir-beta")!;
  const gammaCanvas = root.querySelector<HTMLCanvasElement>(".sir-gamma")!;
  const betaMeanS = root.querySelector<HTMLInputElement>(".beta-mean")!;
  const betaNuS = root.querySelector<HTMLInputElement>(".beta-nu")!;
  const gammaMeanS = root.querySelector<HTMLInputElement>(".gamma-mean")!;
  const gammaNuS = root.querySelector<HTMLInputElement>(".gamma-nu")!;
  const betaMeanV = root.querySelector<HTMLSpanElement>(".beta-mean-v")!;
  const betaNuV = root.querySelector<HTMLSpanElement>(".beta-nu-v")!;
  const gammaMeanV = root.querySelector<HTMLSpanElement>(".gamma-mean-v")!;
  const gammaNuV = root.querySelector<HTMLSpanElement>(".gamma-nu-v")!;

  const setupRange = (s: HTMLInputElement, lo: number, hi: number, val: number) => {
    s.min = String(lo);
    s.max = String(hi);
    s.step = String((hi - lo) / 240);
    s.value = String(val);
  };
  setupRange(betaMeanS, betaRange[0], betaRange[1], betaMean);
  setupRange(gammaMeanS, gammaRange[0], gammaRange[1], gammaMean);
  setupRange(betaNuS, lnLo, lnHi, Math.log(betaNu));
  setupRange(gammaNuS, lnLo, lnHi, Math.log(gammaNu));

  betaMeanS.addEventListener("input", () => {
    betaMean = parseFloat(betaMeanS.value);
    render();
  });
  gammaMeanS.addEventListener("input", () => {
    gammaMean = parseFloat(gammaMeanS.value);
    render();
  });
  betaNuS.addEventListener("input", () => {
    betaNu = Math.exp(parseFloat(betaNuS.value));
    render();
  });
  gammaNuS.addEventListener("input", () => {
    gammaNu = Math.exp(parseFloat(gammaNuS.value));
    render();
  });
  root.querySelector<HTMLButtonElement>(".reset")!.addEventListener("click", () => {
    resetObservations();
    render();
  });
  root.querySelector<HTMLButtonElement>(".clear")!.addEventListener("click", () => {
    observations.length = 0;
    render();
  });

  let mainPlot: Plot | null = null;
  const hitObs = (px: number, py: number): number | null => {
    if (!mainPlot) return null;
    for (let i = 0; i < observations.length; i++) {
      const dx = mainPlot.xPx(observations[i].t) - px;
      const dy = mainPlot.yPx(observations[i].y) - py;
      if (Math.hypot(dx, dy) <= SIR.HIT_RADIUS_PX) return i;
    }
    return null;
  };
  const clampT = (t: number) => clamp(t, SIR.T_DOMAIN[0], SIR.T_DOMAIN[1]);
  const clampY = (y: number) => clamp(y, SIR.Y_VIEW[0], SIR.Y_VIEW[1]);

  mainCanvas.addEventListener("contextmenu", (e) => e.preventDefault());
  mainCanvas.addEventListener("pointerdown", (e) => {
    if (!mainPlot) return;
    const hit = hitObs(e.offsetX, e.offsetY);
    if (hit !== null && (e.shiftKey || e.button === 2)) {
      observations.splice(hit, 1);
      render();
      return;
    }
    if (hit !== null) {
      dragIdx = hit;
      mainCanvas.setPointerCapture(e.pointerId);
      return;
    }
    observations.push({ t: clampT(mainPlot.pxToX(e.offsetX)), y: clampY(mainPlot.pxToY(e.offsetY)) });
    render();
  });
  mainCanvas.addEventListener("pointermove", (e) => {
    if (dragIdx === null || !mainPlot) return;
    observations[dragIdx] = { t: clampT(mainPlot.pxToX(e.offsetX)), y: clampY(mainPlot.pxToY(e.offsetY)) };
    render();
  });
  const endDrag = () => {
    dragIdx = null;
    observations.sort((a, b) => a.t - b.t);
  };
  mainCanvas.addEventListener("pointerup", endDrag);
  mainCanvas.addEventListener("pointercancel", endDrag);

  function updateControls(): void {
    betaMeanV.textContent = betaMean.toFixed(3);
    gammaMeanV.textContent = gammaMean.toFixed(3);
    betaNuV.textContent = `nu=${betaNu.toFixed(0)}`;
    gammaNuV.textContent = `nu=${gammaNu.toFixed(0)}`;
  }

  function render(): void {
    updateControls();
    const betaUnit = clampBetaUnit((betaMean - betaRange[0]) / (betaRange[1] - betaRange[0]));
    const gammaUnit = clampBetaUnit((gammaMean - gammaRange[0]) / (gammaRange[1] - gammaRange[0]));
    const params = { observations, betaUnit, betaNu, gammaUnit, gammaNu };

    const reasons: string[] = [];
    const nFar = observations.filter((p) => p.y < SIR.Y_OOD[0] || p.y > SIR.Y_OOD[1]).length;
    if (nFar > 0) reasons.push(`${nFar} observation(s) beyond training infected-fraction range`);
    if (observations.length > SIR.MAX_CONTEXT_HINT)
      reasons.push(`${observations.length} observations (training used <= ${SIR.MAX_CONTEXT_HINT})`);
    if (observations.length > 0 && observations.length < SIR.MIN_CONTEXT_HINT)
      reasons.push(`${observations.length} observations (training used at least ${SIR.MIN_CONTEXT_HINT})`);
    banner.hidden = reasons.length === 0;
    banner.textContent = reasons.length ? `Out of training distribution: ${reasons.join(" · ")}` : "";

    const ace = sirInfer(model, params, grids);
    const oracle = sirOracle(params, grids, oracleCache, { betaRange, gammaRange, sigmaObs: SIR.SIGMA_OBS });
    const betaPrior = normalize(betaLogPriorOnGrid(grids.betaGrid, betaUnit, betaNu, betaRange[0], betaRange[1]));
    const gammaPrior = normalize(
      betaLogPriorOnGrid(grids.gammaGrid, gammaUnit, gammaNu, gammaRange[0], gammaRange[1]),
    );

    drawMain(ace.predMean, ace.predStd, oracle.yMean, oracle.yStd);
    drawMarginal(betaCanvas, betaRange, grids.betaGrid, oracle.betaPost, ace.betaPost, betaPrior);
    drawMarginal(gammaCanvas, gammaRange, grids.gammaGrid, oracle.gammaPost, ace.gammaPost, gammaPrior);
  }

  function baseMain(): Plot {
    const p = makePlot(mainCanvas, { xDomain: SIR.T_DOMAIN, yDomain: SIR.Y_VIEW });
    p.clear();
    p.rectData(SIR.T_DOMAIN[0], SIR.T_DOMAIN[1], SIR.Y_NORMAL[0], SIR.Y_NORMAL[1], "rgba(37,99,235,0.05)");
    p.hline(0, "#eceef2", 1);
    return p;
  }

  function drawMain(aceMean: number[], aceStd: number[], oracleMean: number[], oracleStd: number[]): void {
    mainPlot = baseMain();
    mainPlot.band(
      grids.tGrid,
      oracleMean.map((m, i) => m - 2 * oracleStd[i]),
      oracleMean.map((m, i) => m + 2 * oracleStd[i]),
      "rgba(22,163,74,0.13)",
    );
    mainPlot.line(grids.tGrid, oracleMean, COL.oracle, 1.7);
    mainPlot.band(
      grids.tGrid,
      aceMean.map((m, i) => m - 2 * aceStd[i]),
      aceMean.map((m, i) => m + 2 * aceStd[i]),
      "rgba(37,99,235,0.14)",
    );
    mainPlot.line(grids.tGrid, aceMean, COL.ace, 1.8);
    mainPlot.dots(
      observations
        .filter((p) => p.y >= SIR.Y_OOD[0] && p.y <= SIR.Y_OOD[1])
        .map((p) => [p.t, p.y] as [number, number]),
      "#111827",
      4,
    );
    mainPlot.dots(
      observations
        .filter((p) => p.y < SIR.Y_OOD[0] || p.y > SIR.Y_OOD[1])
        .map((p) => [p.t, p.y] as [number, number]),
      "#b45309",
      4,
    );
    mainPlot.axes();
    const ctx = mainPlot.ctx;
    ctx.fillStyle = "#9ca3af";
    ctx.font = "11px system-ui";
    ctx.fillText("infected fraction over time", 44, 14);
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
