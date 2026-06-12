/**
 * BO-1D interactive demo (UI layer).
 *
 * Like the GP tab, users edit observations directly on the function plot. The
 * BO-specific part is that ACE also predicts the optimum latents: p(x_opt | D)
 * is drawn along the bottom axis and p(y_opt | D) along the right axis. Each
 * latent can instead be fixed as a known zero-spread context token.
 */

import { BO } from "../config";
import { aceFooter, addInfoButton } from "../explain";
import { clamp, hitPoint, pointOodReasons } from "../interaction";
import { makePlot, type Plot } from "../plot";
import { ACEModel } from "../ace/model";
import { loadWeights } from "../ace/weights";
import { normalize } from "../util";
import { betaLogPriorOnGrid } from "../gaussian/oracle";
import { boInfer, defaultBOGrids, type BOPoint, type BOResult } from "./infer";

const EXPLAINER = {
  title: "About: BO-1D (optimum inference)",
  html: `
    <h3>The task</h3>
    <p>Locate the global minimum of an unknown 1-D function from a few evaluations — the core
    question of Bayesian optimization (BO). What you would most like to have is the posterior over
    the minimum's location and value, p(x<sub>opt</sub> | D) and p(y<sub>opt</sub> | D), given
    the data so far.</p>
    <h3>What ACE is doing</h3>
    <p>Here x<sub>opt</sub> and y<sub>opt</sub> are latent tokens describing the <em>specific</em>
    unknown function, not its class. Training used synthetic functions constructed so that their
    global minimum is known exactly, so the network learned to map observations directly to
    posteriors over the optimum — the densities drawn along the bottom and right axes,
    recomputed in one forward pass per edit. You can also state a runtime prior over where you
    believe the optimum lies, or fix either latent exactly.</p>
    <h3>Compared with the classical approach</h3>
    <p>Under a GP surrogate these posteriors are intractable, so practical BO reasons about the
    optimum indirectly through acquisition functions (expected improvement, entropy-search
    approximations) with their own approximations and inner optimization loops. ACE predicts
    the quantities of interest directly. The prior handling is robust by construction: during
    training, a small fraction of true optima ignored the stated prior (an ε-contaminated
    prior), so a confidently wrong prior shifts the posterior only as far as the data allow,
    rather than overriding them. No exact reference is shown — none is available in closed form
    for this generative process — so read the posteriors structurally: they should tighten as
    observations accumulate and respond sensibly to priors.</p>
    ${aceFooter(
      'See also <a href="https://lacerbi.github.io/blog/2025/just-predict-the-optimum/">Just predict the optimum</a>, a blog post expanding on this idea.',
    )}`,
};

const CSS = `
.bo-root { display: flex; flex-direction: column; gap: 12px; }
.bo-hint { color: var(--muted); margin: 0; }
.bo-legend { font-size: 11px; color: var(--muted); display: flex; gap: 12px; padding: 0 6px; flex-wrap: wrap; }
.bo-legend span::before { content: ""; }
.bo-top { display: flex; gap: 18px; flex-wrap: wrap; align-items: flex-start; }
.bo-main { border: 1px solid var(--line); border-radius: 8px; background: #fff; touch-action: none; }
.bo-controls { display: flex; flex-direction: column; gap: 12px; min-width: 270px; }
.bo-controls fieldset { border: 1px solid var(--line); border-radius: 8px; margin: 0; padding: 8px 10px; }
.bo-controls legend { color: var(--muted); font-size: 12px; padding: 0 4px; }
.bo-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
.bo-row label { width: 46px; color: var(--muted); }
.bo-row input[type=range] { flex: 1; }
.bo-row .val { font-variant-numeric: tabular-nums; color: var(--muted); min-width: 68px; text-align: right; }
.bo-btns { display: flex; gap: 8px; flex-wrap: wrap; }
.bo-btn { font: inherit; padding: 6px 10px; border: 1px solid var(--line); background: #fff;
  border-radius: 6px; cursor: pointer; }
`;

const COL = {
  band: "#2563eb",
  xPost: "#dc2626",
  yPost: "#16a34a",
  prior: "#9ca3af",
  pin: "#b45309",
};
const BETA_UNIT_EPS = 1e-4;

function injectCss(): void {
  if (document.getElementById("bo-style")) return;
  const s = document.createElement("style");
  s.id = "bo-style";
  s.textContent = CSS;
  document.head.appendChild(s);
}

function clampBetaUnit(x: number): number {
  return clamp(x, BETA_UNIT_EPS, 1.0 - BETA_UNIT_EPS);
}

export async function mountBO(el: HTMLElement): Promise<void> {
  injectCss();
  const weights = await loadWeights(`${import.meta.env.BASE_URL}models/bo1d`);
  const model = new ACEModel(weights);
  const grids = defaultBOGrids(model);

  const xRange: [number, number] = [
    model.variables[1].bound_lo,
    model.variables[1].bound_hi,
  ];
  const yOptRange = BO.Y_OPT_RANGE;

  const defaults: BOPoint[] = [
    { x: -0.8, y: 0.25 },
    { x: -0.5, y: 0.05 },
    { x: 0.1, y: -0.28 },
    { x: 0.7, y: 0.2 },
  ];
  const points: BOPoint[] = defaults.map((p) => ({ ...p }));

  let xMean = 0.0;
  let yMean = -0.5;
  let xNu = 2.0;
  let yNu = 2.0;
  let pinX = false;
  let pinY = false;
  let dragIdx: number | null = null;

  el.innerHTML = "";
  const root = document.createElement("div");
  root.className = "bo-root";
  const lnLo = Math.log(BO.NU_RANGE[0]);
  const lnHi = Math.log(BO.NU_RANGE[1]);
  root.innerHTML = `
    <p class="bo-hint">Click empty space to add an observation &middot; drag a point to move it &middot; shift-click to delete.
      Optimum-location and optimum-value posteriors are overlaid on the function plot.</p>
    <div class="bo-legend"><span style="color:${COL.band}">predictive</span>
      <span style="color:${COL.xPost}">x_opt posterior</span>
      <span style="color:${COL.yPost}">y_opt posterior</span>
      <span style="color:${COL.prior}">prior</span></div>
    <div class="bo-top">
      <canvas class="bo-main" width="720" height="430" style="width:720px;height:430px;"></canvas>
      <div class="bo-controls">
        <fieldset>
          <legend>x_opt</legend>
          <label class="bo-row"><input type="checkbox" class="pin-x"/>fix known value</label>
          <div class="bo-row"><label>mean</label><input type="range" class="x-mean"/><span class="val x-mean-v"></span></div>
          <div class="bo-row"><label>conc.</label><input type="range" class="x-nu"/><span class="val x-nu-v"></span></div>
        </fieldset>
        <fieldset>
          <legend>y_opt</legend>
          <label class="bo-row"><input type="checkbox" class="pin-y"/>fix known value</label>
          <div class="bo-row"><label>mean</label><input type="range" class="y-mean"/><span class="val y-mean-v"></span></div>
          <div class="bo-row"><label>conc.</label><input type="range" class="y-nu"/><span class="val y-nu-v"></span></div>
        </fieldset>
        <div class="bo-btns">
          <button class="bo-btn reset">Reset points</button>
          <button class="bo-btn clear">Clear points</button>
          <button class="bo-btn uniform">Uniform priors</button>
        </div>
      </div>
    </div>
  `;
  el.appendChild(root);
  addInfoButton(root.querySelector<HTMLElement>(".bo-hint")!, EXPLAINER);

  const mainCanvas = root.querySelector<HTMLCanvasElement>(".bo-main")!;
  const pinXEl = root.querySelector<HTMLInputElement>(".pin-x")!;
  const pinYEl = root.querySelector<HTMLInputElement>(".pin-y")!;
  const xMeanS = root.querySelector<HTMLInputElement>(".x-mean")!;
  const yMeanS = root.querySelector<HTMLInputElement>(".y-mean")!;
  const xNuS = root.querySelector<HTMLInputElement>(".x-nu")!;
  const yNuS = root.querySelector<HTMLInputElement>(".y-nu")!;
  const xMeanV = root.querySelector<HTMLSpanElement>(".x-mean-v")!;
  const yMeanV = root.querySelector<HTMLSpanElement>(".y-mean-v")!;
  const xNuV = root.querySelector<HTMLSpanElement>(".x-nu-v")!;
  const yNuV = root.querySelector<HTMLSpanElement>(".y-nu-v")!;

  const setupRange = (
    s: HTMLInputElement,
    lo: number,
    hi: number,
    val: number,
    steps = 240,
  ) => {
    s.min = String(lo);
    s.max = String(hi);
    s.step = String((hi - lo) / steps);
    s.value = String(val);
  };
  setupRange(xMeanS, xRange[0], xRange[1], xMean);
  setupRange(yMeanS, yOptRange[0], yOptRange[1], yMean);
  setupRange(xNuS, lnLo, lnHi, Math.log(xNu));
  setupRange(yNuS, lnLo, lnHi, Math.log(yNu));

  xMeanS.addEventListener("input", () => {
    xMean = parseFloat(xMeanS.value);
    render();
  });
  yMeanS.addEventListener("input", () => {
    yMean = parseFloat(yMeanS.value);
    render();
  });
  xNuS.addEventListener("input", () => {
    xNu = Math.exp(parseFloat(xNuS.value));
    render();
  });
  yNuS.addEventListener("input", () => {
    yNu = Math.exp(parseFloat(yNuS.value));
    render();
  });
  pinXEl.addEventListener("change", () => {
    pinX = pinXEl.checked;
    render();
  });
  pinYEl.addEventListener("change", () => {
    pinY = pinYEl.checked;
    render();
  });
  root
    .querySelector<HTMLButtonElement>(".reset")!
    .addEventListener("click", () => {
      points.length = 0;
      points.push(...defaults.map((p) => ({ ...p })));
      render();
    });
  root
    .querySelector<HTMLButtonElement>(".clear")!
    .addEventListener("click", () => {
      points.length = 0;
      render();
    });
  root
    .querySelector<HTMLButtonElement>(".uniform")!
    .addEventListener("click", () => {
      xMean = 0.0;
      yMean = -0.5;
      xNu = 2.0;
      yNu = 2.0;
      xMeanS.value = String(xMean);
      yMeanS.value = String(yMean);
      xNuS.value = String(Math.log(xNu));
      yNuS.value = String(Math.log(yNu));
      render();
    });

  let mainPlot: Plot | null = null;
  const clampX = (x: number) => clamp(x, BO.X_DOMAIN[0], BO.X_DOMAIN[1]);
  const clampY = (y: number) => clamp(y, BO.Y_VIEW[0], BO.Y_VIEW[1]);

  mainCanvas.addEventListener("contextmenu", (e) => e.preventDefault());
  mainCanvas.addEventListener("pointerdown", (e) => {
    const hit = hitPoint(
      points,
      mainPlot,
      e.offsetX,
      e.offsetY,
      BO.HIT_RADIUS_PX,
    );
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
    if (!mainPlot) return;
    points.push({
      x: clampX(mainPlot.pxToX(e.offsetX)),
      y: clampY(mainPlot.pxToY(e.offsetY)),
    });
    render();
  });
  mainCanvas.addEventListener("pointermove", (e) => {
    if (dragIdx === null || !mainPlot) return;
    points[dragIdx] = {
      x: clampX(mainPlot.pxToX(e.offsetX)),
      y: clampY(mainPlot.pxToY(e.offsetY)),
    };
    render();
  });
  const endDrag = () => {
    dragIdx = null;
  };
  mainCanvas.addEventListener("pointerup", endDrag);
  mainCanvas.addEventListener("pointercancel", endDrag);

  function updateControls(): void {
    pinXEl.checked = pinX;
    pinYEl.checked = pinY;
    xMeanV.textContent = pinX ? `x=${xMean.toFixed(2)}` : xMean.toFixed(2);
    yMeanV.textContent = pinY ? `y=${yMean.toFixed(2)}` : yMean.toFixed(2);
    xNuV.textContent = `nu=${xNu.toFixed(0)}`;
    yNuV.textContent = `nu=${yNu.toFixed(0)}`;
    xNuS.disabled = pinX;
    yNuS.disabled = pinY;
  }

  function render(): void {
    updateControls();
    const reasons = pointOodReasons(points, {
      yIsOod: (y) => y < BO.Y_OOD[0] || y > BO.Y_OOD[1],
      yReason: `beyond training y-range (${BO.Y_OOD[0]} to ${BO.Y_OOD[1]})`,
      maxPoints: BO.MAX_CONTEXT_HINT,
      minPoints: BO.MIN_CONTEXT_HINT,
      minReason: (n) =>
        `${n} observations (training used at least ${BO.MIN_CONTEXT_HINT})`,
    });
    const warning = reasons.length
      ? `Out of training distribution: ${reasons.join(" / ")}`
      : "";

    const xUnit = clampBetaUnit((xMean - xRange[0]) / (xRange[1] - xRange[0]));
    const yUnit = clampBetaUnit(
      (yMean - yOptRange[0]) / (yOptRange[1] - yOptRange[0]),
    );
    const res = boInfer(
      model,
      {
        points,
        xPriorUnit: xUnit,
        xPriorNu: xNu,
        yPriorUnit: yUnit,
        yPriorNu: yNu,
        pinXOpt: pinX ? xMean : null,
        pinYOpt: pinY ? yMean : null,
      },
      grids,
    );

    const xPrior = pinX
      ? null
      : normalize(
          betaLogPriorOnGrid(grids.xOptGrid, xUnit, xNu, xRange[0], xRange[1]),
        );
    const yPrior = pinY
      ? null
      : normalize(
          betaLogPriorOnGrid(
            grids.yOptGrid,
            yUnit,
            yNu,
            yOptRange[0],
            yOptRange[1],
          ),
        );
    drawMain(res, xPrior, yPrior, warning);
  }

  function baseMain(): Plot {
    const p = makePlot(mainCanvas, {
      xDomain: BO.X_DOMAIN,
      yDomain: BO.Y_VIEW,
    });
    p.clear();
    p.rectData(
      BO.X_DOMAIN[0],
      BO.X_DOMAIN[1],
      BO.Y_NORMAL[0],
      BO.Y_NORMAL[1],
      "rgba(37,99,235,0.05)",
    );
    p.hline(0, "#eceef2", 1);
    return p;
  }

  function drawBottomDensity(
    p: Plot,
    grid: number[],
    probs: number[],
    color: string,
    fill: string,
    ampScale: number,
  ): void {
    const peak = Math.max(...probs, 1e-12);
    const yr = BO.Y_VIEW[1] - BO.Y_VIEW[0];
    const base = BO.Y_VIEW[0];
    const amp = ampScale * yr;
    const lo = grid.map(() => base);
    const hi = probs.map((v) => base + (v / peak) * amp);
    p.band(grid, lo, hi, fill);
    p.line(grid, hi, color, 1.6);
  }

  function drawRightDensity(
    p: Plot,
    grid: number[],
    probs: number[],
    color: string,
    fill: string,
    ampScale: number,
  ): void {
    const peak = Math.max(...probs, 1e-12);
    const xr = BO.X_DOMAIN[1] - BO.X_DOMAIN[0];
    const right = BO.X_DOMAIN[1];
    const amp = ampScale * xr;
    const xs = probs.map((v) => right - (v / peak) * amp);
    const ctx = p.ctx;
    ctx.beginPath();
    ctx.moveTo(p.xPx(right), p.yPx(grid[0]));
    for (let i = 0; i < grid.length; i++)
      ctx.lineTo(p.xPx(xs[i]), p.yPx(grid[i]));
    ctx.lineTo(p.xPx(right), p.yPx(grid[grid.length - 1]));
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.beginPath();
    for (let i = 0; i < grid.length; i++) {
      const px = p.xPx(xs[i]);
      const py = p.yPx(grid[i]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.stroke();
  }

  function drawMain(
    res: BOResult,
    xPrior: number[] | null,
    yPrior: number[] | null,
    warning: string,
  ): void {
    mainPlot = baseMain();
    const lo = res.bandMean.map((m, i) => m - 2 * res.bandStd[i]);
    const hi = res.bandMean.map((m, i) => m + 2 * res.bandStd[i]);
    mainPlot.band(res.bandX, lo, hi, "rgba(37,99,235,0.14)");
    mainPlot.line(res.bandX, res.bandMean, COL.band, 1.8);

    if (xPrior)
      drawBottomDensity(
        mainPlot,
        res.xOptGrid,
        xPrior,
        COL.prior,
        "rgba(156,163,175,0.13)",
        0.12,
      );
    if (res.xOptPost)
      drawBottomDensity(
        mainPlot,
        res.xOptGrid,
        res.xOptPost,
        COL.xPost,
        "rgba(220,38,38,0.18)",
        0.18,
      );
    else mainPlot.vline(xMean, COL.pin, 2.2, [5, 4]);

    if (yPrior)
      drawRightDensity(
        mainPlot,
        res.yOptGrid,
        yPrior,
        COL.prior,
        "rgba(156,163,175,0.13)",
        0.12,
      );
    if (res.yOptPost)
      drawRightDensity(
        mainPlot,
        res.yOptGrid,
        res.yOptPost,
        COL.yPost,
        "rgba(22,163,74,0.18)",
        0.18,
      );
    else mainPlot.hline(yMean, COL.pin, 2.2, [5, 4]);

    mainPlot.dots(
      points
        .filter((p) => p.y >= BO.Y_OOD[0] && p.y <= BO.Y_OOD[1])
        .map((p) => [p.x, p.y] as [number, number]),
      "#111827",
      4,
    );
    mainPlot.dots(
      points
        .filter((p) => p.y < BO.Y_OOD[0] || p.y > BO.Y_OOD[1])
        .map((p) => [p.x, p.y] as [number, number]),
      COL.pin,
      4,
    );
    mainPlot.axes();
    mainPlot.label("observations and optimum posteriors", 50, 14);
    mainPlot.warning(warning);
  }

  render();
}
