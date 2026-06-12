/**
 * ALINE active-learning demo (UI layer).
 *
 * Default mode: a hidden ground-truth GP function (sampled in-browser by
 * env.ts) answers every query — the user only ever chooses WHERE to sample,
 * never what value comes back. ALINE's acquisition policy is rendered as
 * advice: the π(x | data, goal) distribution along the bottom axis, with
 * markers for ALINE's pick and the classical uncertainty-sampling pick. The
 * user can follow the advice, ignore it, or press "Follow policy" to let the
 * policy drive (animated). The goal selector is live — switching what you
 * want to learn (predict the function vs infer ℓ / σ / kernel) re-scores the
 * candidates instantly, including mid-episode.
 *
 * Secondary mode: "your own data" — free point editing (GP-tab semantics)
 * with the policy advice still live; no hidden truth, so no metrics.
 */

import { ALINE, KERNEL_LABELS } from "../config";
import { aceFooter, addInfoButton } from "../explain";
import { clamp, hitPoint, pointOodReasons } from "../interaction";
import { makePlot, type Plot } from "../plot";
import { ALINEModel } from "../ace/aline";
import { loadWeights } from "../ace/weights";
import { mulberry32 } from "../ace/rng";
import { linspace } from "../util";
import { sampleEpisode, type EpisodeDraw } from "./env";
import {
  alineStep,
  goalActive,
  nearestCandidate,
  type AlineStep,
  type Goal,
  type Obs,
} from "./infer";

interface Point {
  x: number;
  y: number;
}

const SEED0 = 20260612;
const ORACLE_SEED = 777;

// One color per latent, shared by the marginal and metric panels.
const LATENT_COLOR = { ell: "#9333ea", scale: "#0d9488", kernel: "#64748b" };
const LATENT_DIM = { ell: "rgba(147,51,234,0.35)", scale: "rgba(13,148,136,0.35)" };
const KERNEL_SHORT = ["RBF", "M½", "M3/2", "Per"];

const EXPLAINER = {
  title: "About: active learning with ALINE",
  html: `
    <h3>The task</h3>
    <p>Active data acquisition: when each measurement is expensive, it should be chosen
    to teach you the most — <em>about whatever you actually care about</em>. The target
    is rarely "everything": often it is a few parameters of a model (the rest being
    nuisance), or predictions over a region of interest. The same problem appears
    across fields — choosing the stimuli in a behavioral experiment, the next sensor
    placement, the next simulation to run.</p>
    <p>This tab presents the function-learning instance of that general problem: an
    unknown 1D function can be measured at a limited number of locations, and the goal
    may be the <em>whole function</em> (predict it everywhere) or specific
    <em>properties</em> of it — its lengthscale (how fast it varies), its outputscale
    (how large its swings are), or its kernel (which family its variation belongs to).
    Different goals reward different measurements: prediction wants coverage, while
    pinning down the lengthscale favors closely spaced probes, for instance.</p>
    <h3>What this tab is doing</h3>
    <p>The demo draws a random function and hides it; every query — your click, or
    the policy's pick — measures its true value at one location, with a budget of 16
    measurements per episode. ALINE (Huang et al., 2025) is one network that both
    <em>infers</em> and <em>acts</em>. The inference side is exactly the GP-1D ACE
    model: posteriors over the latent properties and a predictive band over the
    function. The new part is a small policy head that scores every candidate
    location by how informative measuring there should be <em>for the currently
    selected goal</em> — the orange distribution along the bottom axis, recomputed in
    one forward pass after every measurement. It was trained with reinforcement
    learning, the reward being the model's own step-to-step improvement in the
    log-probability of the goal (self-estimated information gain). Inside the model a
    goal is just a choice of target tokens, so nothing here is specific to these four
    goals: any subset of the latents or predictions the model knows about could be
    targeted the same way, and switching goals — even halfway through an episode —
    is instant.</p>
    <h3>Compared with the classical approach</h3>
    <p>The green marker shows what uncertainty sampling (query where the predictive
    variance is largest — a strong classical baseline) would pick from the same model.
    Where the orange and green markers separate, the learned policy is deviating from
    the heuristic. Classical pipelines refit a surrogate and re-optimize an acquisition
    rule at every step, and need a different rule for each goal; here one forward pass
    answers "where should I measure next, for this goal?" directly. The honest
    trade-off: the policy is only as good as its training — this demo's policy
    reliably beats random querying, but a well-tuned uncertainty-sampling baseline is
    still competitive on pure prediction, and the goal-targeting differences can be
    subtle.</p>
    ${aceFooter(
      'The acquisition policy follows Huang, Wen, Bharti, Kaski &amp; Acerbi (2025), <em>ALINE: Joint Amortization for Bayesian Inference and Active Data Acquisition</em> (NeurIPS 2025) — <a href="https://arxiv.org/abs/2506.07259">arXiv:2506.07259</a>.',
    )}`,
};

const CSS = `
.al-root { display: flex; flex-direction: column; gap: 12px; }
.al-hint { color: var(--muted); margin: 0; }
.al-top { display: flex; gap: 18px; flex-wrap: wrap; align-items: flex-start; }
.al-plot-col { display: flex; flex-direction: column; gap: 6px; }
.al-main { border: 1px solid var(--line); border-radius: 8px; background: #fff; touch-action: none; }
.al-minis { display: flex; gap: 8px; }
.al-mini { border: 1px solid var(--line); border-radius: 8px; background: #fff; }
.al-status { color: var(--muted); margin: 0; font-size: 12px; font-variant-numeric: tabular-nums; }
.al-controls { display: flex; flex-direction: column; gap: 14px; min-width: 240px; }
.al-controls fieldset { border: 1px solid var(--line); border-radius: 8px; margin: 0; padding: 8px 10px; }
.al-controls legend { color: var(--muted); font-size: 12px; padding: 0 4px; }
.al-goal-btns { display: flex; flex-wrap: wrap; gap: 6px; }
.al-goal-btns button { font: inherit; padding: 4px 8px; border: 1px solid var(--line);
  background: #fff; border-radius: 6px; cursor: pointer; }
.al-goal-btns button.sel { border-color: var(--accent); color: var(--accent); font-weight: 600; }
.al-goal-sep { width: 1px; align-self: stretch; background: var(--line); margin: 2px 3px; }
.al-row { display: flex; align-items: center; gap: 8px; }
.al-btns { display: flex; gap: 8px; flex-wrap: wrap; }
.al-btn { font: inherit; padding: 6px 10px; border: 1px solid var(--line); background: #fff;
  border-radius: 6px; cursor: pointer; }
.al-btn:disabled { opacity: 0.45; cursor: default; }
.al-counter { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
`;

function injectCss(): void {
  if (document.getElementById("al-style")) return;
  const s = document.createElement("style");
  s.id = "al-style";
  s.textContent = CSS;
  document.head.appendChild(s);
}

const raf: (cb: () => void) => void =
  typeof requestAnimationFrame === "function"
    ? (cb) => requestAnimationFrame(() => cb())
    : (cb) => {
        setTimeout(cb, 16);
      };

export async function mountAline(el: HTMLElement): Promise<void> {
  injectCss();
  let model: ALINEModel;
  try {
    const weights = await loadWeights(`${import.meta.env.BASE_URL}models/gp1d_aline`);
    model = new ALINEModel(weights);
  } catch {
    el.innerHTML = `<p class="loading">The ALINE model weights are unavailable here.
      To run this tab locally, export the
      weights with <code>export_weights.py --task gp1d_aline</code> — see the
      "Run locally" section of <code>playground/README.md</code>.</p>`;
    return;
  }

  // --- state ---
  let mode: "episode" | "oracle" = "episode";
  const goal: Goal = { pred: true, ell: false, scale: false, kernel: false };
  let draw: EpisodeDraw | null = null;
  let available: boolean[] = [];
  let obsIdx: number[] = [];
  let episodeSeed = 0;
  let revealTruth = false;
  let history: Array<{ t: number; rmse: number; ell: number; scale: number; kernel: number }> = [];

  const defaultPoints: Point[] = [
    { x: -0.9, y: -0.4 },
    { x: -0.55, y: 0.15 },
    { x: -0.2, y: 0.55 },
    { x: 0.5, y: 0.05 },
  ];
  const points: Point[] = defaultPoints.map((p) => ({ ...p }));
  const oracleRng = mulberry32(ORACLE_SEED);
  const oraclePool = Array.from({ length: ALINE.POOL }, () => 2 * oracleRng() - 1);
  const oracleXStar = Array.from({ length: ALINE.M_PRED }, () => 2 * oracleRng() - 1);
  let dragIdx: number | null = null;

  let lastStep: AlineStep | null = null;
  let candIdx: number[] = []; // candX positions -> pool indices (episode) / oraclePool indices
  let candX: number[] = [];
  let stepMs = 0;
  let epoch = 0;

  const gridX = linspace(ALINE.X_DOMAIN[0], ALINE.X_DOMAIN[1], ALINE.GRID);

  // --- DOM ---
  el.innerHTML = "";
  const root = document.createElement("div");
  root.className = "al-root";
  root.innerHTML = `
    <p class="al-hint">Active learning with ALINE
      (<a href="https://github.com/acerbilab/nanoACE/tree/main/extensions/aline">extensions/aline</a>):
      click to measure a hidden function — the orange curve shows where the learned
      policy would measure next for the selected goal.</p>
    <div class="al-top">
      <div class="al-plot-col">
        <canvas class="al-main" width="660" height="380" style="width:660px;height:380px;"></canvas>
        <div class="al-minis">
          <canvas class="al-mini al-metric" width="214" height="110" style="width:214px;height:110px;"></canvas>
          <canvas class="al-mini al-latents" width="214" height="110" style="width:214px;height:110px;"></canvas>
          <canvas class="al-mini al-kernelpanel" width="214" height="110" style="width:214px;height:110px;"></canvas>
        </div>
        <p class="al-status"></p>
      </div>
      <div class="al-controls">
        <fieldset>
          <legend>mode</legend>
          <label class="al-row"><input type="radio" name="al-mode" class="al-mode-episode" checked/>measure a hidden function</label>
          <label class="al-row"><input type="radio" name="al-mode" class="al-mode-oracle"/>enter your own points</label>
        </fieldset>
        <fieldset>
          <legend>goal — what should the queries teach the model? (shift-click combines)</legend>
          <div class="al-goal-btns">
            <button class="g-pred" title="Improve the predictive band everywhere">predict f(x)</button>
            <span class="al-goal-sep"></span>
            <button class="g-ell" title="Pin down the lengthscale">lengthscale ℓ</button>
            <button class="g-scale" title="Pin down the outputscale">outputscale σ</button>
            <button class="g-kernel" title="Identify the kernel">kernel</button>
          </div>
        </fieldset>
        <fieldset class="al-episode-controls">
          <legend>episode</legend>
          <div class="al-btns">
            <button class="al-btn al-new">New function</button>
            <button class="al-btn al-step" title="Take one policy step (query ALINE's pick)">Step</button>
            <button class="al-btn al-follow" title="Let the policy drive to the end of the budget">Follow policy</button>
            <button class="al-btn al-restart" title="Same hidden function, back to the seed point">Restart</button>
          </div>
          <label class="al-row"><input type="checkbox" class="al-reveal"/>reveal the hidden function</label>
          <span class="al-counter"></span>
        </fieldset>
        <fieldset class="al-oracle-controls" style="display:none">
          <legend>points</legend>
          <div class="al-btns">
            <button class="al-btn al-reset">Reset points</button>
            <button class="al-btn al-clear">Clear points</button>
          </div>
        </fieldset>
      </div>
    </div>
  `;
  el.appendChild(root);
  addInfoButton(root.querySelector<HTMLElement>(".al-hint")!, EXPLAINER);

  const mainCanvas = root.querySelector<HTMLCanvasElement>(".al-main")!;
  const metricCanvas = root.querySelector<HTMLCanvasElement>(".al-metric")!;
  const latentsCanvas = root.querySelector<HTMLCanvasElement>(".al-latents")!;
  const kernelCanvas = root.querySelector<HTMLCanvasElement>(".al-kernelpanel")!;
  const statusEl = root.querySelector<HTMLParagraphElement>(".al-status")!;
  const counterEl = root.querySelector<HTMLSpanElement>(".al-counter")!;
  const modeEpisode = root.querySelector<HTMLInputElement>(".al-mode-episode")!;
  const modeOracle = root.querySelector<HTMLInputElement>(".al-mode-oracle")!;
  const episodeFs = root.querySelector<HTMLFieldSetElement>(".al-episode-controls")!;
  const oracleFs = root.querySelector<HTMLFieldSetElement>(".al-oracle-controls")!;
  const revealBox = root.querySelector<HTMLInputElement>(".al-reveal")!;
  const stepBtn = root.querySelector<HTMLButtonElement>(".al-step")!;
  const followBtn = root.querySelector<HTMLButtonElement>(".al-follow")!;
  const goalBtns = {
    pred: root.querySelector<HTMLButtonElement>(".g-pred")!,
    ell: root.querySelector<HTMLButtonElement>(".g-ell")!,
    scale: root.querySelector<HTMLButtonElement>(".g-scale")!,
    kernel: root.querySelector<HTMLButtonElement>(".g-kernel")!,
  };

  // --- helpers ---
  const stepsTaken = () => Math.max(0, obsIdx.length - 1);
  const budgetLeft = () => mode === "episode" && stepsTaken() < ALINE.T;

  function currentObs(): Obs[] {
    if (mode === "episode") {
      if (!draw) return [];
      const d = draw;
      return obsIdx.map((i) => ({ x: d.poolX[i], y: d.poolY[i] }));
    }
    return points.map((p) => ({ x: p.x, y: p.y }));
  }

  function rebuildCandidates(): void {
    candIdx = [];
    candX = [];
    if (mode === "episode") {
      if (!draw) return;
      for (let i = 0; i < draw.poolX.length; i++) {
        if (available[i]) {
          candIdx.push(i);
          candX.push(draw.poolX[i]);
        }
      }
    } else {
      for (let i = 0; i < oraclePool.length; i++) {
        const near = points.some((p) => Math.abs(p.x - oraclePool[i]) < ALINE.SNAP_EPS);
        if (!near) {
          candIdx.push(i);
          candX.push(oraclePool[i]);
        }
      }
    }
  }

  function recompute(): void {
    rebuildCandidates();
    const obs = currentObs();
    if (obs.length === 0 || candX.length === 0 || !goalActive(goal)) {
      lastStep = null;
      drawAll();
      return;
    }
    const xStar = mode === "episode" && draw ? draw.xStar : oracleXStar;
    const truth =
      mode === "episode" && draw
        ? { gridY: draw.gridY, logEll: draw.logEll, logScale: draw.logScale, kernel: draw.kernel }
        : undefined;
    const t0 = performance.now();
    lastStep = alineStep(
      model,
      { obs, candX, goal: { ...goal }, xStar, gridX, latentGrid: ALINE.LATENT_GRID },
      truth,
    );
    stepMs = performance.now() - t0;
    drawAll();
  }

  function pushHistory(): void {
    if (mode !== "episode" || !lastStep?.metrics) return;
    const m = lastStep.metrics;
    history.push({ t: stepsTaken(), rmse: m.rmse, ell: m.logq.ell, scale: m.logq.scale, kernel: m.logq.kernel });
  }

  function newEpisode(): void {
    epoch += 1;
    draw = sampleEpisode(SEED0 + episodeSeed++, {
      pool: ALINE.POOL,
      grid: ALINE.GRID,
      mPred: ALINE.M_PRED,
    });
    restartEpisode();
  }

  function restartEpisode(): void {
    epoch += 1;
    if (!draw) return;
    available = draw.poolX.map(() => true);
    available[draw.seedIdx] = false;
    obsIdx = [draw.seedIdx];
    revealTruth = false;
    revealBox.checked = false;
    history = [];
    recompute();
    pushHistory();
    drawAll();
  }

  /** Observe pool candidate `poolI` (the only way data enters episode mode). */
  function applyAction(poolI: number): void {
    if (!draw || !budgetLeft() || !available[poolI]) return;
    available[poolI] = false;
    obsIdx.push(poolI);
    recompute();
    pushHistory();
    if (!budgetLeft()) {
      revealTruth = true;
      revealBox.checked = true;
      drawAll();
    }
  }

  function stepOnce(): void {
    if (!lastStep || !budgetLeft()) return;
    applyAction(candIdx[lastStep.argmaxIdx]);
  }

  function followPolicy(): void {
    if (mode !== "episode") return;
    epoch += 1;
    const myEpoch = epoch;
    const tick = () => {
      if (myEpoch !== epoch || !budgetLeft() || !lastStep) return;
      const t0 = performance.now();
      applyAction(candIdx[lastStep.argmaxIdx]);
      if (budgetLeft()) {
        // Pace at FOLLOW_STEP_MS between step starts (compute time included).
        const remaining = Math.max(0, ALINE.FOLLOW_STEP_MS - (performance.now() - t0));
        setTimeout(() => {
          if (myEpoch === epoch) raf(tick);
        }, remaining);
      }
    };
    raf(tick);
  }

  // --- controls ---
  modeEpisode.addEventListener("change", () => {
    if (!modeEpisode.checked) return;
    mode = "episode";
    epoch += 1;
    episodeFs.style.display = "";
    oracleFs.style.display = "none";
    recompute();
  });
  modeOracle.addEventListener("change", () => {
    if (!modeOracle.checked) return;
    mode = "oracle";
    epoch += 1;
    episodeFs.style.display = "none";
    oracleFs.style.display = "";
    recompute();
  });

  // Plain click switches the goal; shift-click combines parameter goals.
  // Predictive + parameter goals together were never trained (xi is either-or),
  // so pred never combines: any click involving it switches exclusively.
  (Object.keys(goalBtns) as Array<keyof Goal>).forEach((key) => {
    goalBtns[key].addEventListener("click", (e) => {
      const combine = e.shiftKey && key !== "pred" && !goal.pred;
      if (combine) {
        const next = { ...goal, [key]: !goal[key] };
        if (!goalActive(next)) return; // at least one goal stays selected
        goal[key] = next[key];
      } else {
        goal.pred = key === "pred";
        goal.ell = key === "ell";
        goal.scale = key === "scale";
        goal.kernel = key === "kernel";
      }
      epoch += 1;
      updateControls();
      recompute();
    });
  });

  root.querySelector<HTMLButtonElement>(".al-new")!.addEventListener("click", newEpisode);
  root.querySelector<HTMLButtonElement>(".al-restart")!.addEventListener("click", restartEpisode);
  stepBtn.addEventListener("click", () => {
    epoch += 1;
    stepOnce();
  });
  followBtn.addEventListener("click", followPolicy);
  revealBox.addEventListener("change", () => {
    revealTruth = revealBox.checked;
    drawAll();
  });
  root.querySelector<HTMLButtonElement>(".al-reset")!.addEventListener("click", () => {
    epoch += 1;
    points.length = 0;
    points.push(...defaultPoints.map((p) => ({ ...p })));
    recompute();
  });
  root.querySelector<HTMLButtonElement>(".al-clear")!.addEventListener("click", () => {
    epoch += 1;
    points.length = 0;
    recompute();
  });

  // --- pointer interaction ---
  let mainPlot: Plot | null = null;
  const clampX = (x: number) => clamp(x, ALINE.X_DOMAIN[0], ALINE.X_DOMAIN[1]);
  const clampY = (y: number) => clamp(y, ALINE.Y_VIEW[0], ALINE.Y_VIEW[1]);

  mainCanvas.addEventListener("contextmenu", (e) => e.preventDefault());
  mainCanvas.addEventListener("pointerdown", (e) => {
    if (!mainPlot) return;
    if (mode === "episode") {
      // Choosing where to sample: snap to the nearest available candidate.
      if (!budgetLeft() || candX.length === 0) return;
      epoch += 1;
      const x = mainPlot.pxToX(e.offsetX);
      applyAction(candIdx[nearestCandidate(candX, x)]);
      return;
    }
    const hit = hitPoint(points, mainPlot, e.offsetX, e.offsetY, ALINE.HIT_RADIUS_PX);
    if (hit !== null && (e.shiftKey || e.button === 2)) {
      points.splice(hit, 1);
      epoch += 1;
      recompute();
      return;
    }
    if (hit !== null) {
      dragIdx = hit;
      mainCanvas.setPointerCapture(e.pointerId);
      return;
    }
    points.push({ x: clampX(mainPlot.pxToX(e.offsetX)), y: clampY(mainPlot.pxToY(e.offsetY)) });
    epoch += 1;
    recompute();
  });
  mainCanvas.addEventListener("pointermove", (e) => {
    if (dragIdx === null || !mainPlot || mode !== "oracle") return;
    points[dragIdx] = { x: clampX(mainPlot.pxToX(e.offsetX)), y: clampY(mainPlot.pxToY(e.offsetY)) };
    recompute();
  });
  const endDrag = () => {
    dragIdx = null;
  };
  mainCanvas.addEventListener("pointerup", endDrag);
  mainCanvas.addEventListener("pointercancel", endDrag);

  // --- rendering ---
  function oodReasons(): string[] {
    const obs = currentObs();
    const reasons = pointOodReasons(obs, {
      yIsOod: (y) => Math.abs(y) > ALINE.Y_OOD,
      yReason: `beyond training y-range (|y| > ${ALINE.Y_OOD})`,
      maxPoints: ALINE.CONTEXT_SOFT,
      maxReason: (n) => `${n} points (episodes trained with ≤ ${ALINE.CONTEXT_SOFT})`,
    });
    if (obs.length > ALINE.CONTEXT_HARD) {
      reasons.push(`${obs.length} points (the base model never saw > ${ALINE.CONTEXT_HARD})`);
    }
    return reasons;
  }

  function updateControls(): void {
    goalBtns.pred.classList.toggle("sel", goal.pred);
    goalBtns.ell.classList.toggle("sel", goal.ell);
    goalBtns.scale.classList.toggle("sel", goal.scale);
    goalBtns.kernel.classList.toggle("sel", goal.kernel);
    const left = budgetLeft();
    stepBtn.disabled = !left;
    followBtn.disabled = !left;
    counterEl.textContent =
      mode === "episode" ? `step ${stepsTaken()}/${ALINE.T}${left ? "" : " — budget spent"}` : "";
  }

  function basePlot(): Plot {
    const p = makePlot(mainCanvas, { xDomain: ALINE.X_DOMAIN, yDomain: ALINE.Y_VIEW });
    p.clear();
    p.rectData(ALINE.X_DOMAIN[0], ALINE.X_DOMAIN[1], ALINE.Y_NORMAL[0], ALINE.Y_NORMAL[1], "rgba(37,99,235,0.05)");
    p.hline(0, "#eceef2", 1);
    return p;
  }

  function drawPolicyOverlay(p: Plot): void {
    if (!lastStep) return;
    // Sort candidates by x for the polyline; this is a pmf over the discrete
    // pool (peak-normalized for display), not a calibrated density.
    const order = candX.map((_, i) => i).sort((a, b) => candX[a] - candX[b]);
    const xs = order.map((i) => candX[i]);
    const probs = order.map((i) => lastStep!.policyProbs[i]);
    const peak = Math.max(...probs, 1e-12);
    const base = ALINE.Y_VIEW[0];
    const amp = ALINE.POLICY_AMP * (ALINE.Y_VIEW[1] - ALINE.Y_VIEW[0]);
    const hi = probs.map((v) => base + (v / peak) * amp);
    p.band(xs, xs.map(() => base), hi, "rgba(234,88,12,0.14)");
    p.line(xs, hi, "#ea580c", 1.6);

    const alineX = candX[lastStep.argmaxIdx];
    const usX = candX[lastStep.usIdx];
    p.vline(usX, "#16a34a", 1.4, [5, 4]);
    p.vline(alineX, "#ea580c", 1.6, [2, 3]);
    p.label("ALINE pick", p.xPx(alineX) + 4, 16, { fill: "#ea580c" });
    p.label("US pick", p.xPx(usX) + 4, 30, { fill: "#16a34a" });
  }

  function drawAll(): void {
    mainPlot = basePlot();
    const obs = currentObs();

    if (!lastStep || obs.length === 0) {
      mainPlot.axes();
      const ctx = mainPlot.ctx;
      ctx.fillStyle = "#9ca3af";
      ctx.font = "14px system-ui";
      ctx.textAlign = "center";
      ctx.fillText(
        mode === "oracle" ? "Add a point to get policy advice." : "Press New function to start an episode.",
        mainPlot.width / 2,
        mainPlot.height / 2,
      );
      ctx.textAlign = "start";
      statusEl.textContent = "";
      updateControls();
      drawMetricPanel();
      drawLatentsPanel();
      drawKernelPanel();
      return;
    }

    // Hidden truth (episode mode, revealed).
    if (mode === "episode" && draw && revealTruth) {
      mainPlot.line(draw.gridX, draw.gridY, "rgba(107,114,128,0.8)", 1.2);
    }

    const lo = lastStep.bandMean.map((m, i) => m - 2 * lastStep!.bandStd[i]);
    const hi = lastStep.bandMean.map((m, i) => m + 2 * lastStep!.bandStd[i]);
    mainPlot.band(gridX, lo, hi, "rgba(37,99,235,0.12)");
    mainPlot.line(gridX, lastStep.bandMean, "#2563eb", 1.4);

    drawPolicyOverlay(mainPlot);

    mainPlot.dots(
      obs.filter((p) => Math.abs(p.y) <= ALINE.Y_OOD).map((p) => [p.x, p.y] as [number, number]),
      "#111827",
      4,
    );
    mainPlot.dots(
      obs.filter((p) => Math.abs(p.y) > ALINE.Y_OOD).map((p) => [p.x, p.y] as [number, number]),
      "#b45309",
      4,
    );
    if (mode === "episode" && draw && obsIdx.length > 0) {
      const s = obsIdx[0];
      mainPlot.dots([[draw.poolX[s], draw.poolY[s]]], "#6b7280", 6);
    }
    mainPlot.axes();

    const reasons = oodReasons();
    mainPlot.warning(reasons.length ? `Out of training distribution: ${reasons.join(" / ")}` : "");

    if (mode === "episode" && draw) {
      const m = lastStep.metrics;
      const truthTxt = revealTruth
        ? ` · truth: ${KERNEL_LABELS[draw.kernel]}, ℓ=${Math.exp(draw.logEll).toFixed(2)}, σ=${Math.exp(draw.logScale).toFixed(2)}`
        : " · truth hidden";
      const logqMean = m ? (m.logq.ell + m.logq.scale + m.logq.kernel) / 3 : null;
      statusEl.textContent =
        `step ${stepsTaken()}/${ALINE.T} · RMSE ${m ? m.rmse.toFixed(3) : "—"} · ` +
        `log q(θ) ${logqMean !== null ? logqMean.toFixed(2) : "—"}${truthTxt} · ${stepMs.toFixed(0)} ms/step`;
    } else {
      statusEl.textContent =
        `${obs.length} point${obs.length === 1 ? "" : "s"} · policy advice live · ` +
        `no ground truth in this mode · ${stepMs.toFixed(0)} ms/step`;
    }
    updateControls();
    drawMetricPanel();
    drawLatentsPanel();
    drawKernelPanel();
  }

  function drawMetricPanel(): void {
    // Predictive goal -> RMSE; parameter goals -> one log q(θ_true) line per
    // TARGETED latent (all three are recorded every step, so switching goals
    // re-renders whichever series without gaps).
    const seriesKeys: Array<"ell" | "scale" | "kernel"> = goal.pred
      ? []
      : (["ell", "scale", "kernel"] as const).filter((k) => goal[k]);
    const values = goal.pred
      ? history.map((h) => h.rmse)
      : seriesKeys.flatMap((k) => history.map((h) => h[k]));
    const p = makePlot(metricCanvas, {
      xDomain: [0, ALINE.T],
      yDomain: yDomainOf(values),
      padding: { l: 30, r: 6, t: 8, b: 14 },
    });
    p.clear();
    p.axes();
    if (mode !== "episode") {
      p.label("no ground truth in this mode", 36, 14, { fill: "#9ca3af" });
      return;
    }
    if (goal.pred) {
      p.label("RMSE vs steps", 36, 14, { fill: "#2563eb" });
      p.line(history.map((h) => h.t), history.map((h) => h.rmse), "#2563eb", 1.4);
      p.dots(history.map((h) => [h.t, h.rmse] as [number, number]), "#2563eb", 2);
      return;
    }
    const names = { ell: "log q(ℓ)", scale: "log q(σ)", kernel: "log q(kernel)" };
    seriesKeys.forEach((k, i) => {
      p.label(names[k], 36, 14 + 17 * i, { fill: LATENT_COLOR[k] });
      p.line(history.map((h) => h.t), history.map((h) => h[k]), LATENT_COLOR[k], 1.4);
      p.dots(history.map((h) => [h.t, h[k]] as [number, number]), LATENT_COLOR[k], 2);
    });
  }

  function yDomainOf(vals: number[]): [number, number] {
    if (vals.length === 0) return [0, 1];
    let lo = Math.min(...vals);
    let hi = Math.max(...vals);
    if (!(hi > lo)) {
      lo -= 0.5;
      hi += 0.5;
    }
    const pad = 0.12 * (hi - lo);
    return [lo - pad, hi + pad];
  }

  function drawLatentsPanel(): void {
    // Both continuous latent posteriors on one panel (they are always computed);
    // the goal-selected ones draw bold, the others dimmed.
    const ellMeta = model.variables[1];
    const scaleMeta = model.variables[2];
    const dE = lastStep?.ell;
    const dS = lastStep?.scale;
    const peak = Math.max(...(dE?.probs ?? [0]), ...(dS?.probs ?? [0]), 1e-12);
    const p = makePlot(latentsCanvas, {
      xDomain: [Math.min(ellMeta.bound_lo, scaleMeta.bound_lo), Math.max(ellMeta.bound_hi, scaleMeta.bound_hi)],
      yDomain: [0, 1.15 * peak],
      padding: { l: 8, r: 6, t: 8, b: 14 },
    });
    p.clear();
    p.axes();
    if (dS) p.line(dS.grid, dS.probs, goal.scale ? LATENT_COLOR.scale : LATENT_DIM.scale, goal.scale ? 2 : 1.2);
    if (dE) p.line(dE.grid, dE.probs, goal.ell ? LATENT_COLOR.ell : LATENT_DIM.ell, goal.ell ? 2 : 1.2);
    p.label(`log ℓ${goal.ell ? " (goal)" : ""}`, 14, 14, { fill: LATENT_COLOR.ell });
    p.label(`log σ${goal.scale ? " (goal)" : ""}`, 14, 31, { fill: LATENT_COLOR.scale });
    if (mode === "episode" && draw && revealTruth) {
      p.vline(draw.logEll, LATENT_COLOR.ell, 1, [4, 3]);
      p.vline(draw.logScale, LATENT_COLOR.scale, 1, [4, 3]);
    }
  }

  function drawKernelPanel(): void {
    const probs = lastStep ? lastStep.kernelProbs : [0, 0, 0, 0];
    const p = makePlot(kernelCanvas, {
      xDomain: [-0.6, probs.length - 0.4],
      yDomain: [0, 1.05],
      padding: { l: 8, r: 6, t: 8, b: 16 },
    });
    p.clear();
    p.axes();
    for (let i = 0; i < probs.length; i++) {
      const isTruth = mode === "episode" && draw !== null && revealTruth && draw.kernel === i;
      const fill = isTruth
        ? "rgba(22,163,74,0.55)"
        : goal.kernel
          ? "rgba(100,116,139,0.6)"
          : "rgba(100,116,139,0.3)";
      p.rectData(i - 0.32, i + 0.32, 0, probs[i], fill);
      p.label(KERNEL_SHORT[i], p.xPx(i) - 10, p.height - 8, { fill: "#6b7280" });
    }
    p.label(`kernel${goal.kernel ? " (goal)" : ""}`, 14, 14, { fill: LATENT_COLOR.kernel });
  }

  // --- boot ---
  updateControls();
  newEpisode();
}
