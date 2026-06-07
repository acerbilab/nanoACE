/**
 * Tiny canvas plotting helper (no dependencies). `makePlot` wraps a canvas with a
 * data→pixel transform and a handful of drawing primitives shared by both demos.
 */

export interface PlotOpts {
  xDomain: [number, number];
  yDomain: [number, number];
  padding?: { l: number; r: number; t: number; b: number };
}

export interface Plot {
  ctx: CanvasRenderingContext2D;
  width: number;
  height: number;
  xPx(x: number): number;
  yPx(y: number): number;
  pxToX(px: number): number;
  pxToY(px: number): number;
  clear(): void;
  rectData(x0: number, x1: number, y0: number, y1: number, fill: string): void;
  line(xs: number[], ys: number[], style: string, width?: number): void;
  band(xs: number[], lo: number[], hi: number[], fill: string): void;
  dots(pts: Array<[number, number]>, fill: string, r?: number): void;
  vline(x: number, style: string, width?: number, dash?: number[]): void;
  hline(y: number, style: string, width?: number, dash?: number[]): void;
  axes(style?: string): void;
}

export function makePlot(canvas: HTMLCanvasElement, opts: PlotOpts): Plot {
  const dpr = window.devicePixelRatio || 1;
  // Derive the CSS size from the inline style first so repeat calls (or panels
  // first drawn while hidden, where clientWidth is 0) stay stable — reading the
  // already-scaled canvas.width back would double-scale.
  const cssW = parseFloat(canvas.style.width) || canvas.clientWidth || canvas.width;
  const cssH = parseFloat(canvas.style.height) || canvas.clientHeight || canvas.height;
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d")!;
  ctx.scale(dpr, dpr);

  const pad = opts.padding ?? { l: 38, r: 12, t: 10, b: 26 };
  const [x0, x1] = opts.xDomain;
  const [y0, y1] = opts.yDomain;
  const plotW = cssW - pad.l - pad.r;
  const plotH = cssH - pad.t - pad.b;

  const xPx = (x: number) => pad.l + ((x - x0) / (x1 - x0)) * plotW;
  const yPx = (y: number) => pad.t + (1 - (y - y0) / (y1 - y0)) * plotH;
  const pxToX = (px: number) => x0 + ((px - pad.l) / plotW) * (x1 - x0);
  const pxToY = (px: number) => y0 + (1 - (px - pad.t) / plotH) * (y1 - y0);

  const plot: Plot = {
    ctx,
    width: cssW,
    height: cssH,
    xPx,
    yPx,
    pxToX,
    pxToY,
    clear() {
      ctx.clearRect(0, 0, cssW, cssH);
    },
    rectData(xa, xb, ya, yb, fill) {
      ctx.fillStyle = fill;
      const px = xPx(xa);
      const py = yPx(yb);
      ctx.fillRect(px, py, xPx(xb) - px, yPx(ya) - py);
    },
    line(xs, ys, style, width = 1.5) {
      ctx.beginPath();
      for (let i = 0; i < xs.length; i++) {
        const px = xPx(xs[i]);
        const py = yPx(ys[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.strokeStyle = style;
      ctx.lineWidth = width;
      ctx.stroke();
    },
    band(xs, lo, hi, fill) {
      ctx.beginPath();
      for (let i = 0; i < xs.length; i++) {
        const px = xPx(xs[i]);
        const py = yPx(hi[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      for (let i = xs.length - 1; i >= 0; i--) ctx.lineTo(xPx(xs[i]), yPx(lo[i]));
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
    },
    dots(pts, fill, r = 4) {
      ctx.fillStyle = fill;
      for (const [x, y] of pts) {
        ctx.beginPath();
        ctx.arc(xPx(x), yPx(y), r, 0, 2 * Math.PI);
        ctx.fill();
      }
    },
    vline(x, style, width = 1, dash = []) {
      ctx.save();
      ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(xPx(x), yPx(y1));
      ctx.lineTo(xPx(x), yPx(y0));
      ctx.strokeStyle = style;
      ctx.lineWidth = width;
      ctx.stroke();
      ctx.restore();
    },
    hline(y, style, width = 1, dash = []) {
      ctx.save();
      ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(xPx(x0), yPx(y));
      ctx.lineTo(xPx(x1), yPx(y));
      ctx.strokeStyle = style;
      ctx.lineWidth = width;
      ctx.stroke();
      ctx.restore();
    },
    axes(style = "#9ca3af") {
      ctx.strokeStyle = style;
      ctx.lineWidth = 1;
      ctx.strokeRect(pad.l, pad.t, plotW, plotH);
    },
  };
  return plot;
}
