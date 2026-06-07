/** Small numeric helpers shared across demos. */

export function linspace(a: number, b: number, n: number): number[] {
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = a + ((b - a) * i) / (n - 1);
  return out;
}

/** Softmax of log-values into a normalized probability vector (sums to 1). */
export function normalize(logp: number[]): number[] {
  let m = -Infinity;
  for (const v of logp) if (v > m) m = v;
  if (!Number.isFinite(m)) return logp.map(() => 1 / logp.length); // all -Inf -> uniform
  const ex = logp.map((v) => Math.exp(v - m));
  let s = 0;
  for (const e of ex) s += e;
  return ex.map((e) => e / s);
}
