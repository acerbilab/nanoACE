/**
 * Weight manifest + blob loading.
 *
 * `export_weights.py` writes `manifest.json` (cfg, derived constants, variable
 * schema, and a tensor table) plus `weights.bin` (float32, little-endian). The
 * byte source is decoupled from decoding: `loadWeights` uses `fetch` in the
 * browser, while `weightsFromBytes` accepts raw bytes so the Node/vitest parity
 * test can feed `fs`-read data.
 */

import type { VariableMeta } from "./schema";

export interface ACECfg {
  x_dim: number;
  d_model: number;
  n_heads: number;
  n_layers: number;
  mlp_hidden: number;
  mdn_components: number;
  head_hidden: number;
  min_scale: number;
}

export interface Derived {
  n_vars: number;
  head_dim: number;
  max_cardinality: number;
  total_disc: number;
  prior_features: number;
}

interface TensorMeta {
  name: string;
  shape: number[];
  offset: number; // in float32 elements
  length: number;
}

export interface Manifest {
  task: string;
  cfg: ACECfg;
  modes: { VALUE: number; PRIOR: number; QUERY: number };
  derived: Derived;
  variables: VariableMeta[];
  tensors: TensorMeta[];
  total_floats: number;
  weights_file: string;
  dtype: string;
  byte_order: string;
}

export interface Tensor {
  shape: number[];
  data: Float32Array;
}

/** Decode an IEEE-754 half (uint16 bit pattern) to a JS number. */
function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * Math.pow(2, -14) * (frac / 1024); // subnormal / zero
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

export class Weights {
  private index = new Map<string, TensorMeta>();
  private cache = new Map<string, Tensor>();
  private fp16: boolean;

  constructor(public manifest: Manifest, private buffer: ArrayBuffer) {
    for (const t of manifest.tensors) this.index.set(t.name, t);
    this.fp16 = manifest.dtype === "float16";
  }

  // Memoized: tensors are fetched once per name per forward (and per row), so we
  // decode/view each tensor a single time. For fp16 this also avoids re-decoding.
  get(name: string): Tensor {
    const hit = this.cache.get(name);
    if (hit) return hit;
    const meta = this.index.get(name);
    if (!meta) throw new Error(`weight not found: ${name}`);
    let data: Float32Array;
    if (this.fp16) {
      const u16 = new Uint16Array(this.buffer, meta.offset * 2, meta.length);
      data = new Float32Array(meta.length);
      for (let i = 0; i < meta.length; i++) data[i] = halfToFloat(u16[i]);
    } else {
      data = new Float32Array(this.buffer, meta.offset * 4, meta.length);
    }
    const tensor: Tensor = { shape: meta.shape, data };
    this.cache.set(name, tensor);
    return tensor;
  }
}

export async function loadWeights(baseUrl: string): Promise<Weights> {
  const [manifest, blob] = await Promise.all([
    fetch(`${baseUrl}/manifest.json`).then((r) => r.json() as Promise<Manifest>),
    fetch(`${baseUrl}/weights.bin`).then((r) => r.arrayBuffer()),
  ]);
  return new Weights(manifest, blob);
}

/** Build Weights from already-read bytes (Node test path). */
export function weightsFromBytes(manifest: Manifest, bytes: Uint8Array): Weights {
  // Copy into a fresh, 0-aligned ArrayBuffer (fs Buffers may be sub-views).
  const ab = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(ab).set(bytes);
  return new Weights(manifest, ab);
}
