/**
 * Variable schema + bounded-latent coordinate helpers.
 *
 * Mirrors `ace.py`: bounded continuous latents live in internal `[-1, 1]` token
 * coordinates; everything else passes through unchanged.
 */

export interface VariableMeta {
  name: string;
  kind: "data" | "latent";
  value_type: "continuous" | "discrete";
  cardinality: number | null;
  transform: string;
  bounds: [number, number] | null;
  is_discrete: boolean;
  is_latent: boolean;
  has_bounds: boolean;
  bound_lo: number;
  bound_hi: number;
  disc_offset: number;
}

export function isBoundedContinuousLatent(v: VariableMeta): boolean {
  return v.is_latent && v.value_type === "continuous" && v.has_bounds;
}

/** Native value -> internal `[-1, 1]` (bounded continuous latents only). */
export function encodeValue(v: VariableMeta, native: number): number {
  if (!isBoundedContinuousLatent(v)) return native;
  return (2.0 * (native - v.bound_lo)) / (v.bound_hi - v.bound_lo) - 1.0;
}

/** Internal `[-1, 1]` -> native value (bounded continuous latents only). */
export function decodeValue(v: VariableMeta, internal: number): number {
  if (!isBoundedContinuousLatent(v)) return internal;
  return v.bound_lo + 0.5 * (internal + 1.0) * (v.bound_hi - v.bound_lo);
}
