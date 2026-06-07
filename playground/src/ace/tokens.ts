/** Mutable builder for a single TokenSet (B=1), shared by the demos. */

import type { TokenSet } from "./model";

export class TokenList {
  varId: number[] = [];
  x: number[][] = [];
  value: number[] = [];
  valueIndex: number[] = [];
  prior: number[][] = [];
  mode: number[] = [];
  mask: boolean[] = [];

  add(
    varId: number,
    mode: number,
    o: { x?: number; value?: number; valueIndex?: number; prior?: [number, number] } = {},
  ): void {
    this.varId.push(varId);
    this.x.push([o.x ?? 0]);
    this.value.push(o.value ?? 0);
    this.valueIndex.push(o.valueIndex ?? 0);
    this.prior.push(o.prior ?? [0, 0]);
    this.mode.push(mode);
    this.mask.push(true);
  }

  get(): TokenSet {
    return {
      varId: this.varId,
      x: this.x,
      value: this.value,
      valueIndex: this.valueIndex,
      prior: this.prior,
      mode: this.mode,
      mask: this.mask,
    };
  }
}
