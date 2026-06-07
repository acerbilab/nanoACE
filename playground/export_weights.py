"""Export a nanoACE checkpoint to a browser-loadable manifest + weight blob.

The TypeScript playground reimplements `ace.py`'s forward pass and loads the
weights produced here. To avoid divergence, all *derived* constants
(`is_discrete`, `disc_offsets`, bounds, ...) are read from a live `ACE` instance
rather than recomputed by hand — the live model is the single source of truth.

Usage (from the project venv):

    python playground/export_weights.py --task gp1d \
        --checkpoint artifacts/gp1d.pt --out playground/public/models/gp1d
    python playground/export_weights.py --task gaussian \
        --checkpoint artifacts/gaussian_toy.pt --out playground/public/models/gaussian

Outputs `<out>/manifest.json` and `<out>/weights.bin` (float16, little-endian).

Weights are stored as float16 to halve the blob size. Parameters are rounded with
torch's `.half().float()` before serialization; `parity.py` applies the SAME
rounding before generating fixtures, so the shipped weights and the parity
references reflect identical values (only float32-vs-float64 arithmetic differs).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

# This script lives in playground/; import the core + task modules from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Task name -> module providing load_checkpoint(path, device) and variables().
TASK_MODULES = {"gp1d": "gp1d", "gaussian": "gaussian_toy"}


def quantize_fp16_inplace(model) -> None:
    """Round every parameter to float16 (kept as float32) so the model evaluates
    with exactly the values that will be shipped as float16. Shared by parity.py."""
    for p in model.parameters():
        p.data = p.data.half().float()


def build_manifest(model, task: str) -> tuple[dict, bytes]:
    """Return (manifest dict, concatenated float32 little-endian weight bytes)."""

    cfg = asdict(model.cfg)
    variables = model.variables
    n_vars = len(variables)

    # Derived constants straight off the live model's registered buffers.
    is_discrete = model.is_discrete.tolist()
    is_latent = model.is_latent.tolist()
    has_bounds = model.has_bounds.tolist()
    bound_lo = model.bound_lo.tolist()
    bound_hi = model.bound_hi.tolist()
    cardinality = model.cardinality.tolist()
    disc_offsets = model.disc_offsets.tolist()
    total_disc = sum(int(v.cardinality or 0) for v in variables if v.value_type == "discrete")
    max_cardinality = max([int(v.cardinality or 0) for v in variables] + [1])

    var_meta = []
    for i, v in enumerate(variables):
        var_meta.append(
            {
                "name": v.name,
                "kind": v.kind,
                "value_type": v.value_type,
                "cardinality": v.cardinality,
                "transform": v.transform,
                "bounds": list(v.bounds) if v.bounds is not None else None,
                "is_discrete": bool(is_discrete[i]),
                "is_latent": bool(is_latent[i]),
                "has_bounds": bool(has_bounds[i]),
                "bound_lo": float(bound_lo[i]),
                "bound_hi": float(bound_hi[i]),
                "disc_offset": int(disc_offsets[i]),
            }
        )

    # Tensor table over the full state_dict, in insertion order. Note `unknown` is a
    # bare nn.Parameter (no .weight/.bias suffix) and MUST be included.
    tensors = []
    blob = bytearray()
    offset = 0  # in float16 elements
    for name, tensor in model.state_dict().items():
        # Parameters are already fp16-rounded (quantize_fp16_inplace), so '<f2' is exact.
        arr = tensor.detach().cpu().contiguous().reshape(-1).numpy().astype("<f2")
        length = int(arr.size)
        tensors.append({"name": name, "shape": list(tensor.shape), "offset": offset, "length": length})
        blob.extend(arr.tobytes())
        offset += length

    manifest = {
        "task": task,
        "cfg": cfg,
        "modes": {"VALUE": 0, "PRIOR": 1, "QUERY": 2},
        "derived": {
            "n_vars": n_vars,
            "head_dim": cfg["d_model"] // cfg["n_heads"],
            "max_cardinality": max_cardinality,
            "total_disc": total_disc,
            "prior_features": 2,
        },
        "variables": var_meta,
        "tensors": tensors,
        "total_floats": offset,
        "weights_file": "weights.bin",
        "dtype": "float16",
        "byte_order": "little",
    }
    return manifest, bytes(blob)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=sorted(TASK_MODULES))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    module = importlib.import_module(TASK_MODULES[args.task])
    model = module.load_checkpoint(args.checkpoint, torch.device("cpu"))
    model.eval()
    quantize_fp16_inplace(model)

    manifest, blob = build_manifest(model, args.task)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / manifest["weights_file"]).write_bytes(blob)

    print(f"[{args.task}] {len(manifest['tensors'])} tensors, "
          f"{manifest['total_floats']:,} floats ({len(blob):,} bytes) -> {out}")


if __name__ == "__main__":
    main()
