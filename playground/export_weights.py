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
    python playground/export_weights.py --task sbi_sir \
        --checkpoint artifacts/sbi_sir.pt --out playground/public/models/sbi_sir
    python playground/export_weights.py --task bo1d \
        --checkpoint artifacts/bo1d.pt --out playground/public/models/bo1d

Outputs `<out>/manifest.json` and `<out>/weights.bin` (float16, little-endian).
The manifest also carries a `provenance` block — the training `config`/`seed`/`step`
read from the checkpoint plus export-time stamps (commit, timestamp, checksum) — so a
deployed blob is traceable to the run that produced it.

Weights are stored as float16 to halve the blob size. Parameters are rounded with
torch's `.half().float()` before serialization; `parity.py` applies the SAME
rounding before generating fixtures, so the shipped weights and the parity
references reflect identical values (only float32-vs-float64 arithmetic differs).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# This script lives in playground/; import the core + task modules from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Task name -> module providing load_checkpoint(path, device) and variables().
# gp1d_arbuffer and gp1d_aline are non-core extension models (extensions/): same
# manifest format, with the extra buf_blocks.* / policy_* tensors flowing through
# the generic state-dict table below (model identity in TS is key-presence).
TASK_MODULES = {
    "gp1d": "gp1d",
    "gaussian": "gaussian_toy",
    "sbi_sir": "sbi_sir",
    "bo1d": "bo1d",
    "gp1d_arbuffer": "extensions.arbuffer.gp1d_arbuffer",
    "gp1d_aline": "extensions.aline.gp1d_aline",
}


def quantize_fp16_inplace(model) -> None:
    """Round every parameter to float16 (kept as float32) so the model evaluates
    with exactly the values that will be shipped as float16. Shared by parity.py."""
    for p in model.parameters():
        p.data = p.data.half().float()


def _git_short_sha() -> str | None:
    """`git rev-parse --short HEAD` for the nanoACE repo, or None if unavailable
    (git missing, not a repo, etc.) — provenance is best-effort, never fatal."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _json_safe(obj):
    """Coerce a value to JSON-native types. Argparse configs are usually already
    clean; this guards a stray Path/Tensor in the checkpoint config from breaking
    the manifest write by stringifying it rather than raising."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# A Windows drive root (`C:\` / `C:/`), a UNC share, or a POSIX root (`\` / `/`).
_ABS_PATH = re.compile(r"^([A-Za-z]:[\\/]|[\\/])")


def _redact_abs_paths(obj):
    """Basename any absolute-path string so the published manifest can't leak an
    OS username or local directory layout. The full training config is shipped
    deliberately (this repo's models carry config + seed to stay reproducible),
    so relative paths like `artifacts/gp1d.pt` are kept — they reveal nothing and
    `base_checkpoint` is useful provenance; only the absolute-path vector is cut.
    Runs after `_json_safe`, so every leaf is already a plain string."""
    if isinstance(obj, dict):
        return {k: _redact_abs_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_abs_paths(v) for v in obj]
    if isinstance(obj, str) and _ABS_PATH.match(obj):
        return re.split(r"[\\/]", obj)[-1]  # keep only the final path component
    return obj


def build_provenance(checkpoint_path: str | Path) -> dict:
    """Training-provenance block: what run produced these weights.

    The training `config`/`seed`/`step` are read straight from the checkpoint
    payload (written by `train.save_checkpoint`; the model-only `load_checkpoint`
    drops them, so we read the raw payload here), alongside export-time stamps the
    checkpoint can't know. Degrades to `{checkpoint_only: true}` when the payload
    carries no `config` (legacy blobs, or intermediate resumable saves that predate
    the final model-only save where `config` lands)."""
    path = Path(checkpoint_path)
    prov: dict = {
        "checkpoint": path.name,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "nanoace_commit": _git_short_sha(),
    }
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config")
    if config is None:
        prov["checkpoint_only"] = True
        return prov
    prov["seed"] = payload.get("seed")
    prov["step"] = payload.get("step")
    # The full config is published deliberately (reproducibility); strip only
    # absolute paths so a public manifest can't carry a username / local layout.
    prov["config"] = _redact_abs_paths(_json_safe(config))
    return prov


def build_manifest(model, task: str, provenance: dict | None = None) -> tuple[dict, bytes]:
    """Return (manifest dict, concatenated float16 little-endian weight bytes)."""

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
        "provenance": provenance,
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

    provenance = build_provenance(args.checkpoint)
    manifest, blob = build_manifest(model, args.task, provenance)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / manifest["weights_file"]).write_bytes(blob)

    cfg_prov = provenance.get("config") or {}
    print(f"[{args.task}] {len(manifest['tensors'])} tensors, "
          f"{manifest['total_floats']:,} floats ({len(blob):,} bytes) -> {out}")
    print(f"  provenance: checkpoint={provenance['checkpoint']} "
          f"commit={provenance.get('nanoace_commit')} steps={cfg_prov.get('steps', 'n/a')}")


if __name__ == "__main__":
    main()
