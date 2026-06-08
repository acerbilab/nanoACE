"""Offline sharded data pool for the expensive examples (generate -> save -> train).

This is the optional offline counterpart to the examples' online samplers. It exists
for GP-1D and BO, whose per-instance physics (GP Cholesky / Matheron planting) is the
expensive part worth caching; Gaussian and SIR are cheap and stay online-only.

Design (see docs/plans/PLAN-offline-data-and-reseed.md):

- **Cache only the expensive physics draws.** A pool stores exactly what an example's
  `draw_instances` produces (a struct-of-arrays per shard), nothing about the
  context/target split or the reveal mask -- those are recomputed at read time, so the
  reveal strategy can change without regenerating the pool.
- **Two functions, one schema.** `write_pool(draw_fn, ...)` generates; `PoolReader` reads
  and returns `Batch`es via the *same* `assemble` the online path uses. `fit` sees the
  identical `(step) -> Batch` interface either way -- no second training code path.
- **Stateless, index-keyed randomness.** The physical-row shuffle and the per-instance
  split decisions are pure functions of `(seed, logical position)` via `ace.mix_int64`
  (no `torch.Generator` state). The logical position is `p = (step - 1) * B + j`, which
  is *batch-size- and steps-independent* (position `p` is the same dataset under any `B`),
  so a pooled run is reproducible and resume-exact from the `step` `fit` already restores.
- **One provenance check.** The manifest carries the `variables()` schema (a hard gate --
  a wrong schema silently misreads the arrays) and a `sha256` of the DGP `gen_config`
  (forceable with `force=True`). This replaces the heavy multi-axis resume-guard matrix.

Simplification: `PoolReader` loads the whole pool into RAM at construction (the shard
files remain the on-disk, inspectable artifact). nanoACE pools fit comfortably; streaming
shards to scale past RAM is a deliberate non-goal here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import torch

from ace import Batch, Variable, mix_int64, mix_seed, reveal_mask_from_index

SCHEMA = "nanoace-pool-v1"

# Distinct splitmix salts so the row-shuffle and split streams are decorrelated namespaces.
_SALT = {"shard": 0xA1, "within": 0xB2, "nctx": 0xC3, "reveal": 0xD4}


# --------------------------------------------------------------------------- #
# Manifest helpers (canonical JSON; variables() repr; DGP config hash)
# --------------------------------------------------------------------------- #


def variables_repr(variables: Sequence[Variable]) -> list[dict]:
    """Serializable, order-preserving view of `variables()` for the manifest/guard."""

    return [
        {
            "name": v.name,
            "kind": v.kind,
            "value_type": v.value_type,
            "cardinality": v.cardinality,
            "transform": v.transform,
            "bounds": list(v.bounds) if v.bounds is not None else None,
        }
        for v in variables
    ]


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def config_hash(gen_config: dict) -> str:
    """16-hex `sha256` of the DGP-only `gen_config` (canonical JSON). Drift => regenerate."""

    return hashlib.sha256(_canonical_json(gen_config).encode("utf-8")).hexdigest()[:16]


def _to_storage(v: torch.Tensor) -> torch.Tensor:
    """Store continuous fields as float32 and integer (categorical) fields as int64."""

    v = v.detach().cpu().contiguous()
    if v.dtype.is_floating_point:
        return v.to(torch.float32)
    return v.to(torch.int64)


# --------------------------------------------------------------------------- #
# Pool generation
# --------------------------------------------------------------------------- #


def _shard_meta(*, cfg_hash: str, seed: int, shard_index: int, start: int, count: int) -> dict:
    return {"schema": SCHEMA, "config_hash": cfg_hash, "seed": int(seed),
            "shard_index": int(shard_index), "start": int(start), "count": int(count)}


def _valid_shard(path: Path, meta: dict) -> bool:
    if not path.exists():
        return False
    try:
        shard = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return shard.get("__meta__") == meta


def write_pool(
    draw_fn: Callable[[int], dict],
    out: str | Path,
    *,
    pool_size: int,
    shard_size: int,
    gen_config: dict,
    variables: Sequence[Variable],
    seed: int,
    force: bool = False,
    log=print,
) -> dict:
    """Generate a sharded finite pool of drawn instances.

    `draw_fn(n) -> dict[str, Tensor]` returns one example's CPU-native struct-of-arrays for
    `n` instances (the example's `draw_instances` bound to its frozen DGP config). Shard `i`
    is produced after `torch.manual_seed(mix_seed(seed, i))`, so a partial build resumes
    deterministically (valid existing shards are skipped). Shards are written atomically
    (temp -> rename); the manifest is written **last**, so "manifest exists => pool complete".
    """

    if pool_size <= 0 or shard_size <= 0:
        raise ValueError("pool_size and shard_size must be positive")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"{manifest_path} exists; pass force=True to rebuild")

    cfg_hash = config_hash(gen_config)
    n_shards = (pool_size + shard_size - 1) // shard_size
    shards, fields_meta = [], None
    for i, start in enumerate(range(0, pool_size, shard_size)):
        count = min(shard_size, pool_size - start)
        fname = f"shard_{i:05d}.pt"
        path = out / fname
        meta = _shard_meta(cfg_hash=cfg_hash, seed=seed, shard_index=i, start=start, count=count)
        if _valid_shard(path, meta) and not force:
            log(f"[skip] {fname} ({count} instances)")
        else:
            torch.manual_seed(mix_seed(seed, i))
            inst = draw_fn(count)
            shard = {k: _to_storage(v) for k, v in inst.items()}
            shard["__meta__"] = meta
            tmp = path.with_suffix(".pt.tmp")
            torch.save(shard, tmp)
            tmp.replace(path)
            log(f"[write] {fname} ({count} instances) [{i + 1}/{n_shards}]")
        if fields_meta is None:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            fields_meta = [
                {"name": k, "shape": list(shard[k].shape[1:]), "dtype": str(shard[k].dtype).replace("torch.", "")}
                for k in shard
                if k != "__meta__"
            ]
        shards.append({"file": fname, "start": start, "count": count})

    manifest = {
        "schema": SCHEMA,
        "pool_size": int(pool_size),
        "shard_size": int(shard_size),
        "seed": int(seed),
        "gen_config": gen_config,
        "config_hash": cfg_hash,
        "variables": variables_repr(variables),
        "fields": fields_meta,
        "shards": shards,
    }
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(manifest_path)
    log(f"[done] manifest -> {manifest_path}")
    return manifest


# --------------------------------------------------------------------------- #
# Pool reading
# --------------------------------------------------------------------------- #


class PoolReader:
    """Read a pool as a `sample_batch(step) -> Batch` thunk for `fit`.

    Validates the manifest on construction: schema and `variables()` are hard gates (a wrong
    token schema would silently misread the cached arrays); a `gen_config` config-hash
    mismatch is refused unless `force=True` (a knowing reuse under changed DGP constants).
    `max_context < N_TOTAL` is required (at least one target). Splits and the "both" shuffle
    are stateless functions of `(seed, p)` with `p = (step - 1) * B + j`.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        assemble: Callable[..., Batch],
        variables: Sequence[Variable],
        gen_config: dict,
        batch_size: int,
        seed: int,
        max_context: int,
        min_context: int,
        latent_context_prob: float,
        device: torch.device | str,
        force: bool = False,
    ):
        self.dir = Path(path)
        self.manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema") != SCHEMA:
            raise ValueError(f"pool schema {self.manifest.get('schema')!r} != {SCHEMA!r}; regenerate the pool")
        if self.manifest.get("variables") != variables_repr(variables):
            raise ValueError(
                "pool variables() mismatch: the cached arrays would be misread under this schema. "
                "Regenerate the pool (NOT overridable by force)."
            )
        want_hash = config_hash(gen_config)
        if self.manifest.get("config_hash") != want_hash:
            msg = (
                f"pool DGP config-hash mismatch (manifest {self.manifest.get('config_hash')}, "
                f"current {want_hash}); the cached data was generated under different DGP constants."
            )
            if not force:
                raise ValueError(msg + " Regenerate the pool, or pass --pool-force to reuse it anyway.")
            print("warning: " + msg + " Reusing it because --pool-force was given.")

        self.n_total = int(self.manifest["gen_config"]["N_TOTAL"])
        if not max_context < self.n_total:
            raise ValueError(f"max_context ({max_context}) must be < N_TOTAL ({self.n_total}) to leave >=1 target")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not 1 <= min_context <= max_context:
            raise ValueError(f"need 1 <= min_context ({min_context}) <= max_context ({max_context})")

        self.assemble = assemble
        self.variables = list(variables)
        self.B = int(batch_size)
        self.seed = int(seed)
        self.max_context = int(max_context)
        self.min_context = int(min_context)
        self.q = 1.0 - float(latent_context_prob)
        self.device = device
        self.pool_size = int(self.manifest["pool_size"])
        self.n_latents = sum(1 for v in self.variables if v.kind == "latent")

        # Load the whole pool into RAM, concatenated in physical (shard) order.
        self.shard_starts = [int(e["start"]) for e in self.manifest["shards"]]
        self.shard_counts = [int(e["count"]) for e in self.manifest["shards"]]
        field_names = [f["name"] for f in self.manifest["fields"]]
        parts: dict[str, list[torch.Tensor]] = {k: [] for k in field_names}
        for e in self.manifest["shards"]:
            shard = torch.load(self.dir / e["file"], map_location="cpu", weights_only=False)
            for k in field_names:
                parts[k].append(shard[k])
        self._fields = {k: torch.cat(v, dim=0) for k, v in parts.items()}
        self._perm_pass: int | None = None
        self._perm: torch.Tensor | None = None

    def _key(self, name: str, *ints: int) -> int:
        """Scalar splitmix-style key for `(seed, name, *ints)`, in `[0, 2**62)`."""

        h = (self.seed + 1) * 0x9E3779B97F4A7C15
        h ^= (_SALT[name] + 1) * 0xBF58476D1CE4E5B9
        for i, x in enumerate(ints):
            h += (int(x) + 1) * (0x94D049BB133111EB + i)
        return h & ((1 << 62) - 1)

    def _index_perm(self, n: int, key: int) -> torch.Tensor:
        return mix_int64(torch.arange(n, dtype=torch.int64) + key).argsort()

    def _pass_perm(self, p: int) -> torch.Tensor:
        """Physical-row visiting order for pass `p`: shard-order + within-shard ("both")."""

        if self._perm_pass == p and self._perm is not None:
            return self._perm
        shard_order = self._index_perm(len(self.shard_starts), self._key("shard", p))
        rows = []
        for s in shard_order.tolist():
            local = self._index_perm(self.shard_counts[s], self._key("within", p, s))
            rows.append(self.shard_starts[s] + local)
        self._perm_pass, self._perm = p, torch.cat(rows)
        return self._perm

    def _physical_rows(self, pos: torch.Tensor) -> torch.Tensor:
        """Map absolute logical positions to physical pool rows (per-pass "both" shuffle)."""

        pass_idx = pos // self.pool_size
        pass_pos = pos % self.pool_size
        out = torch.empty_like(pos)
        for p in torch.unique(pass_idx).tolist():
            m = pass_idx == p
            out[m] = self._pass_perm(int(p))[pass_pos[m]]
        return out

    def _n_context(self, pos: torch.Tensor) -> torch.Tensor:
        span = self.max_context - self.min_context + 1
        mixed = mix_int64(pos + self._key("nctx")) & ((1 << 62) - 1)
        return self.min_context + (mixed % span)

    def __call__(self, step: int) -> Batch:
        start = (int(step) - 1) * self.B
        pos = torch.arange(start, start + self.B, dtype=torch.int64)
        physical = self._physical_rows(pos)
        inst = {k: v.index_select(0, physical) for k, v in self._fields.items()}
        n_context = self._n_context(pos).to(self.device)
        reveal = reveal_mask_from_index(pos + self._key("reveal"), self.n_latents, self.q).to(self.device)
        return self.assemble(
            inst,
            variables=self.variables,
            n_context=n_context,
            reveal_mask=reveal,
            max_context=self.max_context,
            device=self.device,
        )


# --------------------------------------------------------------------------- #
# Build CLI: python data.py <example> --out DIR --pool-size N [...]
# --------------------------------------------------------------------------- #


def _example_module(name: str):
    if name == "gp1d":
        import gp1d as ex
    elif name == "bo1d":
        import bo1d as ex
    else:
        raise SystemExit(f"unknown example {name!r}; choose gp1d or bo1d")
    return ex


def main() -> None:
    p = argparse.ArgumentParser(description="Build a sharded finite training-data pool (generate -> save).")
    p.add_argument("example", choices=("gp1d", "bo1d"))
    p.add_argument("--out", required=True, help="output pool directory")
    p.add_argument("--pool-size", type=int, required=True, help="number of instances in the finite pool")
    p.add_argument("--shard-size", type=int, default=8192, help="instances per shard")
    p.add_argument("--seed", type=int, default=0, help="build seed; shard i uses mix_seed(seed, i)")
    p.add_argument("--force", action="store_true", help="overwrite an existing complete pool")
    args = p.parse_args()

    ex = _example_module(args.example)
    print(f"== build {args.example} pool ==  out={args.out}  pool_size={args.pool_size}  shard_size={args.shard_size}")
    write_pool(
        ex.draw_pool,
        args.out,
        pool_size=args.pool_size,
        shard_size=args.shard_size,
        gen_config=ex.gen_config(),
        variables=ex.variables(),
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
