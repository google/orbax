"""Sharding-config generator for native Orbax checkpoints.

Reads leaf names, shapes, and dtypes from an Orbax checkpoint's metadata
(via `ocp.metadata`) and applies a leading-dim FSDP rule (or an inner-dim
TP rule) to each, emitting the per-tensor sharding JSON that
`get_abstract_state_from_sharding_config` consumes. Use it to produce a
sharding config for a custom Orbax checkpoint, or for a topology the
published configs don't cover.

The published Llama 3.1 sharding configs already live under
`gs://orbax-benchmarks/sharding-configs/...` (and the `llama3-*` YAMLs in
`../configs/` point straight at them); this script is for everything else.

The output JSON shape matches what `get_abstract_state_from_sharding_config`
consumes: per-leaf `{shape, dtype, sharding: {mesh: {shape, axes}, spec}}`.
The mesh it emits is single-axis, so the consuming YAML's
`mesh_config.mesh_axes` must be the matching single axis
(`["fsdp"]` / `["tp"]`) of the same size.

Usage:

  # Against a published or custom Orbax checkpoint, fsdp=16:
  python generate_from_metadata.py \\
      --checkpoint gs://my-bucket/my-ckpt/items \\
      --axis-size 16 --strategy fsdp \\
      --output my_model_fsdp16.json

  # Inner-dim TP variant:
  python generate_from_metadata.py \\
      --checkpoint /local/path/to/items \\
      --axis-size 8 --strategy tp_inner \\
      --output my_model_tp8.json
"""

import argparse
import json
import pathlib

from etils import epath
import numpy as np
from orbax.checkpoint import v1 as ocp
from orbax.checkpoint._src.tree import utils as tree_utils


def _fsdp_spec(
    shape: list[int], axis_name: str, axis_size: int
) -> list[str | None]:
  """Leading-dim FSDP partition spec; falls back to fully replicated.

  Args:
    shape: The leaf shape.
    axis_name: Mesh axis to shard the leading dim along.
    axis_size: Size of that mesh axis.

  Returns:
    A PartitionSpec-style list (axis name or None per dim).
  """
  if len(shape) < 2:
    return [None] * len(shape)
  if shape[0] % axis_size == 0:
    return [axis_name] + [None] * (len(shape) - 1)
  return [None] * len(shape)


def _tp_inner_spec(
    shape: list[int], axis_name: str, axis_size: int
) -> list[str | None]:
  """Inner-dim (last-axis) partition spec; falls back to fully replicated.

  Args:
    shape: The leaf shape.
    axis_name: Mesh axis to shard the last dim along.
    axis_size: Size of that mesh axis.

  Returns:
    A PartitionSpec-style list (axis name or None per dim).
  """
  if len(shape) < 2:
    return [None] * len(shape)
  if shape[-1] % axis_size == 0:
    return [None] * (len(shape) - 1) + [axis_name]
  return [None] * len(shape)


_STRATEGIES = {
    "fsdp": ("fsdp", _fsdp_spec),
    "tp_inner": ("tp", _tp_inner_spec),
}


def build_config(
    checkpoint_path: str, axis_size: int, strategy: str = "fsdp"
) -> dict:
  """Builds the per-leaf sharding map for every leaf in the checkpoint.

  Args:
    checkpoint_path: Path (local or `gs://`) to an Orbax checkpoint dir.
    axis_size: Size of the single mesh axis to shard along.
    strategy: Sharding strategy, "fsdp" or "tp_inner".

  Returns:
    A `{leaf_name: entry}` sharding-config map.

  Raises:
    ValueError: If `strategy` is unknown.
  """
  try:
    axis_name, spec_fn = _STRATEGIES[strategy]
  except KeyError as e:
    raise ValueError(
        f"Unknown strategy {strategy!r}; supported: "
        + ", ".join(sorted(_STRATEGIES))
    ) from e
  metadata = ocp.metadata(epath.Path(checkpoint_path))
  flat = tree_utils.to_flat_dict(metadata.metadata, sep=".")
  mesh = {"shape": [axis_size], "axes": [axis_name]}
  out: dict[str, dict] = {}
  for name, leaf in flat.items():
    shape = list(leaf.shape)
    out[name] = {
        "shape": shape,
        "dtype": np.dtype(leaf.dtype).name,
        "sharding": {
            "mesh": mesh,
            "spec": spec_fn(shape, axis_name, axis_size),
        },
    }
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--checkpoint",
      required=True,
      help="Path (local or gs://) to the Orbax checkpoint directory.",
  )
  parser.add_argument(
      "--axis-size",
      type=int,
      required=True,
      help="Size of the single mesh axis (= number of devices participating).",
  )
  parser.add_argument(
      "--strategy",
      default="fsdp",
      choices=sorted(_STRATEGIES),
      help=(
          "Sharding strategy. `fsdp` splits dim 0 (contiguous shards);"
          " `tp_inner` splits the last dim (strided shards)."
      ),
  )
  parser.add_argument(
      "--output",
      type=pathlib.Path,
      required=True,
      help="Destination JSON path; parent dir is created if missing.",
  )
  args = parser.parse_args()

  cfg = build_config(args.checkpoint, args.axis_size, strategy=args.strategy)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(cfg, indent=2))
  print(f"Wrote {len(cfg)} leaf entries to {args.output}")


if __name__ == "__main__":
  main()
