# Copyright 2026 The Orbax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One-shot prep for the safetensors load benchmark: stage a model + spec.

Downloads a model from the HF Hub (`--repo`) or reads an already-staged copy
(`--local-dir`, may be `gs://`), then optionally mirrors it to GCS (`--gcs`) and
optionally writes the per-tensor sharding JSON the benchmark consumes
(`--sharding-out`, one `--strategy` per run). Example:

  python -m orbax.checkpoint._src.testing.benchmarks.safetensor.prepare \\
      --repo mistralai/Mistral-7B-v0.1 \\
      --gcs gs://orbax-benchmarks/fixtures/mistral7b/ \\
      --axis-size 8 --strategy fsdp \\
      --sharding-out gs://orbax-benchmarks/sharding/mistral7b_fsdp8.json

Needs `huggingface_hub` for `--repo` (plus `HF_TOKEN` for gated models) and an
authenticated `gsutil` for `--gcs`.
"""

from __future__ import annotations

import json
import struct
import subprocess
import tempfile

from absl import app
from absl import flags
from etils import epath

# u64 little-endian header length, then that many bytes of JSON.
_HEADER_LEN_BYTES = 8

# Safetensors dtype string -> the name jnp.dtype() accepts; others are rejected.
_DTYPE_TO_JNP_NAME = {
    "BOOL": "bool_",
    "I8": "int8",
    "U8": "uint8",
    "I16": "int16",
    "U16": "uint16",
    "I32": "int32",
    "U32": "uint32",
    "I64": "int64",
    "U64": "uint64",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "BF16": "bfloat16",
    "F8_E5M2": "float8_e5m2",
    "F8_E4M3": "float8_e4m3fn",
}


def _read_safetensors_header(path: epath.Path) -> dict:
  """Reads the JSON header of one `.safetensors` file (a small ranged read)."""
  with path.open("rb") as f:
    (header_len,) = struct.unpack("<Q", f.read(_HEADER_LEN_BYTES))
    return json.loads(f.read(header_len))


def _fsdp_spec(
    shape: list[int], axis_name: str, axis_size: int
) -> list[str | None]:
  """Leading-dim FSDP spec; replicated when dim 0 isn't divisible by axis_size."""
  if len(shape) >= 2 and shape[0] % axis_size == 0:
    return [axis_name] + [None] * (len(shape) - 1)
  return [None] * len(shape)


def _tp_inner_spec(
    shape: list[int], axis_name: str, axis_size: int
) -> list[str | None]:
  """Last-axis (inner-dim) TP spec; replicated if the last dim isn't divisible."""
  if len(shape) >= 2 and shape[-1] % axis_size == 0:
    return [None] * (len(shape) - 1) + [axis_name]
  return [None] * len(shape)


# strategy -> (mesh axis name, per-tensor spec function)
_STRATEGIES = {
    "fsdp": ("fsdp", _fsdp_spec),
    "tp_inner": ("tp", _tp_inner_spec),
}


def build_config(model_dir: epath.Path, axis_size: int, strategy: str) -> dict:
  """Builds the `{name: {shape, dtype, sharding}}` map for every tensor.

  Args:
    model_dir: Directory of `*.safetensors` files (local or `gs://`).
    axis_size: Size of the single mesh axis to shard along.
    strategy: One of `_STRATEGIES`.

  Returns:
    The sharding-config map `get_abstract_state_from_sharding_config` consumes.

  Raises:
    ValueError: If `strategy` is unknown, no tensors are found, or a dtype is
      unsupported.
  """
  try:
    axis_name, spec_fn = _STRATEGIES[strategy]
  except KeyError as e:
    raise ValueError(
        f"Unknown strategy {strategy!r}; supported: "
        + ", ".join(sorted(_STRATEGIES))
    ) from e
  mesh = {"shape": [axis_size], "axes": [axis_name]}
  files = sorted(model_dir.glob("*.safetensors"))
  if not files:
    raise ValueError(
        f"No `.safetensors` files found under {model_dir}. A cached HF model "
        "lives under $HF_HOME/hub/models--<org>--<repo>/snapshots/<sha>/."
    )
  out: dict[str, dict] = {}
  for f in files:
    for name, info in _read_safetensors_header(f).items():
      if name == "__metadata__":
        continue
      if name in out:
        raise ValueError(
            f"Tensor {name!r} appears in more than one file under {model_dir}; "
            "check the model's index file."
        )
      dtype = info["dtype"]
      if dtype not in _DTYPE_TO_JNP_NAME:
        raise ValueError(
            f"Tensor {name!r} has unsupported dtype {dtype!r}; supported: "
            + ", ".join(sorted(_DTYPE_TO_JNP_NAME))
        )
      shape = list(info["shape"])
      out[name] = {
          "shape": shape,
          "dtype": _DTYPE_TO_JNP_NAME[dtype],
          "sharding": {"mesh": mesh, "spec": spec_fn(shape, axis_name, axis_size)},
      }
  return out


def _snapshot_download(repo_id: str, dest: epath.Path) -> epath.Path:
  """Downloads a repo's `*.safetensors` (+ index) from the HF Hub to `dest`."""
  try:
    from huggingface_hub import snapshot_download  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    raise SystemExit(
        "huggingface_hub is required for --repo; install with "
        "`pip install huggingface_hub`."
    ) from e
  return epath.Path(
      snapshot_download(
          repo_id=repo_id,
          local_dir=str(dest),
          allow_patterns=["*.safetensors", "*.safetensors.index.json"],
      )
  )


def _mirror_to_gcs(local_dir: epath.Path, gcs_uri: str) -> None:
  """Mirrors a model dir to GCS via `gsutil -m rsync` (streams; idempotent)."""
  subprocess.run(
      ["gsutil", "-m", "rsync", "-r", str(local_dir), gcs_uri], check=True
  )


def _write_sharding(
    model_dir: epath.Path, axis_size: int, strategy: str, out: str
) -> int:
  """Builds the sharding config from `model_dir` and writes it to `out`."""
  cfg = build_config(model_dir, axis_size, strategy)
  dest = epath.Path(out)
  if dest.parent and "://" not in out:
    dest.parent.mkdir(parents=True, exist_ok=True)
  dest.write_text(json.dumps(cfg, indent=2))
  return len(cfg)


_REPO = flags.DEFINE_string("repo", None, "HF Hub repo id to download.")
_LOCAL_DIR = flags.DEFINE_string(
    "local_dir",
    None,
    "Already-staged model dir (local or gs://) to read instead.",
)
_GCS = flags.DEFINE_string(
    "gcs", None, "Mirror the model's safetensors to this gs:// bucket."
)
_SHARDING_OUT = flags.DEFINE_string(
    "sharding_out", None, "Write the sharding JSON here (local or gs://)."
)
_AXIS_SIZE = flags.DEFINE_integer(
    "axis_size",
    None,
    "Mesh axis size (= participating devices); needs --sharding_out.",
)
_STRATEGY = flags.DEFINE_enum(
    "strategy",
    "fsdp",
    sorted(_STRATEGIES),
    "fsdp splits dim 0 (contiguous); tp_inner splits the last dim.",
)
_DOWNLOAD_DIR = flags.DEFINE_string(
    "download_dir", None, "Keep the --repo download here instead of a temp dir."
)

flags.mark_flags_as_mutual_exclusion(["repo", "local_dir"], required=True)


def main(argv: list[str]) -> None:
  del argv  # Unused; inputs come from flags.
  if not _GCS.value and not _SHARDING_OUT.value:
    raise app.UsageError("nothing to do: pass --gcs and/or --sharding_out.")
  if _SHARDING_OUT.value and _AXIS_SIZE.value is None:
    raise app.UsageError("--sharding_out requires --axis_size.")

  def _run(model_dir: epath.Path) -> None:
    if _GCS.value:
      print(f">>> Mirroring {model_dir} -> {_GCS.value}")
      _mirror_to_gcs(model_dir, _GCS.value)
    if _SHARDING_OUT.value:
      n = _write_sharding(
          model_dir, _AXIS_SIZE.value, _STRATEGY.value, _SHARDING_OUT.value
      )
      print(f">>> Wrote {n} tensor entries to {_SHARDING_OUT.value}")

  if _REPO.value:
    if _DOWNLOAD_DIR.value:
      _run(_snapshot_download(_REPO.value, epath.Path(_DOWNLOAD_DIR.value)))
    else:
      with tempfile.TemporaryDirectory(prefix="safetensors-prepare-") as td:
        _run(_snapshot_download(_REPO.value, epath.Path(td)))
  else:
    _run(epath.Path(_LOCAL_DIR.value))


if __name__ == "__main__":
  app.run(main)
