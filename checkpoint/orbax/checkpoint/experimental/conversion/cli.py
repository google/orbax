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

"""Command-line interface for Safetensors <-> Orbax checkpoint conversion."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import sys
from typing import TextIO

from absl import app
from absl import flags
import jax
from orbax.checkpoint.experimental.conversion import safetensors_converter


def build_convert_arg_parser(
    prog: str | None = None,
) -> argparse.ArgumentParser:
  """Builds an ArgumentParser for checkpoint conversion."""
  parser = argparse.ArgumentParser(
      prog=prog,
      description=(
          "Convert checkpoints between HuggingFace Safetensors and Orbax"
          " formats using bounded host memory."
      ),
  )
  parser.add_argument(
      "source_path",
      help="Source checkpoint path (.safetensors file or directory).",
  )
  parser.add_argument(
      "dest_path",
      help="Destination checkpoint directory (or .safetensors file path).",
  )
  parser.add_argument(
      "--to-format",
      "--to_format",
      dest="target_format",
      choices=["orbax", "safetensors"],
      default=None,
      help="Target format (auto-detected from source when omitted).",
  )
  parser.add_argument(
      "--overwrite",
      action="store_true",
      default=False,
      help="Overwrite dest_path if it already exists.",
  )
  sep_group = parser.add_mutually_exclusive_group()
  sep_group.add_argument(
      "--key-separator",
      "--key_separator",
      dest="key_separator",
      default=".",
      help="Key delimiter between nested Orbax trees and flat Safetensors.",
  )
  sep_group.add_argument(
      "--flat-keys",
      "--flat_keys",
      dest="flat_keys",
      action="store_true",
      default=False,
      help="Keep/require flat 1-level dictionary keys (key_separator=None).",
  )
  parser.add_argument(
      "--strict-unflatten",
      "--strict_unflatten",
      dest="strict_unflatten",
      action="store_true",
      default=False,
      help="Fail instead of falling back to flat keys on prefix collisions.",
  )
  ckpt_group = parser.add_mutually_exclusive_group()
  ckpt_group.add_argument(
      "--checkpointable-name",
      "--checkpointable_name",
      dest="checkpointable_name",
      default="state",
      help="Orbax checkpointable name (default: 'state').",
  )
  ckpt_group.add_argument(
      "--all-checkpointables",
      "--all_checkpointables",
      dest="all_checkpointables",
      action="store_true",
      default=False,
      help="Convert all PyTree checkpointables in a composite Orbax source.",
  )
  parser.add_argument(
      "--drop-non-tensor-leaves",
      "--drop_non_tensor_leaves",
      dest="drop_non_tensor_leaves",
      action="store_true",
      default=False,
      help="Omit non-tensor leaves (e.g. strings) instead of failing.",
  )
  parser.add_argument(
      "--use-ocdbt",
      "--use_ocdbt",
      dest="use_ocdbt",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Enable OCDBT storage when writing Orbax checkpoints.",
  )
  parser.add_argument(
      "--use-zarr3",
      "--use_zarr3",
      dest="use_zarr3",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Enable Zarr v3 format when writing Orbax checkpoints.",
  )
  parser.add_argument(
      "--use-compression",
      "--use_compression",
      dest="use_compression",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Enable zstd chunk compression when writing Orbax checkpoints.",
  )
  parser.add_argument(
      "--max-shard-size-bytes",
      "--max_shard_size_bytes",
      dest="max_shard_size_bytes",
      type=int,
      default=safetensors_converter.DEFAULT_MAX_SHARD_SIZE_BYTES,
      help="Max bytes per output Safetensors shard (<=0 for single file).",
  )
  parser.add_argument(
      "--max-in-flight-bytes",
      "--max_in_flight_bytes",
      dest="max_in_flight_bytes",
      type=int,
      default=safetensors_converter.DEFAULT_MAX_IN_FLIGHT_BYTES,
      help="Maximum host RAM bytes held in flight during conversion.",
  )
  return parser


def run_convert_cli(
    argv: Sequence[str],
    *,
    prog: str | None = None,
    stdout: TextIO | None = None,
) -> int:
  """Parses CLI args, runs conversion, and prints JSON output.

  Args:
    argv: CLI argument tokens (excluding program/subcommand name).
    prog: Optional program name for usage messages.
    stdout: Output stream for JSON result (defaults to `sys.stdout`).

  Returns:
    Exit status code (`0` on success, `1` on conversion error).
  """
  out = stdout if stdout is not None else sys.stdout
  parser = build_convert_arg_parser(prog=prog)
  parsed = parser.parse_args(list(argv))

  options = safetensors_converter.ConversionOptions(
      overwrite=parsed.overwrite,
      key_separator=None if parsed.flat_keys else parsed.key_separator,
      strict_unflatten=parsed.strict_unflatten,
      checkpointable_name=(
          None if parsed.all_checkpointables else parsed.checkpointable_name
      ),
      drop_non_tensor_leaves=parsed.drop_non_tensor_leaves,
      use_ocdbt=parsed.use_ocdbt,
      use_zarr3=parsed.use_zarr3,
      use_compression=parsed.use_compression,
      max_shard_size_bytes=parsed.max_shard_size_bytes,
      max_in_flight_bytes=parsed.max_in_flight_bytes,
  )
  try:
    result = safetensors_converter.convert_checkpoint(
        parsed.source_path,
        parsed.dest_path,
        target_format=parsed.target_format,
        options=options,
    )
    print(json.dumps(result.to_dict(), indent=2), file=out)
    return 0
  except Exception as err:  # pylint: disable=broad-exception-caught
    payload = {
        "source_path": parsed.source_path,
        "dest_path": parsed.dest_path,
        "error": f"{type(err).__name__}: {err}",
    }
    print(json.dumps(payload, indent=2), file=out)
    return 1


def main(argv: Sequence[str] | None = None) -> None:
  raw_argv = list(sys.argv if argv is None else argv)
  prog = raw_argv[0] if raw_argv else "orbax-convert"
  cli_args = raw_argv[1:]

  def _run() -> int:
    jax.config.update("jax_platforms", "cpu")
    return run_convert_cli(cli_args, prog=prog)

  if flags.FLAGS.is_parsed():
    sys.exit(_run())
  app.run(lambda _: sys.exit(_run()), argv=[prog])


if __name__ == "__main__":
  main()
