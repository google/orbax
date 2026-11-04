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

r"""CLI tool to convert Multi-Tier Checkpointing (MTC) backups to standard Orbax checkpoints.

Usage example:
  python3 convert_mtc_checkpoint.py \
      --input_dir=gs://my-bucket/my-job/2026-09-09_18-00 \
      --overwrite
"""

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging
from etils import epath
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import mtc_checkpoint_converter

FLAGS = flags.FLAGS

_INPUT_DIR = flags.DEFINE_string(
    "input_dir",
    None,
    "Input directory containing MTC backup files (either a specific"
    " timestamp directory or top-level job directory containing timestamp"
    " subdirectories). Supports gs:// and local paths.",
    required=True,
)

_OVERWRITE = flags.DEFINE_boolean(
    "overwrite",
    False,
    "Whether to overwrite target step directories if they already exist.",
)

_STEP = flags.DEFINE_integer(
    "step",
    None,
    "Optional specific step to convert. If omitted, all detected steps will be"
    " converted.",
)

_MAX_WORKERS = flags.DEFINE_integer(
    "max_workers",
    32,
    "Number of worker threads for parallel GCS blob copies.",
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Optional output directory. If omitted, defaults to standard_checkpoints"
    " alongside input_dir.",
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError(f"Unexpected arguments: {argv[1:]}")

  input_dir = epath.Path(_INPUT_DIR.value)
  if _OUTPUT_DIR.value:
    output_dir = epath.Path(_OUTPUT_DIR.value)
  else:
    output_dir = mtc_checkpoint_converter.get_default_output_dir(input_dir)

  print(f"Output directory: {output_dir}")
  logging.info("Starting MTC checkpoint conversion:")
  logging.info("  Input directory : %s", input_dir)
  logging.info("  Output directory: %s", output_dir)
  step_filter = _STEP.value if _STEP.value is not None else "ALL"
  logging.info("  Step filter     : %s", step_filter)
  logging.info("  Overwrite       : %s", _OVERWRITE.value)
  logging.info("  Max workers     : %d", _MAX_WORKERS.value)

  converted = mtc_checkpoint_converter.convert_mtc_backup(
      input_dir=input_dir,
      output_dir=output_dir,
      step=_STEP.value,
      overwrite=_OVERWRITE.value,
      max_workers=_MAX_WORKERS.value,
  )

  logging.info(
      "Successfully converted %d step(s) into standard Orbax checkpoints:",
      len(converted),
  )
  for path in converted:
    logging.info("  - %s", path)
  print(f"\nSuccessfully converted {len(converted)} checkpoint(s):")
  for path in converted:
    print(f"  {path}")


if __name__ == "__main__":
  app.run(main)
