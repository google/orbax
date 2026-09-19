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

"""Merging utility for Orbax checkpoints."""

import asyncio
from collections.abc import Sequence

from absl import app
from absl import flags
from etils import epath
import jax
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.partial import merging


FLAGS = flags.FLAGS

_IN_PATHS = flags.DEFINE_multi_string(
    'in_paths',
    None,
    'Paths of checkpoints to merge.',
)
_OUT_PATH = flags.DEFINE_string(
    'out_path',
    None,
    'Output checkpoint path.',
)
_PER_HOST_MEMORY_LIMIT_BYTES = flags.DEFINE_integer(
    'per_host_memory_limit_bytes',
    None,
    'Memory limit in bytes per CPU host for partial loading and saving.'
    ' Non-uniform memory limits are not supported.',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not _IN_PATHS.value:
    raise app.UsageError('Flag --in_paths must be specified.')
  if _OUT_PATH.value is None:
    raise app.UsageError('Flag --out_path must be specified.')
  if _PER_HOST_MEMORY_LIMIT_BYTES.value is None:
    raise app.UsageError(
        'Flag --per_host_memory_limit_bytes must be specified.'
    )

  if _PER_HOST_MEMORY_LIMIT_BYTES.value <= 0:
    raise ValueError('per_host_memory_limit_bytes must be positive.')

  # Validate input checkpoints.
  layout = orbax_layout.OrbaxLayout()
  for path_str in _IN_PATHS.value:
    path = epath.Path(path_str)
    if not path.exists():
      raise FileNotFoundError(f'Input path {path_str} does not exist.')
    # OrbaxLayout.validate is async.
    try:
      asyncio.run(layout.validate(path))
    except Exception as e:
      raise ValueError(
          f'Input path {path_str} is not a valid checkpoint.'
      ) from e

  # Validate output path.
  out_path = epath.Path(_OUT_PATH.value)
  if out_path.exists():
    if not out_path.is_dir():
      raise ValueError(
          f'Output path {_OUT_PATH.value} exists but is not a directory.'
      )
    if list(out_path.iterdir()):
      raise ValueError(
          f'Output path {_OUT_PATH.value} exists and is not empty.'
      )

  if jax.process_index() == 0:
    out_path.mkdir(parents=True, exist_ok=True)

  merging.merge_checkpoints(
      _IN_PATHS.value,
      _OUT_PATH.value,
      _PER_HOST_MEMORY_LIMIT_BYTES.value,
  )
