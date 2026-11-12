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

"""Resolution utilities bridging `CompressionPreset` and `CompressionSpec`.

Kept separate from `serialization/types.py` and `tensorstore_utils.py` so those
low-level modules do not depend on `orbax.checkpoint.options`.
"""

from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.serialization import types


def resolve_compression_preset(
    *,
    compression: options_lib.CompressionPreset | None = None,
    use_compression: bool | None = None,
) -> options_lib.CompressionPreset:
  """Resolves `compression` and deprecated `use_compression` into a preset.

  Args:
    compression: Explicit compression preset, if provided.
    use_compression: Deprecated boolean toggle (`True` -> `ZSTD`, `False` ->
      `NONE`).

  Returns:
    The resolved `CompressionPreset` (defaults to `CompressionPreset.ZSTD` when
    both arguments are `None`).

  Raises:
    ValueError: If both `use_compression` and `compression` are provided, or if
      `compression` is not a valid `CompressionPreset`.
  """
  if use_compression is not None and compression is not None:
    raise ValueError('Cannot specify both use_compression and compression.')
  if compression is not None:
    if not isinstance(compression, options_lib.CompressionPreset):
      raise ValueError(f'Unsupported compression preset: {compression!r}')
    return compression
  return (
      options_lib.CompressionPreset.ZSTD
      if (use_compression is None or use_compression)
      else options_lib.CompressionPreset.NONE
  )


def compression_spec_from_preset(
    preset: options_lib.CompressionPreset,
) -> types.CompressionSpec:
  """Returns the CompressionSpec corresponding to a CompressionPreset."""
  match preset:
    case options_lib.CompressionPreset.NONE:
      return types.NoCompression()
    case options_lib.CompressionPreset.ZSTD:
      return types.ZstdCompression()
    case _:
      raise ValueError(f'Unsupported compression preset: {preset!r}')
