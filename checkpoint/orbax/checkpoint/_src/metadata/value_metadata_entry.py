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

"""Leaf of a JSON metadata tree."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict

from orbax.checkpoint._src.arrays import types as arrays_types
from orbax.checkpoint._src.metadata import empty_values
from orbax.checkpoint._src.metadata import pytree_metadata_options as pytree_metadata_options_lib


_VALUE_TYPE = 'value_type'
_SKIP_DESERIALIZE = 'skip_deserialize'
_WRITE_SHAPE = 'write_shape'


@dataclasses.dataclass
class ValueMetadataEntry:
  """Represents metadata for a leaf in a tree.

  WARNING: Do not rename this class, as it is saved by its name in the metadata
  storage.

  IMPORTANT: Please make sure that changes in attributes are backwards
  compatible with existing mmetadata in storage.
  """

  value_type: str
  skip_deserialize: bool = False
  write_shape: arrays_types.Shape | None = None

  def to_json(self) -> Dict[str, Any]:
    json_dict = {
        _VALUE_TYPE: self.value_type,
        _SKIP_DESERIALIZE: self.skip_deserialize,
    }
    if self.write_shape is not None:
      # Convert to list because JSON does not support tuples.
      # Make sure to convert back to tuple in `from_json`.
      json_dict[_WRITE_SHAPE] = list(self.write_shape)  # pyrefly: ignore[unsupported-operation]
    return json_dict

  @classmethod
  def from_json(
      cls,
      json_dict: Dict[str, Any],
      pytree_metadata_options: pytree_metadata_options_lib.PyTreeMetadataOptions,
  ) -> ValueMetadataEntry:
    return ValueMetadataEntry(
        value_type=empty_values.override_empty_value_typestr(
            json_dict[_VALUE_TYPE],
            pytree_metadata_options,
        ),
        skip_deserialize=json_dict[_SKIP_DESERIALIZE],
        write_shape=(
            tuple(json_dict[_WRITE_SHAPE])
            if _WRITE_SHAPE in json_dict
            else None
        ),
    )

  @classmethod
  def build(
      cls,
      value_typestr: str,
      *,
      write_shape: arrays_types.Shape | None = None,
  ) -> ValueMetadataEntry:
    """Builds a ValueMetadataEntry for a leaf being saved.

    Args:
      value_typestr: The typestr of the leaf value, as returned by
        `type_handler_registry.get_param_typestr`.
      write_shape: The shape of the chunks written for the leaf, if any.

    Returns:
      A ValueMetadataEntry. `skip_deserialize` is derived from `value_typestr`.
    """
    return ValueMetadataEntry(
        value_type=value_typestr,
        skip_deserialize=empty_values.is_empty_typestr(value_typestr),
        write_shape=write_shape,
    )
