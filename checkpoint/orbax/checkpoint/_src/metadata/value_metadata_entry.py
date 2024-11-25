# Copyright 2024 The Orbax Authors.
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

from orbax.checkpoint._src.metadata import empty_values
from orbax.checkpoint._src.metadata import pytree_metadata_options as pytree_metadata_options_lib
from orbax.checkpoint._src.serialization import types


_VALUE_TYPE = 'value_type'
_SKIP_DESERIALIZE = 'skip_deserialize'


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

  def to_json(self) -> Dict[str, Any]:
    return {
        _VALUE_TYPE: self.value_type,
        _SKIP_DESERIALIZE: self.skip_deserialize,
    }

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
    )

  @classmethod
  def build(
      cls,
      info: types.ParamInfo,
      save_arg: types.SaveArgs,
  ) -> ValueMetadataEntry:
    """Builds a ValueMetadataEntry."""
    del save_arg
    if info.value_typestr is None:
      raise AssertionError(
          'Must set `value_typestr` in `ParamInfo` when saving.'
      )
    skip_deserialize = empty_values.is_empty_typestr(info.value_typestr)
    return ValueMetadataEntry(
        value_type=info.value_typestr, skip_deserialize=skip_deserialize
    )
