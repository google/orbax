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

"""Common constructs for PyTree checkpoint metadata storage."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, TypeAlias

import jax
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.tree import utils as tree_utils

PyTree: TypeAlias = Any

_VALUE_TYPE = 'value_type'
_SKIP_DESERIALIZE = 'skip_deserialize'


@dataclasses.dataclass(kw_only=True)
class PyTreeMetadataOptions:
  """Options for managing PyTree metadata.

  Attributes:
    support_rich_types: [Experimental feature: subject to change without
      notice.] If True, supports NamedTuple and Tuple node types in the
      metadata. Otherwise, a NamedTuple node is converted to dict and Tuple node
      to list.
  """

  # TODO: b/365169723 - Support different namedtuple ser/deser strategies.

  support_rich_types: bool


# Global default options.
PYTREE_METADATA_OPTIONS = PyTreeMetadataOptions(support_rich_types=False)


def serialize_tree(
    tree: PyTree,
    pytree_metadata_options: PyTreeMetadataOptions,
) -> PyTree:
  """Transforms a PyTree to a serializable format.

  IMPORTANT: If `pytree_metadata_options.support_rich_types` is false, the
  returned tree replaces tuple container nodes with list nodes.

  IMPORTANT: If `pytree_metadata_options.support_rich_types` is false, the
  returned tree replaces NamedTuple container nodes with dict
  nodes.

  If `pytree_metadata_options.support_rich_types` is true, then the returned
  tree is the same as the input tree retaining empty nodes as leafs.

  Args:
    tree: The tree to serialize.
    pytree_metadata_options: `PyTreeMetadataOptions` for managing PyTree
      metadata.

  Returns:
    The serialized PyTree.
  """
  if pytree_metadata_options.support_rich_types:
    return jax.tree_util.tree_map(
        lambda x: x,
        tree,
        is_leaf=tree_utils.is_empty_or_leaf,
    )

  return tree_utils.serialize_tree(tree, keep_empty_nodes=True)


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
  def from_json(cls, json_dict: Dict[str, Any]) -> ValueMetadataEntry:
    return ValueMetadataEntry(
        value_type=json_dict[_VALUE_TYPE],
        skip_deserialize=json_dict[_SKIP_DESERIALIZE],
    )

  @classmethod
  def build(
      cls,
      info: type_handlers.ParamInfo,
      save_arg: type_handlers.SaveArgs,
  ) -> ValueMetadataEntry:
    """Builds a ValueMetadataEntry."""
    del save_arg
    if info.value_typestr is None:
      raise AssertionError(
          'Must set `value_typestr` in `ParamInfo` when saving.'
      )
    skip_deserialize = type_handlers.is_empty_typestr(info.value_typestr)
    return ValueMetadataEntry(
        value_type=info.value_typestr, skip_deserialize=skip_deserialize
    )
