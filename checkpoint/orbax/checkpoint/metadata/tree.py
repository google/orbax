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

"""Utilities for working with Orbax metadata."""

import dataclasses
import enum
import functools
import operator
from typing import Any, Dict, Hashable, List, Optional, Tuple, TypeVar, Union
import jax
from orbax.checkpoint import tree as tree_utils
from orbax.checkpoint import type_handlers
from orbax.checkpoint import utils


_KEY_NAME = 'key'
_KEY_TYPE = 'key_type'
_VALUE_TYPE = 'value_type'
_SKIP_DESERIALIZE = 'skip_deserialize'

_TREE_METADATA_KEY = 'tree_metadata'
_KEY_METADATA_KEY = 'key_metadata'
_VALUE_METADATA_KEY = 'value_metadata'
_USE_ZARR3 = 'use_zarr3'

PyTree = Any
TupleKeyPathStr = Tuple[str, ...]
KeyEntry = TypeVar('KeyEntry', bound=Hashable)
KeyPath = tuple[KeyEntry, ...]


class _KeyType(enum.Enum):
  """Enum representing PyTree key type."""

  SEQUENCE = 1
  DICT = 2

  def to_json(self) -> int:
    return self.value

  @classmethod
  def from_json(cls, value: int) -> '_KeyType':
    return cls(value)


def _get_key_metadata_type(key: Any) -> _KeyType:
  """Translates the JAX key class into a proto enum."""
  if tree_utils.is_sequence_key(key):
    return _KeyType.SEQUENCE
  elif tree_utils.is_dict_key(key):
    return _KeyType.DICT
  else:
    raise ValueError(f'Unsupported KeyEntry: {type(key)}: "{key}"')


def _keypath_from_key_type(key_name: str, key_type: _KeyType) -> Any:
  """Converts from Key in TreeMetadata to JAX keypath class."""
  if key_type == _KeyType.SEQUENCE:
    return jax.tree_util.SequenceKey(int(key_name))
  elif key_type == _KeyType.DICT:
    return jax.tree_util.DictKey(key_name)
  else:
    raise ValueError(f'Unsupported KeyEntry: {key_type}')


@dataclasses.dataclass
class NestedKeyMetadataEntry:
  """Represents a key at a single level of nesting."""
  nested_key_name: str
  key_type: _KeyType

  def to_json(self) -> Dict[str, Union[str, int]]:
    return {
        _KEY_NAME: self.nested_key_name,
        _KEY_TYPE: self.key_type.to_json(),
    }

  @classmethod
  def from_json(
      cls, json_dict: Dict[str, Union[str, int]]
  ) -> 'NestedKeyMetadataEntry':
    return NestedKeyMetadataEntry(
        nested_key_name=json_dict[_KEY_NAME],
        key_type=_KeyType.from_json(json_dict[_KEY_TYPE]),
    )


@dataclasses.dataclass
class KeyMetadataEntry:
  """Represents metadata for a key (all levels of nesting)."""
  nested_key_metadata_entries: List[NestedKeyMetadataEntry]

  def to_json(self) -> Tuple[Dict[str, Union[str, int]], ...]:
    return tuple(
        [entry.to_json() for entry in self.nested_key_metadata_entries]
    )

  @classmethod
  def from_json(
      cls, json_dict: Tuple[Dict[str, Union[str, int]], ...]
  ) -> 'KeyMetadataEntry':
    return KeyMetadataEntry(
        [NestedKeyMetadataEntry.from_json(entry) for entry in json_dict]
    )

  @classmethod
  def build(cls, keypath: KeyPath) -> 'KeyMetadataEntry':
    return KeyMetadataEntry([
        NestedKeyMetadataEntry(
            str(tree_utils.get_key_name(k)), _get_key_metadata_type(k)
        )
        for k in keypath
    ])


@dataclasses.dataclass
class ValueMetadataEntry:
  """Represents metadata for a leaf in a tree."""
  value_type: str
  skip_deserialize: bool = False

  def to_json(self) -> Dict[str, Any]:
    return {
        _VALUE_TYPE: self.value_type,
        _SKIP_DESERIALIZE: self.skip_deserialize,
    }

  @classmethod
  def from_json(cls, json_dict: Dict[str, Any]) -> 'ValueMetadataEntry':
    return ValueMetadataEntry(
        value_type=json_dict[_VALUE_TYPE],
        skip_deserialize=json_dict[_SKIP_DESERIALIZE],
    )

  @classmethod
  def build(
      cls,
      value: Any,
      save_arg: type_handlers.SaveArgs,
      registry: type_handlers.TypeHandlerRegistry,
  ) -> 'ValueMetadataEntry':
    """Builds a ValueMetadataEntry."""
    if utils.is_supported_empty_aggregation_type(value):
      typestr = type_handlers.get_empty_value_typestr(value)
      skip_deserialize = True
    else:
      try:
        handler = registry.get(type(value))
        typestr = handler.typestr()
        skip_deserialize = save_arg.aggregate
      except ValueError:
        # Not an error because users' training states often have a bunch of
        # random unserializable objects in them (empty states, optimizer
        # objects, etc.). An error occurring due to a missing TypeHandler
        # will be surfaced elsewhere.
        typestr = type_handlers.RESTORE_TYPE_NONE
        skip_deserialize = True
    return ValueMetadataEntry(
        value_type=typestr, skip_deserialize=skip_deserialize
    )


@dataclasses.dataclass
class TreeMetadataEntry:
  """Represents metadata for a named key/value pair in a tree."""
  keypath: str
  key_metadata: KeyMetadataEntry
  value_metadata: ValueMetadataEntry

  def to_json(self) -> Dict[str, Any]:
    return {
        self.keypath: {
            _KEY_METADATA_KEY: self.key_metadata.to_json(),
            _VALUE_METADATA_KEY: self.value_metadata.to_json(),
        }
    }

  @classmethod
  def from_json(
      cls, keypath: str, json_dict: Dict[str, Any]
  ) -> 'TreeMetadataEntry':
    return TreeMetadataEntry(
        keypath,
        KeyMetadataEntry.from_json(json_dict[_KEY_METADATA_KEY]),
        ValueMetadataEntry.from_json(json_dict[_VALUE_METADATA_KEY]),
    )

  @classmethod
  def build(
      cls,
      keypath: KeyPath,
      value: Any,
      save_arg: type_handlers.SaveArgs,
      type_handler_registry: type_handlers.TypeHandlerRegistry,
  ) -> 'TreeMetadataEntry':
    """Builds a TreeMetadataEntry."""
    key_metadata = KeyMetadataEntry.build(keypath)
    value_metadata = ValueMetadataEntry.build(
        value, save_arg, type_handler_registry
    )
    return TreeMetadataEntry(
        str(tuple([str(tree_utils.get_key_name(k)) for k in keypath])),
        key_metadata,
        value_metadata,
    )

  def jax_keypath(self) -> KeyPath:
    keypath = []
    for nested_key_entry in self.key_metadata.nested_key_metadata_entries:
      nested_key_name = nested_key_entry.nested_key_name
      key_type = nested_key_entry.key_type
      keypath.append(_keypath_from_key_type(nested_key_name, key_type))
    return tuple(keypath)


@dataclasses.dataclass
class TreeMetadata:
  """Metadata representation of a PyTree."""

  tree_metadata_entries: List[TreeMetadataEntry]
  use_zarr3: bool

  @classmethod
  def build(
      cls,
      tree: PyTree,
      *,
      type_handler_registry: type_handlers.TypeHandlerRegistry,
      save_args: Optional[PyTree] = None,
      use_zarr3: bool = False,
  ) -> 'TreeMetadata':
    """Builds the tree metadata."""
    if save_args is None:
      save_args = jax.tree.map(
          lambda _: type_handlers.SaveArgs(),
          tree,
          is_leaf=tree_utils.is_empty_or_leaf,
      )
    flat_with_keys, _ = jax.tree_util.tree_flatten_with_path(
        tree, is_leaf=tree_utils.is_empty_or_leaf
    )
    flat_save_args_with_keys, _ = jax.tree_util.tree_flatten_with_path(
        save_args, is_leaf=tree_utils.is_empty_or_leaf
    )
    tree_metadata_entries = []
    for (keypath, value), (_, save_arg) in zip(
        flat_with_keys, flat_save_args_with_keys
    ):
      tree_metadata_entries.append(
          TreeMetadataEntry.build(
              keypath, value, save_arg, type_handler_registry
          )
      )
    return TreeMetadata(tree_metadata_entries, use_zarr3)

  def to_json(self) -> Dict[str, Any]:
    """Returns a JSON representation of the metadata.

    Uses JSON format::
      {
          _TREE_METADATA_KEY: {
            "(top_level_key, lower_level_key)": {
                _KEY_METADATA_KEY: (
                    {_KEY_NAME: "top_level_key", _KEY_TYPE: <_KeyType (int)>},
                    {_KEY_NAME: "lower_level_key", _KEY_TYPE: <_KeyType (int)>},
                )
                _VALUE_METADATA_KEY: {
                    _VALUE_TYPE: "jax.Array",
                    _SKIP_DESERIALIZE: True/False,
                }
            }
            ...
        }
      }
    """
    return {
        _TREE_METADATA_KEY: functools.reduce(
            operator.ior,
            [entry.to_json() for entry in self.tree_metadata_entries],
            {},
        ),
        _USE_ZARR3: self.use_zarr3,
    }

  @classmethod
  def from_json(cls, json_dict: Dict[str, Any]) -> 'TreeMetadata':
    """Convert the TreeMetadata from a JSON representation."""
    use_zarr3 = False
    if _USE_ZARR3 in json_dict:
      use_zarr3 = json_dict[_USE_ZARR3]

    tree_metadata_entries = []
    for keypath, json_tree_metadata_entry in json_dict[
        _TREE_METADATA_KEY
    ].items():
      tree_metadata_entries.append(
          TreeMetadataEntry.from_json(keypath, json_tree_metadata_entry)
      )
    return TreeMetadata(
        tree_metadata_entries,
        use_zarr3=use_zarr3,
    )

  def as_nested_tree(self, *, keep_empty_nodes: bool) -> Dict[str, Any]:
    """Converts to a nested tree, with values of ValueMetadataEntry."""

    def _maybe_as_empty_value(value_metadata: ValueMetadataEntry) -> Any:
      if not keep_empty_nodes and type_handlers.is_empty_typestr(
          value_metadata.value_type
      ):
        # Return node as the empty value itself rather than as
        # a dataclass representation.
        return type_handlers.get_empty_value_from_typestr(
            value_metadata.value_type
        )
      return value_metadata

    return tree_utils.from_flattened_with_keypath([
        (entry.jax_keypath(), _maybe_as_empty_value(entry.value_metadata))
        for entry in self.tree_metadata_entries
    ])
