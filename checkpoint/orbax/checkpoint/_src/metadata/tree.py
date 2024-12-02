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

"""Main module for PyTree checkpoint metadata storage."""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import enum
import functools
import operator
from typing import Any, Dict, Hashable, List, Optional, Tuple, TypeAlias, TypeVar, Union

from absl import logging
from etils import epath
import jax
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.metadata import empty_values
from orbax.checkpoint._src.metadata import pytree_metadata_options as pytree_metadata_options_lib
from orbax.checkpoint._src.metadata import tree_rich_types
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.metadata import value_metadata_entry
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.serialization import types
from orbax.checkpoint._src.tree import utils as tree_utils

ValueMetadataEntry: TypeAlias = value_metadata_entry.ValueMetadataEntry
PyTreeMetadataOptions: TypeAlias = (
    pytree_metadata_options_lib.PyTreeMetadataOptions
)
PyTree: TypeAlias = Any
KeyEntry = TypeVar('KeyEntry', bound=Hashable)
KeyPath: TypeAlias = tuple[KeyEntry, ...]

PYTREE_METADATA_OPTIONS = pytree_metadata_options_lib.PYTREE_METADATA_OPTIONS

_KEY_NAME = 'key'
_KEY_TYPE = 'key_type'
_TREE_METADATA_KEY = 'tree_metadata'
_KEY_METADATA_KEY = 'key_metadata'
_VALUE_METADATA_KEY = 'value_metadata'
_USE_ZARR3 = 'use_zarr3'
_STORE_ARRAY_DATA_EQUAL_TO_FILL_VALUE = 'store_array_data_equal_to_fill_value'
_VALUE_METADATA_TREE = 'value_metadata_tree'


class KeyType(enum.Enum):
  """Enum representing PyTree key type."""

  SEQUENCE = 1
  DICT = 2

  def to_json(self) -> int:
    return self.value

  @classmethod
  def from_json(cls, value: int) -> KeyType:
    return cls(value)


def _get_key_metadata_type(key: Any) -> KeyType:
  """Translates the JAX key class into a proto enum."""
  if tree_utils.is_sequence_key(key):
    return KeyType.SEQUENCE
  elif tree_utils.is_dict_key(key):
    return KeyType.DICT
  else:
    raise ValueError(f'Unsupported KeyEntry: {type(key)}: "{key}"')


def _keypath_from_key_type(key_name: str, key_type: KeyType) -> Any:
  """Converts from Key in InternalTreeMetadata to JAX keypath class."""
  if key_type == KeyType.SEQUENCE:
    return jax.tree_util.SequenceKey(int(key_name))
  elif key_type == KeyType.DICT:
    return jax.tree_util.DictKey(key_name)
  else:
    raise ValueError(f'Unsupported KeyEntry: {key_type}')


@dataclasses.dataclass
class NestedKeyMetadataEntry:
  """Represents a key at a single level of nesting."""

  nested_key_name: str
  key_type: KeyType

  def to_json(self) -> Dict[str, Union[str, int]]:
    return {
        _KEY_NAME: self.nested_key_name,
        _KEY_TYPE: self.key_type.to_json(),
    }

  @classmethod
  def from_json(
      cls, json_dict: Dict[str, Union[str, int]]
  ) -> NestedKeyMetadataEntry:
    return NestedKeyMetadataEntry(
        nested_key_name=json_dict[_KEY_NAME],
        key_type=KeyType.from_json(json_dict[_KEY_TYPE]),
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
  ) -> KeyMetadataEntry:
    return KeyMetadataEntry(
        [NestedKeyMetadataEntry.from_json(entry) for entry in json_dict]
    )

  @classmethod
  def build(cls, keypath: KeyPath) -> KeyMetadataEntry:
    return KeyMetadataEntry([
        NestedKeyMetadataEntry(
            str(tree_utils.get_key_name(k)), _get_key_metadata_type(k)
        )
        for k in keypath
    ])


@dataclasses.dataclass
class InternalTreeMetadataEntry:
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
  ) -> InternalTreeMetadataEntry:
    return InternalTreeMetadataEntry(
        keypath,
        KeyMetadataEntry.from_json(json_dict[_KEY_METADATA_KEY]),
        ValueMetadataEntry.from_json(
            json_dict[_VALUE_METADATA_KEY],
            pytree_metadata_options=PyTreeMetadataOptions(
                support_rich_types=False  # Always in legacy mode.
            ),
        ),
    )

  @classmethod
  def build(
      cls,
      keypath: KeyPath,
      info: types.ParamInfo,
      save_arg: types.SaveArgs,
  ) -> InternalTreeMetadataEntry:
    """Builds a InternalTreeMetadataEntry."""
    return InternalTreeMetadataEntry(
        keypath=str(tuple([str(tree_utils.get_key_name(k)) for k in keypath])),
        key_metadata=KeyMetadataEntry.build(keypath),
        value_metadata=ValueMetadataEntry.build(info, save_arg),
    )

  def jax_keypath(self) -> KeyPath:
    keypath = []
    for nested_key_entry in self.key_metadata.nested_key_metadata_entries:
      nested_key_name = nested_key_entry.nested_key_name
      key_type = nested_key_entry.key_type
      keypath.append(_keypath_from_key_type(nested_key_name, key_type))
    return tuple(keypath)


@dataclasses.dataclass
class InternalTreeMetadata:
  """Metadata representation of a PyTree."""

  tree_metadata_entries: List[InternalTreeMetadataEntry]
  use_zarr3: bool
  store_array_data_equal_to_fill_value: bool
  pytree_metadata_options: PyTreeMetadataOptions
  value_metadata_tree: PyTree | None = None

  def __post_init__(self):
    logging.vlog(
        1,
        'Created InternalTreeMetadata with pytree_metadata_options=%s, has'
        ' tree_metadata_entries=%s, has rich typed value_metadata_tree=%s',
        self.pytree_metadata_options,
        len(self.tree_metadata_entries),
        self.value_metadata_tree is not None,
    )

  @classmethod
  def build(
      cls,
      param_infos: PyTree,
      *,
      save_args: Optional[PyTree] = None,
      use_zarr3: bool = False,
      pytree_metadata_options: PyTreeMetadataOptions = (
          PYTREE_METADATA_OPTIONS
      ),
  ) -> InternalTreeMetadata:
    """Returns an InternalTreeMetadata instance."""
    if save_args is None:
      save_args = jax.tree.map(
          lambda _: types.SaveArgs(),
          param_infos,
          is_leaf=tree_utils.is_empty_or_leaf,
      )
    flat_info_with_keys, _ = jax.tree_util.tree_flatten_with_path(
        param_infos, is_leaf=tree_utils.is_empty_or_leaf
    )
    flat_save_args_with_keys, _ = jax.tree_util.tree_flatten_with_path(
        save_args, is_leaf=tree_utils.is_empty_or_leaf
    )
    tree_metadata_entries = []
    for (keypath, info), (_, save_arg) in zip(
        flat_info_with_keys, flat_save_args_with_keys
    ):
      tree_metadata_entries.append(
          InternalTreeMetadataEntry.build(keypath, info, save_arg)
      )
    value_metadata_tree = None
    if pytree_metadata_options.support_rich_types:
      value_metadata_tree = jax.tree_util.tree_map(
          ValueMetadataEntry.build,
          param_infos,
          save_args,
          is_leaf=tree_utils.is_empty_or_leaf,
      )
      logging.vlog(
          1,
          'Created rich typed value_metadata_tree from param_infos and'
          ' save_args.',
      )
    return InternalTreeMetadata(
        tree_metadata_entries=tree_metadata_entries,
        use_zarr3=use_zarr3,
        store_array_data_equal_to_fill_value=ts_utils.STORE_ARRAY_DATA_EQUAL_TO_FILL_VALUE,
        pytree_metadata_options=pytree_metadata_options,
        value_metadata_tree=value_metadata_tree,
    )

  def to_json(self) -> Dict[str, Any]:
    """Returns a JSON representation of the metadata.

    Uses JSON format::
    ```
      {
          _TREE_METADATA_KEY: {
            "(top_level_key, lower_level_key)": {
                _KEY_METADATA_KEY: (
                    {_KEY_NAME: "top_level_key", _KEY_TYPE: <KeyType
                    (int)>},
                    {_KEY_NAME: "lower_level_key", _KEY_TYPE: <KeyType
                    (int)>},
                )
                _VALUE_METADATA_KEY: {
                    _VALUE_TYPE: "jax.Array",
                    _SKIP_DESERIALIZE: True/False,
                }
            }
            ...
          },
          _USE_ZARR3: True/False,
          _STORE_ARRAY_DATA_EQUAL_TO_FILL_VALUE: True,
          _VALUE_METADATA_TREE: '{
            "mu_nu": {
              "category": "namedtuple",
              "module": "orbax.checkpoint._src.testing.test_tree_utils",
              "clazz": "MuNu",
              "entries": [
                {
                  "key": "mu",
                  "value": {
                    "category": "custom",
                    "clazz": "ValueMetadataEntry",
                    "data": {
                      "value_type": "jax.Array",
                      "skip_deserialize": false
                    }
                  }
                },
                {
                  "key": "nu",
                  "value": {
                    "category": "custom",
                    "clazz": "ValueMetadataEntry",
                    "data": {
                      "value_type": "np.ndarray",
                      "skip_deserialize": false
                    }
                  }
                }
              ]
            },
            "my_tuple": {
              "category": "custom",
              "clazz": "tuple",
              "entries": [
                  {
                    "category": "custom",
                    "clazz": "ValueMetadataEntry",
                    "data": {
                      "value_type": "np.ndarray",
                      "skip_deserialize": false
                    }
                  }
              ]
            }
          }'
      }
    ```
    """
    json_object = {
        _TREE_METADATA_KEY: functools.reduce(
            operator.ior,
            [entry.to_json() for entry in self.tree_metadata_entries],
            {},
        ),
        _USE_ZARR3: self.use_zarr3,
        _STORE_ARRAY_DATA_EQUAL_TO_FILL_VALUE: (
            self.store_array_data_equal_to_fill_value
        ),
    }
    # TODO: b/365169723 - Support versioned evolution of metadata storage.
    if (
        self.pytree_metadata_options.support_rich_types
        and self.value_metadata_tree is not None
    ):
      json_object[_VALUE_METADATA_TREE] = (
          tree_rich_types.value_metadata_tree_to_json_str(
              self.value_metadata_tree
          )
      )
      logging.vlog(
          1, 'Serialized rich typed value_metadata_tree to json_object.'
      )
    return json_object

  @classmethod
  def from_json(
      cls,
      json_dict: Dict[str, Any],
      pytree_metadata_options: PyTreeMetadataOptions = (
          PYTREE_METADATA_OPTIONS
      ),
  ) -> InternalTreeMetadata:
    """Returns an InternalTreeMetadata instance from its JSON representation."""
    use_zarr3 = json_dict.get(_USE_ZARR3, False)
    store_array_data_equal_to_fill_value = json_dict.get(
        _STORE_ARRAY_DATA_EQUAL_TO_FILL_VALUE, False
    )

    tree_metadata_entries = []
    for keypath, json_tree_metadata_entry in json_dict[
        _TREE_METADATA_KEY
    ].items():
      tree_metadata_entries.append(
          InternalTreeMetadataEntry.from_json(keypath, json_tree_metadata_entry)
      )
    # TODO: b/365169723 - Support versioned evolution of metadata storage.
    value_metadata_tree = None
    if (
        pytree_metadata_options.support_rich_types
        and _VALUE_METADATA_TREE in json_dict
    ):
      value_metadata_tree = tree_rich_types.value_metadata_tree_from_json_str(
          json_dict[_VALUE_METADATA_TREE]
      )
      logging.info(
          'Deserialized rich typed value_metadata_tree from json_dict.'
      )
    return InternalTreeMetadata(
        tree_metadata_entries=tree_metadata_entries,
        use_zarr3=use_zarr3,
        pytree_metadata_options=pytree_metadata_options,
        value_metadata_tree=value_metadata_tree,
        store_array_data_equal_to_fill_value=store_array_data_equal_to_fill_value,
    )

  def as_nested_tree(self) -> Dict[str, Any]:
    """Converts to a nested tree, with leaves of ValueMetadataEntry."""
    # TODO: b/365169723 - Support versioned evolution of metadata storage.
    if (
        self.pytree_metadata_options.support_rich_types
        and self.value_metadata_tree is not None
    ):
      logging.info(
          'Returning rich typed value_metadata_tree from InternalTreeMetadata.'
      )
      return self.value_metadata_tree

    return tree_utils.from_flattened_with_keypath([
        (entry.jax_keypath(), entry.value_metadata)
        for entry in self.tree_metadata_entries
    ])

  def as_user_metadata(
      self,
      directory: epath.Path,
      type_handler_registry: types.TypeHandlerRegistry,
      *,
      use_ocdbt: bool = True,
  ) -> PyTree:
    """Returns a user-facing PyTree with leaves of `ValueMetadataEntry`.

    The returned PyTree conforms to the structure of `self` InternalTreeMetadata
    but the `ValueMetadataEntry` leaves are derived from the checkpoints in
    `directory`.

    Args:
      directory: The directory to read the checkpoint from.
      type_handler_registry: `TypeHandlerRegistry` whose registered
        TypeHandlers' metadata are used to build the `ValueMetadataEntry`
        leaves. See `TypeHandler.metadata()`.
      use_ocdbt: Whether to use OCDBT for reading the metadata from tensorstore.
    """
    flat_param_infos = {}
    flat_restore_types = {}
    reference_metadata_tree = self.as_nested_tree()
    ts_context = ts_utils.get_ts_context(use_ocdbt=use_ocdbt)
    for keypath, value_meta in tree_utils.to_flat_dict(
        reference_metadata_tree
    ).items():
      param_name = '.'.join(keypath)
      flat_param_infos[keypath] = types.ParamInfo(
          name=param_name,
          path=directory / param_name,
          parent_dir=directory,
          skip_deserialize=value_meta.skip_deserialize,
          is_ocdbt_checkpoint=use_ocdbt,
          use_zarr3=self.use_zarr3,
          ts_context=ts_context,
      )
      flat_restore_types[keypath] = value_meta.value_type

    flat_metadatas = {}
    batched_param_infos = collections.defaultdict(list)
    batched_keypaths = collections.defaultdict(list)
    for keypath in flat_param_infos:
      param_info = flat_param_infos[keypath]
      restore_type = flat_restore_types[keypath]
      if param_info.skip_deserialize:
        if empty_values.is_empty_typestr(restore_type):
          flat_metadatas[keypath] = empty_values.get_empty_value_from_typestr(
              restore_type, self.pytree_metadata_options
          )
        else:
          flat_metadatas[keypath] = value_metadata.Metadata(
              name=param_info.name, directory=param_info.parent_dir
          )
      else:
        batched_keypaths[restore_type].append(keypath)
        batched_param_infos[restore_type].append(param_info)

    metadata_ops = []
    for restore_type, param_infos in batched_param_infos.items():
      handler = type_handler_registry.get(restore_type)
      metadata_ops.append(handler.metadata(param_infos))

    async def _get_metadata():
      return await asyncio.gather(*metadata_ops)

    batched_metadatas = asyncio_utils.run_sync(_get_metadata())
    for keypath_batch, metadata_batch in zip(
        batched_keypaths.values(), batched_metadatas
    ):
      for keypath, value in zip(keypath_batch, metadata_batch):
        flat_metadatas[keypath] = value
    return tree_utils.from_flat_dict(
        flat_metadatas, target=reference_metadata_tree
    )


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
