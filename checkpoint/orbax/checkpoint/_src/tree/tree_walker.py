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

"""TreeWalker class for PyTree structure parsing, argument alignment, and matching."""

import dataclasses
from typing import Any, Optional, Tuple
from absl import logging
from etils import epath
import jax
from orbax.checkpoint import utils
from orbax.checkpoint._src.engine import async_io_engine
from orbax.checkpoint._src.metadata import empty_values
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import limits
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.serialization import type_handler_registry as type_handler_registry_lib
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.serialization import types
from orbax.checkpoint._src.tree import structure_utils as tree_structure_utils
from orbax.checkpoint._src.tree import utils as tree_utils

PyTree = Any
ParamInfo = types.ParamInfo
TypeHandlerRegistry = types.TypeHandlerRegistry
PLACEHOLDER = type_handlers.PLACEHOLDER
BatchRequest = async_io_engine.BatchRequest


class PartialSaveError(Exception):
  """Raised when there is an error during partial saving."""


class PartialSaveReplacementError(PartialSaveError):
  """Raised when a replacement is attempted during partial saving."""


def filter_partial_save_requests(
    item: PyTree,
    value_metadata_tree: PyTree,
    batch_requests: list[BatchRequest],
) -> list[BatchRequest]:
  """Filters BatchRequests to only include additions compared to on-disk metadata."""
  tree_diff = tree_structure_utils.tree_difference(item, value_metadata_tree)

  additions = set()

  def _handle_diffs(keypath, diff):
    keypath = tree_utils.tuple_path_from_keypath(keypath)
    if diff.lhs is not None:  # Leaf is present in the current item
      if diff.rhs is None:  # Leaf was not in the on-disk metadata
        additions.add(keypath)
      else:  # Leaf was also in the on-disk metadata
        raise PartialSaveReplacementError(
            f'Key "{keypath}" was found in the on-disk PyTree metadata and'
            ' supplied item. Partial saving currently does not support'
            ' REPLACEMENT. Please reach out to the Orbax team if you need'
            ' this feature.'
        )

  jax.tree.map_with_path(
      _handle_diffs,
      tree_diff,
      is_leaf=lambda x: isinstance(x, tree_structure_utils.Diff),
  )

  logging.info(
      '[process=%d] Found the following additions during partial save: %s',
      multihost.process_index(),
      additions,
  )

  # Filter out requests that don't have any additions.
  filtered_requests = []
  for request in batch_requests:
    filtered_items = []
    for key, value, info, arg in zip(
        request.keys, request.values, request.infos, request.args
    ):
      for add in additions:
        # Additions may be a prefix/parent of the key.
        if add == key[: len(add)]:
          filtered_items.append((key, value, info, arg))
    if filtered_items:
      keys, values, infos, args = zip(*filtered_items)
      filtered_requests.append(
          dataclasses.replace(
              request,
              keys=list(keys),
              values=list(values),
              infos=list(infos),
              args=list(args),
          )
      )

  return filtered_requests


def align_structures_for_omission(
    item: PyTree,
    serialized_item: PyTree,
    value_metadata_tree: PyTree,
    restore_args: PyTree,
    *,
    support_rich_types: bool,
) -> Tuple[PyTree, PyTree]:
  """Aligns structures specified in `item`. Skips omitted leaves."""
  if not support_rich_types:
    # Replace empty containers with scalar values (zeros). During saving,
    # some empty containers (like named tuples) were given
    # ValueMetadataEntries as if they were scalars. We normalize these
    # containers to scalars so that tree_trim is none the wiser.
    serialized_item = jax.tree.map(
        lambda v: 0 if empty_values.is_empty_container(v) else v,
        serialized_item,
        is_leaf=tree_utils.is_empty_or_leaf,
    )

  value_metadata_tree = tree_structure_utils.tree_trim(
      serialized_item, value_metadata_tree, strict=False
  )
  value_metadata_tree = value_metadata_tree.unsafe_structure

  if restore_args is not None:
    restore_args = tree_structure_utils.tree_trim(
        item, restore_args, strict=True
    )

  return value_metadata_tree, restore_args


def align_structures_for_placeholders(
    serialized_item: PyTree, value_metadata_tree: PyTree
) -> PyTree:
  """Aligns leaves from `item`, populating placeholders where appropriate."""
  diff = (
      tree_structure_utils.tree_difference(
          serialized_item,
          value_metadata_tree,
          is_leaf=tree_utils.is_empty_or_leaf,
          leaves_equal=lambda a, b: True,
      )
      or {}
  )
  for keypath, value_diff in tree_utils.to_flat_dict(
      diff, is_leaf=lambda x: isinstance(x, tree_structure_utils.Diff)
  ).items():
    if value_diff.lhs is PLACEHOLDER and value_diff.rhs is None:
      parent = value_metadata_tree
      for key in keypath[:-1]:
        parent = parent[key]
      parent[keypath[-1]] = PLACEHOLDER
    else:
      formatted_diff = tree_structure_utils.format_tree_diff(
          diff, source_label='Item', target_label='Metadata'
      )
      raise ValueError(
          'User-provided restore item and on-disk value metadata tree'
          f' structures do not match:\n{formatted_diff}\nIf this mismatch is'
          ' intentional, pass `partial_restore=True` to only restore'
          ' parameters found in `item`.'
      )
  return jax.tree.map(
      lambda v, i: PLACEHOLDER if type_handlers.is_placeholder(i) else v,
      value_metadata_tree,
      serialized_item,
  )


def get_param_infos(
    item: PyTree,
    directory: epath.Path,
    *,
    use_zarr3: bool,
    enable_pinned_host_transfer: bool,
    type_handler_registry: TypeHandlerRegistry,
    pytree_metadata_options: tree_metadata.PyTreeMetadataOptions,
    is_prioritized_key_fn: Optional[types.IsPrioritizedKeyFn] = None,
    use_compression: Optional[bool] = True,
    use_ocdbt: bool = True,
    ocdbt_target_data_file_size: Optional[int] = None,
    device_host_byte_limiter: Optional[limits.ByteLimiter] = None,
    raise_array_data_missing_error: bool = True,
    concurrent_bytes: Optional[int] = None,
) -> PyTree:
  """Returns parameter information for elements in `item`."""
  names = tree_utils.get_param_names(item)
  ts_context = ts_utils.get_ts_context(use_ocdbt=use_ocdbt)

  def _param_info(keypath, name, value):
    if isinstance(value, tree_metadata.ValueMetadataEntry):
      skip_deserialize = value.skip_deserialize
    elif isinstance(value, type(PLACEHOLDER)):
      skip_deserialize = True
    else:
      skip_deserialize = False
    return ParamInfo(
        name=name,
        keypath=keypath,
        parent_dir=directory,
        skip_deserialize=skip_deserialize,
        is_ocdbt_checkpoint=use_ocdbt,
        use_compression=use_compression,
        use_zarr3=use_zarr3,
        enable_pinned_host_transfer=enable_pinned_host_transfer,
        ocdbt_target_data_file_size=ocdbt_target_data_file_size,
        byte_limiter=limits.get_byte_limiter(concurrent_bytes),
        device_host_byte_limiter=device_host_byte_limiter,
        ts_context=ts_context,
        value_typestr=type_handler_registry_lib.get_param_typestr(
            value, type_handler_registry, pytree_metadata_options
        ),
        raise_array_data_missing_error=raise_array_data_missing_error,
        is_prioritized_key_fn=is_prioritized_key_fn,
    )

  return jax.tree.map_with_path(
      _param_info, names, item, is_leaf=utils.is_empty_or_leaf
  )


def validate_and_align_restore_structures(
    *,
    item: PyTree,
    serialized_item: PyTree,
    value_metadata_tree: PyTree,
    restore_args: PyTree,
    partial_restore: bool,
    support_rich_types: bool,
) -> Tuple[PyTree, PyTree, PyTree]:
  """Validates structures match and aligns them for omission or placeholders.

  Args:
    item: User-provided target structure, or None.
    serialized_item: Serialized version of `item`.
    value_metadata_tree: On-disk value metadata tree structure.
    restore_args: User-provided restore args structure.
    partial_restore: If True, skips omitted parameters.
    support_rich_types: If True, enables rich type support.

  Returns:
    Tuple of (aligned_item, aligned_value_metadata_tree, aligned_restore_args).
  """
  if item is None:
    item = value_metadata_tree
  elif partial_restore:
    value_metadata_tree, restore_args = align_structures_for_omission(
        item,
        serialized_item,
        value_metadata_tree,
        restore_args,
        support_rich_types=support_rich_types,
    )
  elif any(
      type_handlers.is_placeholder(leaf) for leaf in jax.tree.leaves(item)
  ):
    value_metadata_tree = align_structures_for_placeholders(
        serialized_item, value_metadata_tree
    )
  else:
    # Deserialize value metadata tree to the same structure as item to allow
    # for comparison with item that contains rich types.
    if support_rich_types:
      value_metadata_tree = tree_utils.deserialize_tree(
          value_metadata_tree, item
      )
    # is_empty_or_leaf is necessary here to treat empty nodes (e.g. empty
    # dicts, lists, custom nodes) as leaves, as they do not contain any
    # actual data to be restored, but are needed to maintain the structure.
    diff = tree_structure_utils.tree_difference(
        serialized_item,
        value_metadata_tree,
        is_leaf=tree_utils.is_empty_or_leaf,
        leaves_equal=lambda a, b: True,
    )
    if diff is not None:
      formatted_diff = tree_structure_utils.format_tree_diff(
          diff, source_label='Item', target_label='Metadata'
      )
      raise ValueError(
          'User-provided restore item and on-disk value metadata tree'
          f' structures do not match:\n{formatted_diff}\nIf this mismatch is'
          ' intentional, pass `partial_restore=True` to only restore'
          ' parameters found in `item`.'
      )
  return item, value_metadata_tree, restore_args
