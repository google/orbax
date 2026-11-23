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

"""Utilities for validating and aligning PyTree structures during restore.

`BasePyTreeCheckpointHandler.restore` compares the user-provided `item` against
the on-disk value metadata tree. Depending on the restore mode, the two trees
must match exactly, may differ only at `PLACEHOLDER` leaves, or the metadata
tree is trimmed to the leaves present in `item` (`partial_restore=True`).
"""

from __future__ import annotations

from typing import Any, Tuple

import jax
from orbax.checkpoint._src.metadata import empty_values
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.tree import structure_utils as tree_structure_utils
from orbax.checkpoint._src.tree import utils as tree_utils

PyTree = Any
PLACEHOLDER = type_handlers.PLACEHOLDER


def _structure_mismatch_error(diff: PyTree) -> ValueError:
  formatted_diff = tree_structure_utils.format_tree_diff(
      diff, source_label='Item', target_label='Metadata'
  )
  return ValueError(
      'User-provided restore item and on-disk value metadata tree'
      f' structures do not match:\n{formatted_diff}\nIf this mismatch is'
      ' intentional, pass `partial_restore=True` to only restore'
      ' parameters found in `item`.'
  )


def _align_structures_for_omission(
    item: PyTree,
    serialized_item: PyTree,
    value_metadata_tree: PyTree,
    restore_args: PyTree,
    *,
    support_rich_types: bool,
) -> Tuple[PyTree, PyTree]:
  """Trims the metadata tree and `restore_args` to the leaves in `item`.

  Used for `partial_restore=True`: leaves omitted from `item` are not restored.

  Args:
    item: The user-provided restore target.
    serialized_item: `item` serialized with the handler's
      `PyTreeMetadataOptions`.
    value_metadata_tree: The on-disk value metadata tree.
    restore_args: The user-provided restore args, or None.
    support_rich_types: Whether the handler supports rich types.

  Returns:
    A tuple of (trimmed value metadata tree, trimmed restore args).
  """
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


def _align_structures_for_placeholders(
    serialized_item: PyTree, value_metadata_tree: PyTree
) -> PyTree:
  """Marks metadata leaves as `PLACEHOLDER` where `item` has placeholders.

  `item` may only differ from the metadata tree at `PLACEHOLDER` leaves that are
  absent from the metadata; any other mismatch is an error.

  Args:
    serialized_item: The serialized user-provided restore target.
    value_metadata_tree: The on-disk value metadata tree. May be modified in
      place.

  Returns:
    The value metadata tree with `PLACEHOLDER` at every leaf that is a
    placeholder in `serialized_item`.

  Raises:
    ValueError: If the structures differ other than at placeholders.
  """
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
      raise _structure_mismatch_error(diff)
  return jax.tree.map(
      lambda v, i: PLACEHOLDER if type_handlers.is_placeholder(i) else v,
      value_metadata_tree,
      serialized_item,
  )


def validate_and_align_restore_structures(
    item: PyTree | None,
    serialized_item: PyTree,
    value_metadata_tree: PyTree,
    restore_args: PyTree,
    *,
    partial_restore: bool,
    support_rich_types: bool,
) -> Tuple[PyTree, PyTree, PyTree]:
  """Validates that `item` matches the metadata tree and aligns them.

  Args:
    item: The user-provided restore target, or None to restore the tree as
      saved.
    serialized_item: `item` serialized with the handler's
      `PyTreeMetadataOptions`.
    value_metadata_tree: The on-disk value metadata tree.
    restore_args: The user-provided restore args, or None.
    partial_restore: Whether to skip leaves omitted from `item`.
    support_rich_types: Whether the handler supports rich types.

  Returns:
    A tuple of (item, value metadata tree, restore args) aligned for restore.

  Raises:
    ValueError: If `item` does not match the metadata tree for the requested
      restore mode.
  """
  if item is None:
    item = value_metadata_tree
  elif partial_restore:
    value_metadata_tree, restore_args = _align_structures_for_omission(
        item,
        serialized_item,
        value_metadata_tree,
        restore_args,
        support_rich_types=support_rich_types,
    )
  elif any(
      type_handlers.is_placeholder(leaf) for leaf in jax.tree.leaves(item)
  ):
    value_metadata_tree = _align_structures_for_placeholders(
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
      raise _structure_mismatch_error(diff)
  return item, value_metadata_tree, restore_args
