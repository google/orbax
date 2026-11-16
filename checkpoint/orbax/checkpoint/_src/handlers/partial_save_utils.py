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

"""Utilities for handling partial save logic and snapshot merging."""

import asyncio
import functools
import json
from typing import Any, Iterable, Mapping, Set, Tuple

from absl import logging
from etils import epath
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint._src.path import format_utils
from orbax.checkpoint._src.path.snapshot import snapshot
from orbax.checkpoint._src.tree import utils as tree_utils


class PartialSaveError(Exception):
  """Raised when there is an error during partial saving."""


class PartialSaveReplacementError(PartialSaveError):
  """Raised when a replacement is attempted during partial saving."""


def _is_prefix(t1: Tuple[Any, ...], t2: Tuple[Any, ...]) -> bool:
  """Checks if tuple t1 is a prefix of tuple t2."""
  return len(t1) < len(t2) and t2[: len(t1)] == t1


async def _read_pending_metadatas(
    directory: epath.Path,
    pytree_metadata_options: tree_metadata.PyTreeMetadataOptions,
) -> list[tree_metadata.InternalTreeMetadata]:
  """Reads and merges all _METADATA files from pending snapshot directories."""
  tmp_dir = directory.parent
  if not tmp_dir.name.endswith(atomicity_types.TMP_DIR_SUFFIX):
    raise ValueError(
        'Expected temporary directory name to end with '
        f'{atomicity_types.TMP_DIR_SUFFIX}, but got {tmp_dir.name}.'
        'Partial saving requires a TemporaryPath class that supports snapshots.'
    )
  base_name = tmp_dir.name[: -len(atomicity_types.TMP_DIR_SUFFIX)]
  partial_path = tmp_dir.parent / base_name

  # Glob for metadata files written by previous partial saves in this session.
  pending_dirs = await snapshot.list_pending_dirs(partial_path)
  pending_metadata_files = []
  for d in pending_dirs:
    meta_file = d / directory.name / format_utils.PYTREE_METADATA_FILE
    if await async_path.exists(meta_file):
      pending_metadata_files.append(meta_file)

  async def get_tree_metadata(meta_file: epath.Path):
    return tree_metadata.InternalTreeMetadata.from_json(
        json.loads(await async_path.read_text(meta_file)),
        pytree_metadata_options=pytree_metadata_options,
    )

  return await asyncio.gather(
      *[get_tree_metadata(meta_file) for meta_file in pending_metadata_files]
  )


async def _merge_pending_metadatas(
    internal_metas: list[tree_metadata.InternalTreeMetadata],
) -> Set[Tuple[Any, ...]]:
  """Merges pending metadata from previous partial saves."""
  merge_trees = lambda a, b: a.merge(b, overwrite=True)
  merged_metadata = (
      functools.reduce(merge_trees, internal_metas).tree_metadata_entries
      if internal_metas
      else []
  )
  return {
      tree_utils.tuple_path_from_keypath(entry.jax_keypath())
      for entry in merged_metadata
  }


def _validate_keys(
    keys: Iterable[Any], merged_metadata_keys: Set[Any]
) -> Set[Any]:
  """Validates that keys do not conflict with previously saved metadata keys."""
  for key in keys:
    is_exact_match = key in merged_metadata_keys
    has_prefix_conflict = isinstance(key, tuple) and any(
        isinstance(mt, tuple) and (_is_prefix(key, mt) or _is_prefix(mt, key))
        for mt in merged_metadata_keys
    )
    if is_exact_match or has_prefix_conflict:
      raise PartialSaveReplacementError(
          f'Key "{key!r}" was found in a previous partial save in this session.'
          ' Partial saving currently does not support REPLACEMENT.'
      )
  return set(keys)


async def get_partial_save_additions(
    directory: epath.Path,
    flat_item: Mapping[Any, Any],
    pytree_metadata_options: tree_metadata.PyTreeMetadataOptions,
) -> Set[Any]:
  """Gets keys from the current save that are additions to the partial save.

  This method checks the keys in `flat_item` against metadata from previously
  completed partial saves within the same checkpoint session.

  Args:
    directory: The directory of the current partial save.
    flat_item: The flattened current PyTree.
    pytree_metadata_options: The PyTree metadata options to use for parsing
      metadata.

  Returns:
    A set of keys that are additions to the partial save.

  Raises:
    ValueError: If the directory is not a temporary directory.
    PartialSaveReplacementError: If a key in `flat_item` matches a key in a
      previous partial save.
  """
  internal_metas = await _read_pending_metadatas(
      directory, pytree_metadata_options=pytree_metadata_options
  )
  merged_metadata_keys = await _merge_pending_metadatas(internal_metas)
  additions = _validate_keys(flat_item, merged_metadata_keys)
  logging.info(
      '[process=%d] Found the following additions during partial save: %s',
      multihost.process_index(),
      additions,
  )
  return additions
