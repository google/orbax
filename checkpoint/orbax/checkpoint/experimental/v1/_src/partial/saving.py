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

"""Defines free-function interface for partial saving and finalizing."""

import ast
import asyncio
import dataclasses
import itertools
import json
from typing import Any, Awaitable, Callable
from etils import epath

from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.handlers import base_pytree_checkpoint_handler
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import format_utils
from orbax.checkpoint._src.path import utils as ocp_path_utils
from orbax.checkpoint._src.path.snapshot import snapshot
from orbax.checkpoint._src.tree import structure_utils as tree_structure_utils
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import global_registration  # pylint: disable=unused-import
from orbax.checkpoint.experimental.v1._src.handlers import pytree_handler
from orbax.checkpoint.experimental.v1._src.handlers import stateful_checkpointable_handler
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.metadata import serialization as metadata_serialization
from orbax.checkpoint.experimental.v1._src.partial import path as partial_path_lib
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.saving import execution
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.synchronization import synchronization
from orbax.checkpoint.experimental.v1._src.synchronization import types as async_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


STATE_CHECKPOINTABLE_KEY = checkpoint_layout.STATE_CHECKPOINTABLE_KEY
ORBAX_CHECKPOINT_INDICATOR_FILE = orbax_layout.ORBAX_CHECKPOINT_INDICATOR_FILE
CHECKPOINT_METADATA_FILENAME = metadata_serialization._CHECKPOINT_METADATA_FILENAME  # pylint: disable=protected-access
PYTREE_METADATA_FILE = format_utils.PYTREE_METADATA_FILE


StatefulCheckpointableHandler = (
    stateful_checkpointable_handler.StatefulCheckpointableHandler
)
BasePyTreeCheckpointHandler = (
    base_pytree_checkpoint_handler.BasePyTreeCheckpointHandler
)


@dataclasses.dataclass
class _PartialSavePyTree(handler_types.StatefulCheckpointable):
  """Wraps a PyTree to signal that it should be saved in partial mode."""

  state: tree_types.PyTree

  def __post_init__(self):
    self.handler = pytree_handler.PyTreeHandler(partial_save_mode=True)

  async def save(
      self, directory: path_types.PathAwaitingCreation
  ) -> Awaitable[None]:
    return await self.handler.save(directory, self.state)

  async def load(self, directory: path_types.Path) -> Awaitable[None]:
    raise NotImplementedError('Partial load is not supported via this wrapper.')


def save(
    path: path_types.PathLike,
    state: tree_types.PyTreeOf[tree_types.Leaf],  # pyrefly: ignore[bad-specialization]
    *,
    custom_metadata: tree_types.JsonType | None = None,
):
  """Partially saves a PyTree.

  This function allows for incrementally updating a checkpoint. It is designed
  to be called multiple times. The first call initiates a new partial save
  "session" in a temporary location. Subsequent calls will update this session
  by modifying the checkpoint in place.

  The operation is atomic; if it is interrupted, the previous version of the
  partial save will be preserved.

  IMPORTANT: The checkpoint is not finalized at the target `path` until
  :py:func:`.finalize` is called. The intermediate checkpoints are
  temporary and should not be used directly.

  ### Workflow

  A typical partial save workflow involves one or more calls to
  :py:func:`.save` followed by a single call to :py:func:`~.finalize`::

    path = '/path/to/my/checkpoint'

    # The first call creates a temporary directory:
    # '/path/to/my/checkpoint.partial_save'
    # Note: the exact temporary directory name is an implementation detail that
    # depends on the file system and should not be relied on.
    ocp.partial.save(path, {'layer1': ..., 'step': 1})

    # A subsequent call reads the previous version and applies new updates
    # to the temporary directory:
    # '/path/to/my/checkpoint.partial_save'
    ocp.partial.save(path, {'layer2': ..., 'metrics': ...})

    # This call commits the latest version to the final destination at
    # '/path/to/my/checkpoint'.
    ocp.partial.finalize(path)

  ### Additions vs. Replacements

  The provided `state` represents a set of updates.
  - If a key in `state` (e.g., 'metrics') does not exist in the on-disk
    checkpoint, it is treated as an **addition**. In other words, the sets of
    keys of the on-disk PyTree and the provided `state` are disjoint.
  - If a key (e.g., 'step') already exists, its value is **replaced**. In other
    words, the sets of keys of the on-disk PyTree and the provided `state`
    overlap. Replacements are currently NOT supported. Please reach out to the
    Orbax team if you need this functionality.

  See :py:func:`~.v1.save` for general
  PyTree saving documentation.

  Args:
    path: The path to save the checkpoint to.
    state: A PyTree representing the additions to be applied to the on-disk
      checkpoint.
    custom_metadata: User-provided custom metadata. This will be merged with any
      existing custom metadata. Values from this dictionary will overwrite
      existing values if keys conflict.
  """
  save_async(
      path,
      state,
      custom_metadata=custom_metadata,
  ).result()


def save_async(
    path: path_types.PathLike,
    state: tree_types.PyTreeOf[tree_types.Leaf],  # pyrefly: ignore[bad-specialization]
    *,
    custom_metadata: tree_types.JsonType | None = None,
) -> async_types.AsyncResponse[None]:
  """Partially saves a PyTree asynchronously.

  Unlike :py:func:`.save`, this function returns an
  :py:class:`.AsyncResponse`
  immediately after scheduling the save operation. The actual writing to disk
  happens in a background thread. You can use `response.result()` to block
  until the operation is complete.

  This function allows for incrementally updating a checkpoint. It is designed
  to be called multiple times. The first call initiates a new partial save
  "session" in a temporary location. Subsequent calls will update this session
  by creating a new version that includes all previous changes plus the new
  ones.

  The operation is atomic; if it is interrupted, the previous version of the
  partial save will be preserved.

  IMPORTANT: The checkpoint is not finalized at the target `path` until
  :py:func:`.finalize` is called. The intermediate checkpoints are
  temporary and may be garbage collected in certain environments.

  ### Workflow

  A typical partial save workflow involves one or more calls to
  :py:func:`.save_async` followed by a single call to
  :py:func:`.finalize`::

    path = '/path/to/my/checkpoint'

    # The first call creates a temporary directory and returns immediately.
    response1 = ocp.partial.save_async(path, {'layer1': ..., 'step': 1})

    # A subsequent call also returns immediately. Orbax ensures that this
    # operation waits for the first one to complete before starting.
    response2 = ocp.partial.save_async(
        path, {'layer2': ..., 'metrics': ...}
    )

    # Wait for all async partial saves to complete before finalizing.
    response1.result()
    response2.result()

    # This call commits the latest version to the final destination at
    # '/path/to/my/checkpoint'.
    ocp.partial.finalize(path)

  ### Additions vs. Replacements

  The provided `state` represents a set of updates.
  - If a key in `state` (e.g., 'metrics') does not exist in the on-disk
    checkpoint, it is treated as an **addition**.
  - If a key (e.g., 'step') already exists, its value is **replaced**.
    Replacements are currently NOT supported. Please reach out to the Orbax team
    if you need this functionality.

  See :py:func:`~.v1.save_async` for general
  PyTree saving documentation.

  Args:
    path: The path to save the checkpoint to.
    state: The PyTree to save. This may be any JAX PyTree consisting of
      supported leaf types (see :py:class:`~.v1.tree.Leaf`).
      Default supported leaf types include `jax.Array`, `np.ndarray`,
      simple types like `int`, `float`, `str`, and empty nodes.
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.

  Returns:
    An :py:class:`.AsyncResponse` that can be used to block until the save is
    complete.
    Blocking can be done using `response.result()`, which returns `None`.

  Raises:
    FileExistsError: If a finalized checkpoint already exists at `path`. To
      overwrite, it must be deleted first.
  """
  ctx = context_lib.get_context()
  path = ctx.file_options.path_class(path)
  if path.exists():
    raise FileExistsError(f'Finalized checkpoint already exists at {path}.')

  return execution.save_checkpointables_impl(
      partial_path_lib.add_partial_save_suffix(path),
      {STATE_CHECKPOINTABLE_KEY: _PartialSavePyTree(state)},
      overwrite=False,
      custom_metadata=custom_metadata,
      async_origin=True,
      partial_save=True,
  )


async def _read_first_metadata(
    pending_dirs: list[epath.Path],
) -> tree_metadata.InternalTreeMetadata | None:
  """Reads metadata from the first pending directory."""
  if not pending_dirs:
    return None

  for item in await async_path.iterdir(pending_dirs[0]):
    if not await async_path.is_dir(item):
      continue
    first_meta_path = item / PYTREE_METADATA_FILE
    if await async_path.exists(first_meta_path):
      try:
        return tree_metadata.InternalTreeMetadata.from_json(
            json.loads(await async_path.read_text(first_meta_path)),
            pytree_metadata_options=tree_metadata.PYTREE_METADATA_OPTIONS,
        )
      except json.JSONDecodeError as e:
        raise ValueError(
            'Failed to read metadata from first metadata file'
            f' {first_meta_path}.'
        ) from e
  return None


def _is_prefix(t1: tuple[str, ...], t2: tuple[str, ...]) -> bool:
  return len(t1) < len(t2) and t2[: len(t1)] == t1


def _filter_conflicting_keys(d: dict[str, Any]) -> dict[str, Any]:
  """Filters metadata keys that conflict due to parent-child relationships.

  When merging metadata from multiple partial saves, we might encounter
  conflicting entries. For example, one partial save might save 'a/b' as a
  leaf, while another saves 'a/b/c' as a leaf. This is a conflict because
  'a/b' cannot be both a leaf and an intermediate node containing 'c'. This
  function resolves the conflict by removing metadata for 'a/b', keeping
  'a/b/c', and implicitly treating 'a/b' as an intermediate node.

  Args:
    d: A dictionary of metadata.

  Returns:
    The filtered metadata dictionary.
  """
  keys = list(d.keys())
  to_remove = set()

  parsed_keys = {}
  for k in keys:
    try:
      parsed_keys[k] = ast.literal_eval(k)
    except (ValueError, SyntaxError):
      parsed_keys[k] = k

  for k1, k2 in itertools.permutations(keys, 2):
    t1, t2 = parsed_keys[k1], parsed_keys[k2]
    if isinstance(t1, tuple) and isinstance(t2, tuple):
      if _is_prefix(t1, t2):
        to_remove.add(k1)
    elif isinstance(k1, str) and isinstance(k2, str):
      if k2.startswith((k1 + '.', k1 + '/')):
        to_remove.add(k1)

  for k in to_remove:
    del d[k]
  return d


async def _rename_or_merge_json(
    src: epath.Path, dst: epath.Path, merge_fn: Callable[[Any, Any], Any]
):
  """Tries to rename src to dst, otherwise merges them as JSONs using merge_fn."""
  try:
    await async_path.rename(src, dst)
  except FileExistsError:
    pass
  else:
    return

  src_meta = json.loads(await async_path.read_text(src))
  dst_meta = json.loads(await async_path.read_text(dst))

  merged_meta = merge_fn(src_meta, dst_meta)

  await async_path.write_text(dst, json.dumps(merged_meta))
  await async_path.unlink(src)


async def _merge_pytree_metadata(src_item: epath.Path, dst_item: epath.Path):
  """Merges PyTree metadata files (_METADATA or _sharding)."""

  def _merge_fn(src_meta, dst_meta):
    merged = tree_structure_utils.merge_trees(
        dst_meta, src_meta, overwrite=True
    )
    if 'tree_metadata' in merged:
      merged['tree_metadata'] = _filter_conflicting_keys(
          merged['tree_metadata']
      )
    return merged

  await _rename_or_merge_json(src_item, dst_item, _merge_fn)


async def _rename_ocdbt_process_dir(
    item: epath.Path, pytree_dst: epath.Path, uuid_str: str
):
  """Renames an ocdbt.process_ directory to avoid collisions across partial saves."""
  # To avoid collisions across different partial save pending directories,
  # we append the pending dir's UUID to the original process ID.
  # We must avoid using '_' in the new ID because `ocdbt_utils.py` splits
  # the directory name by '_' to extract the process ID.
  new_name = f'{item.name}{uuid_str.replace("-", "")}'
  await async_path.rename(item, pytree_dst / new_name)


async def _merge_array_metadatas(src_dir: epath.Path, dst_dir: epath.Path):
  """Merges array_metadatas JSON files (process_0, process_1, etc.)."""
  await async_path.mkdir(dst_dir, parents=True, exist_ok=True)

  async def _process_child(src_child: epath.Path):
    dst_child = dst_dir / src_child.name

    def _merge_fn(src_meta, dst_meta):
      src_arr_meta = src_meta.get('array_metadatas', [])
      dst_arr_meta = dst_meta.get('array_metadatas', [])
      dst_arr_meta.extend(src_arr_meta)
      dst_meta['array_metadatas'] = dst_arr_meta
      return dst_meta

    await _rename_or_merge_json(src_child, dst_child, _merge_fn)

  await asyncio.gather(*[
      _process_child(src_child)
      for src_child in await async_path.iterdir(src_dir)
  ])


async def _recursive_merge(src: epath.Path, dst: epath.Path):
  """Recursively merges src into dst."""
  if not await async_path.exists(src):
    return

  try:
    await async_path.rename(src, dst)
  except FileExistsError:
    pass
  else:
    return

  if await async_path.is_dir(src):
    items = await async_path.iterdir(src)
    await asyncio.gather(
        *[_recursive_merge(item, dst / item.name) for item in items]
    )
    await async_path.rmtree(src)
    return

  raise FileExistsError(
      f'File collision on {src.name} during finalize. Overwriting destination '
      'file is not allowed.'
  )


async def _merge_pytree_directory(
    pytree_src: epath.Path,
    partial_path: epath.Path,
    uuid_str: str,
):
  """Merges a single pending pytree directory into the destination."""
  if not await async_path.exists(pytree_src):
    return

  pytree_dst = partial_path / pytree_src.name
  await async_path.mkdir(pytree_dst, parents=True, exist_ok=True)

  async def _merge_item(item_path: epath.Path):
    if item_path.name in [PYTREE_METADATA_FILE, '_sharding']:
      await _merge_pytree_metadata(item_path, pytree_dst / item_path.name)
    elif item_path.name.startswith('ocdbt.process_'):
      await _rename_ocdbt_process_dir(item_path, pytree_dst, uuid_str)
    elif item_path.name == 'array_metadatas':
      await _merge_array_metadatas(item_path, pytree_dst / item_path.name)
    else:
      await _recursive_merge(item_path, pytree_dst / item_path.name)

  await asyncio.gather(
      *[_merge_item(item) for item in await async_path.iterdir(pytree_src)]
  )

  await async_path.rmtree(pytree_src)


async def _merge_checkpoint_metadata(src: epath.Path, dst: epath.Path):
  """Merges checkpoint metadata."""

  def _merge_fn(src_meta, dst_meta):
    return tree_structure_utils.merge_trees(dst_meta, src_meta, overwrite=True)

  await _rename_or_merge_json(src, dst, _merge_fn)


async def _merge_indicator_file(src: epath.Path, dst: epath.Path):
  """Merges the Orbax checkpoint indicator file."""
  try:
    await async_path.rename(src, dst)
  except FileExistsError:
    await async_path.unlink(src)




async def _is_pytree_dir(item: epath.Path) -> bool:
  """Returns True if the item is a PyTree directory."""
  return await async_path.is_dir(item) and await async_path.exists(
      item / PYTREE_METADATA_FILE
  )


async def _merge_all(partial_path: epath.Path):
  """Merges all pending directories into the partial path."""

  # Each partial save call results in a new pending directory containing unique
  # PyTree keypaths and corresponding data. During finalization, all pending
  # directories are merged to form the final checkpoint state.
  # Ensure deterministic merge order (alphabetical glob + timestamp).
  pending_dirs = sorted(await snapshot.list_pending_dirs(partial_path))

  first_metadata = await _read_first_metadata(pending_dirs)
  use_zarr3 = first_metadata.use_zarr3 if first_metadata is not None else False

  pytree_directories = []

  for p_dir in pending_dirs:
    uuid_str = snapshot.get_uuid_from_pending_dir_name(p_dir.name)

    async def _process_item(item: epath.Path, uuid_str: str):
      if item.name == CHECKPOINT_METADATA_FILENAME:
        await _merge_checkpoint_metadata(item, partial_path / item.name)
      elif item.name == ORBAX_CHECKPOINT_INDICATOR_FILE:
        await _merge_indicator_file(item, partial_path / item.name)
      elif await _is_pytree_dir(item):
        pytree_directories.append(item.name)
        await _merge_pytree_directory(item, partial_path, uuid_str)
      else:
        await _recursive_merge(item, partial_path / item.name)

    await asyncio.gather(*[
        _process_item(item, uuid_str)
        for item in await async_path.iterdir(p_dir)
    ])

    await async_path.rmtree(p_dir)

  # 3. Call PyTreeHandler.finalize to perform OCDBT merge.
  # This merges the individual ocdbt.process_xxx directories into a single
  # valid manifest for the final partial state.
  handler = BasePyTreeCheckpointHandler(use_zarr3=use_zarr3)
  for pytree_dir_name in pytree_directories:
    await asyncio.to_thread(handler.finalize, partial_path / pytree_dir_name)


def finalize(path: path_types.PathLike) -> None:
  """Finalizes a partially-saved checkpoint, making it permanent and readable.

  This function commits all changes made during a partial save session,
  concluding the transaction. It should be called once after all desired
  :py:func:`.save` operations are complete.

  The finalization process is atomic. It renames the temporary, versioned
  partial save directory to the final target `path`, making the updated
  checkpoint "live".

  IMPORTANT: Until `finalize` is called, the checkpoint at the target `path`
  is not created or modified. All changes are buffered in a temporary location.
  This function is what makes those changes permanent.


  ### Example::
    path = '/path/to/my/checkpoint'

    # These calls write to a temporary, versioned directory, not the final path.
    ocp.partial.save(path, {'step': 1})
    ocp.partial.save_checkpointables(path, {'metrics': ...})

    # This call performs the atomic rename, making the checkpoint available at
    # '/path/to/my/checkpoint'.
    ocp.partial.finalize(path)

  Args:
    path: The final, target path of the checkpoint to be finalized. This should
      be the same path that was passed to :py:func:`~.save` calls.

  Raises:
    FileExistsError: If a finalized checkpoint already exists at `path`. To
      overwrite, it must be deleted first.
    FileNotFoundError: If no partial save session is found for the given `path`.
      This can happen if :py:func:`.save` was not called first.
  """
  context = context_lib.get_context()
  path = context.file_options.path_class(path)
  if partial_path_lib.is_partial_save_path(path):
    final_path = partial_path_lib.remove_partial_save_suffix(path)
    partial_path = path
  else:
    final_path = path
    partial_path = partial_path_lib.add_partial_save_suffix(path)

  async def _finalize_impl():
    await multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'OcpPartialSaving:finalize_path_existence_start',
            prefix=context.multiprocessing_options.barrier_sync_key_prefix,
        ),
        operation_id=synchronization.get_operation_id(),
        processes=context.multiprocessing_options.active_processes,
    )
    if await async_path.exists(final_path):
      raise FileExistsError(
          f'Finalized checkpoint already exists at {final_path}.'
      )
    elif not await async_path.exists(partial_path):
      raise FileNotFoundError(
          f'Partial save path {partial_path} does not exist.'
      )

    await multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'OcpPartialSaving:finalize_path_rename_start',
            prefix=context.multiprocessing_options.barrier_sync_key_prefix,
        ),
        operation_id=synchronization.get_operation_id(),
        processes=context.multiprocessing_options.active_processes,
    )

    finalize_failed = False
    finalize_error = None
    if multihost.is_primary_host(context.multiprocessing_options.primary_host):
      try:
        await _merge_all(partial_path)
        await async_path.rename(partial_path, final_path)
      except (ValueError, OSError) as e:
        finalize_failed = True
        finalize_error = e

    finalize_failed = multihost.broadcast_one_to_all(
        finalize_failed,
        is_source=multihost.is_primary_host(
            context.multiprocessing_options.primary_host
        ),
    )

    await multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'OcpPartialSaving:finalize_rename_complete',
            prefix=context.multiprocessing_options.barrier_sync_key_prefix,
        ),
        operation_id=synchronization.get_operation_id(),
        processes=context.multiprocessing_options.active_processes,
    )

    if finalize_failed:
      raise finalize_error or OSError('Partial checkpoint finalization failed.')

  asyncio_utils.run_sync(_finalize_impl())
