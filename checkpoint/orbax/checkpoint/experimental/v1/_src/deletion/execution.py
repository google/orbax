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

"""Coordinated execution and responses for checkpoint deletion."""

import asyncio
from collections.abc import Callable
import dataclasses
from typing import Any
import uuid

from absl import logging
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.deletion import errors
from orbax.checkpoint.experimental.v1._src.deletion import metadata
from orbax.checkpoint.experimental.v1._src.deletion import path_utils
from orbax.checkpoint.experimental.v1._src.deletion import validation
from orbax.checkpoint.experimental.v1._src.metadata import serialization as metadata_serialization
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.synchronization import synchronization
from orbax.checkpoint.experimental.v1._src.synchronization import thread_utils


@dataclasses.dataclass
class _Plan:
  path: path_types.Path
  destination: path_types.Path | None
  record: dict[str, Any] | None = None
  resuming: bool = False


async def _prepare(
    path: path_types.Path,
    name: str | None,
    missing_ok: bool,
    context: context_lib.Context,
    operation_id: str,
    managed: bool,
    whole_relocation_name: str | None,
) -> _Plan | None:
  """Resolves the target and metadata transition without changing files."""
  validation.validate_arguments(name, missing_ok)
  await path_utils.check_path(path)
  if not await async_path.exists(path):
    if missing_ok:
      return None
    raise FileNotFoundError(f'Checkpoint does not exist: {path}.')
  await validation.validate_root(path, context=context, managed=managed)
  if name is None:
    target = await path_utils.destination(
        path, context, operation_id, whole_relocation_name=whole_relocation_name
    )
    if target is not None and await async_path.exists(target):
      raise FileExistsError(f'Deletion destination already exists: {target}.')
    return _Plan(path, target)
  record = await metadata.read_record(path)
  if record is not None:
    if record['checkpointable_name'] != name:
      raise errors.DeletionRecoveryError(
          f'Checkpoint at {path} has a deletion for a different item.'
      )
    destination = record['destination']
    target = (
        context.file_options.path_class(destination) if destination else None
    )
    if target is not None:
      try:
        await path_utils.check_recovery_destination(
            target, path, name, record['operation_id']
        )
      except ValueError as e:
        raise errors.DeletionRecoveryError(
            'Invalid recovery destination.'
        ) from e
    return _Plan(path, target, record, resuming=True)
  try:
    original = await validation.prepare_item(path, name)
  except FileNotFoundError:
    # Only the absent item is optional, not a missing checkpoint metadata file.
    metadata_path = metadata_serialization.checkpoint_metadata_file_path(path)
    if missing_ok and await async_path.exists(metadata_path):
      if not await async_path.exists(path / name):
        return None
    raise
  target = await path_utils.destination(
      path / name, context, operation_id, checkpoint_root=path
  )
  return _Plan(
      path,
      target,
      {
          'operation_id': operation_id,
          'checkpointable_name': name,
          'destination': str(target) if target is not None else None,
          'original_metadata': original,
          'updated_metadata': metadata.updated_metadata(original, name),
      },
  )


async def _begin(plan: _Plan | None, context: context_lib.Context) -> None:
  """Publishes partial-deletion intent before returning an async response."""
  if plan is not None and plan.record is not None and not plan.resuming:
    await metadata.write_json(
        plan.path / metadata.RECORD_FILENAME,
        plan.record,
        exclusive=True,
        permission_mode=context.file_options.path_permission_mode,
    )


async def _execute(
    plan: _Plan | None, whole_delete: Callable[[], None] | None
) -> None:
  """Removes the payload, then commits metadata and clears partial intent."""
  if plan is None:
    return
  logging.info(
      'Deleting checkpoint%s at %s.',
      'able ' + repr(plan.record['checkpointable_name']) if plan.record else '',
      plan.path,
  )
  if plan.record is None:
    if whole_delete is not None:
      if plan.destination is not None and await async_path.exists(
          plan.destination
      ):
        raise FileExistsError(
            f'Deletion destination already exists: {plan.destination}.'
        )
      # The managed v0 step deleter blocks; keep it off the event loop.
      await asyncio.to_thread(whole_delete)
    else:
      await path_utils.delete_path(
          plan.path, plan.destination, keep_markers_until_last=True
      )
    return
  record = plan.record
  item = plan.path / record['checkpointable_name']
  # An absent item was already removed or relocated. A trash cleaner may also
  # have removed the relocated copy since; neither should block finishing.
  if await async_path.exists(item):
    await path_utils.delete_path(item, plan.destination)
  await metadata.write_json(
      metadata_serialization.checkpoint_metadata_file_path(plan.path),
      record['updated_metadata'],
  )
  await async_path.unlink(plan.path / metadata.RECORD_FILENAME)
  await metadata.sync_directory(plan.path)


def start(
    path: path_types.PathLike,
    *,
    checkpointable_name: str | None,
    missing_ok: bool,
    managed: bool = False,
    whole_delete: Callable[[], None] | None = None,
    whole_relocation_name: str | None = None,
) -> thread_utils.BackgroundThreadRunner[bool]:
  """Prepares deletion on the calling thread; cleanup runs in a worker."""
  context = context_lib.get_context()
  path = context.file_options.path_class(path)
  opts = context.multiprocessing_options

  async def setup() -> tuple[_Plan | None, str]:
    await synchronization.synchronize_next_operation_id(
        prefix=opts.barrier_sync_key_prefix,
        processes=opts.active_processes,
    )
    operation_id = synchronization.get_operation_id()
    # Only the primary publishes its plan. UUIDs avoid trash collisions across
    # restarts, when the process-local synchronization counter starts over.
    destination_id = uuid.uuid4().hex
    plan = await _prepare(
        path,
        checkpointable_name,
        missing_ok,
        context,
        destination_id,
        managed,
        whole_relocation_name,
    )
    await multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'delete:prepare', prefix=opts.barrier_sync_key_prefix
        ),
        operation_id=operation_id,
        processes=opts.active_processes,
    )
    if multihost.is_primary_host(opts.primary_host):
      await _begin(plan, context)
    await multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'delete:begin', prefix=opts.barrier_sync_key_prefix
        ),
        operation_id=operation_id,
        processes=opts.active_processes,
    )
    return plan, operation_id

  plan, operation_id = asyncio_utils.run_sync(setup())

  async def run() -> bool:
    if multihost.is_primary_host(opts.primary_host):
      await _execute(plan, whole_delete)
    await multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'delete:finalize', prefix=opts.barrier_sync_key_prefix
        ),
        operation_id=operation_id,
        processes=opts.active_processes,
    )
    return plan is not None

  return thread_utils.BackgroundThreadRunner[bool](run())
