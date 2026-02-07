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

"""OCDBT utilities for Orbax checkpointing."""

import asyncio
import re
import threading
from typing import Optional, Union

from absl import logging
from etils import epath
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
import tensorstore as ts

_SHARDING_SUFFIX_RE = r'/\d+(\.\d+)*$'  # /0, /0.0, /1.0.1, etc.
_ZARRAY_SUFFIX_RE = r'/\.zarray$'
_ZARRAY_SUFFIX = '/.zarray'


async def _validate_params(
    ts_kv_store: ts.KvStore,
    use_zarr3: bool,
) -> None:
  """Validates the params present in tensorstore KvStore.

  Supports zarr2.

  NOTE: Support for zarr3 will be added later.

  Args:
    ts_kv_store: Open kvstore to validate, with transaction if applicable.
    use_zarr3: If True, use zarr3 driver, otherwise, use zarr driver.
  """
  # TODO: b/362328389 - Add support for zarr3.
  if use_zarr3:
    logging.info(
        'Param validation support for Zarr3 will be added later (b/362328389).'
    )
    return

  raw_ts_params = await ts_kv_store.list()
  if not raw_ts_params:
    # TODO: b/361090820 - Raise error once we confirm that Bennu writing empty
    # states is a bug.
    # e.g. //learning/deepmind/jax/roc/formats/roc_orbax:roc_orbax_test
    logging.info(
        'Skipping param validation: No params found in TensorStore'
        ' KvStore: %s.',
        ts_kv_store,
    )
    return

  process_index = multihost.process_index()
  current_thread_name = threading.current_thread().name
  # [a/.zarray, a/0, b.zarray, b/0.0, b/0.1, c/0, d/.zarray] -> {a, b, d}
  with_zarray = set()
  # [a/.zarray, a/0, b.zarray, b/0.0, b/0.1, c/0, d/.zarray] -> {a, b, c}
  without_zarray = set()
  for ts_param in raw_ts_params:
    ts_param = ts_param.decode('utf-8')
    if logging.vlog_is_on(1):
      logging.vlog(
          1,
          '[process=%s][thread=%s] Validating raw param: %s',
          process_index,
          current_thread_name,
          ts_param,
      )
    # b/0.0 -> b, a/0 -> a, a/.zarray -> a/.zarray
    ts_param = re.sub(_SHARDING_SUFFIX_RE, '', ts_param)
    if ts_param.endswith(_ZARRAY_SUFFIX):
      # a/.zarray -> a
      ts_param = re.sub(_ZARRAY_SUFFIX_RE, '', ts_param)
      with_zarray.add(ts_param)
      if logging.vlog_is_on(1):
        logging.vlog(
            1,
            '[process=%s][thread=%s] Collecting param with .zarray: %s',
            process_index,
            current_thread_name,
            ts_param,
        )
    else:
      # b -> b
      without_zarray.add(ts_param)
      if logging.vlog_is_on(1):
        logging.vlog(
            1,
            '[process=%s][thread=%s] Collecting param without .zarray: %s',
            process_index,
            current_thread_name,
            ts_param,
        )

  unique = with_zarray | without_zarray
  logging.vlog(
      1,
      '[process=%s][thread=%s] Validating params in TensorStore KvStore.',
      process_index,
      current_thread_name,
  )
  missing_params = unique - without_zarray
  if missing_params:
    formatted_missing_params = ' \n'.join(sorted(missing_params))
    raise ValueError(
        f'Save failed: {len(missing_params)}/{len(unique)} params are missing'
        f' in checkpoint:\n{formatted_missing_params}.\nTensorstore KvStore:'
        f' {ts_kv_store}.'
    )
  missing_zarrays = unique - with_zarray
  if missing_zarrays:
    formatted_missing_zarrays = ' \n'.join(sorted(missing_zarrays))
    raise ValueError(
        f'Save failed: {len(missing_zarrays)}/{len(unique)} params are missing'
        f' .zarray in checkpoint:\n{formatted_missing_zarrays}.\nTensorstore'
        f' KvStore: {ts_kv_store}.'
    )


async def merge_ocdbt_per_process_files(
    directory: epath.Path,
    ts_context: ts.Context,
    use_zarr3: bool,
    enable_validation: bool = True,
):
  """Merges OCDBT files written to per-process subdirectories.

  With Tensorstore's OCDBT format, arrays are initially written to per-process
  subdirectories, depending on which host is doing the writing. This function
  can be called to merge the per-process files into a global key-value store.

  The original per-process subdirectories are not and should not be deleted -
  the global kvstore continues to reference them.

  NOTE: If no suitable subdirs with OCDBT checkpoints are found, this function
  does not raise any error and no merged checkpoint is created.

  Args:
    directory: checkpoint location.
    ts_context: Tensorstore context.
    use_zarr3: If True, use zarr3 driver, otherwise, use zarr driver for params
      validation.
    enable_validation: If True, validate params after merging. May have a
      performance impact.
  """
  open_ops = []
  for process_dir in directory.glob(f'{ts_utils.PROCESS_SUBDIR_PREFIX}*'):
    child_kvstore_tspec = ts_utils.build_kvstore_tspec_for_merge(
        directory.as_posix(),
        process_dir.name,
    )
    logging.vlog(1, 'child_kvstore_tspec: %s', child_kvstore_tspec)
    open_ops.append(ts_utils.open_kv_store(child_kvstore_tspec, ts_context))
  if not open_ops:  # No per-process OCDBT checkpoint found!
    logging.warning(
        '[process=%s][thread=%s] Skipping merge of OCDBT checkpoints: No'
        ' per-process OCDBT checkpoint subdirs found in %s, ',
        multihost.process_index(),
        threading.current_thread().name,
        directory,
    )
    return

  parent_kvstore_tspec = ts_utils.build_kvstore_tspec(
      directory.as_posix(), use_ocdbt=True
  )
  ts_utils.add_ocdbt_write_options(parent_kvstore_tspec)
  open_ops.append(ts_utils.open_kv_store(parent_kvstore_tspec, ts_context))

  opened = await asyncio.gather(*open_ops)
  parent, children = opened[-1], opened[:-1]
  copy_ops = []
  txn = ts.Transaction(atomic=True)
  for child in children:
    copy_ops.append(
        child.experimental_copy_range_to(parent.with_transaction(txn))
    )
  await asyncio.gather(*copy_ops)

  # Validate merged params.
  if enable_validation:
    await _validate_params(parent.with_transaction(txn), use_zarr3=use_zarr3)
  await txn.commit_async()


def get_process_index_for_subdir(
    use_ocdbt: bool,
    override_ocdbt_process_id: Optional[str] = None,
) -> Optional[Union[int, str]]:
  """If OCDBT + merge feature is in use, returns a process index."""
  if use_ocdbt:
    return override_ocdbt_process_id or multihost.process_index()
  else:
    return None
