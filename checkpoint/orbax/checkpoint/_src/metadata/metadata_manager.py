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

"""MetadataManager class for Orbax PyTree checkpointing."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Optional
import uuid

from absl import logging
from etils import epath
import jax
from orbax.checkpoint._src.logging import event_tracking
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import format_utils
from orbax.checkpoint._src.serialization import ocdbt_utils
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.serialization import types



class MetadataManager:
  """Manages file-system and metadata persistence for PyTree checkpoints."""

  async def get_param_infos_with_write_shape(
      self,
      param_infos: Any,
      checkpoint_dir: epath.Path,
      *,
      array_metadata_store: Any | None,
      primary_host: int | None,
  ) -> Any:
    """Returns `param_infos` updated with `write_shape`."""
    if array_metadata_store is None:
      return param_infos
    if not multihost.is_primary_host(primary_host):
      return param_infos
    # Extract write_shape from ArrayMetadata for current process_index.
    process_index = multihost.process_index()
    array_metadatas = await array_metadata_store.read(
        checkpoint_dir, process_index=process_index
    )
    if array_metadatas is None:
      jax_array_param_info = type_handlers.any_jax_array_param_info(param_infos)
      if jax_array_param_info is not None:
        raise ValueError(
            f'No ArrayMetadata found for process_index={process_index} in the'
            f' checkpoint directory: {checkpoint_dir}. But input PyTree'
            ' contains at least one jax.Array param_info:'
            f' {jax_array_param_info}.'
        )
      return param_infos

    assert isinstance(array_metadatas, list)
    array_metadatas_cache = {
        array_metadata.param_name: array_metadata
        for array_metadata in array_metadatas
    }

    def update_param_info(param_info: types.ParamInfo) -> types.ParamInfo:
      if not type_handlers.represents_jax_array(param_info):
        return param_info
      if param_info.name not in array_metadatas_cache:
        raise ValueError(
            f'No ArrayMetadata found for param_info: {param_info}, checkpoint'
            f' directory: {checkpoint_dir}, process_index={process_index}.'
        )
      return param_info.replace(
          write_shape=array_metadatas_cache[param_info.name].write_shape
      )

    return jax.tree.map(update_param_info, param_infos)

  async def write_metadata_file(
      self,
      directory: epath.Path,
      internal_tree_metadata: tree_metadata.InternalTreeMetadata,
      *,
      primary_host: int | None,
      pytree_metadata_options: Any,
  ) -> None:
    """Writes the pytree metadata file (`_METADATA`)."""
    if multihost.is_primary_host(primary_host):
      metadata_write_start_time = time.time()
      path = directory / format_utils.PYTREE_METADATA_FILE

      logging.vlog(
          1,
          'Writing pytree metadata file: %s with pytree_metadata_options: %s',
          path,
          pytree_metadata_options,
      )
      await async_path.write_text(
          path,
          json.dumps(internal_tree_metadata.to_json()),
      )
      jax.monitoring.record_event_duration_secs(
          '/jax/checkpoint/write/async/metadata_write_duration_secs',
          time.time() - metadata_write_start_time,
      )

  async def read_metadata_file(
      self, directory: epath.Path, *, pytree_metadata_options: Any
  ) -> tree_metadata.InternalTreeMetadata:
    """Reads metadata file and returns internal tree metadata."""
    path = directory / format_utils.PYTREE_METADATA_FILE
    if not await async_path.exists(path):
      raise FileNotFoundError(
          f'Metadata file (named {format_utils.PYTREE_METADATA_FILE}) does not'
          f' exist at {directory}.'
      )
    logging.vlog(
        1,
        'Reading pytree metadata file: %s with pytree_metadata_options: %s',
        path,
        pytree_metadata_options,
    )
    metadata = tree_metadata.InternalTreeMetadata.from_json(
        json.loads(await async_path.read_text(path)),
        pytree_metadata_options=pytree_metadata_options,
    )

    # Log the read event for the checkpoint to the DM log.
    event_tracking.record_read_metadata_event(directory)

    return metadata


  async def finalize_async(
      self,
      directory: epath.Path,
      *,
      array_metadata_store: Any | None,
      primary_host: int | None,
      array_metadata_validator: Any,
      use_zarr3: bool,
      enable_post_merge_validation: bool,
  ) -> None:
    """Finalizes checkpoint save (merging OCDBT and validating ArrayMetadata)."""
    start_time = time.time()
    finalize_coros = []
    if array_metadata_store is not None:
      if primary_host is None:
        logging.log_first_n(
            logging.INFO,
            '[process=%s] Skipped cross-host ArrayMetadata validation'
            ' because all hosts are primary (e.g. local storage).',
            1,  # log only once
            multihost.process_index(),
        )
      elif multihost.is_primary_host(primary_host):
        finalize_coros.append(
            array_metadata_store_lib.validate_all_array_metadatas(
                array_metadata_validator,
                array_metadata_store,
                directory,
            )
        )

    async def merge_ocdbt_per_process_files():
      merge_start_time = time.time()
      ts_context = ts_utils.get_ts_context(use_ocdbt=True)
      await ocdbt_utils.merge_ocdbt_per_process_files(
          directory,
          ts_context=ts_context,
          use_zarr3=use_zarr3,
          enable_validation=enable_post_merge_validation,
      )
      jax.monitoring.record_event_duration_secs(
          '/jax/checkpoint/write/async/ocdbt_merge_duration_secs',
          time.time() - merge_start_time,
      )

    finalize_coros.append(merge_ocdbt_per_process_files())

    await asyncio.gather(*finalize_coros)
    end_time = time.time()
    logging.info(
        '[process=%s][thread=%s] Pytree save finalize (merge_ocdbt +'
        ' ArrayMetadata validation) completed. Time taken: %fs. use_zarr3=%s,'
        ' enable_post_merge_validation=%s, directory=%s',
        multihost.process_index(),
        threading.current_thread().name,
        end_time - start_time,
        use_zarr3,
        enable_post_merge_validation,
        directory,
    )
