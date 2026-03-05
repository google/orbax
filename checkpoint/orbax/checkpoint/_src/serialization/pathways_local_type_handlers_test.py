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

"""Test for Pathways local type handlers."""

from typing import Any, Sequence

from absl import flags
import jax
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import pathways as multihost_pathways
from orbax.checkpoint._src.serialization import local_type_handlers_test
from orbax.checkpoint._src.serialization import ocdbt_utils
from orbax.checkpoint._src.serialization import pathways_handler_registry
from orbax.checkpoint._src.serialization import type_handler_registry
from orbax.checkpoint._src.testing import multiprocess_test
import tensorstore as ts


USE_COLOCATED_PYTHON = flags.DEFINE_boolean(
    'use_colocated_python',
    False,
    'Whether to use colocated Python.',
)


class FakeArrayMetadataStore(array_metadata_store_lib.Store):
  """A fake in-memory store that mimics the real checkpoint store API."""

  def __init__(self):
    self._data = {}

  async def write(
      self,
      checkpoint_dir: Any,
      array_metadatas: Sequence[Any],
      process_index: int,
  ) -> None:
    """Simulates writing metadata to storage."""
    if checkpoint_dir not in self._data:
      self._data[checkpoint_dir] = []
    self._data[checkpoint_dir].extend(array_metadatas)

  async def read(
      self, checkpoint_dir: Any, process_index: int | None = None
  ) -> Any:
    """Simulates reading metadata from storage."""
    return {0: self._data.get(checkpoint_dir, [])}


class PathwaysLocalTypeHandlersTest(
    local_type_handlers_test.LocalTypeHandlersTest,
):

  def setUp(self):
    super().setUp()
    self.assertTrue(multihost.is_pathways_backend())

  def validate_topology(self):
    self.assertEqual(jax.device_count(), 8)
    self.assertGreater(multihost_pathways.worker_count(None), 1)

  def get_array_handler(self):
    pathways_handler_registry.register_pathways_handlers(
        checkpointing_impl=pathways_handler_registry.CheckpointingImpl.from_options(
            use_colocated_python=USE_COLOCATED_PYTHON.value,
            use_remote_python=True,  # Fallback
        ),
        primary_host=None,
        replica_id=None,
        use_replica_parallel=False,
        thinmint_testing=True,
        array_metadata_store=FakeArrayMetadataStore(),
    )
    return type_handler_registry.get_type_handler(jax.Array)

  def validate_paths(self):
    # Array files should not exist at the global path level.
    self.assertFalse((self.base_directory / 'manifest.ocdbt').exists())
    for worker_id in range(multihost_pathways.worker_count(None)):
      self.assertTrue((self.base_directory / f'local_{worker_id}').exists())

  async def finalize_save(
      self, *, ts_context: ts.Context, use_zarr3: bool, use_ocdbt: bool
  ):
    if use_ocdbt:
      for worker_id in range(multihost_pathways.worker_count(None)):
        await ocdbt_utils.merge_ocdbt_per_process_files(
            self.base_directory / f'local_{worker_id}',
            ts_context=ts_context,
            use_zarr3=use_zarr3,
        )
      test_utils.sync_global_processes(
          'local_serialization:merge_ocdbt_complete'
      )


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  multiprocess_test.main()
