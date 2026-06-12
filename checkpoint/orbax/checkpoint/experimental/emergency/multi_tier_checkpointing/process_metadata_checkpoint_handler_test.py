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

"""Tests for ProcessMetadataCheckpointHandler.

This file tests the functionality of ProcessMetadataCheckpointHandler,
specifically its ability to save and restore process metadata.
"""

import asyncio
from typing import List
from unittest import mock

from absl import logging
from absl.testing import flagsaver
from absl.testing import parameterized
from etils import epath
import jax
from orbax.checkpoint import test_utils
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import process_metadata_checkpoint_handler


ProcessMetadataCheckpointHandler = (
    process_metadata_checkpoint_handler.ProcessMetadataCheckpointHandler
)
ProcessMetadataSaveArgs = (
    process_metadata_checkpoint_handler.ProcessMetadataSaveArgs
)
ProcessMetadataRestoreArgs = (
    process_metadata_checkpoint_handler.ProcessMetadataRestoreArgs
)
_MESH_METADATA_FILE_NAME = 'mesh_metadata.json'


class TestProcessMetadataCheckpointHandler(ProcessMetadataCheckpointHandler):

  def __init__(
      self,
  ):
    super().__init__()
    self._handler = (
        process_metadata_checkpoint_handler.ProcessMetadataCheckpointHandler
    )


class ProcessMetadataCheckpointHandlerTest(
    parameterized.TestCase, multiprocess_test.MultiProcessTest
):

  def setUp(self):
    super().setUp()
    self.enter_context(
        flagsaver.flagsaver(experimental_orbax_use_distributed_process_id=True)
    )
    if not multihost.is_runtime_to_distributed_ids_initialized():
      multihost.initialize_runtime_to_distributed_ids()
    if not multihost.is_distributed_to_device_ids_initialized():
      multihost.initialize_distributed_to_device_ids()
    self.global_mesh = self.make_global_mesh()
    self.directory = epath.Path(
        self.create_tempdir(
            name=f'checkpointing_test_pid{multihost.process_index()}'
        ).full_path
    )
    logging.info(
        'self.directory=%s',
        self.directory,
    )
    test_utils.sync_global_processes(
        'ProcessMetadataCheckpointHandlerTest:setup_complete'
    )

  def tearDown(self):
    super().tearDown()
    test_utils.sync_global_processes(
        'ProcessMetadataCheckpointHandlerTest:teardown_complete'
    )

  def make_global_mesh(self, replica_axis_index: int = 0) -> jax.sharding.Mesh:
    if replica_axis_index not in [0, 1]:
      raise ValueError(
          'replica_axis_index must be 0 or 1 for this test. Got:'
          f' {replica_axis_index}.'
      )
    self.assertEqual(jax.device_count(), 8)
    self.assertEqual(jax.process_count(), 4)
    self.assertEqual(jax.local_device_count(), 2)

    # setup global mesh info for 2-slice tests
    slice_processes = [{0, 1}, {2, 3}]
    mesh = test_utils.get_fake_global_mesh_for_slices(
        slice_processes, replica_axis_index
    )
    if replica_axis_index == 0:
      assert mesh.devices.shape == (2, 4), mesh.devices.shape
    if replica_axis_index == 1:
      assert mesh.devices.shape == (4, 2), mesh.devices.shape
    return mesh

  def assert_process_metadata_equals(
      self,
      global_mesh: jax.sharding.Mesh,
      distributed_to_device_ids: List[List[int]],
      device_ids: List[int],
  ):
    self.assertListEqual(
        multihost.distributed_to_device_ids(),
        distributed_to_device_ids,
    )

    self.assertListEqual(
        device_ids,
        [int(id) for id in global_mesh.device_ids.flatten()],
    )

  @parameterized.parameters((0,), (1,))
  def test_save_restore(self, replica_axis_index: int):
    global_mesh = self.make_global_mesh(replica_axis_index=replica_axis_index)
    handler = ProcessMetadataCheckpointHandler()

    handler.save(
        self.directory,
        ProcessMetadataSaveArgs(global_mesh),
    )
    restored = handler.restore(
        self.directory,
        ProcessMetadataRestoreArgs(),
    )
    self.assert_process_metadata_equals(
        global_mesh, restored[0], restored[1]
    )

  def test_async_save_restore(self):
    handler = TestProcessMetadataCheckpointHandler()

    async def run_async_test():
      async_futures = await handler.async_save(
          self.directory,
          ProcessMetadataSaveArgs(self.global_mesh),
      )
      for future in async_futures:
        future.result()
      self.assertTrue((self.directory / _MESH_METADATA_FILE_NAME).exists())
      restored = handler.restore(
          self.directory,
          ProcessMetadataRestoreArgs(),
      )
      self.assert_process_metadata_equals(
          self.global_mesh, restored[0], restored[1]
      )

    asyncio_utils.run_sync(run_async_test())
    handler.close()

  def test_async_save_uses_injected_distributed_to_device_ids_fn(self):
    directory = epath.Path(self.create_tempdir().full_path)
    distributed_to_device_ids = [[10, 11]]
    save_args = ProcessMetadataSaveArgs(global_mesh=mock.sentinel.mesh)
    commit_future = object()
    save_result = object()
    handler = ProcessMetadataCheckpointHandler(
        distributed_to_device_ids_fn=lambda: distributed_to_device_ids,
    )
    mock_save_process_metadata = mock.Mock(return_value=save_result)

    with mock.patch.object(
        process_metadata_checkpoint_handler.mesh_consistency,
        'save_process_metadata',
        new=mock_save_process_metadata,
    ), mock.patch.object(
        process_metadata_checkpoint_handler.future,
        'CommitFutureAwaitingContractedSignals',
        return_value=commit_future,
    ) as mock_commit_future:
      result = asyncio.run(handler.async_save(directory, save_args))

    self.assertEqual(result, [commit_future])
    mock_save_process_metadata.assert_called_once_with(
        directory,
        mock.sentinel.mesh,
        distributed_to_device_ids,
    )
    mock_commit_future.assert_called_once_with(
        save_result,
        name='process_metadata_ch_save',
    )

  def test_restore_delegates_to_read_process_metadata(self):
    directory = epath.Path(self.create_tempdir().full_path)
    restored = ([[1, 2]], [1, 2])
    handler = ProcessMetadataCheckpointHandler()

    with mock.patch.object(
        process_metadata_checkpoint_handler.mesh_consistency,
        'read_process_metadata',
        return_value=restored,
    ) as mock_read_process_metadata:
      result = handler.restore(
          directory,
          ProcessMetadataRestoreArgs(),
      )

    self.assertEqual(result, restored)
    mock_read_process_metadata.assert_called_once_with(directory)


if __name__ == '__main__':
  multiprocess_test.main()
