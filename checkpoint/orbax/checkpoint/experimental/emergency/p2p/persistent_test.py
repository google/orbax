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

from unittest import mock

from absl.testing import absltest
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint.experimental.emergency import checkpoint_manager as emergency_checkpoint_manager
from orbax.checkpoint.experimental.emergency.p2p import args as p2p_args_lib
from orbax.checkpoint.experimental.emergency.p2p import persistent

Mesh = jax.sharding.Mesh


class MockJaxClient:
  runtime_type = 'tpu'


class MockDevice:

  def __init__(self, process_index, slice_index):
    self.id = process_index
    self.process_index = process_index
    self.slice_index = slice_index
    self.client = MockJaxClient()

  def __repr__(self):
    return (
        f'MockDevice(id={self.id}, pi={self.process_index},'
        f' si={self.slice_index})'
    )


class PersistentCheckpointManagerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(mock.patch.object(jax, 'process_count', return_value=4))
    self.mock_client = mock.MagicMock()
    self.enter_context(
        mock.patch.object(
            multihost,
            'get_jax_distributed_client',
            return_value=self.mock_client,
        )
    )

    devices = np.array([
        [MockDevice(0, 0), MockDevice(1, 0)],
        [MockDevice(2, 1), MockDevice(3, 1)],
    ])
    self.enter_context(
        mock.patch.object(
            jax, 'devices', return_value=devices.flatten().tolist()
        )
    )

    self.directory = epath.Path(self.create_tempdir().full_path)
    self.mesh = mock.Mock(
        spec=jax.sharding.Mesh,
        devices=devices,
        axis_names=('replica', 'data'),
        shape={'replica': 2, 'data': 2},
        shape_tuple=devices.shape,
        size=devices.size,
    )
    self.options = emergency_checkpoint_manager.CheckpointManagerOptions()

  def _patch_process_index(
      self, in_primary_slice=True, process_index=0, replica_id=0
  ):
    self.enter_context(
        mock.patch.object(
            jax,
            'process_index',
            return_value=process_index,
        )
    )
    self.enter_context(
        mock.patch(
            'orbax.checkpoint._src.multihost.multislice.process_replica_id',
            return_value=replica_id,
        )
    )
    self.enter_context(
        mock.patch(
            'orbax.checkpoint._src.multihost.multislice.in_replica',
            return_value=in_primary_slice,
        )
    )
    self.enter_context(
        mock.patch(
            'orbax.checkpoint._src.multihost.multislice.replica_devices',
            return_value=self.mesh.devices.flatten(),
        )
    )
    self.enter_context(
        mock.patch(
            'orbax.checkpoint._src.multihost.multislice.primary_process_in_replica',
            return_value=0,
        )
    )
    active_processes = {process_index}
    self.enter_context(
        mock.patch(
            'orbax.checkpoint._src.multihost.multihost.unique_processes_from_devices',
            return_value=active_processes,
        )
    )
    self.enter_context(
        mock.patch.object(
            multihost, 'process_index', return_value=process_index
        )
    )

  def test_init_in_primary_slice(self):
    self._patch_process_index(process_index=0)
    manager = persistent.PersistentCheckpointManager(
        self.directory, self.mesh, replica_axis_index=0, options=self.options
    )
    self.assertTrue(manager._in_primary_slice)
    self.assertIsNotNone(manager._manager)
    manager.close()

  def test_init_not_in_primary_slice(self):
    self._patch_process_index(
        process_index=2, in_primary_slice=False, replica_id=1
    )
    manager = persistent.PersistentCheckpointManager(
        self.directory, self.mesh, replica_axis_index=0, options=self.options
    )
    self.assertFalse(manager._in_primary_slice)
    self.assertIsNotNone(manager._manager)
    manager.close()

  def test_save_in_primary_slice_saves(self):
    self._patch_process_index(process_index=0)
    manager = persistent.PersistentCheckpointManager(
        self.directory, self.mesh, replica_axis_index=0, options=self.options
    )
    manager._manager = mock.MagicMock()
    args = p2p_args_lib.Composite(
        state=args_lib.PyTreeSave({'a': jax.device_put(1)})
    )
    manager.save(1, args)
    manager._manager.save.assert_called_once()
    manager.close()

  def test_save_not_in_primary_slice_does_not_save(self):
    self._patch_process_index(
        process_index=2, in_primary_slice=False, replica_id=1
    )
    manager = persistent.PersistentCheckpointManager(
        self.directory, self.mesh, replica_axis_index=0, options=self.options
    )
    manager._manager = mock.MagicMock()
    args = p2p_args_lib.Composite(
        state=args_lib.PyTreeSave({'a': jax.device_put(1)})
    )
    manager.save(1, args)
    manager._manager.save.assert_not_called()
    manager.close()

  def test_save_and_restore(self):
    self._patch_process_index(process_index=0)
    # persistent checkpoint manager with multiprocessing only works with a
    # unified storage.
    self.enter_context(mock.patch.object(jax, 'process_count', return_value=1))
    devices = np.array([
        [MockDevice(0, 0)],
    ])
    mesh = mock.Mock(
        spec=jax.sharding.Mesh,
        devices=devices,
        axis_names=('replica', 'data'),
        shape={'replica': 1, 'data': 1},
        shape_tuple=devices.shape,
        size=devices.size,
    )
    manager = persistent.PersistentCheckpointManager(
        self.directory, mesh, replica_axis_index=0, options=self.options
    )

    arr = jax.device_put(np.arange(self.mesh.size, dtype=np.int32))
    state = {'a': arr, 'b': jax.device_put(1)}
    args = p2p_args_lib.Composite(state=args_lib.PyTreeSave(state))

    self.assertTrue(manager.save(1, args))
    manager.wait_until_finished()

    self.assertFalse((self.directory / '1' / 'default').exists())
    self.assertTrue((self.directory / '1' / 'state').exists())

    def _to_abstract(x):
      if isinstance(x, jax.Array):
        return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding)
      return x

    abstract_state = jax.tree.map(_to_abstract, state)
    restored = manager.restore(
        1, args=p2p_args_lib.Composite(state=abstract_state)
    )
    restored_state = restored.state
    test_utils.assert_tree_equal(self, state, restored_state)
    manager.close()

  def test_delete_in_primary_slice_deletes(self):
    self._patch_process_index(process_index=0)
    manager = persistent.PersistentCheckpointManager(
        self.directory, self.mesh, replica_axis_index=0, options=self.options
    )
    manager._manager = mock.MagicMock()
    manager.delete(1)
    manager._manager.delete.assert_called_once_with(1)
    manager.close()

  def test_delete_not_in_primary_slice_does_not_delete(self):
    self._patch_process_index(
        process_index=2, in_primary_slice=False, replica_id=1
    )
    manager = persistent.PersistentCheckpointManager(
        self.directory, self.mesh, replica_axis_index=0, options=self.options
    )
    manager._manager = mock.MagicMock()
    manager.delete(1)
    manager._manager.delete.assert_not_called()
    manager.close()

  def test_wait_until_finished_calls_manager(self):
    self._patch_process_index(process_index=0)
    manager = persistent.PersistentCheckpointManager(
        self.directory, self.mesh, replica_axis_index=0, options=self.options
    )
    manager._manager = mock.MagicMock()
    manager.wait_until_finished()
    manager._manager.wait_until_finished.assert_called_once()
    manager.close()

  def test_check_for_errors_calls_manager(self):
    self._patch_process_index(process_index=0)
    manager = persistent.PersistentCheckpointManager(
        self.directory, self.mesh, replica_axis_index=0, options=self.options
    )
    manager._manager = mock.MagicMock()
    manager.check_for_errors()
    manager._manager.check_for_errors.assert_called_once()
    manager.close()

  def test_close_calls_manager(self):
    self._patch_process_index(process_index=0)
    manager = persistent.PersistentCheckpointManager(
        self.directory, self.mesh, replica_axis_index=0, options=self.options
    )
    manager._manager = mock.MagicMock()
    manager.close()
    manager._manager.close.assert_called_once()


if __name__ == '__main__':
  absltest.main()
