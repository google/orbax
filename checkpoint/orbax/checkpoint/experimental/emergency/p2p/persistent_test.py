# Copyright 2025 The Orbax Authors.
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
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint.experimental.emergency.p2p import persistent

Mesh = jax.sharding.Mesh


class MockDevice:

  def __init__(self, process_index):
    self.process_index = process_index

  def __repr__(self):
    return f'MockDevice(pi={self.process_index})'


class PersistentCheckpointManagerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(multihost, 'get_jax_distributed_client')
    )
    self.directory = epath.Path(self.create_tempdir().full_path)
    devices = np.array([
        [MockDevice(0), MockDevice(1)],
        [MockDevice(2), MockDevice(3)],
    ])
    self.mesh = mock.Mock(
        spec=jax.sharding.Mesh,
        devices=devices,
        axis_names=('x',),
        shape={'x': devices.shape[0]},
        shape_tuple=devices.shape,
        size=devices.size,
    )
    self.options = checkpoint_manager.CheckpointManagerOptions()

  def _patch_multihost_multislice(
      self, in_primary_slice=True, process_index=0, replica_id=0
  ):
    self.enter_context(
        mock.patch(
            'orbax.checkpoint._src.multihost.multihost.process_index',
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
    self.enter_context(
        mock.patch(
            'orbax.checkpoint._src.multihost.multihost.unique_processes_from_devices',
            return_value={0},
        )
    )

  def test_init_in_primary_slice(self):
    self._patch_multihost_multislice(in_primary_slice=True)
    manager = persistent.PersistentCheckpointManager(
        self.directory, self.mesh, replica_axis_index=0, options=self.options
    )
    self.assertTrue(manager._in_primary_slice)
    self.assertIsNotNone(manager._manager)
    manager.close()

  def test_init_not_in_primary_slice(self):
    self._patch_multihost_multislice(in_primary_slice=False)
    manager = persistent.PersistentCheckpointManager(
        self.directory, self.mesh, replica_axis_index=0, options=self.options
    )
    self.assertFalse(manager._in_primary_slice)
    self.assertIsNotNone(manager._manager)
    manager.close()

  def test_save_in_primary_slice_saves(self):
    self._patch_multihost_multislice(in_primary_slice=True)
    manager = persistent.PersistentCheckpointManager(
        self.directory, self.mesh, replica_axis_index=0, options=self.options
    )
    manager._manager = mock.MagicMock()
    args = args_lib.Composite(
        state=args_lib.PyTreeSave({'a': jax.device_put(1)})
    )
    manager.save(1, args)
    manager._manager.save.assert_called_once()
    manager.close()

  def test_save_not_in_primary_slice_does_not_save(self):
    self._patch_multihost_multislice(in_primary_slice=False)
    manager = persistent.PersistentCheckpointManager(
        self.directory, self.mesh, replica_axis_index=0, options=self.options
    )
    manager._manager = mock.MagicMock()
    args = args_lib.Composite(
        state=args_lib.PyTreeSave({'a': jax.device_put(1)})
    )
    manager.save(1, args)
    manager._manager.save.assert_not_called()
    manager.close()

  def test_save_and_restore(self):
    self._patch_multihost_multislice(in_primary_slice=True)
    manager = persistent.PersistentCheckpointManager(
        self.directory, self.mesh, replica_axis_index=0, options=self.options
    )

    arr = jax.device_put(np.arange(self.mesh.size, dtype=np.int32))
    state = {'a': arr, 'b': jax.device_put(1)}
    args = args_lib.Composite(state=args_lib.PyTreeSave(state))

    self.assertTrue(manager.save(1, args))
    manager.wait_until_finished()

    def _to_abstract(x):
      if isinstance(x, jax.Array):
        return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding)
      return x

    abstract_state = jax.tree.map(_to_abstract, state)
    restored = manager.restore(1, args=args_lib.Composite(state=abstract_state))
    restored_state = restored.state

    self.assertIsInstance(restored_state['a'], jax.Array)
    self.assertIsInstance(restored_state['b'], jax.Array)
    np.testing.assert_array_equal(state['a'], restored_state['a'])
    self.assertEqual(state['b'], restored_state['b'])
    manager.close()

  def test_delete_in_primary_slice_deletes(self):
    self._patch_multihost_multislice(in_primary_slice=True)
    manager = persistent.PersistentCheckpointManager(
        self.directory, self.mesh, replica_axis_index=0, options=self.options
    )
    manager._manager = mock.MagicMock()
    manager.delete(1)
    manager._manager.delete.assert_called_once_with(1)
    manager.close()

  def test_delete_not_in_primary_slice_does_not_delete(self):
    self._patch_multihost_multislice(in_primary_slice=False)
    manager = persistent.PersistentCheckpointManager(
        self.directory, self.mesh, replica_axis_index=0, options=self.options
    )
    manager._manager = mock.MagicMock()
    manager.delete(1)
    manager._manager.delete.assert_not_called()
    manager.close()

  def test_wait_until_finished_calls_manager(self):
    self._patch_multihost_multislice(in_primary_slice=True)
    manager = persistent.PersistentCheckpointManager(
        self.directory, self.mesh, replica_axis_index=0, options=self.options
    )
    manager._manager = mock.MagicMock()
    manager.wait_until_finished()
    manager._manager.wait_until_finished.assert_called_once()
    manager.close()

  def test_check_for_errors_calls_manager(self):
    self._patch_multihost_multislice(in_primary_slice=True)
    manager = persistent.PersistentCheckpointManager(
        self.directory, self.mesh, replica_axis_index=0, options=self.options
    )
    manager._manager = mock.MagicMock()
    manager.check_for_errors()
    manager._manager.check_for_errors.assert_called_once()
    manager.close()

  def test_close_calls_manager(self):
    self._patch_multihost_multislice(in_primary_slice=True)
    manager = persistent.PersistentCheckpointManager(
        self.directory, self.mesh, replica_axis_index=0, options=self.options
    )
    manager._manager = mock.MagicMock()
    manager.close()
    manager._manager.close.assert_called_once()


if __name__ == '__main__':
  absltest.main()
