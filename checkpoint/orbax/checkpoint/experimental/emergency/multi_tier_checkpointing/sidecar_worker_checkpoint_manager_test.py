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

"""Unit tests for the worker-side colocated checkpoint manager."""

from unittest import mock

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import args as args_lib
from orbax.checkpoint.experimental.emergency import replicator_checkpoint_manager as rcm_lib
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import (
    sidecar_worker_checkpoint_manager as sidecar_lib,
)


class SidecarWorkerCheckpointManagerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.manager = sidecar_lib.WorkerCheckpointManagerRaw.__new__(
        sidecar_lib.WorkerCheckpointManagerRaw
    )
    self.manager._rcm = mock.Mock()

  def test_save_delegates_to_rcm_and_packs_result(self):
    state = {'weights': np.arange(4)}
    step_array = jnp.asarray(7, dtype=np.int32)
    force_array = jnp.asarray(True)
    packed_result = object()
    self.manager._rcm.save.return_value = True

    with mock.patch.object(
        sidecar_lib.colocated_utils,
        'make_scalar_on_like',
        return_value=packed_result,
    ) as mock_make_scalar:
      result = self.manager.save(step_array, force_array, state)

    self.assertIs(result, packed_result)
    self.manager._rcm.save.assert_called_once()
    args, kwargs = self.manager._rcm.save.call_args
    self.assertEqual(args[0], 7)
    self.assertTrue(kwargs['force'])
    self.assertIsInstance(kwargs['args'], args_lib.Composite)
    self.assertIsInstance(kwargs['args']['state'], args_lib.PyTreeSave)
    self.assertIs(kwargs['args']['state'].item, state)
    mock_make_scalar.assert_called_once_with(True, step_array, dtype=jnp.bool_)

  def test_should_save_delegates_to_rcm_and_packs_result(self):
    step_array = jnp.asarray(11, dtype=np.int32)
    packed_result = object()
    self.manager._rcm.should_save.return_value = True

    with mock.patch.object(
        sidecar_lib.colocated_utils,
        'make_scalar_on_like',
        return_value=packed_result,
    ) as mock_make_scalar:
      result = self.manager.should_save(step_array)

    self.assertIs(result, packed_result)
    self.manager._rcm.should_save.assert_called_once_with(11)
    mock_make_scalar.assert_called_once_with(True, step_array, dtype=jnp.bool_)

  def test_restore_infer_unwraps_state(self):
    restored_state = {'weights': np.arange(2, dtype=np.float32)}
    self.manager._rcm.restore.return_value = args_lib.Composite(
        state=restored_state
    )

    result = self.manager.restore_infer(jnp.asarray(5, dtype=np.int32))

    self.assertEqual(result, restored_state)
    self.manager._rcm.restore.assert_called_once()
    args, _ = self.manager._rcm.restore.call_args
    self.assertEqual(args[0], 5)
    self.assertIsInstance(args[1], args_lib.Composite)
    self.assertIsInstance(args[1]['state'], args_lib.PyTreeRestore)
    self.assertIsNone(args[1]['state'].item)
    self.assertIsNone(args[1]['state'].restore_args)

  def test_restore_infer_negative_step_requests_latest(self):
    self.manager._rcm.restore.return_value = args_lib.Composite(state={'x': 1})

    self.manager.restore_infer(jnp.asarray(-1, dtype=np.int32))

    args, _ = self.manager._rcm.restore.call_args
    self.assertIsNone(args[0])

  def test_latest_step_delegates_to_rcm_and_packs_result(self):
    dummy_array = jnp.asarray(True)
    packed_result = object()
    self.manager._rcm.latest_step.return_value = 9

    with mock.patch.object(
        sidecar_lib.colocated_utils,
        'make_scalar_on_like',
        return_value=packed_result,
    ) as mock_make_scalar:
      result = self.manager.latest_step(dummy_array)

    self.assertIs(result, packed_result)
    self.manager._rcm.latest_step.assert_called_once_with()
    mock_make_scalar.assert_called_once_with(9, dummy_array, dtype=jnp.int32)

  def test_latest_step_returns_zero_when_no_step_exists(self):
    dummy_array = jnp.asarray(True)
    packed_result = object()
    self.manager._rcm.latest_step.return_value = None

    with mock.patch.object(
        sidecar_lib.colocated_utils,
        'make_scalar_on_like',
        return_value=packed_result,
    ) as mock_make_scalar:
      result = self.manager.latest_step(dummy_array)

    self.assertIs(result, packed_result)
    mock_make_scalar.assert_called_once_with(0, dummy_array, dtype=jnp.int32)

  def test_is_saving_in_progress_delegates_to_rcm_and_packs_result(self):
    dummy_array = jnp.asarray(True)
    packed_result = object()
    self.manager._rcm.is_saving_in_progress.return_value = True

    with mock.patch.object(
        sidecar_lib.colocated_utils,
        'make_scalar_on_like',
        return_value=packed_result,
    ) as mock_make_scalar:
      result = self.manager.is_saving_in_progress(dummy_array)

    self.assertIs(result, packed_result)
    self.manager._rcm.is_saving_in_progress.assert_called_once_with()
    mock_make_scalar.assert_called_once_with(True, dummy_array, dtype=jnp.bool_)

  def test_wait_until_finished_blocks_and_packs_true_result(self):
    dummy_array = jnp.asarray(True)
    packed_result = object()

    with mock.patch.object(
        sidecar_lib.colocated_utils,
        'make_scalar_on_like',
        return_value=packed_result,
    ) as mock_make_scalar:
      result = self.manager.wait_until_finished(dummy_array)

    self.assertIs(result, packed_result)
    self.manager._rcm.wait_until_finished.assert_called_once_with()
    mock_make_scalar.assert_called_once_with(True, dummy_array, dtype=jnp.bool_)

  def test_init_reconstructs_cpu_mesh_from_local_devices(self):
    local_devices = [mock.Mock(id=1), mock.Mock(id=2)]
    cpu_mesh = object()

    with mock.patch.object(
        sidecar_lib.colocated_transport,
        'install_pathways_colocated_serialization_patch',
    ) as mock_install_patch, mock.patch.object(
        sidecar_lib.signaling_client,
        'mark_pathways_colocated_runtime_active',
    ) as mock_mark_sidecar_runtime, mock.patch.object(
        sidecar_lib,
        'epath',
    ) as mock_epath, mock.patch.object(
        sidecar_lib.jax,
        'devices',
        return_value=local_devices,
    ) as mock_devices, mock.patch.object(
        sidecar_lib.jax.sharding,
        'Mesh',
        return_value=cpu_mesh,
    ) as mock_mesh, mock.patch.object(
        rcm_lib, 'ReplicatorCheckpointManager'
    ) as mock_rcm, mock.patch.object(
        rcm_lib, 'ReplicatorCheckpointManagerOptions', return_value='options'
    ):
      mock_epath.Path.return_value = '/tmp/local'
      sidecar_lib.WorkerCheckpointManagerRaw(
          local_directory='/tmp/local',
          mesh_shape=(2,),
          mesh_axis_names=('x',),
          mesh_axis_types=None,
          save_interval_steps=3,
      )

    mock_install_patch.assert_called_once_with()
    mock_mark_sidecar_runtime.assert_called_once_with()
    mock_devices.assert_called_once_with()
    mock_mesh.assert_called_once()
    np.testing.assert_array_equal(
        mock_mesh.call_args.args[0], np.array(local_devices)
    )
    self.assertEqual(mock_mesh.call_args.args[1], ('x',))
    self.assertIsNone(mock_mesh.call_args.kwargs['axis_types'])
    mock_rcm.assert_called_once()
    self.assertIs(mock_rcm.call_args.kwargs['global_mesh'], cpu_mesh)
    self.assertTrue(mock_rcm.call_args.kwargs['_is_sidecar'])

  def test_close_delegates_to_rcm_and_packs_result(self):
    dummy_array = jnp.asarray(True)
    packed_result = object()

    with mock.patch.object(
        sidecar_lib.colocated_utils,
        'make_scalar_on_like',
        return_value=packed_result,
    ):
      result = self.manager.close(dummy_array)

    self.assertIs(result, packed_result)
    self.manager._rcm.close.assert_called_once()


if __name__ == '__main__':
  absltest.main()
