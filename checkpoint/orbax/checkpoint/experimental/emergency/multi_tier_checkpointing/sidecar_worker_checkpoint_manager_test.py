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
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import args as args_lib
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import replicator_checkpoint_manager as rcm_lib
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
    self.manager._enable_async_checkpointing = True
    self.manager._save_concurrent_gb = None
    self.manager._restore_concurrent_gb = None

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

  def test_save_skips_protocol_no_checkpoint_sentinel(self):
    state = {'weights': np.arange(4)}
    step_array = jnp.asarray(
        sidecar_lib.colocated_utils.NO_STEP_SENTINEL, dtype=np.int32
    )
    force_array = jnp.asarray(True)
    packed_result = object()

    with mock.patch.object(
        sidecar_lib.colocated_utils,
        'make_scalar_on_like',
        return_value=packed_result,
    ) as mock_make_scalar:
      result = self.manager.save(step_array, force_array, state)

    self.assertIs(result, packed_result)
    self.manager._rcm.save.assert_not_called()
    mock_make_scalar.assert_called_once_with(False, step_array, dtype=jnp.bool_)

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

  def test_should_save_skips_protocol_no_checkpoint_sentinel(self):
    step_array = jnp.asarray(
        sidecar_lib.colocated_utils.NO_STEP_SENTINEL, dtype=np.int32
    )
    packed_result = object()

    with mock.patch.object(
        sidecar_lib.colocated_utils,
        'make_scalar_on_like',
        return_value=packed_result,
    ) as mock_make_scalar:
      result = self.manager.should_save(step_array)

    self.assertIs(result, packed_result)
    self.manager._rcm.should_save.assert_not_called()
    mock_make_scalar.assert_called_once_with(False, step_array, dtype=jnp.bool_)

  def test_restore_infer_unwraps_state(self):
    restored_state = {'weights': np.arange(2, dtype=np.float32)}
    self.manager._rcm.restore.return_value = args_lib.Composite(
        state=restored_state
    )

    result = self.manager.restore_infer(
        jnp.asarray(5, dtype=np.int32), jnp.asarray(True)
    )

    self.assertEqual(result, restored_state)
    self.manager._rcm.restore.assert_called_once()
    args, _ = self.manager._rcm.restore.call_args
    self.assertEqual(args[0], 5)
    self.assertIsInstance(args[1], args_lib.Composite)
    self.assertIsInstance(args[1]['state'], args_lib.PyTreeRestore)
    self.assertIsNone(args[1]['state'].item)
    self.assertIsNone(args[1]['state'].restore_args)
    self.assertTrue(args[1]['state'].partial_restore)

  def test_restore_infer_negative_step_requests_latest(self):
    self.manager._rcm.restore.return_value = args_lib.Composite(state={'x': 1})

    self.manager.restore_infer(
        jnp.asarray(-1, dtype=np.int32), jnp.asarray(False)
    )

    args, _ = self.manager._rcm.restore.call_args
    self.assertIsNone(args[0])

  def test_restore_infer_rejects_protocol_no_checkpoint_sentinel(self):
    with self.assertRaisesRegex(ValueError, 'cannot restore step 0'):
      self.manager.restore_infer(
          jnp.asarray(
              sidecar_lib.colocated_utils.NO_STEP_SENTINEL, dtype=np.int32
          ),
          jnp.asarray(False),
      )

    self.manager._rcm.restore.assert_not_called()

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

  def test_latest_step_returns_sentinel_when_no_step_exists(self):
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
    mock_make_scalar.assert_called_once_with(
        sidecar_lib.colocated_utils.NO_STEP_SENTINEL,
        dummy_array,
        dtype=jnp.int32,
    )

  def test_all_steps_returns_sorted_fixed_size_array(self):
    dummy_array = jnp.asarray(0, dtype=jnp.int32)
    self.manager._rcm.all_steps.return_value = [5, 1, 4]

    result = self.manager.all_steps(dummy_array)

    steps_array = np.asarray(result)
    max_steps = sidecar_lib.colocated_utils.MAX_TRACKED_STEPS
    self.assertEqual(steps_array.shape, (max_steps,))
    self.assertEqual(steps_array.dtype, np.int32)
    expected = [1, 4, 5] + [sidecar_lib.colocated_utils.NO_STEP_SENTINEL] * (
        max_steps - 3
    )
    np.testing.assert_array_equal(
        steps_array, np.asarray(expected, dtype=np.int32)
    )
    self.manager._rcm.all_steps.assert_called_once_with()

  def test_all_steps_result_uses_replicated_sharding(self):
    device = jax.devices()[0]
    mesh = jax.sharding.Mesh(np.array([device]), ('worker',))
    worker_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('worker')
    )
    dummy_array = jax.device_put(
        jnp.asarray([0], dtype=jnp.int32), worker_sharding
    )
    self.manager._rcm.all_steps.return_value = [5]

    result = self.manager.all_steps(dummy_array)

    self.assertEqual(result.sharding.mesh, mesh)
    self.assertEqual(result.sharding.spec, jax.sharding.PartitionSpec())

  def test_all_steps_does_not_use_device_put(self):
    dummy_array = jnp.asarray(0, dtype=jnp.int32)
    self.manager._rcm.all_steps.return_value = [5]

    with mock.patch.object(
        sidecar_lib.jax, 'device_put', side_effect=AssertionError
    ):
      result = self.manager.all_steps(dummy_array)

    np.testing.assert_array_equal(
        np.asarray(result)[:2],
        np.asarray([5, sidecar_lib.colocated_utils.NO_STEP_SENTINEL]),
    )

  def test_array_on_sharding_returns_requested_slice(self):
    value = np.arange(8, dtype=np.int32)
    captured = {}

    def fake_make_array_from_callback(shape, sharding, data_callback, *, dtype):
      del sharding
      captured['shape'] = shape
      captured['dtype'] = dtype
      return data_callback((slice(2, 5),))

    with mock.patch.object(
        sidecar_lib.jax,
        'make_array_from_callback',
        side_effect=fake_make_array_from_callback,
    ):
      result = sidecar_lib._array_on_sharding(value, mock.sentinel.sharding)

    self.assertEqual(captured['shape'], value.shape)
    self.assertEqual(captured['dtype'], value.dtype)
    np.testing.assert_array_equal(result, np.asarray([2, 3, 4], dtype=np.int32))

  def test_all_steps_limits_to_latest_max_steps(self):
    dummy_array = jnp.asarray(0, dtype=jnp.int32)
    max_steps = sidecar_lib.colocated_utils.MAX_TRACKED_STEPS
    self.manager._rcm.all_steps.return_value = list(range(1, max_steps + 3))

    result = self.manager.all_steps(dummy_array)

    steps_array = np.asarray(result)
    self.assertEqual(steps_array.shape, (max_steps,))
    expected = list(range(3, max_steps + 3))
    np.testing.assert_array_equal(
        steps_array, np.asarray(expected, dtype=np.int32)
    )

  def test_all_steps_returns_all_sentinels_when_no_steps_exist(self):
    dummy_array = jnp.asarray(0, dtype=jnp.int32)
    self.manager._rcm.all_steps.return_value = []

    result = self.manager.all_steps(dummy_array)

    steps_array = np.asarray(result)
    max_steps = sidecar_lib.colocated_utils.MAX_TRACKED_STEPS
    self.assertEqual(steps_array.shape, (max_steps,))
    expected = [sidecar_lib.colocated_utils.NO_STEP_SENTINEL] * max_steps
    np.testing.assert_array_equal(
        steps_array, np.asarray(expected, dtype=np.int32)
    )

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

  def test_init_bootstraps_sidecar_components(self):
    local_devices = [
        mock.Mock(spec=jax.Device, id=11),
        mock.Mock(spec=jax.Device, id=12),
    ]
    with mock.patch.object(
        sidecar_lib.colocated_transport,
        'install_pathways_colocated_serialization_patch',
    ) as mock_install_patch, mock.patch.object(
        sidecar_lib.signaling_client,
        'mark_pathways_colocated_runtime_active',
    ) as mock_mark_sidecar_runtime, mock.patch.object(
        sidecar_lib, 'epath'
    ), mock.patch.object(
        sidecar_lib.colocated_transport,
        'resolve_colocated_cpu_devices',
        return_value=local_devices,
    ) as mock_resolve_cpu_devices, mock.patch.object(
        sidecar_lib.jax.sharding, 'Mesh'
    ), mock.patch.object(
        rcm_lib, 'ReplicatorCheckpointManager'
    ), mock.patch.object(
        rcm_lib, 'ReplicatorCheckpointManagerOptions'
    ):
      sidecar_lib.WorkerCheckpointManagerRaw(
          local_directory='/tmp/local',
          mesh_shape=(2,),
          mesh_axis_names=('x',),
          save_interval_steps=3,
          mesh_device_ids=(101, 102),
          mesh_axis_types=None,
          distributed_to_device_ids=((101, 102),),
          enable_async_checkpointing=False,
      )
    mock_install_patch.assert_called_once_with()
    mock_mark_sidecar_runtime.assert_called_once_with()
    mock_resolve_cpu_devices.assert_called_once_with((101, 102))

  def test_init_reconstructs_cpu_mesh(self):
    local_devices = [
        mock.Mock(spec=jax.Device, id=11),
        mock.Mock(spec=jax.Device, id=12),
    ]
    cpu_mesh = object()
    with mock.patch.object(
        sidecar_lib.colocated_transport,
        'install_pathways_colocated_serialization_patch',
    ), mock.patch.object(
        sidecar_lib.signaling_client,
        'mark_pathways_colocated_runtime_active',
    ), mock.patch.object(
        sidecar_lib, 'epath'
    ), mock.patch.object(
        sidecar_lib.colocated_transport,
        'resolve_colocated_cpu_devices',
        return_value=local_devices,
    ), mock.patch.object(
        sidecar_lib.jax.sharding, 'Mesh', return_value=cpu_mesh
    ) as mock_mesh, mock.patch.object(
        rcm_lib, 'ReplicatorCheckpointManager'
    ), mock.patch.object(
        rcm_lib, 'ReplicatorCheckpointManagerOptions'
    ):
      sidecar_lib.WorkerCheckpointManagerRaw(
          local_directory='/tmp/local',
          mesh_shape=(2,),
          mesh_axis_names=('x',),
          save_interval_steps=3,
          mesh_device_ids=(101, 102),
          mesh_axis_types=None,
          distributed_to_device_ids=((101, 102),),
          enable_async_checkpointing=False,
      )
    mock_mesh.assert_called_once()
    np.testing.assert_array_equal(
        mock_mesh.call_args.args[0], np.array(local_devices)
    )
    self.assertEqual(mock_mesh.call_args.args[1], ('x',))
    self.assertIsNone(mock_mesh.call_args.kwargs['axis_types'])

  def test_init_configures_replicator_checkpoint_manager(self):
    cpu_mesh = object()
    local_devices = [
        mock.Mock(spec=jax.Device, id=11),
        mock.Mock(spec=jax.Device, id=12),
    ]
    with mock.patch.object(
        sidecar_lib.colocated_transport,
        'install_pathways_colocated_serialization_patch',
    ), mock.patch.object(
        sidecar_lib.signaling_client,
        'mark_pathways_colocated_runtime_active',
    ), mock.patch.object(
        sidecar_lib, 'epath'
    ), mock.patch.object(
        sidecar_lib.colocated_transport,
        'resolve_colocated_cpu_devices',
        return_value=local_devices,
    ), mock.patch.object(
        sidecar_lib.jax.sharding, 'Mesh', return_value=cpu_mesh
    ), mock.patch.object(
        rcm_lib, 'ReplicatorCheckpointManager'
    ) as mock_rcm, mock.patch.object(
        rcm_lib, 'ReplicatorCheckpointManagerOptions', return_value='options'
    ) as mock_options:
      sidecar_lib.WorkerCheckpointManagerRaw(
          local_directory='/tmp/local',
          mesh_shape=(2,),
          mesh_axis_names=('x',),
          save_interval_steps=3,
          mesh_device_ids=(101, 102),
          mesh_axis_types=None,
          distributed_to_device_ids=((101, 102),),
          enable_async_checkpointing=False,
      )
    mock_options.assert_called_once_with(
        save_interval_steps=3,
        enable_async_checkpointing=False,
        save_concurrent_gb=None,
        restore_concurrent_gb=None,
    )
    mock_rcm.assert_called_once()
    self.assertIs(mock_rcm.call_args.kwargs['global_mesh'], cpu_mesh)
    self.assertTrue(mock_rcm.call_args.kwargs['_is_sidecar'])
    distributed_to_device_ids_fn = mock_rcm.call_args.kwargs[
        '_distributed_to_device_ids_fn'
    ]
    self.assertEqual(distributed_to_device_ids_fn(), [[11, 12]])

  def test_init_rejects_duplicate_mesh_device_ids(self):
    with mock.patch.object(
        sidecar_lib.colocated_transport,
        'install_pathways_colocated_serialization_patch',
    ), mock.patch.object(
        sidecar_lib.signaling_client,
        'mark_pathways_colocated_runtime_active',
    ):
      with self.assertRaisesRegex(ValueError, 'mesh_device_ids must be unique'):
        sidecar_lib.WorkerCheckpointManagerRaw(
            local_directory='/tmp/local',
            mesh_shape=(2,),
            mesh_axis_names=('x',),
            save_interval_steps=3,
            mesh_device_ids=(101, 101),
        )

  def test_init_rejects_duplicate_remapped_distributed_ids(self):
    local_devices = [
        mock.Mock(spec=jax.Device, id=11),
        mock.Mock(spec=jax.Device, id=12),
    ]

    with mock.patch.object(
        sidecar_lib.colocated_transport,
        'install_pathways_colocated_serialization_patch',
    ), mock.patch.object(
        sidecar_lib.signaling_client,
        'mark_pathways_colocated_runtime_active',
    ), mock.patch.object(
        sidecar_lib.colocated_transport,
        'resolve_colocated_cpu_devices',
        return_value=local_devices,
    ), mock.patch.object(
        sidecar_lib.jax.sharding,
        'Mesh',
        return_value=mock.Mock(),
    ):
      with self.assertRaisesRegex(
          ValueError, 'distributed_to_device_ids must be unique'
      ):
        sidecar_lib.WorkerCheckpointManagerRaw(
            local_directory='/tmp/local',
            mesh_shape=(2,),
            mesh_axis_names=('x',),
            save_interval_steps=3,
            mesh_device_ids=(101, 102),
            distributed_to_device_ids=((101,), (101,)),
        )

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
