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

"""Internal unit tests for colocated controller helpers."""

from __future__ import annotations

from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import args as args_lib
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import (
    colocated_controller as controller_lib,
)
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import (
    sidecar_worker_checkpoint_manager,
)


class ColocatedControllerInternalTest(absltest.TestCase):

  def test_step_arr_on_state_devices_supports_single_device_sharding(self):
    device = jax.devices()[0]
    sharding = jax.sharding.SingleDeviceSharding(device)
    state = {'x': jax.device_put(jnp.arange(2, dtype=jnp.int32), sharding)}

    step_arr = controller_lib._step_arr_on_state_devices(4, state)

    self.assertEqual(np.asarray(step_arr).item(), 4)
    self.assertIsInstance(step_arr.sharding, jax.sharding.SingleDeviceSharding)

  def test_init_preserves_global_mesh_device_order(self):
    fake_device_1 = mock.Mock(id=1)
    fake_device_2 = mock.Mock(id=2)
    fake_mesh = mock.Mock()
    fake_mesh.devices = np.array([fake_device_2, fake_device_1])
    fake_mesh.axis_names = ('x',)
    fake_mesh.axis_types = (jax.sharding.AxisType.Auto,)
    worker_manager = mock.Mock()
    dummy = object()
    cpu_devices = (mock.Mock(id=101), mock.Mock(id=102))
    with mock.patch.object(
        sidecar_worker_checkpoint_manager,
        'WorkerCheckpointManager',
        return_value=worker_manager,
    ) as mock_worker_manager, mock.patch.object(
        controller_lib.colocated_transport,
        'install_pathways_colocated_serialization_patch',
    ) as mock_install_patch, mock.patch.object(
        controller_lib.colocated_transport,
        'unique_colocated_cpu_devices',
        return_value=cpu_devices,
    ) as mock_unique_cpus, mock.patch.object(
        controller_lib.dispatchers,
        'get_dummy_input_array',
        return_value=dummy,
    ), mock.patch.object(
        controller_lib.ColocatedController,
        '_specialize_scalar_worker_calls',
        autospec=True,
    ):
      controller = controller_lib.ColocatedController(
          local_directory=mock.Mock(),
          global_mesh=fake_mesh,
          options=mock.Mock(save_interval_steps=3),
          persistent_directory=None,
          handler_registry=None,
          checkpoint_manager_options_fn=mock.Mock(),
      )

    self.assertIs(controller._dummy, dummy)
    mock_install_patch.assert_called_once_with()
    mock_unique_cpus.assert_called_once_with((fake_device_2, fake_device_1))
    kwargs = mock_worker_manager.call_args.kwargs
    self.assertEqual(kwargs['mesh_shape'], (2,))
    self.assertEqual(kwargs['mesh_axis_names'], ('x',))
    self.assertEqual(kwargs['mesh_axis_types'], tuple(fake_mesh.axis_types))
    self.assertEqual(kwargs['save_interval_steps'], 3)
    self.assertIsInstance(kwargs['local_directory'], str)

  def test_init_rejects_empty_colocated_cpu_devices(self):
    fake_mesh = mock.Mock()
    fake_mesh.devices = np.array([mock.Mock(id=1)])
    fake_mesh.axis_names = ('x',)
    fake_mesh.axis_types = (jax.sharding.AxisType.Auto,)

    with mock.patch.object(
        sidecar_worker_checkpoint_manager,
        'WorkerCheckpointManager',
        return_value=mock.Mock(),
    ), mock.patch.object(
        controller_lib.colocated_transport,
        'install_pathways_colocated_serialization_patch',
    ), mock.patch.object(
        controller_lib.colocated_transport,
        'unique_colocated_cpu_devices',
        return_value=(),
    ), self.assertRaisesRegex(
        ValueError, 'requires at least one colocated CPU device'
    ):
      controller_lib.ColocatedController(
          local_directory=mock.Mock(),
          global_mesh=fake_mesh,
          options=mock.Mock(save_interval_steps=3),
          persistent_directory=None,
          handler_registry=None,
          checkpoint_manager_options_fn=mock.Mock(),
      )

  def _make_controller_for_specialization(self):
    controller = controller_lib.ColocatedController.__new__(
        controller_lib.ColocatedController
    )
    device = jax.devices()[0]
    controller._worker_save_call = None
    controller._colocated_cpu_devices = (device,)
    controller._colocated_cpu_ids = frozenset({device.id})
    return controller, device

  def _make_controller_for_restore(self):
    controller, device = self._make_controller_for_specialization()
    controller._dummy = jax.device_put(
        jnp.array(True), jax.sharding.SingleDeviceSharding(device)
    )
    controller._persistent_checkpoint_manager = None
    mesh = jax.sharding.Mesh(np.array([device]), ('d',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    return controller, sharding

  def test_get_worker_save_call_specializes_with_cpu_specs_and_devices(self):
    controller, device = self._make_controller_for_specialization()
    mesh = jax.sharding.Mesh(np.array([device]), ('d',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    step_arr = jax.device_put(jnp.array(1, dtype=jnp.int32), sharding)
    force_arr = jax.device_put(jnp.array(True, dtype=jnp.bool_), sharding)
    state = {'x': jax.device_put(jnp.arange(2, dtype=jnp.int32), sharding)}

    captured = {}
    sentinel = object()

    def _specialize_fn(*, in_specs, out_specs_fn, devices):
      captured['in_specs'] = in_specs
      captured['out_specs'] = out_specs_fn
      captured['devices'] = devices
      return sentinel

    controller._worker_manager = mock.Mock()
    controller._worker_manager.save.specialize = _specialize_fn

    result = controller._get_worker_save_call(step_arr, force_arr, state)

    self.assertIs(result, sentinel)
    self.assertEqual(captured['devices'], controller._colocated_cpu_devices)
    spec_leaves = [
        leaf
        for leaf in jax.tree.leaves(captured['in_specs'])
        if isinstance(leaf, jax.ShapeDtypeStruct)
    ]
    self.assertNotEmpty(spec_leaves)
    for leaf in spec_leaves:
      self.assertEqual({d.platform for d in leaf.sharding.device_set}, {'cpu'})
      self.assertEqual({d.id for d in leaf.sharding.device_set}, {device.id})

  def test_prepare_restore_target_shardings_uses_restore_args(self):
    controller, sharding = self._make_controller_for_restore()
    restore_args = {'x': type_handlers.ArrayRestoreArgs(sharding=sharding)}

    target_shardings = controller._prepare_restore_target_shardings(
        args_lib.PyTreeRestore(item=None, restore_args=restore_args),
    )

    self.assertEqual(target_shardings['x'], sharding)

  def test_rebuild_restored_state_unwraps_single_mapping_wrapper(self):
    controller, sharding = self._make_controller_for_restore()
    del sharding
    restored_state = {'params': {'weights': jnp.arange(2, dtype=jnp.float32)}}
    template_state = {'weights': jax.ShapeDtypeStruct((2,), jnp.float32)}

    rebuilt = controller._rebuild_restored_state(restored_state, template_state)

    self.assertEqual(
        jax.tree.structure(rebuilt), jax.tree.structure(template_state)
    )
    np.testing.assert_array_equal(
        rebuilt['weights'], restored_state['params']['weights']
    )

  def test_rebuild_restored_state_rejects_arbitrary_suffix_mapping(self):
    controller, sharding = self._make_controller_for_restore()
    del sharding
    restored_state = {
        'params': {'encoder': {'weights': jnp.arange(2, dtype=jnp.float32)}}
    }
    template_state = {'weights': jax.ShapeDtypeStruct((2,), jnp.float32)}

    with self.assertRaisesRegex(
        ValueError,
        'does not match the caller template structure',
    ):
      controller._rebuild_restored_state(restored_state, template_state)

  def test_latest_step_returns_none_for_worker_sentinel(self):
    controller, _ = self._make_controller_for_restore()
    controller._worker_manager = mock.Mock()
    controller._worker_manager.latest_step.return_value = object()

    with mock.patch.object(
        controller_lib.colocated_utils,
        'require_unanimous_scalar_result',
        return_value=0,
    ) as mock_unanimous:
      latest_step = controller.latest_step()

    self.assertIsNone(latest_step)
    controller._worker_manager.latest_step.assert_called_once_with(
        controller._dummy
    )
    mock_unanimous.assert_called_once()

  def test_latest_step_retries_transient_runtime_error(self):
    controller, _ = self._make_controller_for_restore()
    controller._worker_manager = mock.Mock()
    controller._worker_manager.latest_step.side_effect = [
        RuntimeError('transient'),
        object(),
    ]

    with mock.patch.object(
        controller_lib.colocated_utils,
        'require_unanimous_scalar_result',
        return_value=12,
    ), mock.patch.object(controller_lib.time, 'sleep') as mock_sleep:
      latest_step = controller.latest_step()

    self.assertEqual(latest_step, 12)
    self.assertEqual(controller._worker_manager.latest_step.call_count, 2)
    mock_sleep.assert_called_once_with(1)

  def test_should_save_returns_worker_vote(self):
    controller, _ = self._make_controller_for_restore()
    controller._worker_manager = mock.Mock()
    controller._worker_manager.should_save.return_value = object()

    with mock.patch.object(
        controller_lib.colocated_utils,
        'require_unanimous_scalar_result',
        return_value=True,
    ) as mock_unanimous:
      should_save = controller.should_save(9)

    self.assertTrue(should_save)
    controller._worker_manager.should_save.assert_called_once()
    step_arg = controller._worker_manager.should_save.call_args.args[0]
    self.assertEqual(np.asarray(step_arg).item(), 9)
    mock_unanimous.assert_called_once()

  def test_is_saving_in_progress_returns_worker_vote(self):
    controller, _ = self._make_controller_for_restore()
    controller._worker_manager = mock.Mock()
    controller._worker_manager.is_saving_in_progress.return_value = object()

    with mock.patch.object(
        controller_lib.colocated_utils,
        'require_unanimous_scalar_result',
        return_value=False,
    ) as mock_unanimous:
      in_progress = controller.is_saving_in_progress()

    self.assertFalse(in_progress)
    controller._worker_manager.is_saving_in_progress.assert_called_once_with(
        controller._dummy
    )
    mock_unanimous.assert_called_once()

  def test_wait_until_finished_blocks_on_worker_result(self):
    controller, _ = self._make_controller_for_restore()
    worker_result = object()
    controller._worker_manager = mock.Mock()
    controller._worker_manager.wait_until_finished.return_value = worker_result

    with mock.patch.object(
        controller_lib.jax, 'block_until_ready'
    ) as mock_block:
      controller.wait_until_finished()

    controller._worker_manager.wait_until_finished.assert_called_once_with(
        controller._dummy
    )
    mock_block.assert_called_once_with(worker_result)

  def test_close_shuts_down_persistent_and_worker_managers(self):
    controller, _ = self._make_controller_for_restore()
    worker_result = object()
    controller._worker_manager = mock.Mock()
    controller._worker_manager.close.return_value = worker_result
    controller._persistent_checkpoint_manager = mock.Mock()

    with mock.patch.object(
        controller_lib.jax, 'block_until_ready'
    ) as mock_block:
      controller.close()

    controller._persistent_checkpoint_manager.close.assert_called_once_with()
    controller._worker_manager.close.assert_called_once_with(controller._dummy)
    mock_block.assert_called_once_with(worker_result)

  def test_restore_uses_worker_restore_infer_and_reshards(self):
    controller, sharding = self._make_controller_for_restore()
    mesh = sharding.mesh
    cpu_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    restored_cpu_state = {
        'params': {
            'weights': jax.device_put(
                jnp.arange(2, dtype=jnp.float32),
                cpu_sharding,
            )
        }
    }
    template_state = {
        'weights': jax.ShapeDtypeStruct((2,), jnp.float32, sharding=sharding)
    }
    restore_args = {
        'weights': type_handlers.ArrayRestoreArgs(sharding=sharding)
    }
    controller._worker_manager = mock.Mock()
    controller._worker_manager.restore_infer.return_value = restored_cpu_state

    result = controller.restore(
        7,
        args_lib.Composite(
            state=args_lib.PyTreeRestore(
                item=template_state,
                restore_args=restore_args,
            )
        ),
        default_item_mode=True,
    )

    controller._worker_manager.restore_infer.assert_called_once()
    self.assertEqual(
        jax.tree.structure(result), jax.tree.structure(template_state)
    )
    np.testing.assert_array_equal(
        np.asarray(result['weights']), np.arange(2, dtype=np.float32)
    )

  def test_restore_none_step_uses_latest_sentinel(self):
    controller, sharding = self._make_controller_for_restore()
    mesh = sharding.mesh
    cpu_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    restored_cpu_state = {
        'weights': jax.device_put(
            jnp.arange(2, dtype=jnp.float32),
            cpu_sharding,
        )
    }
    template_state = {
        'weights': jax.ShapeDtypeStruct((2,), jnp.float32, sharding=sharding)
    }
    restore_args = {
        'weights': type_handlers.ArrayRestoreArgs(sharding=sharding)
    }
    controller._worker_manager = mock.Mock()
    controller._worker_manager.restore_infer.return_value = restored_cpu_state

    controller.restore(
        None,
        args_lib.Composite(
            state=args_lib.PyTreeRestore(
                item=template_state,
                restore_args=restore_args,
            )
        ),
        default_item_mode=True,
    )

    step_arg = controller._worker_manager.restore_infer.call_args.args[0]
    self.assertEqual(np.asarray(step_arg).item(), -1)

  def test_restore_none_step_with_dataset_resolves_single_explicit_step(self):
    controller, sharding = self._make_controller_for_restore()
    mesh = sharding.mesh
    cpu_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    restored_cpu_state = {
        'weights': jax.device_put(
            jnp.arange(2, dtype=jnp.float32),
            cpu_sharding,
        )
    }
    template_state = {
        'weights': jax.ShapeDtypeStruct((2,), jnp.float32, sharding=sharding)
    }
    restore_args = {
        'weights': type_handlers.ArrayRestoreArgs(sharding=sharding)
    }
    controller._worker_manager = mock.Mock()
    controller._worker_manager.restore_infer.return_value = restored_cpu_state
    controller._persistent_checkpoint_manager = mock.Mock()
    controller._persistent_checkpoint_manager.restore.return_value = (
        args_lib.Composite(dataset='dataset-state')
    )
    controller.latest_step = mock.Mock(return_value=11)

    result = controller.restore(
        None,
        args_lib.Composite(
            state=args_lib.PyTreeRestore(
                item=template_state,
                restore_args=restore_args,
            ),
            dataset=mock.Mock(),
        ),
        default_item_mode=False,
    )

    step_arg = controller._worker_manager.restore_infer.call_args.args[0]
    self.assertEqual(np.asarray(step_arg).item(), 11)
    controller._persistent_checkpoint_manager.restore.assert_called_once()
    self.assertEqual(
        controller._persistent_checkpoint_manager.restore.call_args.args[0], 11
    )
    self.assertEqual(result['dataset'], 'dataset-state')

  def test_save_no_precheck_should_save_only_saves_dataset_on_success(self):
    controller, device = self._make_controller_for_specialization()
    mesh = jax.sharding.Mesh(np.array([device]), ('d',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state = {'x': jax.device_put(jnp.arange(2, dtype=jnp.int32), sharding)}
    controller.should_save = mock.Mock(side_effect=AssertionError('unexpected'))
    controller._prepare_state_for_save = mock.Mock(return_value=state)
    controller._invoke_specialized_call = mock.Mock(return_value=object())
    controller._get_worker_save_call = mock.Mock(return_value=object())
    controller._save_persistent_dataset = mock.Mock()

    with mock.patch.object(
        controller_lib.colocated_utils,
        'require_unanimous_scalar_result',
        return_value=False,
    ):
      saved = controller.save(
          7,
          args_lib.Composite(
              state=args_lib.PyTreeSave({'x': 1}),
              dataset=mock.Mock(),
          ),
      )

    self.assertFalse(saved)
    controller.should_save.assert_not_called()
    controller._save_persistent_dataset.assert_not_called()

  def test_restore_retries_once_on_runtime_error(self):
    controller, sharding = self._make_controller_for_restore()
    mesh = sharding.mesh
    cpu_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    restored_cpu_state = {
        'params': {
            'weights': jax.device_put(
                jnp.arange(2, dtype=jnp.float32),
                cpu_sharding,
            )
        }
    }
    template_state = {
        'weights': jax.ShapeDtypeStruct((2,), jnp.float32, sharding=sharding)
    }
    restore_args = {
        'weights': type_handlers.ArrayRestoreArgs(sharding=sharding)
    }
    controller._worker_manager = mock.Mock()
    controller._worker_manager.restore_infer.side_effect = [
        RuntimeError('transient'),
        restored_cpu_state,
    ]

    result = controller.restore(
        7,
        args_lib.Composite(
            state=args_lib.PyTreeRestore(
                item=template_state,
                restore_args=restore_args,
            )
        ),
        default_item_mode=True,
    )

    self.assertEqual(controller._worker_manager.restore_infer.call_count, 2)
    np.testing.assert_array_equal(
        np.asarray(result['weights']), np.arange(2, dtype=np.float32)
    )

  def test_restore_does_not_retry_non_retriable_error(self):
    controller, sharding = self._make_controller_for_restore()
    template_state = {
        'weights': jax.ShapeDtypeStruct((2,), jnp.float32, sharding=sharding)
    }
    restore_args = {
        'weights': type_handlers.ArrayRestoreArgs(sharding=sharding)
    }
    controller._worker_manager = mock.Mock()
    controller._worker_manager.restore_infer.side_effect = ValueError(
        'bad restore'
    )

    with self.assertRaisesRegex(ValueError, 'bad restore'):
      controller.restore(
          7,
          args_lib.Composite(
              state=args_lib.PyTreeRestore(
                  item=template_state,
                  restore_args=restore_args,
              )
          ),
          default_item_mode=True,
      )

    self.assertEqual(controller._worker_manager.restore_infer.call_count, 1)

  def test_invoke_specialized_call_retries_once_on_runtime_error(self):
    controller = controller_lib.ColocatedController.__new__(
        controller_lib.ColocatedController
    )
    setattr(controller, '_worker_save_call', object())
    rebuilt = object()
    build_call = mock.Mock(return_value=rebuilt)
    call = mock.Mock(side_effect=[RuntimeError('transient'), object()])

    result = controller._invoke_specialized_call(
        op_name='save',
        cache_attr='_worker_save_call',
        build_call=build_call,
        call=call,
    )

    self.assertIsNotNone(result)
    self.assertEqual(call.call_count, 2)
    self.assertEqual(build_call.call_count, 2)
    self.assertIs(getattr(controller, '_worker_save_call'), None)

  def test_invoke_specialized_call_does_not_retry_non_retriable_error(self):
    controller = controller_lib.ColocatedController.__new__(
        controller_lib.ColocatedController
    )
    sentinel = object()
    setattr(controller, '_worker_save_call', sentinel)
    build_call = mock.Mock(return_value=sentinel)
    call = mock.Mock(side_effect=ValueError('bad specialization'))

    with self.assertRaisesRegex(ValueError, 'bad specialization'):
      controller._invoke_specialized_call(
          op_name='save',
          cache_attr='_worker_save_call',
          build_call=build_call,
          call=call,
      )

    self.assertEqual(call.call_count, 1)
    self.assertEqual(build_call.call_count, 1)
    self.assertIs(getattr(controller, '_worker_save_call'), sentinel)


if __name__ == '__main__':
  absltest.main()
