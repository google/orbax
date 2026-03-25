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
        controller_lib,
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
    self.assertNotIn('cpu_mesh', kwargs)

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

  def test_get_or_create_specialized_worker_call_builds_in_specs_once(self):
    controller = controller_lib.ColocatedController.__new__(
        controller_lib.ColocatedController
    )
    controller._worker_save_call = None

    device = jax.devices()[0]
    mesh = jax.sharding.Mesh(np.array([device]), ('d',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    spec = jax.ShapeDtypeStruct((), jnp.int32, sharding=sharding)

    controller._colocated_cpu_ids = frozenset({device.id})
    controller._colocated_cpu_devices = (device,)

    build_calls = {'count': 0}
    specialize_calls = {'count': 0}

    def _build_in_specs():
      build_calls['count'] += 1
      return ((spec,), {})

    def _specialize_fn(*, in_specs, out_specs_fn, devices):
      del in_specs, out_specs_fn, devices
      specialize_calls['count'] += 1
      return object()

    first = controller._get_or_create_specialized_worker_call(
        cache_attr='_worker_save_call',
        in_specs_builder=_build_in_specs,
        out_specs_fn=lambda *_: spec,
        specialize_fn=_specialize_fn,
        tree_name='test_specs',
    )
    second = controller._get_or_create_specialized_worker_call(
        cache_attr='_worker_save_call',
        in_specs_builder=_build_in_specs,
        out_specs_fn=lambda *_: spec,
        specialize_fn=_specialize_fn,
        tree_name='test_specs',
    )

    self.assertIs(first, second)
    self.assertEqual(build_calls['count'], 1)
    self.assertEqual(specialize_calls['count'], 1)

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

  def test_rebuild_restored_state_uses_template_structure(self):
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


if __name__ == '__main__':
  absltest.main()
