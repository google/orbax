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

import collections
import threading
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import args as args_lib
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.metadata import value as metadata_value
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import (
    colocated_controller as controller_lib,
)
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import (
    sidecar_worker_checkpoint_manager,
)


EmptyState = collections.namedtuple('EmptyState', [])


def _steps_array(steps: list[int]) -> np.ndarray:
  padded = steps + [controller_lib.colocated_utils.NO_STEP_SENTINEL] * (
      controller_lib.colocated_utils.MAX_TRACKED_STEPS - len(steps)
  )
  return np.asarray(padded, dtype=np.int32)


def _options(**overrides) -> mock.Mock:
  defaults = dict(
      save_interval_steps=3,
      step_name_format=None,
      should_save_fn=None,
      preservation_policy=None,
      enable_async_checkpointing=True,
      save_concurrent_gb=None,
      restore_concurrent_gb=None,
  )
  defaults.update(overrides)
  return mock.Mock(spec=list(defaults.keys()), **defaults)


class ColocatedControllerInternalTest(parameterized.TestCase):

  def test_step_arr_on_state_devices_supports_single_device_sharding(self):
    device = jax.devices()[0]
    sharding = jax.sharding.SingleDeviceSharding(device)
    state = {'x': jax.device_put(jnp.arange(2, dtype=jnp.int32), sharding)}

    step_arr = controller_lib._step_arr_on_state_devices(4, state)

    self.assertEqual(np.asarray(step_arr).item(), 4)
    self.assertIsInstance(step_arr.sharding, jax.sharding.SingleDeviceSharding)

  def test_init_preserves_global_mesh_device_order(self):
    fake_device_1 = mock.Mock(spec=jax.Device, id=1)
    fake_device_2 = mock.Mock(spec=jax.Device, id=2)
    fake_mesh = mock.create_autospec(jax.sharding.Mesh, instance=True)
    fake_mesh.devices = np.array([fake_device_2, fake_device_1])
    fake_mesh.axis_names = ('x',)
    fake_mesh.axis_types = (jax.sharding.AxisType.Auto,)
    worker_manager = mock.Mock(
        spec=sidecar_worker_checkpoint_manager.WorkerCheckpointManager
    )
    dummy = object()
    worker_cpu_devices = (mock.Mock(spec=jax.Device, id=101),)
    state_cpu_devices = (
        mock.Mock(spec=jax.Device, id=101),
        mock.Mock(spec=jax.Device, id=102),
    )
    topology = mock.Mock(spec=controller_lib.pathways_topology.Topology)
    topology.worker_cpu_devices.return_value = worker_cpu_devices
    topology.remap_distributed_device_ids.return_value = [[102], [101]]
    fake_cpu_mesh = mock.create_autospec(jax.sharding.Mesh, instance=True)
    fake_cpu_mesh.devices = np.array(state_cpu_devices)
    mock_worker_manager = self.enter_context(
        mock.patch.object(
            sidecar_worker_checkpoint_manager,
            'WorkerCheckpointManager',
            return_value=worker_manager,
        )
    )
    mock_install_patch = self.enter_context(
        mock.patch.object(
            controller_lib.colocated_transport,
            'install_pathways_colocated_serialization_patch',
        )
    )
    mock_topology_from_devices = self.enter_context(
        mock.patch.object(
            controller_lib.pathways_topology.Topology,
            'from_devices',
            return_value=topology,
        )
    )
    mock_colocated_cpu_mesh = self.enter_context(
        mock.patch.object(
            controller_lib.colocated_transport,
            'colocated_cpu_mesh',
            return_value=fake_cpu_mesh,
        )
    )
    mock_get_dummy_input_array = self.enter_context(
        mock.patch.object(
            controller_lib.dispatchers,
            'get_dummy_input_array',
            return_value=dummy,
        )
    )
    self.enter_context(
        mock.patch.object(
            controller_lib.ColocatedController,
            '_specialize_worker_calls',
            autospec=True,
        )
    )
    controller = controller_lib.ColocatedController(
        local_directory=mock.Mock(spec=epath.Path),
        global_mesh=fake_mesh,
        options=_options(),
        persistent_directory=None,
        handler_registry=None,
        checkpoint_manager_options_fn=mock.Mock(),
    )

    self.assertIs(controller._dummy, dummy)
    mock_install_patch.assert_called_once_with()
    mock_topology_from_devices.assert_called_once_with(
        (fake_device_2, fake_device_1)
    )
    mock_colocated_cpu_mesh.assert_called_once_with(fake_mesh)
    topology.remap_distributed_device_ids.assert_called_once_with(
        (fake_device_2, fake_device_1),
        (state_cpu_devices[0], state_cpu_devices[1]),
    )
    topology.worker_cpu_devices.assert_called_once_with()
    mock_get_dummy_input_array.assert_called_once_with(worker_cpu_devices)
    mock_worker_manager.assert_called_once_with(
        local_directory=mock.ANY,
        mesh_shape=(2,),
        mesh_axis_names=('x',),
        mesh_device_ids=(101, 102),
        mesh_axis_types=tuple(fake_mesh.axis_types),
        save_interval_steps=3,
        distributed_to_device_ids=((102,), (101,)),
        enable_async_checkpointing=True,
        save_concurrent_gb=None,
        restore_concurrent_gb=None,
    )

  def test_init_rejects_empty_colocated_cpu_devices(self):
    fake_mesh = mock.create_autospec(jax.sharding.Mesh, instance=True)
    fake_mesh.devices = np.array([mock.Mock(spec=jax.Device, id=1)])
    fake_mesh.axis_names = ('x',)
    fake_mesh.axis_types = (jax.sharding.AxisType.Auto,)
    fake_cpu_mesh = mock.create_autospec(jax.sharding.Mesh, instance=True)
    fake_cpu_mesh.devices = np.array([mock.Mock(spec=jax.Device, id=101)])
    topology = mock.Mock(spec=controller_lib.pathways_topology.Topology)
    topology.worker_cpu_devices.return_value = ()
    topology.remap_distributed_device_ids.return_value = [[101]]

    self.enter_context(
        mock.patch.object(
            sidecar_worker_checkpoint_manager,
            'WorkerCheckpointManager',
            autospec=True,
        )
    )
    self.enter_context(
        mock.patch.object(
            controller_lib.colocated_transport,
            'install_pathways_colocated_serialization_patch',
        )
    )
    self.enter_context(
        mock.patch.object(
            controller_lib.pathways_topology.Topology,
            'from_devices',
            return_value=topology,
        )
    )
    self.enter_context(
        mock.patch.object(
            controller_lib.colocated_transport,
            'colocated_cpu_mesh',
            return_value=fake_cpu_mesh,
        )
    )
    with self.assertRaisesRegex(
        ValueError, 'requires at least one colocated CPU device'
    ):
      controller_lib.ColocatedController(
          local_directory=mock.Mock(spec=epath.Path),
          global_mesh=fake_mesh,
          options=_options(),
          persistent_directory=None,
          handler_registry=None,
          checkpoint_manager_options_fn=mock.Mock(),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='step_name_format',
          name='step_name_format',
          value=mock.Mock(spec=object),
      ),
      dict(
          testcase_name='should_save_fn',
          name='should_save_fn',
          value=lambda *_: True,
      ),
      dict(
          testcase_name='preservation_policy',
          name='preservation_policy',
          value=mock.Mock(spec=object),
      ),
  )
  def test_init_rejects_unsupported_custom_colocated_options(self, name, value):
    with self.assertRaisesRegex(NotImplementedError, name):
      controller_lib._validate_colocated_options(_options(**{name: value}))

  def _make_controller_for_specialization(self):
    controller = controller_lib.ColocatedController.__new__(
        controller_lib.ColocatedController
    )
    device = jax.devices()[0]
    controller._local_directory = 'local'
    controller._worker_save_call = None
    controller._worker_save_call_in_specs = None
    controller._worker_cpu_devices = (device,)
    controller._state_cpu_devices = (device,)
    controller._colocated_cpu_ids = frozenset({device.id})
    controller._save_lifecycle = controller_lib._ControllerSaveHandoffLifecycle(
        worker_wait_until_finished=lambda: (  # pylint: disable=unnecessary-lambda
            controller._worker_wait_until_finished()
        ),
        save_persistent_dataset=lambda step, dataset_args, *, force: (
            controller._save_persistent_dataset(
                step, dataset_args, force=force
            )
        ),
    )
    controller._enable_async_checkpointing = True
    return controller, device

  def _join_pending_handoff(self, controller):
    pending_save = controller._pending_save
    self.assertIsNotNone(pending_save)
    thread = pending_save.handoff_thread
    if thread is not None:
      thread.join(timeout=5)
      self.assertFalse(thread.is_alive())

  def _make_controller_for_restore(self):
    controller, device = self._make_controller_for_specialization()
    controller._dummy = jax.device_put(
        jnp.array(True), jax.sharding.SingleDeviceSharding(device)
    )
    controller._persistent_checkpoint_manager = None
    mesh = jax.sharding.Mesh(np.array([device]), ('d',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    controller._worker_manager = mock.Mock()
    controller._worker_manager.restore_infer.specialize.side_effect = (
        lambda *args, **kwargs: controller._worker_manager.restore_infer
    )
    return controller, sharding

  def test_all_steps_specializes_with_replicated_sharding(self):
    controller, device = self._make_controller_for_specialization()
    controller._worker_cpu_devices = (device,)
    controller._worker_manager = mock.Mock()
    for name in (
        'latest_step',
        'should_save',
        'wait_until_finished',
        'check_for_errors',
        'close',
        'is_saving_in_progress',
    ):
      prop = getattr(controller._worker_manager, name)
      prop.specialize.return_value = object()
    mesh = jax.sharding.Mesh(np.array([device]), ('worker',))
    worker_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('worker')
    )
    arg_spec = jax.ShapeDtypeStruct((1,), jnp.int32, sharding=worker_sharding)
    captured = {}

    def _all_steps_specialize(*, out_specs_fn, devices):
      captured['out_spec'] = out_specs_fn(arg_spec)
      captured['devices'] = devices
      return object()

    controller._worker_manager.all_steps.specialize.side_effect = (
        _all_steps_specialize
    )

    controller._specialize_worker_calls()

    self.assertEqual(
        captured['out_spec'].shape,
        (controller_lib.colocated_utils.MAX_TRACKED_STEPS,),
    )
    self.assertEqual(captured['out_spec'].sharding.mesh, mesh)
    self.assertEqual(
        captured['out_spec'].sharding.spec, jax.sharding.PartitionSpec()
    )
    self.assertEqual(captured['devices'], (device,))

  def _set_worker_restore_result(
      self,
      controller,
      *,
      return_value=None,
      side_effect=None,
  ):
    controller._worker_manager = mock.Mock()
    controller._worker_manager.restore_infer.specialize.return_value = (
        controller._worker_manager.restore_infer
    )
    if side_effect is not None:
      controller._worker_manager.restore_infer.side_effect = side_effect
    else:
      controller._worker_manager.restore_infer.return_value = return_value

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
    self.assertEqual(captured['devices'], controller._worker_cpu_devices)
    spec_leaves = [
        leaf
        for leaf in jax.tree.leaves(captured['in_specs'])
        if isinstance(leaf, jax.ShapeDtypeStruct)
    ]
    self.assertNotEmpty(spec_leaves)
    for leaf in spec_leaves:
      self.assertEqual({d.platform for d in leaf.sharding.device_set}, {'cpu'})
      self.assertEqual({d.id for d in leaf.sharding.device_set}, {device.id})
    self.assertEqual(
        controller._worker_save_call_in_specs, captured['in_specs']
    )

  def test_get_worker_save_call_reuses_matching_specs_and_rebuilds_changed_specs(
      self,
  ):
    controller, device = self._make_controller_for_specialization()
    mesh = jax.sharding.Mesh(np.array([device]), ('d',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    step_arr = jax.device_put(jnp.array(1, dtype=jnp.int32), sharding)
    force_arr = jax.device_put(jnp.array(True, dtype=jnp.bool_), sharding)
    state = {'x': jax.device_put(jnp.arange(2, dtype=jnp.int32), sharding)}
    changed_state = {
        'x': jax.device_put(jnp.arange(3, dtype=jnp.int32), sharding)
    }
    first_call = object()
    second_call = object()
    controller._worker_manager = mock.Mock()
    controller._worker_manager.save.specialize = mock.Mock(
        side_effect=[first_call, second_call]
    )

    result_1 = controller._get_worker_save_call(step_arr, force_arr, state)
    result_2 = controller._get_worker_save_call(step_arr, force_arr, state)
    result_3 = controller._get_worker_save_call(
        step_arr, force_arr, changed_state
    )

    self.assertIs(result_1, first_call)
    self.assertIs(result_2, first_call)
    self.assertIs(result_3, second_call)
    self.assertEqual(controller._worker_manager.save.specialize.call_count, 2)

  def test_prepare_restore_target_shardings_uses_restore_args(self):
    _, sharding = self._make_controller_for_restore()
    restore_args = {'x': type_handlers.ArrayRestoreArgs(sharding=sharding)}

    target_shardings = controller_lib._prepare_restore_target_shardings(
        args_lib.PyTreeRestore(item=None, restore_args=restore_args),
    )

    self.assertEqual(target_shardings['x'], sharding)

  def test_rebuild_restored_state_restores_empty_namedtuple_from_none(self):
    controller, sharding = self._make_controller_for_restore()
    del sharding
    restored_state = {
        'opt_state': [None, {'count': jnp.asarray(7, dtype=jnp.int32)}],
        'weights': jnp.arange(2, dtype=jnp.float32),
    }
    template_state = {
        'opt_state': [
            EmptyState(),
            {'count': jax.ShapeDtypeStruct((), jnp.int32)},
        ],
        'weights': jax.ShapeDtypeStruct((2,), jnp.float32),
    }

    rebuilt = controller._rebuild_restored_state(restored_state, template_state)

    self.assertEqual(
        jax.tree.structure(rebuilt), jax.tree.structure(template_state)
    )
    self.assertEqual(rebuilt['opt_state'][0], EmptyState())
    np.testing.assert_array_equal(
        np.asarray(rebuilt['weights']), np.arange(2, dtype=np.float32)
    )

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

  def test_rebuild_restored_state_unwraps_state_mapping_wrapper(self):
    controller, sharding = self._make_controller_for_restore()
    del sharding
    restored_state = {
        'metadata': {'step': jnp.asarray(7, dtype=jnp.int32)},
        'state': {'weights': jnp.arange(2, dtype=jnp.float32)},
    }
    template_state = {'weights': jax.ShapeDtypeStruct((2,), jnp.float32)}

    rebuilt = controller._rebuild_restored_state(restored_state, template_state)

    self.assertEqual(
        jax.tree.structure(rebuilt), jax.tree.structure(template_state)
    )
    np.testing.assert_array_equal(
        rebuilt['weights'], restored_state['state']['weights']
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
    controller._worker_manager.all_steps.return_value = object()

    with mock.patch.object(
        controller_lib.colocated_utils,
        'array_result_values',
        return_value=[
            _steps_array([]),
            _steps_array([]),
        ],
    ) as mock_array_values:
      latest_step = controller.latest_step()

    self.assertIsNone(latest_step)
    controller._worker_manager.all_steps.assert_called_once()
    mock_array_values.assert_called_once()

  def test_latest_step_returns_highest_common_step(self):
    controller, _ = self._make_controller_for_restore()
    controller._worker_manager = mock.Mock()
    controller._worker_manager.all_steps.return_value = object()

    with mock.patch.object(
        controller_lib.colocated_utils,
        'array_result_values',
        return_value=[
            _steps_array([3, 5]),
            _steps_array([1, 5]),
            _steps_array([5]),
        ],
    ):
      latest_step = controller.latest_step()

    self.assertEqual(latest_step, 5)
    controller._worker_manager.all_steps.assert_called_once()

  def test_latest_step_returns_none_when_no_common_positive_step(self):
    controller, _ = self._make_controller_for_restore()
    controller._worker_manager = mock.Mock()
    controller._worker_manager.all_steps.return_value = object()

    with mock.patch.object(
        controller_lib.colocated_utils,
        'array_result_values',
        return_value=[
            _steps_array([3]),
            _steps_array([4]),
        ],
    ):
      latest_step = controller.latest_step()

    self.assertIsNone(latest_step)

  def test_latest_step_retries_transient_runtime_error(self):
    controller, _ = self._make_controller_for_restore()
    controller._worker_manager = mock.Mock()
    controller._worker_manager.all_steps.side_effect = [
        RuntimeError('transient'),
        object(),
    ]

    with mock.patch.object(
        controller_lib.colocated_utils,
        'array_result_values',
        return_value=[_steps_array([12])],
    ), mock.patch.object(controller_lib.time, 'sleep') as mock_sleep:
      latest_step = controller.latest_step()

    self.assertEqual(latest_step, 12)
    self.assertEqual(controller._worker_manager.all_steps.call_count, 2)
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

  def test_should_save_skips_protocol_no_checkpoint_sentinel(self):
    controller, _ = self._make_controller_for_restore()
    controller._worker_manager = mock.Mock()

    should_save = controller.should_save(
        controller_lib.colocated_utils.NO_STEP_SENTINEL
    )

    self.assertFalse(should_save)
    controller._worker_manager.should_save.assert_not_called()

  def test_save_skips_state_preparation_when_workers_will_not_save(self):
    controller, _ = self._make_controller_for_restore()

    with mock.patch.object(
        controller, 'should_save', return_value=False
    ) as mock_should_save, mock.patch.object(
        controller, '_prepare_state_for_save'
    ) as mock_prepare, mock.patch.object(
        controller, '_finish_pending_save'
    ) as mock_finish_pending:
      saved = controller.save(
          9,
          args_lib.Composite(state=args_lib.PyTreeSave(item={'x': 1})),
      )

    self.assertFalse(saved)
    mock_should_save.assert_called_once_with(9)
    mock_prepare.assert_not_called()
    mock_finish_pending.assert_not_called()

  def test_save_skips_protocol_no_checkpoint_sentinel_even_when_forced(self):
    controller, _ = self._make_controller_for_restore()

    with mock.patch.object(
        controller, 'should_save'
    ) as mock_should_save, mock.patch.object(
        controller, '_prepare_state_for_save'
    ) as mock_prepare, mock.patch.object(
        controller, '_finish_pending_save'
    ) as mock_finish_pending:
      saved = controller.save(
          controller_lib.colocated_utils.NO_STEP_SENTINEL,
          args_lib.Composite(state=args_lib.PyTreeSave(item={'x': 1})),
          force=True,
      )

    self.assertFalse(saved)
    mock_should_save.assert_not_called()
    mock_prepare.assert_not_called()
    mock_finish_pending.assert_not_called()

  def test_save_tracks_pending_worker_result_until_lifecycle_drain(self):
    controller, _ = self._make_controller_for_restore()
    state = {'x': jax.device_put(jnp.arange(1, dtype=jnp.int32))}
    result = object()
    handoff_started = threading.Event()
    release_handoff = threading.Event()

    def _block_until_ready(value):
      if value is result:
        handoff_started.set()
        if not release_handoff.wait(timeout=5):
          raise TimeoutError('handoff test did not release worker result')
      return value

    with mock.patch.object(
        controller, 'should_save', return_value=True
    ), mock.patch.object(
        controller, '_prepare_state_for_save', return_value=state
    ), mock.patch.object(
        controller, '_get_worker_save_call', return_value=lambda *_: result
    ), mock.patch.object(
        controller_lib.jax,
        'block_until_ready',
        side_effect=_block_until_ready,
    ), mock.patch.object(
        controller, '_save_persistent_dataset'
    ) as mock_save_dataset, mock.patch.object(
        controller_lib.colocated_utils,
        'require_unanimous_scalar_result',
        return_value=True,
    ), mock.patch.object(
        controller, '_worker_wait_until_finished'
    ):
      saved = controller.save(
          9,
          args_lib.Composite(state=args_lib.PyTreeSave(item=state)),
      )
      self.assertTrue(saved)
      self.assertTrue(handoff_started.wait(timeout=5))
      self.assertIsNotNone(controller._pending_save)
      self.assertIsNotNone(controller._pending_save.keepalive)
      self.assertTrue(controller.is_saving_in_progress())
      mock_save_dataset.assert_not_called()

      release_handoff.set()
      controller._finish_pending_save()

      self.assertIsNone(controller._pending_save)
      mock_save_dataset.assert_called_once()

  def test_save_defers_worker_result_until_lifecycle_drain(self):
    controller, _ = self._make_controller_for_restore()
    state = {'x': jax.device_put(jnp.arange(1, dtype=jnp.int32))}
    result = object()
    handoff_started = threading.Event()
    release_handoff = threading.Event()

    def _block_until_ready(value):
      if value is result:
        handoff_started.set()
        if not release_handoff.wait(timeout=5):
          raise TimeoutError('handoff test did not release worker result')
      return value

    with mock.patch.object(
        controller, 'should_save', return_value=True
    ), mock.patch.object(
        controller, '_prepare_state_for_save', return_value=state
    ), mock.patch.object(
        controller, '_get_worker_save_call', return_value=lambda *_: result
    ), mock.patch.object(
        controller_lib.jax,
        'block_until_ready',
        side_effect=_block_until_ready,
    ), mock.patch.object(
        controller, '_save_persistent_dataset'
    ), mock.patch.object(
        controller_lib.colocated_utils,
        'require_unanimous_scalar_result',
        return_value=True,
    ), mock.patch.object(
        controller, '_worker_wait_until_finished'
    ) as mock_worker_wait:
      saved = controller.save(
          9,
          args_lib.Composite(state=args_lib.PyTreeSave(item=state)),
      )

      self.assertTrue(saved)
      self.assertTrue(handoff_started.wait(timeout=5))
      self.assertIsNotNone(controller._pending_save)
      mock_worker_wait.assert_not_called()

      wait_errors = []
      wait_done = threading.Event()

      def _wait_until_finished():
        try:
          controller.wait_until_finished()
        except Exception as e:  # pylint: disable=broad-exception-caught
          wait_errors.append(e)
        finally:
          wait_done.set()

      wait_thread = threading.Thread(target=_wait_until_finished)
      wait_thread.start()
      self.assertFalse(wait_done.wait(timeout=0.1))
      mock_worker_wait.assert_not_called()

      release_handoff.set()
      wait_thread.join(timeout=5)
      self.assertFalse(wait_thread.is_alive())

    self.assertEmpty(wait_errors)
    mock_worker_wait.assert_called_once_with()
    self.assertIsNone(controller._pending_save)

  def test_worker_handoff_releases_keepalive_after_worker_result(self):
    controller, _ = self._make_controller_for_restore()
    keepalive = (mock.sentinel.state,)
    result = object()
    pending_save = controller_lib._PendingSave(
        step=9,
        force=False,
        result=result,
        keepalive=keepalive,
        dataset_args=None,
        start_time=0.0,
    )

    with mock.patch.object(
        controller_lib.jax, 'block_until_ready'
    ) as mock_block, mock.patch.object(
        controller_lib.colocated_utils,
        'require_unanimous_scalar_result',
        return_value=True,
    ):
      saved = controller._save_lifecycle.await_handoff(pending_save)

    self.assertTrue(saved)
    mock_block.assert_called_once_with(result)
    self.assertIsNone(pending_save.keepalive)

  def test_worker_handoff_releases_keepalive_when_worker_does_not_save(
      self,
  ):
    controller, _ = self._make_controller_for_restore()
    result = object()
    pending_save = controller_lib._PendingSave(
        step=9,
        force=False,
        result=result,
        keepalive=(mock.sentinel.state,),
        dataset_args=None,
        start_time=0.0,
    )

    with mock.patch.object(
        controller_lib.jax, 'block_until_ready'
    ), mock.patch.object(
        controller_lib.colocated_utils,
        'require_unanimous_scalar_result',
        return_value=False,
    ):
      saved = controller._save_lifecycle.await_handoff(pending_save)

    self.assertFalse(saved)
    self.assertIsNone(pending_save.keepalive)

  def test_worker_handoff_rejects_mixed_handoff_ack(self):
    controller, _ = self._make_controller_for_restore()
    result = object()
    pending_save = controller_lib._PendingSave(
        step=9,
        force=False,
        result=result,
        keepalive=(mock.sentinel.state,),
        dataset_args=None,
        start_time=0.0,
    )

    with mock.patch.object(
        controller_lib.jax, 'block_until_ready'
    ), mock.patch.object(
        controller_lib.colocated_utils,
        'require_unanimous_scalar_result',
        side_effect=ValueError('mixed votes'),
    ):
      with self.assertRaisesRegex(ValueError, 'mixed votes'):
        controller._save_lifecycle.await_handoff(pending_save)

    self.assertIsNone(pending_save.keepalive)

  def test_sync_save_drains_worker_result_before_returning(self):
    controller, _ = self._make_controller_for_restore()
    controller._enable_async_checkpointing = False
    state = {'x': jax.device_put(jnp.arange(1, dtype=jnp.int32))}
    result = object()
    worker_wait_result = object()
    controller._worker_manager.wait_until_finished.return_value = (
        worker_wait_result
    )

    with mock.patch.object(
        controller, 'should_save', return_value=True
    ), mock.patch.object(
        controller, '_prepare_state_for_save', return_value=state
    ), mock.patch.object(
        controller, '_get_worker_save_call', return_value=lambda *_: result
    ), mock.patch.object(
        controller_lib.jax, 'block_until_ready'
    ) as mock_block, mock.patch.object(
        controller_lib.colocated_utils,
        'require_unanimous_scalar_result',
        return_value=True,
    ):
      saved = controller.save(
          9,
          args_lib.Composite(state=args_lib.PyTreeSave(item=state)),
      )

    self.assertTrue(saved)
    mock_block.assert_has_calls([
        mock.call(result),
        mock.call(worker_wait_result),
    ])
    self.assertIsNone(controller._pending_save)

  def test_save_defers_persistent_dataset_until_worker_result(self):
    controller, _ = self._make_controller_for_restore()
    state = {'x': jax.device_put(jnp.arange(1, dtype=jnp.int32))}
    result = object()
    dataset_arg = mock.Mock()
    controller._persistent_checkpoint_manager = mock.Mock()

    with mock.patch.object(
        controller, 'should_save', return_value=True
    ), mock.patch.object(
        controller, '_prepare_state_for_save', return_value=state
    ), mock.patch.object(
        controller, '_get_worker_save_call', return_value=lambda *_: result
    ), mock.patch.object(
        controller_lib.colocated_utils,
        'require_unanimous_scalar_result',
        return_value=True,
    ):
      saved = controller.save(
          9,
          args_lib.Composite(
              state=args_lib.PyTreeSave(item=state),
              dataset=dataset_arg,
          ),
      )
      self.assertTrue(saved)
      controller._persistent_checkpoint_manager.save.assert_not_called()

      controller._finish_pending_save()

      controller._persistent_checkpoint_manager.save.assert_called_once_with(
          9,
          args=args_lib.Composite(dataset=dataset_arg),
          force=False,
      )
      self.assertIsNone(controller._pending_save)

  def test_finish_pending_save_re_raises_completed_handoff_error(self):
    controller, _ = self._make_controller_for_restore()
    state = {'x': jax.device_put(jnp.arange(1, dtype=jnp.int32))}
    result = object()
    handoff_started = threading.Event()
    release_handoff = threading.Event()

    def _block_until_ready(value):
      if value is result:
        handoff_started.set()
        if not release_handoff.wait(timeout=5):
          raise TimeoutError('handoff test did not release worker result')
      return value

    with mock.patch.object(
        controller, 'should_save', return_value=True
    ), mock.patch.object(
        controller, '_prepare_state_for_save', return_value=state
    ), mock.patch.object(
        controller, '_get_worker_save_call', return_value=lambda *_: result
    ), mock.patch.object(
        controller_lib.jax,
        'block_until_ready',
        side_effect=_block_until_ready,
    ), mock.patch.object(
        controller_lib.colocated_utils,
        'require_unanimous_scalar_result',
        side_effect=RuntimeError('colocated save handoff failed'),
    ):
      saved = controller.save(
          9,
          args_lib.Composite(state=args_lib.PyTreeSave(item=state)),
      )
      self.assertTrue(saved)
      self.assertTrue(handoff_started.wait(timeout=5))
      self.assertIsNotNone(controller._pending_save)

      release_handoff.set()
      self._join_pending_handoff(controller)
      with self.assertRaisesRegex(
          RuntimeError, 'colocated save handoff failed'
      ):
        controller._finish_pending_save()

    self.assertIsNotNone(controller._pending_save)

  def test_finish_pending_save_waits_for_worker_completion_when_saved(self):
    controller, _ = self._make_controller_for_restore()
    state = {'x': jax.device_put(jnp.arange(1, dtype=jnp.int32))}
    result = object()
    worker_wait_result = object()
    controller._worker_manager.wait_until_finished.return_value = (
        worker_wait_result
    )
    handoff_started = threading.Event()
    release_handoff = threading.Event()

    def _block_until_ready(value):
      if value is result:
        handoff_started.set()
        if not release_handoff.wait(timeout=5):
          raise TimeoutError('handoff test did not release worker result')
      return value

    with mock.patch.object(
        controller, 'should_save', return_value=True
    ), mock.patch.object(
        controller, '_prepare_state_for_save', return_value=state
    ), mock.patch.object(
        controller, '_get_worker_save_call', return_value=lambda *_: result
    ), mock.patch.object(
        controller_lib.jax,
        'block_until_ready',
        side_effect=_block_until_ready,
    ) as mock_block, mock.patch.object(
        controller_lib.colocated_utils,
        'require_unanimous_scalar_result',
        return_value=True,
    ):
      saved = controller.save(
          9,
          args_lib.Composite(state=args_lib.PyTreeSave(item=state)),
      )
      self.assertTrue(saved)
      self.assertTrue(handoff_started.wait(timeout=5))

      release_handoff.set()
      controller._finish_pending_save()

    mock_block.assert_has_calls([
        mock.call(result),
        mock.call(worker_wait_result),
    ])
    self.assertIsNone(controller._pending_save)

  def test_finish_pending_save_skips_worker_poll_when_not_saved(self):
    controller, _ = self._make_controller_for_restore()
    state = {'x': jax.device_put(jnp.arange(1, dtype=jnp.int32))}
    result = object()
    handoff_started = threading.Event()
    release_handoff = threading.Event()

    def _block_until_ready(value):
      if value is result:
        handoff_started.set()
        if not release_handoff.wait(timeout=5):
          raise TimeoutError('handoff test did not release worker result')
      return value

    with mock.patch.object(
        controller, 'should_save', return_value=True
    ), mock.patch.object(
        controller, '_prepare_state_for_save', return_value=state
    ), mock.patch.object(
        controller, '_get_worker_save_call', return_value=lambda *_: result
    ), mock.patch.object(
        controller_lib.jax,
        'block_until_ready',
        side_effect=_block_until_ready,
    ), mock.patch.object(
        controller_lib.colocated_utils,
        'require_unanimous_scalar_result',
        return_value=False,
    ):
      saved = controller.save(
          9,
          args_lib.Composite(state=args_lib.PyTreeSave(item=state)),
      )
      self.assertTrue(saved)
      self.assertTrue(handoff_started.wait(timeout=5))

      release_handoff.set()
      controller._finish_pending_save()

    controller._worker_manager.wait_until_finished.assert_not_called()
    self.assertIsNone(controller._pending_save)

  def test_finish_pending_save_releases_keepalive_even_when_worker_fails(
      self,
  ):
    controller, _ = self._make_controller_for_restore()
    state = {'x': jax.device_put(jnp.arange(1, dtype=jnp.int32))}
    result = object()

    with mock.patch.object(
        controller, 'should_save', return_value=True
    ), mock.patch.object(
        controller, '_prepare_state_for_save', return_value=state
    ), mock.patch.object(
        controller, '_get_worker_save_call', return_value=lambda *_: result
    ), mock.patch.object(
        controller_lib.colocated_utils,
        'require_unanimous_scalar_result',
        return_value=True,
    ), mock.patch.object(
        controller,
        '_worker_wait_until_finished',
        side_effect=RuntimeError('worker wait failed'),
    ):
      controller.save(
          9,
          args_lib.Composite(state=args_lib.PyTreeSave(item=state)),
      )

      with self.assertRaisesRegex(RuntimeError, 'worker wait failed'):
        controller._finish_pending_save()

      self.assertIsNone(controller._pending_save)

  def test_wait_until_finished_finishes_pending_save(self):
    controller, _ = self._make_controller_for_restore()

    with mock.patch.object(
        controller, '_finish_pending_save'
    ) as mock_finish_pending, mock.patch.object(
        controller, '_worker_wait_until_finished'
    ) as mock_worker_wait:
      controller.wait_until_finished()

    mock_finish_pending.assert_called_once_with()
    mock_worker_wait.assert_called_once_with()

  def test_wait_until_finished_waits_on_persistent_manager_after_pending_save(
      self,
  ):
    controller, _ = self._make_controller_for_restore()
    persistent_manager = mock.Mock()
    controller._persistent_checkpoint_manager = persistent_manager
    order = []

    def _finish_pending():
      order.append('finish_pending')

    persistent_manager.wait_until_finished.side_effect = lambda: order.append(
        'persistent'
    )

    with mock.patch.object(
        controller, '_finish_pending_save', side_effect=_finish_pending
    ), mock.patch.object(controller, '_worker_wait_until_finished'):
      controller.wait_until_finished()

    self.assertEqual(order, ['finish_pending', 'persistent'])

  def test_wait_until_finished_skips_worker_poll_when_pending_save_existed(
      self,
  ):
    controller, _ = self._make_controller_for_restore()

    def _finish():
      pass

    with mock.patch.object(
        controller._save_lifecycle, 'has_pending_save', return_value=True
    ), mock.patch.object(
        controller, '_finish_pending_save', side_effect=_finish
    ), mock.patch.object(
        controller, '_worker_wait_until_finished'
    ) as mock_worker_wait:
      controller.wait_until_finished()

    mock_worker_wait.assert_not_called()

  def test_close_finishes_pending_save(self):
    controller, _ = self._make_controller_for_restore()

    with mock.patch.object(
        controller, '_finish_pending_save'
    ) as mock_finish_pending, mock.patch.object(
        controller, '_worker_close'
    ) as mock_worker_close:
      controller.close()

    mock_finish_pending.assert_called_once_with()
    mock_worker_close.assert_called_once_with()

  def test_close_waits_for_pending_save_even_when_worker_close_fails(self):
    controller, _ = self._make_controller_for_restore()

    with mock.patch.object(
        controller, '_finish_pending_save'
    ) as mock_finish_pending, mock.patch.object(
        controller,
        '_worker_close',
        side_effect=RuntimeError('worker close failed'),
    ):
      with self.assertRaisesRegex(RuntimeError, 'worker close failed'):
        controller.close()

    mock_finish_pending.assert_called_once_with()

  def test_close_calls_worker_close_even_when_pending_save_fails(self):
    controller, _ = self._make_controller_for_restore()

    with mock.patch.object(
        controller,
        '_finish_pending_save',
        side_effect=RuntimeError('finish failed'),
    ), mock.patch.object(
        controller, '_worker_close'
    ) as mock_worker_close:
      with self.assertRaisesRegex(RuntimeError, 'finish failed'):
        controller.close()

    mock_worker_close.assert_called_once_with()

  def test_check_for_errors_surfaces_completed_handoff_error_without_polling(
      self,
  ):
    controller, _ = self._make_controller_for_restore()
    state = {'x': jax.device_put(jnp.arange(1, dtype=jnp.int32))}
    result = object()
    handoff_started = threading.Event()
    release_handoff = threading.Event()

    def _block_until_ready(value):
      if value is result:
        handoff_started.set()
        if not release_handoff.wait(timeout=5):
          raise TimeoutError('handoff test did not release worker result')
      return value

    with mock.patch.object(
        controller, 'should_save', return_value=True
    ), mock.patch.object(
        controller, '_prepare_state_for_save', return_value=state
    ), mock.patch.object(
        controller, '_get_worker_save_call', return_value=lambda *_: result
    ), mock.patch.object(
        controller_lib.jax,
        'block_until_ready',
        side_effect=_block_until_ready,
    ), mock.patch.object(
        controller_lib.colocated_utils,
        'require_unanimous_scalar_result',
        side_effect=RuntimeError('colocated save handoff failed'),
    ), mock.patch.object(
        controller, '_worker_wait_until_finished'
    ) as mock_worker_wait, mock.patch.object(
        controller_lib.logging, 'exception'
    ):
      saved = controller.save(
          9,
          args_lib.Composite(state=args_lib.PyTreeSave(item=state)),
      )
      self.assertTrue(saved)
      self.assertTrue(handoff_started.wait(timeout=5))
      self.assertIsNotNone(controller._pending_save)

      release_handoff.set()
      self._join_pending_handoff(controller)
      with self.assertRaisesRegex(
          RuntimeError, 'colocated save handoff failed'
      ):
        controller.check_for_errors()

      mock_worker_wait.assert_not_called()

  def test_close_surfaces_subsequent_worker_error_after_pending_save_error(
      self,
  ):
    controller, _ = self._make_controller_for_restore()

    with mock.patch.object(
        controller,
        '_finish_pending_save',
        side_effect=RuntimeError('finish failed'),
    ), mock.patch.object(
        controller,
        '_worker_close',
        side_effect=RuntimeError('worker close failed'),
    ), mock.patch.object(
        controller_lib.logging, 'exception'
    ) as mock_log:
      with self.assertRaisesRegex(RuntimeError, 'finish failed'):
        controller.close()

    mock_log.assert_called_once()
    self.assertIn('worker close failed', str(mock_log.call_args))

  def test_close_executes_worker_and_persistent_close_even_on_failure(self):
    controller, _ = self._make_controller_for_restore()
    persistent_manager = mock.Mock()
    controller._persistent_checkpoint_manager = persistent_manager
    persistent_manager.close.side_effect = RuntimeError('persistent failed')

    with mock.patch.object(
        controller,
        '_finish_pending_save',
        side_effect=RuntimeError('finish failed'),
    ), mock.patch.object(
        controller,
        '_worker_close',
        side_effect=RuntimeError('worker close failed'),
    ), mock.patch.object(
        controller_lib.logging, 'exception'
    ) as mock_log:
      with self.assertRaisesRegex(RuntimeError, 'finish failed'):
        controller.close()

    persistent_manager.close.assert_called_once_with()
    self.assertEqual(mock_log.call_count, 2)
    logged = [str(call) for call in mock_log.call_args_list]
    self.assertTrue(any('persistent failed' in msg for msg in logged))
    self.assertTrue(any('worker close failed' in msg for msg in logged))

  def test_close_waits_for_handoff_even_when_worker_close_fails(self):
    controller, _ = self._make_controller_for_restore()
    state = {'x': jax.device_put(jnp.arange(1, dtype=jnp.int32))}
    result = object()
    handoff_started = threading.Event()
    release_handoff = threading.Event()

    def _block_until_ready(value):
      if value is result:
        handoff_started.set()
        if not release_handoff.wait(timeout=5):
          raise TimeoutError('handoff test did not release worker result')
      return value

    with mock.patch.object(
        controller, 'should_save', return_value=True
    ), mock.patch.object(
        controller, '_prepare_state_for_save', return_value=state
    ), mock.patch.object(
        controller, '_get_worker_save_call', return_value=lambda *_: result
    ), mock.patch.object(
        controller_lib.jax,
        'block_until_ready',
        side_effect=_block_until_ready,
    ), mock.patch.object(
        controller_lib.colocated_utils,
        'require_unanimous_scalar_result',
        return_value=True,
    ), mock.patch.object(
        controller,
        '_worker_wait_until_finished',
    ) as mock_worker_wait, mock.patch.object(
        controller,
        '_worker_close',
        side_effect=RuntimeError('worker close failed'),
    ):
      saved = controller.save(
          9,
          args_lib.Composite(state=args_lib.PyTreeSave(item=state)),
      )
      self.assertTrue(saved)
      self.assertTrue(handoff_started.wait(timeout=5))
      assert controller._pending_save is not None
      self.assertIsNotNone(controller._pending_save.keepalive)

      close_errors = []
      close_done = threading.Event()

      def _close():
        try:
          controller.close()
        except Exception as e:  # pylint: disable=broad-exception-caught
          close_errors.append(e)
        finally:
          close_done.set()

      close_thread = threading.Thread(target=_close)
      close_thread.start()
      self.assertFalse(close_done.wait(timeout=0.1))
      mock_worker_wait.assert_not_called()

      release_handoff.set()
      close_thread.join(timeout=5)
      self.assertFalse(close_thread.is_alive())

    self.assertLen(close_errors, 1)
    self.assertIsInstance(close_errors[0], RuntimeError)
    self.assertIn('worker close failed', str(close_errors[0]))
    mock_worker_wait.assert_called_once_with()
    self.assertIsNone(controller._pending_save)

  def test_close_waits_for_handoff_and_persistent_close_even_on_worker_error(
      self,
  ):
    controller, _ = self._make_controller_for_restore()
    state = {'x': jax.device_put(jnp.arange(1, dtype=jnp.int32))}
    result = object()
    persistent_manager = mock.Mock()
    controller._persistent_checkpoint_manager = persistent_manager
    handoff_started = threading.Event()
    release_handoff = threading.Event()

    def _block_until_ready(value):
      if value is result:
        handoff_started.set()
        if not release_handoff.wait(timeout=5):
          raise TimeoutError('handoff test did not release worker result')
      return value

    with mock.patch.object(
        controller, 'should_save', return_value=True
    ), mock.patch.object(
        controller, '_prepare_state_for_save', return_value=state
    ), mock.patch.object(
        controller, '_get_worker_save_call', return_value=lambda *_: result
    ), mock.patch.object(
        controller_lib.jax,
        'block_until_ready',
        side_effect=_block_until_ready,
    ), mock.patch.object(
        controller_lib.colocated_utils,
        'require_unanimous_scalar_result',
        return_value=True,
    ), mock.patch.object(
        controller,
        '_worker_wait_until_finished',
    ) as mock_worker_wait, mock.patch.object(
        controller,
        '_worker_close',
        side_effect=RuntimeError('worker close failed'),
    ):
      saved = controller.save(
          9,
          args_lib.Composite(state=args_lib.PyTreeSave(item=state)),
      )
      self.assertTrue(saved)
      self.assertTrue(handoff_started.wait(timeout=5))
      assert controller._pending_save is not None
      self.assertIsNotNone(controller._pending_save.keepalive)

      close_errors = []
      close_done = threading.Event()

      def _close():
        try:
          controller.close()
        except Exception as e:  # pylint: disable=broad-exception-caught
          close_errors.append(e)
        finally:
          close_done.set()

      close_thread = threading.Thread(target=_close)
      close_thread.start()
      self.assertFalse(close_done.wait(timeout=0.1))
      mock_worker_wait.assert_not_called()
      persistent_manager.close.assert_not_called()

      release_handoff.set()
      close_thread.join(timeout=5)
      self.assertFalse(close_thread.is_alive())

    self.assertLen(close_errors, 1)
    self.assertIsInstance(close_errors[0], RuntimeError)
    self.assertIn('worker close failed', str(close_errors[0]))
    mock_worker_wait.assert_called_once_with()
    persistent_manager.close.assert_called_once_with()
    self.assertIsNone(controller._pending_save)

  def test_close_waits_for_handoff_and_worker_close_even_on_persistent_error(
      self,
  ):
    controller, _ = self._make_controller_for_restore()
    state = {'x': jax.device_put(jnp.arange(1, dtype=jnp.int32))}
    result = object()
    persistent_manager = mock.Mock()
    controller._persistent_checkpoint_manager = persistent_manager
    persistent_manager.close.side_effect = RuntimeError('persistent failed')
    handoff_started = threading.Event()
    release_handoff = threading.Event()

    def _block_until_ready(value):
      if value is result:
        handoff_started.set()
        if not release_handoff.wait(timeout=5):
          raise TimeoutError('handoff test did not release worker result')
      return value

    with mock.patch.object(
        controller, 'should_save', return_value=True
    ), mock.patch.object(
        controller, '_prepare_state_for_save', return_value=state
    ), mock.patch.object(
        controller, '_get_worker_save_call', return_value=lambda *_: result
    ), mock.patch.object(
        controller_lib.jax,
        'block_until_ready',
        side_effect=_block_until_ready,
    ), mock.patch.object(
        controller_lib.colocated_utils,
        'require_unanimous_scalar_result',
        return_value=True,
    ), mock.patch.object(
        controller,
        '_worker_wait_until_finished',
    ) as mock_worker_wait, mock.patch.object(
        controller,
        '_worker_close',
    ) as mock_worker_close:
      saved = controller.save(
          9,
          args_lib.Composite(state=args_lib.PyTreeSave(item=state)),
      )
      self.assertTrue(saved)
      self.assertTrue(handoff_started.wait(timeout=5))
      assert controller._pending_save is not None
      self.assertIsNotNone(controller._pending_save.keepalive)

      close_errors = []
      close_done = threading.Event()

      def _close():
        try:
          controller.close()
        except Exception as e:  # pylint: disable=broad-exception-caught
          close_errors.append(e)
        finally:
          close_done.set()

      close_thread = threading.Thread(target=_close)
      close_thread.start()
      self.assertFalse(close_done.wait(timeout=0.1))
      mock_worker_wait.assert_not_called()
      mock_worker_close.assert_not_called()

      release_handoff.set()
      close_thread.join(timeout=5)
      self.assertFalse(close_thread.is_alive())

    self.assertLen(close_errors, 1)
    self.assertIsInstance(close_errors[0], RuntimeError)
    self.assertIn('persistent failed', str(close_errors[0]))
    mock_worker_wait.assert_called_once_with()
    mock_worker_close.assert_called_once_with()
    self.assertIsNone(controller._pending_save)

  def test_prepare_state_for_save_does_not_block_on_colocated_state(self):
    controller, _ = self._make_controller_for_restore()
    state = {'x': jax.device_put(jnp.arange(1, dtype=jnp.int32))}

    with mock.patch.object(
        controller_lib,
        '_serialize_for_colocated_transport',
        side_effect=lambda value: value,
    ), mock.patch.object(
        controller_lib.colocated_transport,
        'to_colocated_python',
        return_value=state,
    ), mock.patch.object(
        controller_lib.jax, 'block_until_ready'
    ) as mock_block:
      result = controller._prepare_state_for_save(
          9,
          args_lib.Composite(state=args_lib.PyTreeSave(item=state)),
      )

    self.assertIs(result, state)
    mock_block.assert_not_called()

  def test_is_saving_in_progress_returns_worker_vote(self):
    controller, _ = self._make_controller_for_restore()
    controller._worker_manager = mock.Mock()
    result = object()
    controller._worker_manager.is_saving_in_progress.return_value = result

    with mock.patch.object(
        controller_lib.colocated_utils,
        'scalar_result_values',
        return_value=[False, False],
    ) as mock_values:
      in_progress = controller.is_saving_in_progress()

    self.assertFalse(in_progress)
    controller._worker_manager.is_saving_in_progress.assert_called_once_with(
        controller._dummy
    )
    mock_values.assert_called_once_with(
        result, op_name='is_saving_in_progress'
    )

  def test_is_saving_in_progress_allows_mixed_worker_status(self):
    controller, _ = self._make_controller_for_restore()
    controller._worker_manager = mock.Mock()
    result = object()
    controller._worker_manager.is_saving_in_progress.return_value = result

    with mock.patch.object(
        controller_lib.colocated_utils,
        'scalar_result_values',
        return_value=[True, True, False, False, False, False, False, True],
    ):
      self.assertTrue(controller.is_saving_in_progress())

  def test_is_saving_in_progress_rejects_non_bool_worker_status(self):
    controller, _ = self._make_controller_for_restore()
    controller._worker_manager = mock.Mock()
    controller._worker_manager.is_saving_in_progress.return_value = object()

    with mock.patch.object(
        controller_lib.colocated_utils,
        'scalar_result_values',
        return_value=[False, 1],
    ):
      with self.assertRaisesRegex(TypeError, 'expected worker results'):
        controller.is_saving_in_progress()

  def test_is_saving_in_progress_includes_pending_save(self):
    controller, _ = self._make_controller_for_restore()
    controller._worker_manager = mock.Mock()
    controller._pending_save = controller_lib._PendingSave(
        step=9,
        force=False,
        result=mock.sentinel.result,
        keepalive=(mock.sentinel.state,),
        dataset_args=None,
        start_time=0.0,
    )

    self.assertTrue(controller.is_saving_in_progress())

    controller._worker_manager.is_saving_in_progress.assert_not_called()

  def test_is_saving_in_progress_advances_dataset_save_after_handoff(self):
    controller, _ = self._make_controller_for_restore()
    controller._pending_save = controller_lib._PendingSave(
        step=9,
        force=True,
        result=mock.sentinel.result,
        keepalive=None,
        dataset_args=args_lib.Composite(dataset=mock.sentinel.dataset),
        start_time=0.0,
        handoff_done=True,
        handoff_saved=True,
    )

    with mock.patch.object(
        controller, '_save_persistent_dataset'
    ) as mock_save_dataset, mock.patch.object(
        controller, '_worker_is_saving_in_progress', return_value=False
    ) as mock_worker_in_progress:
      self.assertFalse(controller.is_saving_in_progress())

    mock_save_dataset.assert_called_once_with(
        9,
        args_lib.Composite(dataset=mock.sentinel.dataset),
        force=True,
    )
    mock_worker_in_progress.assert_called_once_with()
    assert controller._pending_save is not None
    self.assertTrue(controller._pending_save.saved)

  def test_is_saving_in_progress_polls_worker_after_keepalive_handoff(self):
    controller, _ = self._make_controller_for_restore()
    controller._pending_save = controller_lib._PendingSave(
        step=9,
        force=False,
        result=mock.sentinel.result,
        keepalive=None,
        dataset_args=None,
        start_time=0.0,
        handoff_done=True,
        handoff_saved=True,
        saved=True,
    )

    with mock.patch.object(
        controller, '_worker_is_saving_in_progress', return_value=False
    ) as mock_worker_in_progress:
      self.assertFalse(controller.is_saving_in_progress())

    mock_worker_in_progress.assert_called_once_with()

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
        'metadata': {
            'step': jax.device_put(
                jnp.asarray(7, dtype=jnp.int32), cpu_sharding
            )
        },
        'state': {
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
    self._set_worker_restore_result(
        controller, return_value=restored_cpu_state
    )

    result = controller.restore(
        7,
        args_lib.Composite(
            state=args_lib.PyTreeRestore(
                item=template_state,
                restore_args=restore_args,
                partial_restore=True,
            )
        ),
        default_item_mode=True,
    )

    controller._worker_manager.wait_until_finished.assert_not_called()
    controller._worker_manager.restore_infer.assert_called_once()
    specialize_kwargs = (
        controller._worker_manager.restore_infer.specialize.call_args.kwargs
    )
    self.assertIn('in_specs', specialize_kwargs)
    self.assertEqual(
        specialize_kwargs['devices'], controller._state_cpu_devices
    )
    self.assertNotIn('out_specs_fn', specialize_kwargs)
    _, partial_restore_arg = (
        controller._worker_manager.restore_infer.call_args.args
    )
    self.assertTrue(np.asarray(partial_restore_arg).item())
    self.assertEqual(
        jax.tree.structure(result), jax.tree.structure(template_state)
    )
    np.testing.assert_array_equal(
        np.asarray(result['weights']), np.arange(2, dtype=np.float32)
    )

  def test_restore_rejects_protocol_no_checkpoint_sentinel(self):
    controller, sharding = self._make_controller_for_restore()
    template_state = {
        'weights': jax.ShapeDtypeStruct((2,), jnp.float32, sharding=sharding)
    }
    restore_args = {
        'weights': type_handlers.ArrayRestoreArgs(sharding=sharding)
    }

    with self.assertRaisesRegex(ValueError, 'cannot restore step 0'):
      controller.restore(
          controller_lib.colocated_utils.NO_STEP_SENTINEL,
          args_lib.Composite(
              state=args_lib.PyTreeRestore(
                  item=template_state,
                  restore_args=restore_args,
              )
          ),
          default_item_mode=True,
      )

    controller._worker_manager.restore_infer.specialize.assert_not_called()

  def test_restore_rejects_shape_mismatch_before_final_remap(self):
    controller, sharding = self._make_controller_for_restore()
    mesh = sharding.mesh
    cpu_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    restored_cpu_state = {
        'params': {
            'weights': jax.device_put(
                jnp.arange(3, dtype=jnp.float32),
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
    self._set_worker_restore_result(controller, return_value=restored_cpu_state)

    with self.assertRaisesRegex(ValueError, 'unexpected shape'):
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

  def test_restore_rejects_dtype_mismatch_before_final_remap(self):
    controller, sharding = self._make_controller_for_restore()
    mesh = sharding.mesh
    cpu_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    restored_cpu_state = {
        'params': {
            'weights': jax.device_put(
                jnp.arange(2, dtype=jnp.int32),
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
    self._set_worker_restore_result(controller, return_value=restored_cpu_state)

    with self.assertRaisesRegex(ValueError, 'unexpected dtype'):
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

  def test_restore_rejects_restore_args_shape_conflict_with_item(self):
    controller, sharding = self._make_controller_for_restore()
    template_state = {
        'weights': jax.ShapeDtypeStruct((2,), jnp.float32, sharding=sharding)
    }
    restore_args = {
        'weights': type_handlers.ArrayRestoreArgs(
            sharding=sharding,
            global_shape=(3,),
            dtype=jnp.float32,
        )
    }

    with self.assertRaisesRegex(ValueError, 'shape transforms'):
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

  def test_restore_rejects_restore_args_structure_conflict_with_item(self):
    controller, sharding = self._make_controller_for_restore()
    template_state = {
        'params': {
            'weights': jax.ShapeDtypeStruct(
                (2,), jnp.float32, sharding=sharding
            )
        }
    }
    restore_args = {
        'weights': type_handlers.ArrayRestoreArgs(
            sharding=sharding,
            global_shape=(2,),
            dtype=jnp.float32,
        )
    }

    with self.assertRaisesRegex(ValueError, 'same pytree structure'):
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

  def test_restore_uses_item_sharding_when_restore_args_missing(self):
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
    self._set_worker_restore_result(controller, return_value=restored_cpu_state)

    result = controller.restore(
        7,
        args_lib.Composite(state=args_lib.PyTreeRestore(item=template_state)),
        default_item_mode=True,
    )

    self.assertEqual(
        jax.tree.structure(result), jax.tree.structure(template_state)
    )
    np.testing.assert_array_equal(
        np.asarray(result['weights']), np.arange(2, dtype=np.float32)
    )

  def test_restore_uses_metadata_item_when_restore_args_missing(self):
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
        'weights': metadata_value.ArrayMetadata(
            name='weights',
            directory=None,
            shape=(2,),
            sharding=sharding_metadata.from_jax_sharding(sharding),
            dtype=jnp.float32,
        )
    }
    self._set_worker_restore_result(controller, return_value=restored_cpu_state)

    result = controller.restore(
        7,
        args_lib.Composite(state=args_lib.PyTreeRestore(item=template_state)),
        default_item_mode=True,
    )

    np.testing.assert_array_equal(
        np.asarray(result['weights']), np.arange(2, dtype=np.float32)
    )

  def test_restore_accepts_worker_none_for_empty_namedtuple(self):
    controller, sharding = self._make_controller_for_restore()
    mesh = sharding.mesh
    cpu_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    restored_cpu_state = {
        'opt_state': [
            None,
            {
                'count': jax.device_put(
                    jnp.asarray(7, dtype=jnp.int32), cpu_sharding
                )
            },
        ],
        'weights': jax.device_put(
            jnp.arange(2, dtype=jnp.float32), cpu_sharding
        ),
    }
    template_state = {
        'opt_state': [
            EmptyState(),
            {'count': jax.ShapeDtypeStruct((), jnp.int32, sharding=sharding)},
        ],
        'weights': jax.ShapeDtypeStruct((2,), jnp.float32, sharding=sharding),
    }
    restore_args = {
        'opt_state': [
            EmptyState(),
            {'count': type_handlers.ArrayRestoreArgs(sharding=sharding)},
        ],
        'weights': type_handlers.ArrayRestoreArgs(sharding=sharding),
    }
    captured = {}
    self._set_worker_restore_result(
        controller, return_value=restored_cpu_state
    )

    def _specialize_fn(*, in_specs, devices):
      captured['in_specs'] = in_specs
      captured['devices'] = devices
      return controller._worker_manager.restore_infer

    controller._worker_manager.restore_infer.specialize.side_effect = (
        _specialize_fn
    )

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

    self.assertIn('in_specs', captured)
    self.assertEqual(captured['devices'], controller._state_cpu_devices)
    self.assertEqual(
        jax.tree.structure(result), jax.tree.structure(template_state)
    )
    self.assertEqual(result['opt_state'][0], EmptyState())
    np.testing.assert_array_equal(
        np.asarray(result['opt_state'][1]['count']), 7
    )
    np.testing.assert_array_equal(
        np.asarray(result['weights']), np.arange(2, dtype=np.float32)
    )

  def test_restore_no_item_rebuilds_empty_namedtuple_from_restore_args(self):
    controller, sharding = self._make_controller_for_restore()
    mesh = sharding.mesh
    cpu_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    restored_cpu_state = {
        'opt_state': [None],
        'weights': jax.device_put(
            jnp.arange(2, dtype=jnp.float32), cpu_sharding
        ),
    }
    restore_args = {
        'opt_state': [EmptyState()],
        'weights': type_handlers.ArrayRestoreArgs(
            sharding=sharding,
            global_shape=(2,),
            dtype=jnp.float32,
        ),
    }
    self._set_worker_restore_result(
        controller, return_value=restored_cpu_state
    )

    result = controller.restore(
        7,
        args_lib.Composite(
            state=args_lib.PyTreeRestore(
                restore_args=restore_args,
            )
        ),
        default_item_mode=True,
    )

    self.assertEqual(
        jax.tree.structure(result), jax.tree.structure(restore_args)
    )
    self.assertEqual(result['opt_state'][0], EmptyState())
    np.testing.assert_array_equal(
        np.asarray(result['weights']), np.arange(2, dtype=np.float32)
    )

  def test_restore_none_step_resolves_single_explicit_step(self):
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
    self._set_worker_restore_result(
        controller, return_value=restored_cpu_state
    )
    controller.latest_step = mock.Mock(return_value=11)

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

    step_arg, partial_restore_arg = (
        controller._worker_manager.restore_infer.call_args.args
    )
    self.assertEqual(np.asarray(step_arg).item(), 11)
    self.assertFalse(np.asarray(partial_restore_arg).item())
    controller.latest_step.assert_called_once_with()

  def test_restore_none_step_raises_when_no_workers_have_checkpoints(self):
    controller, sharding = self._make_controller_for_restore()
    template_state = {
        'weights': jax.ShapeDtypeStruct((2,), jnp.float32, sharding=sharding)
    }
    restore_args = {
        'weights': type_handlers.ArrayRestoreArgs(sharding=sharding)
    }
    controller.latest_step = mock.Mock(return_value=None)

    with self.assertRaisesRegex(FileNotFoundError, 'No steps found'):
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

    controller._worker_manager.restore_infer.assert_not_called()

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
    self._set_worker_restore_result(
        controller, return_value=restored_cpu_state
    )
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

    step_arg, partial_restore_arg = (
        controller._worker_manager.restore_infer.call_args.args
    )
    self.assertEqual(np.asarray(step_arg).item(), 11)
    self.assertFalse(np.asarray(partial_restore_arg).item())
    controller._persistent_checkpoint_manager.restore.assert_called_once()
    self.assertEqual(
        controller._persistent_checkpoint_manager.restore.call_args.args[0], 11
    )
    self.assertEqual(result['dataset'], 'dataset-state')

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
    self._set_worker_restore_result(
        controller,
        side_effect=[
            RuntimeError('transient'),
            restored_cpu_state,
        ],
    )

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

  def test_restore_does_not_retry_type_error(self):
    controller, sharding = self._make_controller_for_restore()
    template_state = {
        'weights': jax.ShapeDtypeStruct((2,), jnp.float32, sharding=sharding)
    }
    restore_args = {
        'weights': type_handlers.ArrayRestoreArgs(sharding=sharding)
    }
    first_restore_call = mock.Mock(side_effect=TypeError('bad restore type'))
    controller._worker_manager.restore_infer.specialize.side_effect = None
    controller._worker_manager.restore_infer.specialize.return_value = (
        first_restore_call
    )

    with self.assertRaisesRegex(TypeError, 'bad restore type'):
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

    controller._worker_manager.restore_infer.specialize.assert_called_once()
    first_restore_call.assert_called_once()

  def test_restore_does_not_retry_non_retriable_error(self):
    controller, sharding = self._make_controller_for_restore()
    template_state = {
        'weights': jax.ShapeDtypeStruct((2,), jnp.float32, sharding=sharding)
    }
    restore_args = {
        'weights': type_handlers.ArrayRestoreArgs(sharding=sharding)
    }
    self._set_worker_restore_result(
        controller,
        side_effect=ValueError('bad restore'),
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


if __name__ == '__main__':
  absltest.main()

