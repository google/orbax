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

"""Tests for ArrayHandler device list validation and dispatcher restores."""

from collections.abc import Sequence
import dataclasses
from typing import Any
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import options as options_lib
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.serialization import jax_array_handlers
from orbax.checkpoint._src.serialization import limits
from orbax.checkpoint._src.serialization import type_handler_registry
from orbax.checkpoint._src.serialization import types

ArrayRestoreArgs = jax_array_handlers.ArrayRestoreArgs

# Each test array has 256 float32 values (1 KiB) and is evenly sharded over
# all devices. The local dispatcher runs everything in this process, so all
# devices belong to a single worker, which then holds 1 KiB per array.
_ARRAY_SIZE = 256
_ARRAY_BYTES = _ARRAY_SIZE * 4


@dataclasses.dataclass(frozen=True)
class _DispatchCall:
  """A call recorded by `_LocalDispatcher`."""

  func_name: str
  result_specs: Any
  func_kwargs: dict[str, Any]


class _LocalDispatcher(dispatchers.Dispatcher):
  """Runs dispatched functions in-process and records each call."""

  def __init__(self):
    self.calls: list[_DispatchCall] = []

  def name(self) -> str:
    return 'local'

  def device_to_host(self, arrays: Any) -> Any:
    return arrays

  def dispatch(
      self,
      func,
      *,
      input_arrays=None,
      result_specs=None,
      func_args=(),
      func_kwargs=None,
  ):
    func_kwargs = func_kwargs or {}
    self.calls.append(_DispatchCall(func.__name__, result_specs, func_kwargs))
    if input_arrays is not None:
      func_args = (input_arrays, *func_args)
    return func(*func_args, **func_kwargs)

  def calls_to(self, func_name: str) -> list[_DispatchCall]:
    """Returns the calls that dispatched `func_name`, in order."""
    return [call for call in self.calls if call.func_name == func_name]

  def batches(self, func_name: str) -> list[list[str]]:
    """Returns the parameter names passed to each call of `func_name`."""
    return [
        [info.name for info in call.func_kwargs['infos']]
        for call in self.calls_to(func_name)
    ]


class _LocalColocatedDispatcher(
    _LocalDispatcher, dispatchers.ColocatedPythonDispatcher
):
  """ColocatedPythonDispatcher that runs dispatched functions in-process."""


def _param_info(
    name: str, directory: epath.Path, **kwargs: Any
) -> types.ParamInfo:
  return types.ParamInfo(
      name=name,
      parent_dir=directory,
      keypath=(jax.tree_util.DictKey(name),),
      is_ocdbt_checkpoint=False,
      **kwargs,
  )


def _sharded_arrays(num_arrays: int) -> list[jax.Array]:
  sharding = jax.sharding.NamedSharding(
      jax.sharding.Mesh(np.asarray(jax.devices()), ('x',)),
      jax.sharding.PartitionSpec('x'),
  )
  return [
      jax.device_put(np.arange(_ARRAY_SIZE, dtype=np.float32) + i, sharding)
      for i in range(num_arrays)
  ]


def _distinct_arrays(num_arrays: int) -> list[jax.Array]:
  """Returns test arrays that differ in values, shape and sharding.

  Array `i` has shape `(256 / 2**i, 2**i)` and is split along its first axis
  over all devices, with the device order rotated by `i`.

  Args:
    num_arrays: The number of arrays to return.
  """
  arrays = []
  for i in range(num_arrays):
    mesh = jax.sharding.Mesh(np.roll(np.asarray(jax.devices()), i), ('x',))
    value = np.arange(_ARRAY_SIZE, dtype=np.float32) + i
    arrays.append(
        jax.device_put(
            value.reshape(_ARRAY_SIZE // 2**i, 2**i),
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x')),
        )
    )
  return arrays


def _restore_args(arrays: Sequence[jax.Array]) -> list[ArrayRestoreArgs]:
  return [
      ArrayRestoreArgs(sharding=a.sharding, global_shape=a.shape, dtype=a.dtype)
      for a in arrays
  ]


class _StepPriorityCallback(types.DefaultSerializationStatusCallback):
  """Gives 'step' `step_priority` and every other array ASYNC_DEPRIORITIZED."""

  def __init__(self, step_priority: types.TransferPriority):
    self._step_priority = step_priority

  def key_priority(self, keypath: Any) -> types.TransferPriority:
    if keypath[-1].key == 'step':
      return self._step_priority
    return types.TransferPriority.ASYNCHRONOUS_DEPRIORITIZED


class DeviceListValidationTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def setUp(self):
    super().setUp()
    test_utils.set_tensorstore_driver_for_test()
    self.directory = epath.Path(self.create_tempdir().full_path)

  @parameterized.named_parameters(
      dict(testcase_name='no_limit', device_host_max_bytes=None),
      # With the default callback, all arrays still go into one dispatch.
      dict(testcase_name='limit', device_host_max_bytes=10 * _ARRAY_BYTES),
  )
  async def test_colocated_save_fails_before_dispatching(
      self, device_host_max_bytes
  ):
    if jax.device_count() < 2:
      self.skipTest('Requires at least 2 devices.')
    params = _sharded_arrays(1)[0]
    step = jax.device_put(
        np.int32(1), jax.sharding.SingleDeviceSharding(jax.devices()[1])
    )
    limiter = limits.get_byte_limiter(device_host_max_bytes)
    infos = [
        _param_info('params', self.directory, device_host_byte_limiter=limiter),
        _param_info('step', self.directory, device_host_byte_limiter=limiter),
    ]
    dispatcher = dispatchers.ColocatedPythonDispatcher()
    handler = jax_array_handlers.ArrayHandler(dispatcher=dispatcher)

    with mock.patch.object(
        dispatcher, 'dispatch', autospec=True, spec_set=True
    ) as mock_dispatch, mock.patch.object(
        dispatcher, 'device_to_host', autospec=True, spec_set=True
    ) as mock_device_to_host:
      with self.assertRaises(ValueError) as cm:
        await handler.serialize([params, step], infos)

    message = str(cm.exception)
    self.assertIn(f'{jax.device_count()} device(s): params', message)
    self.assertIn('1 device(s): step', message)
    mock_dispatch.assert_not_called()
    mock_device_to_host.assert_not_called()

  async def test_colocated_save_allows_replicated_scalar_on_same_mesh(self):
    if jax.device_count() < 2:
      self.skipTest('Requires at least 2 devices.')
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('x',))
    params = jax.device_put(
        np.arange(_ARRAY_SIZE, dtype=np.float32),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x')),
    )
    step = jax.device_put(
        np.int32(1),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
    )
    infos = [
        _param_info('params', self.directory),
        _param_info('step', self.directory),
    ]
    dispatcher = _LocalColocatedDispatcher()
    handler = jax_array_handlers.ArrayHandler(dispatcher=dispatcher)

    futures = await handler.serialize([params, step], infos)
    for f in futures:
      f.result()

    self.assertEqual(
        dispatcher.batches('_worker_serialize_arrays'), [['params', 'step']]
    )

  async def test_colocated_save_allows_device_lists_in_separate_dispatches(
      self,
  ):
    if jax.device_count() < 2:
      self.skipTest('Requires at least 2 devices.')
    params = _sharded_arrays(1)[0]
    step = jax.device_put(
        np.int32(1), jax.sharding.SingleDeviceSharding(jax.devices()[1])
    )
    limiter = limits.get_byte_limiter(10 * _ARRAY_BYTES)
    infos = [
        _param_info('params', self.directory, device_host_byte_limiter=limiter),
        _param_info('step', self.directory, device_host_byte_limiter=limiter),
    ]
    dispatcher = _LocalColocatedDispatcher()
    handler = jax_array_handlers.ArrayHandler(
        dispatcher=dispatcher,
        callback=_StepPriorityCallback(types.TransferPriority.SYNCHRONOUS),
    )

    futures = await handler.serialize([params, step], infos)
    for f in futures:
      f.result()

    self.assertEqual(
        dispatcher.batches('_worker_serialize_arrays'), [['step'], ['params']]
    )

  async def test_colocated_save_fails_when_a_dispatch_mixes_device_lists(self):
    if jax.device_count() < 2:
      self.skipTest('Requires at least 2 devices.')
    params = _sharded_arrays(1)[0]
    step = jax.device_put(
        np.int32(1), jax.sharding.SingleDeviceSharding(jax.devices()[1])
    )
    # Large enough for both arrays to share one memory-budgeted dispatch.
    limiter = limits.get_byte_limiter(10 * _ARRAY_BYTES)
    infos = [
        _param_info('params', self.directory, device_host_byte_limiter=limiter),
        _param_info('step', self.directory, device_host_byte_limiter=limiter),
    ]
    dispatcher = _LocalColocatedDispatcher()
    handler = jax_array_handlers.ArrayHandler(
        dispatcher=dispatcher,
        callback=_StepPriorityCallback(
            types.TransferPriority.ASYNCHRONOUS_DEPRIORITIZED
        ),
    )

    futures = await handler.serialize([params, step], infos)
    with self.assertRaises(ValueError) as cm:
      for f in futures:
        f.result()

    message = str(cm.exception)
    self.assertIn(f'{jax.device_count()} device(s): params', message)
    self.assertIn('1 device(s): step', message)
    self.assertEmpty(dispatcher.batches('_worker_serialize_arrays'))

  async def test_colocated_save_fails_on_different_mesh_device_order(self):
    if jax.device_count() < 2:
      self.skipTest('Requires at least 2 devices.')
    forward_mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('x',))
    reverse_mesh = jax.sharding.Mesh(np.asarray(jax.devices()[::-1]), ('x',))
    a = jax.device_put(
        np.arange(_ARRAY_SIZE, dtype=np.float32),
        jax.sharding.NamedSharding(
            forward_mesh, jax.sharding.PartitionSpec('x')
        ),
    )
    b = jax.device_put(
        np.arange(_ARRAY_SIZE, dtype=np.float32),
        jax.sharding.NamedSharding(
            reverse_mesh, jax.sharding.PartitionSpec('x')
        ),
    )
    infos = [
        _param_info('a', self.directory),
        _param_info('b', self.directory),
    ]
    dispatcher = dispatchers.ColocatedPythonDispatcher()
    handler = jax_array_handlers.ArrayHandler(dispatcher=dispatcher)

    with mock.patch.object(
        dispatcher, 'dispatch', autospec=True, spec_set=True
    ) as mock_dispatch, mock.patch.object(
        dispatcher, 'device_to_host', autospec=True, spec_set=True
    ) as mock_device_to_host:
      with self.assertRaises(ValueError) as cm:
        await handler.serialize([a, b], infos)

    message = str(cm.exception)
    self.assertIn('found 2 device lists', message)
    self.assertIn(f'{jax.device_count()} device(s): a', message)
    self.assertIn(f'{jax.device_count()} device(s): b', message)
    mock_dispatch.assert_not_called()
    mock_device_to_host.assert_not_called()

  async def test_non_colocated_dispatcher_skips_validation(self):
    if jax.device_count() < 2:
      self.skipTest('Requires at least 2 devices.')
    params = _sharded_arrays(1)[0]
    step = jax.device_put(
        np.int32(1), jax.sharding.SingleDeviceSharding(jax.devices()[1])
    )
    infos = [
        _param_info('params', self.directory),
        _param_info('step', self.directory),
    ]
    dispatcher = _LocalDispatcher()
    handler = jax_array_handlers.ArrayHandler(dispatcher=dispatcher)

    futures = await handler.serialize([params, step], infos)
    for f in futures:
      f.result()

    self.assertEqual(
        dispatcher.batches('_worker_serialize_arrays'), [['params', 'step']]
    )

  async def test_colocated_restore_fails_before_batching_or_dispatching(self):
    if jax.device_count() < 2:
      self.skipTest('Requires at least 2 devices.')
    params = _sharded_arrays(1)[0]
    step = jax.device_put(
        np.int32(1), jax.sharding.SingleDeviceSharding(jax.devices()[0])
    )
    # Limit is tight enough that params and step would fall in separate batches
    # if batching ran before validation.
    limiter = limits.get_byte_limiter(_ARRAY_BYTES)
    infos = [
        _param_info('params', self.directory, device_host_byte_limiter=limiter),
        _param_info('step', self.directory, device_host_byte_limiter=limiter),
    ]
    restore_args = [
        ArrayRestoreArgs(
            sharding=params.sharding,
            global_shape=params.shape,
            dtype=params.dtype,
        ),
        ArrayRestoreArgs(
            sharding=step.sharding, global_shape=step.shape, dtype=step.dtype
        ),
    ]
    dispatcher = dispatchers.ColocatedPythonDispatcher()
    handler = jax_array_handlers.ArrayHandler(dispatcher=dispatcher)

    with mock.patch.object(
        dispatcher, 'dispatch', autospec=True, spec_set=True
    ) as mock_dispatch:
      with self.assertRaises(ValueError) as cm:
        await handler.deserialize(infos, restore_args)

    message = str(cm.exception)
    self.assertIn(f'{jax.device_count()} device(s): params', message)
    self.assertIn('1 device(s): step', message)
    mock_dispatch.assert_not_called()

  async def test_colocated_restore_fails_on_different_mesh_device_order(self):
    if jax.device_count() < 2:
      self.skipTest('Requires at least 2 devices.')
    forward_mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('x',))
    reverse_mesh = jax.sharding.Mesh(np.asarray(jax.devices()[::-1]), ('x',))
    infos = [
        _param_info('a', self.directory),
        _param_info('b', self.directory),
    ]
    restore_args = [
        ArrayRestoreArgs(
            sharding=jax.sharding.NamedSharding(
                forward_mesh, jax.sharding.PartitionSpec('x')
            ),
            global_shape=(_ARRAY_SIZE,),
            dtype=np.dtype(np.float32),
        ),
        ArrayRestoreArgs(
            sharding=jax.sharding.NamedSharding(
                reverse_mesh, jax.sharding.PartitionSpec('x')
            ),
            global_shape=(_ARRAY_SIZE,),
            dtype=np.dtype(np.float32),
        ),
    ]
    dispatcher = dispatchers.ColocatedPythonDispatcher()
    handler = jax_array_handlers.ArrayHandler(dispatcher=dispatcher)

    with mock.patch.object(
        dispatcher, 'dispatch', autospec=True, spec_set=True
    ) as mock_dispatch:
      with self.assertRaises(ValueError) as cm:
        await handler.deserialize(infos, restore_args)

    message = str(cm.exception)
    self.assertIn('found 2 device lists', message)
    self.assertIn(f'{jax.device_count()} device(s): a', message)
    self.assertIn(f'{jax.device_count()} device(s): b', message)
    mock_dispatch.assert_not_called()


class DispatcherRestoreBatchingTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def setUp(self):
    super().setUp()
    test_utils.set_tensorstore_driver_for_test()
    self.directory = epath.Path(self.create_tempdir().full_path)

  def _infos(
      self, names: list[str], device_host_max_bytes: int | None = None
  ) -> list[types.ParamInfo]:
    limiter = limits.get_byte_limiter(device_host_max_bytes)
    return [
        _param_info(name, self.directory, device_host_byte_limiter=limiter)
        for name in names
    ]

  @parameterized.named_parameters(
      dict(
          testcase_name='no_limit',
          device_host_max_bytes=None,
          expected_batches=[['a0', 'a1', 'a2', 'a3']],
      ),
      dict(
          testcase_name='two_arrays_per_batch',
          device_host_max_bytes=int(2.5 * _ARRAY_BYTES),
          expected_batches=[['a0', 'a1'], ['a2', 'a3']],
      ),
      dict(
          testcase_name='array_larger_than_limit',
          device_host_max_bytes=_ARRAY_BYTES // 2,
          expected_batches=[['a0'], ['a1'], ['a2'], ['a3']],
      ),
  )
  async def test_restore_is_batched_by_device_host_limit(
      self, device_host_max_bytes, expected_batches
  ):
    names = ['a0', 'a1', 'a2', 'a3']
    arrays = _distinct_arrays(len(names))
    futures = await jax_array_handlers.ArrayHandler().serialize(
        arrays, self._infos(names)
    )
    for f in futures:
      f.result()
    dispatcher = _LocalDispatcher()
    handler = jax_array_handlers.ArrayHandler(dispatcher=dispatcher)
    restore_args = _restore_args(arrays)

    restored = await handler.deserialize(
        self._infos(names, device_host_max_bytes), restore_args
    )

    calls = dispatcher.calls_to('_sync_deserialize_arrays')
    self.assertEqual(
        [[info.name for info in call.func_kwargs['infos']] for call in calls],
        expected_batches,
    )
    # Each call gets the same slice of every per-parameter argument. The arrays
    # differ in shape and sharding, so misaligned slices would not match.
    args_by_name = dict(zip(names, restore_args))
    expected_args = [
        [args_by_name[name] for name in batch] for batch in expected_batches
    ]
    self.assertEqual(
        [call.func_kwargs['args'] for call in calls], expected_args
    )
    self.assertEqual(
        [call.func_kwargs['shardings'] for call in calls],
        [[arg.sharding for arg in batch] for batch in expected_args],
    )
    self.assertEqual(
        [[(s.shape, s.sharding) for s in call.result_specs] for call in calls],
        [
            [(arg.global_shape, arg.sharding) for arg in batch]
            for batch in expected_args
        ],
    )
    # Restored arrays are returned in their original order.
    test_utils.assert_tree_equal(self, arrays, list(restored))

  @parameterized.named_parameters(
      dict(
          testcase_name='host_offloaded_without_limit',
          memory_kind='pinned_host',
          device_host_max_bytes=None,
          expect_warning=True,
      ),
      dict(
          testcase_name='host_offloaded_with_limit',
          memory_kind='pinned_host',
          device_host_max_bytes=int(2.5 * _ARRAY_BYTES),
          expect_warning=False,
      ),
      dict(
          testcase_name='device_without_limit',
          memory_kind='device',
          device_host_max_bytes=None,
          expect_warning=False,
      ),
  )
  async def test_warns_about_unbounded_host_offloaded_restore(
      self, memory_kind, device_host_max_bytes, expect_warning
  ):
    names = ['a0', 'a1']
    arrays = _sharded_arrays(len(names))
    futures = await jax_array_handlers.ArrayHandler().serialize(
        arrays, self._infos(names)
    )
    for f in futures:
      f.result()
    handler = jax_array_handlers.ArrayHandler(dispatcher=_LocalDispatcher())
    restore_args = [
        ArrayRestoreArgs(
            sharding=a.sharding.with_memory_kind(memory_kind),
            global_shape=a.shape,
            dtype=a.dtype,
        )
        for a in arrays
    ]

    with mock.patch.object(
        jax_array_handlers.logging, 'warning', autospec=True
    ) as mock_warning:
      restored = await handler.deserialize(
          self._infos(names, device_host_max_bytes), restore_args
      )

    warned = any(
        'restore_device_host_concurrent_gb' in call.args[0]
        for call in mock_warning.call_args_list
    )
    self.assertEqual(warned, expect_warning)
    test_utils.assert_tree_equal(self, arrays, list(restored))


class PyTreeCheckpointHandlerRestoreLimitTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    test_utils.set_tensorstore_driver_for_test()
    self.directory = epath.Path(self.create_tempdir().full_path)

  @parameterized.named_parameters(
      dict(
          testcase_name='restore_limit_only',
          save_gb=None,
          restore_gb=1,
          expected_max_bytes=10**9,
      ),
      dict(
          testcase_name='save_limit_does_not_bound_restore',
          save_gb=2,
          restore_gb=None,
          expected_max_bytes=None,
      ),
      dict(
          testcase_name='explicit_restore_limit',
          save_gb=2,
          restore_gb=1,
          expected_max_bytes=10**9,
      ),
      dict(
          testcase_name='auto_save_limit',
          save_gb='auto',
          restore_gb=None,
          expected_max_bytes=None,
      ),
      dict(
          testcase_name='no_limits',
          save_gb=None,
          restore_gb=None,
          expected_max_bytes=None,
      ),
  )
  def test_dispatcher_restore_limit(
      self, save_gb, restore_gb, expected_max_bytes
  ):
    tree = dict(zip(['a0', 'a1'], _sharded_arrays(2)))
    save_handler = pytree_checkpoint_handler.PyTreeCheckpointHandler(
        use_ocdbt=False
    )
    self.addCleanup(save_handler.close)
    save_handler.save(
        self.directory, args=pytree_checkpoint_handler.PyTreeSaveArgs(tree)
    )
    dispatcher = _LocalDispatcher()
    registry = type_handler_registry.create_type_handler_registry(
        (jax.Array, jax_array_handlers.ArrayHandler(dispatcher=dispatcher))
    )
    handler = pytree_checkpoint_handler.PyTreeCheckpointHandler(
        save_device_host_concurrent_gb=save_gb,
        restore_device_host_concurrent_gb=restore_gb,
        # Required when save_device_host_concurrent_gb is "auto".
        memory_limit_options=options_lib.MemoryLimitOptions(
            max_transfer_concurrent_gb=80
        ),
        type_handler_registry=registry,
    )
    self.addCleanup(handler.close)
    restore_args = jax.tree.map(
        lambda a: ArrayRestoreArgs(
            sharding=a.sharding, global_shape=a.shape, dtype=a.dtype
        ),
        tree,
    )

    restored = handler.restore(
        self.directory,
        args=pytree_checkpoint_handler.PyTreeRestoreArgs(
            restore_args=restore_args
        ),
    )

    [call] = dispatcher.calls_to('_sync_deserialize_arrays')
    limiter = call.func_kwargs['infos'][0].device_host_byte_limiter
    max_bytes = (
        limiter.max_bytes
        if isinstance(limiter, limits.LimitInFlightBytes)
        else None
    )
    self.assertEqual(max_bytes, expected_max_bytes)
    test_utils.assert_tree_equal(self, tree, restored)

  @parameterized.parameters(0, -1, 'auto')
  def test_invalid_restore_limit_raises(self, restore_gb):
    with self.assertRaisesRegex(
        ValueError, 'restore_device_host_concurrent_gb must be a positive'
    ):
      pytree_checkpoint_handler.PyTreeCheckpointHandler(
          restore_device_host_concurrent_gb=restore_gb
      )


if __name__ == '__main__':
  absltest.main()
