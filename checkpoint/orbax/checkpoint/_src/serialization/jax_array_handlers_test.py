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
import os
from typing import Any
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
from jax.extend import backend as jax_backend
import numpy as np
from orbax.checkpoint import options as options_lib
from orbax.checkpoint import pathways as ocp_pathways
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.multihost import colocated_transport
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.serialization import jax_array_handlers
from orbax.checkpoint._src.serialization import limits
from orbax.checkpoint._src.serialization import type_handler_registry
from orbax.checkpoint._src.serialization import types
from orbax.checkpoint._src.serialization import worker_memory_utils

ArrayRestoreArgs = jax_array_handlers.ArrayRestoreArgs

_REQUIRED_CPU_DEVICES = 8
_PREV_NUM_CPU_DEVICES: int | None = None


def setUpModule():
  global _PREV_NUM_CPU_DEVICES
  prev = int(getattr(jax.config, 'jax_num_cpu_devices'))
  if (
      prev >= _REQUIRED_CPU_DEVICES
      or '--xla_force_host_platform_device_count'
      in os.environ.get('XLA_FLAGS', '')
  ):
    return
  try:
    jax.config.update('jax_num_cpu_devices', _REQUIRED_CPU_DEVICES)
    _PREV_NUM_CPU_DEVICES = prev
  except RuntimeError:
    if jax.default_backend() == 'cpu' and (
        jax.device_count() < _REQUIRED_CPU_DEVICES
    ):
      jax_backend.clear_backends()
      jax.config.update('jax_num_cpu_devices', _REQUIRED_CPU_DEVICES)
      _PREV_NUM_CPU_DEVICES = prev
      worker_memory_utils._device_to_worker_ids.cache_clear()  # pylint: disable=protected-access
      colocated_transport._get_cpu_device_map.cache_clear()  # pylint: disable=protected-access


def tearDownModule():
  global _PREV_NUM_CPU_DEVICES
  if _PREV_NUM_CPU_DEVICES is not None:
    jax_backend.clear_backends()
    jax.config.update('jax_num_cpu_devices', _PREV_NUM_CPU_DEVICES)
    worker_memory_utils._device_to_worker_ids.cache_clear()  # pylint: disable=protected-access
    colocated_transport._get_cpu_device_map.cache_clear()  # pylint: disable=protected-access
    _PREV_NUM_CPU_DEVICES = None

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
      dict(
          testcase_name='limit_default_callback',
          device_host_max_bytes=10 * _ARRAY_BYTES,
      ),
      # Even when save priority or async-copy settings would split `params` and
      # `step` into separate dispatches, colocated saves must reject the mixed
      # device lists upfront so they cannot write an unrestorable checkpoint.
      dict(
          testcase_name='limit_separate_priority_batches',
          device_host_max_bytes=_ARRAY_BYTES,
          step_priority=types.TransferPriority.SYNCHRONOUS,
      ),
      dict(
          testcase_name='limit_deprioritized_batches',
          device_host_max_bytes=_ARRAY_BYTES,
          step_priority=types.TransferPriority.ASYNCHRONOUS_DEPRIORITIZED,
      ),
      dict(
          testcase_name='copy_device_arrays_for_async_transfer',
          device_host_max_bytes=None,
          copy_device_arrays=True,
      ),
  )
  async def test_colocated_save_fails_before_dispatching(
      self,
      device_host_max_bytes,
      step_priority=None,
      copy_device_arrays=False,
  ):
    if jax.device_count() < 2:
      self.skipTest('Requires at least 2 devices.')
    params = _sharded_arrays(1)[0]
    step = jax.device_put(
        np.int32(1), jax.sharding.SingleDeviceSharding(jax.devices()[1])
    )
    limiter = limits.get_byte_limiter(device_host_max_bytes)
    infos = [
        _param_info(
            'params',
            self.directory,
            device_host_byte_limiter=limiter,
            copy_device_arrays_for_async_transfer=copy_device_arrays,
        ),
        _param_info(
            'step',
            self.directory,
            device_host_byte_limiter=limiter,
            copy_device_arrays_for_async_transfer=copy_device_arrays,
        ),
    ]
    callback = (
        _StepPriorityCallback(step_priority)
        if step_priority is not None
        else None
    )
    dispatcher = dispatchers.ColocatedPythonDispatcher()
    handler = jax_array_handlers.ArrayHandler(
        dispatcher=dispatcher, callback=callback
    )

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
    self.assertFalse((self.directory / '_sharding').exists())

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
    infos = [
        _param_info('params', self.directory),
        _param_info('step', self.directory),
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
    # Limit is tight enough that params and step would fall in separate batches
    # if batching ran before validation.
    handler = jax_array_handlers.ArrayHandler(
        dispatcher=dispatcher,
        dispatcher_restore_concurrent_bytes=_ARRAY_BYTES,
    )

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

  def _infos(self, names: list[str]) -> list[types.ParamInfo]:
    return [_param_info(name, self.directory) for name in names]

  @parameterized.named_parameters(
      dict(
          testcase_name='no_limit',
          restore_concurrent_bytes=None,
          expected_batches=[['a0', 'a1', 'a2', 'a3']],
      ),
      dict(
          testcase_name='two_arrays_per_batch',
          restore_concurrent_bytes=int(2.5 * _ARRAY_BYTES),
          expected_batches=[['a0', 'a1'], ['a2', 'a3']],
      ),
      dict(
          testcase_name='array_larger_than_limit',
          restore_concurrent_bytes=_ARRAY_BYTES // 2,
          expected_batches=[['a0'], ['a1'], ['a2'], ['a3']],
      ),
  )
  async def test_restore_is_batched_by_dispatcher_limit(
      self, restore_concurrent_bytes, expected_batches
  ):
    names = ['a0', 'a1', 'a2', 'a3']
    arrays = _distinct_arrays(len(names))
    futures = await jax_array_handlers.ArrayHandler().serialize(
        arrays, self._infos(names)
    )
    for f in futures:
      f.result()
    dispatcher = _LocalDispatcher()
    handler = jax_array_handlers.ArrayHandler(
        dispatcher=dispatcher,
        dispatcher_restore_concurrent_bytes=restore_concurrent_bytes,
    )
    restore_args = _restore_args(arrays)

    restored = await handler.deserialize(self._infos(names), restore_args)

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

  async def test_batched_restore_preserves_typed_prng_keys(self):
    names = ['k0', 'k1', 'k2']
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.asarray(jax.devices()), ('x',)),
        jax.sharding.PartitionSpec('x'),
    )
    keys = [
        jax.device_put(
            jax.random.split(
                jax.random.key(i, impl='threefry2x32'), _ARRAY_SIZE
            ),
            sharding,
        )
        for i in range(len(names))
    ]
    key_bytes = _ARRAY_SIZE * keys[0].dtype.itemsize
    metadata_store = array_metadata_store_lib.Store()
    save_handler = jax_array_handlers.ArrayHandler(
        array_metadata_store=metadata_store
    )
    futures = await save_handler.serialize(keys, self._infos(names))
    for f in futures:
      f.result()

    dispatcher = _LocalDispatcher()
    handler = jax_array_handlers.ArrayHandler(
        array_metadata_store=metadata_store,
        dispatcher=dispatcher,
        dispatcher_restore_concurrent_bytes=int(2.5 * key_bytes),
    )

    restored = await handler.deserialize(
        self._infos(names), _restore_args(keys)
    )

    self.assertEqual(
        dispatcher.batches('_sync_deserialize_arrays'),
        [['k0', 'k1'], ['k2']],
    )
    for r in restored:
      self.assertTrue(jax.dtypes.issubdtype(r.dtype, jax.dtypes.prng_key))
    test_utils.assert_tree_equal(self, keys, list(restored))

  @parameterized.named_parameters(
      dict(
          testcase_name='colocated_pinned_host_without_limit',
          dispatcher_cls=_LocalColocatedDispatcher,
          memory_kind='pinned_host',
          restore_concurrent_bytes=None,
          expect_warning=True,
      ),
      dict(
          testcase_name='colocated_pinned_host_with_limit',
          dispatcher_cls=_LocalColocatedDispatcher,
          memory_kind='pinned_host',
          restore_concurrent_bytes=int(2.5 * _ARRAY_BYTES),
          expect_warning=False,
      ),
      dict(
          testcase_name='colocated_unpinned_host_without_limit',
          dispatcher_cls=_LocalColocatedDispatcher,
          memory_kind='unpinned_host',
          restore_concurrent_bytes=None,
          expect_warning=True,
      ),
      dict(
          testcase_name='colocated_unpinned_host_with_limit',
          dispatcher_cls=_LocalColocatedDispatcher,
          memory_kind='unpinned_host',
          restore_concurrent_bytes=int(2.5 * _ARRAY_BYTES),
          expect_warning=False,
      ),
      dict(
          testcase_name='colocated_device_without_limit',
          dispatcher_cls=_LocalColocatedDispatcher,
          memory_kind='device',
          restore_concurrent_bytes=None,
          expect_warning=False,
      ),
      dict(
          testcase_name='non_colocated_host_offloaded_without_limit',
          dispatcher_cls=_LocalDispatcher,
          memory_kind='pinned_host',
          restore_concurrent_bytes=None,
          expect_warning=False,
      ),
  )
  async def test_warns_about_unbounded_host_offloaded_restore(
      self,
      dispatcher_cls,
      memory_kind,
      restore_concurrent_bytes,
      expect_warning,
  ):
    names = ['a0', 'a1']
    arrays = _sharded_arrays(len(names))
    futures = await jax_array_handlers.ArrayHandler().serialize(
        arrays, self._infos(names)
    )
    for f in futures:
      f.result()
    handler = jax_array_handlers.ArrayHandler(
        dispatcher=dispatcher_cls(),
        dispatcher_restore_concurrent_bytes=restore_concurrent_bytes,
    )
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
      restored = await handler.deserialize(self._infos(names), restore_args)

    warned = any(
        'dispatcher_restore_concurrent_bytes' in call.args[0]
        for call in mock_warning.call_args_list
    )
    self.assertEqual(warned, expect_warning)
    test_utils.assert_tree_equal(self, arrays, list(restored))


class ArrayHandlerRestoreLimitTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    test_utils.set_tensorstore_driver_for_test()
    self.directory = epath.Path(self.create_tempdir().full_path)

  @parameterized.named_parameters(
      dict(
          testcase_name='restore_limit_only',
          save_gb=None,
          restore_bytes=int(2.5 * _ARRAY_BYTES),
          expected_batches=[['a0', 'a1'], ['a2', 'a3']],
      ),
      dict(
          testcase_name='save_limit_does_not_bound_restore',
          save_gb=1,
          restore_bytes=None,
          expected_batches=[['a0', 'a1', 'a2', 'a3']],
      ),
      dict(
          testcase_name='explicit_restore_limit_with_save_limit',
          save_gb=2,
          restore_bytes=int(2.5 * _ARRAY_BYTES),
          expected_batches=[['a0', 'a1'], ['a2', 'a3']],
      ),
      dict(
          testcase_name='auto_save_limit_does_not_bound_restore',
          save_gb='auto',
          restore_bytes=None,
          expected_batches=[['a0', 'a1', 'a2', 'a3']],
      ),
      # `transforms` restores through a separate code path.
      dict(
          testcase_name='transforms_restore_limit',
          save_gb=None,
          restore_bytes=int(2.5 * _ARRAY_BYTES),
          expected_batches=[['a0', 'a1'], ['a2', 'a3']],
          transforms={},
      ),
      dict(
          testcase_name='transforms_no_limit',
          save_gb=None,
          restore_bytes=None,
          expected_batches=[['a0', 'a1', 'a2', 'a3']],
          transforms={},
      ),
  )
  def test_pytree_restore_uses_array_handler_limit(
      self, save_gb, restore_bytes, expected_batches, transforms=None
  ):
    names = ['a0', 'a1', 'a2', 'a3']
    tree = dict(zip(names, _sharded_arrays(len(names))))
    save_handler = pytree_checkpoint_handler.PyTreeCheckpointHandler(
        use_ocdbt=False
    )
    self.addCleanup(save_handler.close)
    save_handler.save(
        self.directory, args=pytree_checkpoint_handler.PyTreeSaveArgs(tree)
    )
    dispatcher = _LocalDispatcher()
    registry = type_handler_registry.create_type_handler_registry((
        jax.Array,
        jax_array_handlers.ArrayHandler(
            dispatcher=dispatcher,
            dispatcher_restore_concurrent_bytes=restore_bytes,
        ),
    ))
    handler = pytree_checkpoint_handler.PyTreeCheckpointHandler(
        save_device_host_concurrent_gb=save_gb,
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
            item=tree, restore_args=restore_args, transforms=transforms
        ),
    )

    self.assertEqual(
        dispatcher.batches('_sync_deserialize_arrays'), expected_batches
    )
    test_utils.assert_tree_equal(self, tree, restored)

  def test_register_type_handlers_plumbs_restore_limit(self):
    names = ['a0', 'a1', 'a2', 'a3']
    tree = dict(zip(names, _sharded_arrays(len(names))))
    save_handler = pytree_checkpoint_handler.PyTreeCheckpointHandler(
        use_ocdbt=False
    )
    self.addCleanup(save_handler.close)
    save_handler.save(
        self.directory, args=pytree_checkpoint_handler.PyTreeSaveArgs(tree)
    )
    saved_handlers = [
        (ty, type_handler_registry.get_type_handler(ty))
        for ty in (int, float, bytes, np.number, np.ndarray, jax.Array)
    ]

    def _restore_global_handlers():
      for ty, h in saved_handlers:
        type_handler_registry.register_type_handler(ty, h, override=True)

    self.addCleanup(_restore_global_handlers)
    created_dispatchers: list[_LocalColocatedDispatcher] = []

    class _CapturedColocatedDispatcher(_LocalColocatedDispatcher):

      def __init__(self):
        super().__init__()
        created_dispatchers.append(self)

    with mock.patch.object(
        dispatchers, 'ColocatedPythonDispatcher', _CapturedColocatedDispatcher
    ):
      ocp_pathways.register_type_handlers(
          checkpointing_impl=ocp_pathways.CheckpointingImpl.COLOCATED_PYTHON,
          dispatcher_restore_concurrent_bytes=int(2.5 * _ARRAY_BYTES),
      )
      handler = pytree_checkpoint_handler.PyTreeCheckpointHandler(
          use_ocdbt=False
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
              item=tree, restore_args=restore_args
          ),
      )

    self.assertLen(created_dispatchers, 1)
    dispatcher = created_dispatchers[0]
    self.assertEqual(
        dispatcher.batches('_sync_deserialize_arrays'),
        [['a0', 'a1'], ['a2', 'a3']],
    )
    test_utils.assert_tree_equal(self, tree, restored)

  @parameterized.parameters(0, -1, 1.5, True, 'auto')
  def test_invalid_restore_limit_raises(self, restore_bytes):
    with self.assertRaisesRegex(
        ValueError,
        'dispatcher_restore_concurrent_bytes must be a positive int or None',
    ):
      jax_array_handlers.ArrayHandler(
          dispatcher_restore_concurrent_bytes=restore_bytes
      )


if __name__ == '__main__':
  absltest.main()
