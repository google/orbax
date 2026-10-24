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

"""Tests for splitting arrays into batches that fit a worker memory budget."""

from collections.abc import Callable, Sequence
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.serialization import worker_memory_utils

P = jax.sharding.PartitionSpec

_KIB = 1024
_NUM_WORKERS = 2
_DEVICES_PER_WORKER = 4


def _worker_devices(worker_id: int) -> list[jax.Device]:
  """Returns the devices that `_TwoWorkerDispatcher` places on `worker_id`."""
  start = worker_id * _DEVICES_PER_WORKER
  return jax.devices()[start : start + _DEVICES_PER_WORKER]


class _TwoWorkerDispatcher(dispatchers.Dispatcher):
  """Fake dispatcher for 8 devices on 2 workers, 4 consecutive devices each.

  Batching dispatches one function, which maps an array of device IDs to the
  IDs of the workers that the devices are local to. This fake computes that
  mapping directly instead of running the function on each worker.
  """

  def name(self) -> str:
    return 'two_worker'

  def device_to_host(self, arrays: Any) -> Any:
    return arrays

  def dispatch(
      self,
      func: Callable[..., Any],
      *,
      input_arrays: Any = None,
      result_specs: Any = None,
      func_args: tuple[Any, ...] = (),
      func_kwargs: dict[str, Any] | None = None,
  ) -> Any:
    del func, func_args, func_kwargs
    worker_id_by_device_id = {
        device.id: i // _DEVICES_PER_WORKER
        for i, device in enumerate(jax.devices())
    }
    worker_ids = np.asarray(
        [worker_id_by_device_id[int(i)] for i in np.asarray(input_arrays)],
        dtype=np.int32,
    )
    return jax.device_put(worker_ids, result_specs.sharding)


def _spec(
    num_bytes: int,
    partition_spec: jax.sharding.PartitionSpec,
    devices: Sequence[jax.Device] | None = None,
) -> jax.ShapeDtypeStruct:
  """Returns a spec for a float32 vector of `num_bytes` on `devices`."""
  mesh = jax.sharding.Mesh(np.asarray(devices or jax.devices()), ('x',))
  return jax.ShapeDtypeStruct(
      (num_bytes // 4,),
      np.float32,
      sharding=jax.sharding.NamedSharding(mesh, partition_spec),
  )


def _sharded_spec(worker_bytes: int) -> jax.ShapeDtypeStruct:
  """Returns a spec sharded over all devices, using `worker_bytes` per worker.

  Each worker holds 4 of the 8 shards, i.e. half of the array.

  Args:
    worker_bytes: The memory that the spec uses on each worker.
  """
  return _spec(2 * worker_bytes, P('x'))


def _batches(
    specs: Sequence[jax.ShapeDtypeStruct], worker_memory_budget: int
) -> list[list[int]]:
  return list(
      worker_memory_utils.next_memory_budgeted_batch_indices(
          specs, worker_memory_budget, dispatcher=_TwoWorkerDispatcher()
      )
  )


class NextMemoryBudgetedBatchIndicesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.device_count() != _NUM_WORKERS * _DEVICES_PER_WORKER:
      self.skipTest('Requires 8 devices.')

  def test_mixed_sharded_and_replicated_specs(self):
    # 2 KiB arrays. Each worker holds 4 of the 8 shards of a sharded array
    # (1 KiB), and 4 copies of a replicated array (8 KiB).
    sharded = _spec(2 * _KIB, P('x'))
    replicated = _spec(2 * _KIB, P())
    specs = [replicated, sharded, sharded, replicated]

    batches = _batches(specs, worker_memory_budget=12 * _KIB)

    # 8 + 1 + 1 = 10 KiB per worker fits the budget, another 8 KiB does not.
    self.assertEqual(batches, [[0, 1, 2], [3]])

  def test_budget_applies_to_each_worker(self):
    # Replicated over the 4 devices of one worker: 4 KiB on that worker only.
    on_worker_0 = _spec(_KIB, P(), _worker_devices(0))
    on_worker_1 = _spec(_KIB, P(), _worker_devices(1))
    specs = [on_worker_0, on_worker_1, on_worker_0]

    batches = _batches(specs, worker_memory_budget=6 * _KIB)

    # The first two specs use 4 KiB on different workers. The third would bring
    # worker 0 to 8 KiB.
    self.assertEqual(batches, [[0, 1], [2]])

  @parameterized.named_parameters(
      dict(
          testcase_name='first',
          worker_kib=[5, 1, 1],
          expected_batches=[[0], [1, 2]],
      ),
      dict(
          testcase_name='middle',
          worker_kib=[1, 5, 1],
          expected_batches=[[0], [1], [2]],
      ),
      dict(
          testcase_name='last',
          worker_kib=[1, 1, 5],
          expected_batches=[[0, 1], [2]],
      ),
  )
  def test_spec_over_budget_gets_own_batch(self, worker_kib, expected_batches):
    specs = [_sharded_spec(kib * _KIB) for kib in worker_kib]

    batches = _batches(specs, worker_memory_budget=4 * _KIB)

    self.assertEqual(batches, expected_batches)

  def test_usage_resets_after_split(self):
    specs = [_sharded_spec(_KIB)] * 5

    # Fits 2 specs (2 KiB per worker), but not 3 (3 KiB).
    batches = _batches(specs, worker_memory_budget=int(2.5 * _KIB))

    # If usage carried over after the first split, every later spec would get
    # its own batch.
    self.assertEqual(batches, [[0, 1], [2, 3], [4]])

  @parameterized.named_parameters(
      dict(
          testcase_name='usage_equals_budget',
          worker_memory_budget=2 * _KIB,
          expected_batches=[[0], [1]],
      ),
      dict(
          testcase_name='usage_below_budget',
          worker_memory_budget=2 * _KIB + 1,
          expected_batches=[[0, 1]],
      ),
  )
  def test_usage_must_stay_below_budget(
      self, worker_memory_budget, expected_batches
  ):
    # Together, the specs use 2 KiB per worker.
    specs = [_sharded_spec(_KIB), _sharded_spec(_KIB)]

    batches = _batches(specs, worker_memory_budget)

    self.assertEqual(batches, expected_batches)


if __name__ == '__main__':
  absltest.main()
