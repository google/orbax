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

"""Tests for colocated utility helpers."""

from __future__ import annotations

import dataclasses
from typing import Optional
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import (
    colocated_utils,
)


@dataclasses.dataclass
class _FakeShard:
  data: np.ndarray


@dataclasses.dataclass
class _FakeResult:
  addressable_shards: list[_FakeShard]


@dataclasses.dataclass(frozen=True)
class _FakeDevice:
  id: int
  process_index: int = 0
  virtual_task_index: Optional[int] = None
  slice_index: Optional[int] = None


class ColocatedUtilsTest(absltest.TestCase):

  def _replicated_array(self, value):
    device = jax.devices()[0]
    mesh = jax.sharding.Mesh(np.array([device]), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    return jax.device_put(value, sharding)

  def test_device_list_signature_named_sharding(self):
    device = jax.devices()[0]
    mesh = jax.sharding.Mesh(np.array([device]), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    signature = colocated_utils.device_list_signature(sharding)

    self.assertEqual(signature, ((device.platform, device.id),))

  def test_device_list_signature_single_device_sharding(self):
    device = jax.devices()[0]
    sharding = jax.sharding.SingleDeviceSharding(device)

    signature = colocated_utils.device_list_signature(sharding)

    self.assertEqual(signature, ((device.platform, device.id),))

  def test_require_unanimous_scalar_result_success(self):
    result = _FakeResult(
        addressable_shards=[
            _FakeShard(np.asarray(True)),
            _FakeShard(np.asarray(True)),
        ]
    )

    value = colocated_utils.require_unanimous_scalar_result(  # pytype: disable=wrong-arg-types
        result, op_name='test_op'
    )

    self.assertTrue(value)

  def test_require_unanimous_scalar_result_raises_on_disagreement(self):
    result = _FakeResult(
        addressable_shards=[
            _FakeShard(np.asarray(4, dtype=np.int32)),
            _FakeShard(np.asarray(5, dtype=np.int32)),
        ]
    )

    with self.assertRaisesRegex(RuntimeError, 'workers disagreed'):
      colocated_utils.require_unanimous_scalar_result(result, op_name='test_op')  # pytype: disable=wrong-arg-types

  def test_assert_arrays_on_platform(self):
    arr = self._replicated_array(jnp.array([1, 2], dtype=jnp.int32))

    colocated_utils.assert_arrays_on_platform(
        {'x': arr},
        expected_platform=jax.devices()[0].platform,
        tree_name='test_tree',
    )

  def test_assert_arrays_on_allowed_cpu_ids(self):
    arr = self._replicated_array(jnp.array([1, 2], dtype=jnp.int32))
    allowed_ids = frozenset(d.id for d in arr.sharding.device_set)

    colocated_utils.assert_arrays_on_allowed_cpu_ids(
        {'x': arr},
        allowed_ids=allowed_ids,
        tree_name='test_tree',
    )

  def test_assert_specs_on_allowed_cpu_ids(self):
    arr = self._replicated_array(jnp.array([1, 2], dtype=jnp.int32))
    allowed_ids = frozenset(d.id for d in arr.sharding.device_set)
    spec = jax.ShapeDtypeStruct(arr.shape, arr.dtype, sharding=arr.sharding)

    colocated_utils.assert_specs_on_allowed_cpu_ids(
        {'x': spec},
        allowed_ids=allowed_ids,
        tree_name='test_specs',
    )

  def test_make_scalar_on_like(self):
    array = self._replicated_array(jnp.array(1, dtype=jnp.int32))

    scalar = colocated_utils.make_scalar_on_like(7, array, dtype=jnp.int32)

    self.assertEqual(np.asarray(scalar).item(), 7)

  def test_make_scalar_on_like_delegates_to_colocated_transport(self):
    array = self._replicated_array(jnp.array(1, dtype=jnp.int32))
    sentinel = object()

    with mock.patch.object(
        colocated_utils.colocated_transport,
        'make_scalar_array_like',
        return_value=sentinel,
    ) as mock_make_scalar:
      result = colocated_utils.make_scalar_on_like(7, array, dtype=jnp.int32)

    self.assertIs(result, sentinel)
    mock_make_scalar.assert_called_once_with(7, array, dtype=jnp.int32)

  def test_compute_distributed_to_device_ids_sorted_by_worker_key(self):
    devices = [
        _FakeDevice(id=74, virtual_task_index=1, slice_index=1),
        _FakeDevice(id=72, virtual_task_index=0, slice_index=1),
        _FakeDevice(id=2, virtual_task_index=1, slice_index=0),
        _FakeDevice(id=0, virtual_task_index=0, slice_index=0),
        _FakeDevice(id=1, virtual_task_index=0, slice_index=0),
    ]

    distributed = colocated_utils.compute_distributed_to_device_ids(devices)  # pytype: disable=wrong-arg-types

    self.assertEqual(distributed, [[0, 1], [2], [72], [74]])


if __name__ == '__main__':
  absltest.main()
