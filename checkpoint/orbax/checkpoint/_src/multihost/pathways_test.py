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

"""Tests for pathways multihost utilities."""

from __future__ import annotations

import dataclasses
from typing import Optional

from absl.testing import absltest
import numpy as np
from orbax.checkpoint._src.multihost import pathways


@dataclasses.dataclass(frozen=True)
class _FakeDevice:
  id: int
  platform: str = 'tpu'
  process_index: int = 0
  virtual_task_index: Optional[int] = None
  task_id: Optional[int] = None
  slice_index: Optional[int] = None
  coords: Optional[tuple[int, ...]] = None
  core_on_chip: Optional[int] = None
  repr_text: Optional[str] = None

  def __repr__(self):
    if self.repr_text is not None:
      return self.repr_text
    parts = [f'id={self.id}']
    parts.append(f'platform={self.platform}')
    if self.virtual_task_index is not None:
      parts.append(f'vtask={self.virtual_task_index}')
    if self.task_id is not None:
      parts.append(f'task_id={self.task_id}')
    if self.slice_index is not None:
      parts.append(f'slice={self.slice_index}')
    if self.coords is not None:
      parts.append(f'coords={self.coords}')
    if self.core_on_chip is not None:
      parts.append(f'core_on_chip={self.core_on_chip}')
    return f'FakeDevice({", ".join(parts)})'


class PathwaysTest(absltest.TestCase):

  def tearDown(self):
    pathways.worker_count.cache_clear()
    super().tearDown()

  def test_worker_count_uses_grouping(self):
    class _FakeMesh:

      @property
      def devices(self):
        return np.array(
            (
                _FakeDevice(id=0, virtual_task_index=0, slice_index=0),
                _FakeDevice(id=1, virtual_task_index=0, slice_index=0),
                _FakeDevice(id=72, virtual_task_index=0, slice_index=1),
            ),
            dtype=object,
        )

    self.assertEqual(pathways.worker_count(_FakeMesh()), 2)  # pytype: disable=wrong-arg-types

  def test_group_devices_by_worker_uses_vtask_and_slice(self):
    devices = [
        _FakeDevice(id=0, virtual_task_index=0, slice_index=0),
        _FakeDevice(id=1, virtual_task_index=0, slice_index=0),
        _FakeDevice(id=2, virtual_task_index=1, slice_index=0),
        _FakeDevice(id=72, virtual_task_index=0, slice_index=1),
    ]
    grouped = pathways.group_devices_by_worker(devices)  # pytype: disable=wrong-arg-types

    self.assertLen(grouped, 3)
    self.assertEqual([d.id for d in grouped[(0, 0)]], [0, 1])
    self.assertEqual([d.id for d in grouped[(1, 0)]], [2])
    self.assertEqual([d.id for d in grouped[(0, 1)]], [72])

  def test_group_devices_by_worker_parses_repr_fallback(self):
    device = _FakeDevice(
        id=10,
        repr_text='Device(TPU, logical_task=3, slice=1)',
    )
    grouped = pathways.group_devices_by_worker([device])  # pytype: disable=wrong-arg-types
    self.assertIn((3, 1), grouped)
    self.assertEqual(grouped[(3, 1)][0].id, 10)

  def test_group_devices_by_worker_handles_none_attr_values(self):
    device = _FakeDevice(
        id=11,
        virtual_task_index=None,
        slice_index=None,
        repr_text='Device(TPU, logical_task=4, slice=2)',
    )
    grouped = pathways.group_devices_by_worker([device])  # pytype: disable=wrong-arg-types
    self.assertIn((4, 2), grouped)
    self.assertEqual(grouped[(4, 2)][0].id, 11)

  def test_group_devices_by_worker_uses_task_id_fallback(self):
    device = _FakeDevice(
        id=13,
        task_id=5,
        repr_text='Device(TPU, no_repr_task)',
    )
    grouped = pathways.group_devices_by_worker([device])  # pytype: disable=wrong-arg-types
    self.assertIn((5,), grouped)
    self.assertEqual(grouped[(5,)][0].id, 13)

  def test_group_devices_by_worker_falls_back_to_process_index(self):
    device = _FakeDevice(
        id=12,
        process_index=7,
        virtual_task_index=None,
        slice_index=None,
        repr_text='Device(TPU, no_task_metadata)',
    )
    grouped = pathways.group_devices_by_worker([device])  # pytype: disable=wrong-arg-types
    self.assertIn((7,), grouped)
    self.assertEqual(grouped[(7,)][0].id, 12)

  def test_compute_distributed_to_device_ids_sorted_by_worker_key(self):
    devices = [
        _FakeDevice(id=74, virtual_task_index=1, slice_index=1),
        _FakeDevice(id=72, virtual_task_index=0, slice_index=1),
        _FakeDevice(id=2, virtual_task_index=1, slice_index=0),
        _FakeDevice(id=0, virtual_task_index=0, slice_index=0),
        _FakeDevice(id=1, virtual_task_index=0, slice_index=0),
    ]
    distributed = pathways.compute_distributed_to_device_ids(devices)  # pytype: disable=wrong-arg-types

    self.assertEqual(distributed, [[0, 1], [2], [72], [74]])


if __name__ == '__main__':
  absltest.main()
