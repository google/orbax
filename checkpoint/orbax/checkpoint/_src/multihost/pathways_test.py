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
from unittest import mock

from absl.testing import absltest
import numpy as np
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.multihost import pathways
from orbax.checkpoint._src.serialization import worker_memory_utils


@dataclasses.dataclass(frozen=True)
class _FakeDevice:
  id: int
  platform: str = 'tpu'
  process_index: int = 0
  virtual_task_index: Optional[int] = None
  task_id: Optional[int] = None
  slice_index: Optional[int] = None
  repr_logical_task: Optional[int] = None
  repr_vtask: Optional[int] = None
  coords: Optional[tuple[int, ...]] = None
  core_on_chip: Optional[int] = None

  def __repr__(self):
    parts = [f'id={self.id}']
    parts.append(f'platform={self.platform}')
    if self.virtual_task_index is not None:
      parts.append(f'vtask={self.virtual_task_index}')
    elif self.repr_vtask is not None:
      parts.append(f'vtask={self.repr_vtask}')
    if self.task_id is not None:
      parts.append(f'task_id={self.task_id}')
    if self.repr_logical_task is not None:
      parts.append(f'logical_task={self.repr_logical_task}')
    if self.slice_index is not None:
      parts.append(f'slice={self.slice_index}')
    if self.coords is not None:
      parts.append(f'coords={self.coords}')
    if self.core_on_chip is not None:
      parts.append(f'core_on_chip={self.core_on_chip}')
    return f'FakeDevice({", ".join(parts)})'


class PathwaysTest(absltest.TestCase):

  def tearDown(self):
    pathways._WARNED_REPR_PATTERNS.clear()  # pylint: disable=protected-access
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

    self.assertEqual(pathways.worker_count(_FakeMesh()), 2)

  def test_worker_count_preserves_best_effort_slice_only_compatibility(self):
    class _FakeMesh:

      @property
      def devices(self):
        return np.array(
            (
                _FakeDevice(id=0, slice_index=5),
                _FakeDevice(id=1, slice_index=5),
            ),
            dtype=object,
        )

    with mock.patch.object(pathways.logging, 'warning') as warning:
      self.assertEqual(pathways.worker_count(_FakeMesh()), 1)

    warning.assert_called_once()
    self.assertIn(
        'worker_count() may not be accurate',
        warning.call_args.args[0],
    )

  def test_group_devices_by_worker_uses_vtask_and_slice(self):
    devices = [
        _FakeDevice(id=0, virtual_task_index=0, slice_index=0),
        _FakeDevice(id=1, virtual_task_index=0, slice_index=0),
        _FakeDevice(id=2, virtual_task_index=1, slice_index=0),
        _FakeDevice(id=72, virtual_task_index=0, slice_index=1),
    ]
    grouped = pathways.group_devices_by_worker(devices)  # pyrefly: ignore[bad-argument-type]

    self.assertLen(grouped, 3)
    self.assertEqual([d.id for d in grouped[(0, 0)]], [0, 1])
    self.assertEqual([d.id for d in grouped[(1, 0)]], [2])
    self.assertEqual([d.id for d in grouped[(0, 1)]], [72])

  def test_group_devices_by_worker_uses_task_id_and_slice_index(self):
    devices = [
        _FakeDevice(id=0, task_id=0, slice_index=0),
        _FakeDevice(id=1, task_id=0, slice_index=0),
        _FakeDevice(id=72, task_id=0, slice_index=1),
    ]
    grouped = pathways.group_devices_by_worker(devices)  # pyrefly: ignore[bad-argument-type]

    self.assertLen(grouped, 2)
    self.assertEqual([d.id for d in grouped[(0, 0)]], [0, 1])
    self.assertEqual([d.id for d in grouped[(0, 1)]], [72])

  def test_group_devices_by_worker_uses_vtask_repr_fallback_from_main(self):
    devices = [
        _FakeDevice(id=0, repr_vtask=0, slice_index=0),
        _FakeDevice(id=1, repr_vtask=0, slice_index=0),
        _FakeDevice(id=2, repr_vtask=1, slice_index=0),
    ]

    grouped = pathways.group_devices_by_worker(devices)  # pyrefly: ignore[bad-argument-type]

    self.assertLen(grouped, 2)
    self.assertEqual([d.id for d in grouped[(0, 0)]], [0, 1])
    self.assertEqual([d.id for d in grouped[(1, 0)]], [2])

  def test_group_devices_by_worker_warns_on_repr_fallback(self):
    devices = [
        _FakeDevice(id=0, task_id=0, repr_logical_task=0, slice_index=0),
        _FakeDevice(id=2, task_id=0, repr_logical_task=1, slice_index=0),
    ]

    with mock.patch.object(pathways.logging, 'warning') as warning:
      grouped = pathways.group_devices_by_worker(devices)  # pyrefly: ignore[bad-argument-type]

    self.assertLen(grouped, 2)
    warning.assert_called_once()
    self.assertIn(
        'Pathways worker-key inference fell back to repr parsing',
        warning.call_args.args[0],
    )
    self.assertEqual(warning.call_args.args[1], r'logical_task=(\d+)')

  def test_group_devices_by_worker_does_not_warn_without_repr_fallback(self):
    devices = [
        _FakeDevice(id=0, task_id=0, slice_index=0),
        _FakeDevice(id=1, task_id=0, slice_index=0),
    ]

    with mock.patch.object(pathways.logging, 'warning') as warning:
      grouped = pathways.group_devices_by_worker(devices)  # pyrefly: ignore[bad-argument-type]

    self.assertLen(grouped, 1)
    warning.assert_not_called()

  def test_group_devices_by_worker_prefers_logical_task_repr_over_task_id(self):
    devices = [
        _FakeDevice(id=0, task_id=0, repr_logical_task=0, slice_index=0),
        _FakeDevice(id=2, task_id=0, repr_logical_task=1, slice_index=0),
    ]

    grouped = pathways.group_devices_by_worker(devices)  # pyrefly: ignore[bad-argument-type]

    self.assertLen(grouped, 2)
    self.assertEqual([d.id for d in grouped[(0, 0)]], [0])
    self.assertEqual([d.id for d in grouped[(1, 0)]], [2])

  def test_group_devices_by_worker_uses_task_id_fallback(self):
    device = _FakeDevice(
        id=13,
        task_id=5,
    )
    grouped = pathways.group_devices_by_worker([device])  # pyrefly: ignore[bad-argument-type]
    self.assertIn((5,), grouped)
    self.assertEqual(grouped[(5,)][0].id, 13)

  def test_group_devices_by_worker_raises_on_slice_only_metadata(self):
    device = _FakeDevice(
        id=12,
        slice_index=7,
    )

    with self.assertRaisesRegex(
        ValueError, 'requires a task identifier; slice_index alone is ambiguous'
    ):
      pathways.group_devices_by_worker([device])  # pyrefly: ignore[bad-argument-type]

  def test_group_devices_by_worker_raises_without_worker_metadata(self):
    device = _FakeDevice(
        id=12,
        process_index=7,
        virtual_task_index=None,
        task_id=None,
        slice_index=None,
    )

    with self.assertRaisesRegex(
        ValueError, 'Unable to infer Pathways worker key from device attributes'
    ):
      pathways.group_devices_by_worker([device])  # pyrefly: ignore[bad-argument-type]

  def test_worker_primary_host(self):
    devices = (_FakeDevice(id=0), _FakeDevice(id=1), _FakeDevice(id=2))
    dispatcher = mock.create_autospec(
        dispatchers.ColocatedPythonDispatcher, instance=True
    )
    for mapping, primary_host, expected in (
        # Worker 0 holds some of the devices: it stays the primary host.
        ({0: 0, 1: 0, 2: 1}, 0, 0),
        # Worker 0 holds none of the devices (e.g. its slice is down): the
        # lowest worker holding them is elected.
        ({0: 9, 1: 8, 2: 8}, 0, 8),
        # A non-zero configured primary host that is live is kept.
        ({0: 2, 1: 3, 2: 4}, 3, 3),
        # Empty mapping: nothing to elect from.
        ({}, 0, 0),
    ):
      with self.subTest(mapping=mapping, primary_host=primary_host):
        with mock.patch.object(
            worker_memory_utils, '_device_to_worker_ids', return_value=mapping
        ) as mock_map:
          elected = pathways.worker_primary_host(
              dispatcher,
              devices,  # pyrefly: ignore[bad-argument-type]
              primary_host,
          )
          self.assertEqual(elected, expected)
          mock_map.assert_called_once_with(dispatcher, devices)


if __name__ == '__main__':
  absltest.main()
