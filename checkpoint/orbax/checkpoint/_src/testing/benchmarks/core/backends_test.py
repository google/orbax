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

"""Tests for backends.is_simdevice; other backends are faked by device name."""

from unittest import mock

from absl.testing import absltest
import jax
from orbax.checkpoint._src.testing.benchmarks.core import backends


class _FakeDevice:
  """A device that only has a name."""

  def __init__(self, name: str):
    self._name = name

  def __str__(self) -> str:
    return self._name


class IsSimDeviceTest(absltest.TestCase):

  def test_cpu(self):
    self.assertFalse(backends.is_simdevice())

  def test_simdevice(self):
    with mock.patch.object(
        jax,
        'devices',
        autospec=True,
        return_value=[_FakeDevice('SimDevice(0)')],
    ):
      self.assertTrue(backends.is_simdevice())

  def test_tpu(self):
    with mock.patch.object(
        jax,
        'devices',
        autospec=True,
        return_value=[_FakeDevice('TPU_0(process=0,(0,0,0,0))')],
    ):
      self.assertFalse(backends.is_simdevice())


if __name__ == '__main__':
  absltest.main()
