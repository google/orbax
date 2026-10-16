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

"""Tests for simdevice_backend, with the SimDevice plugin mocked out."""

import atexit
import dataclasses
import os
import sys
import types
from unittest import mock

from absl.testing import absltest
import jax
from orbax.checkpoint._src.testing.benchmarks.core import configs
from orbax.checkpoint._src.testing.benchmarks.core import simdevice_backend
import yaml

_CONFIG = configs.SimDeviceConfig(device_type='v6e', topology=[2, 2, 1])

# What the real plugin leaves in JAX_PLATFORMS once it is imported.
_SIMDEVICE_PLATFORMS = 'simdevice,cpu'


class RenderTopologyYamlTest(absltest.TestCase):

  def test_render(self):
    rendered = simdevice_backend.render_topology_yaml(
        configs.SimDeviceConfig(
            device_type='v7x', topology=[4, 4, 1], num_processes=2
        )
    )

    self.assertEqual(
        yaml.safe_load(rendered),
        {
            'device': {'type': 'v7x'},
            'topology': {'mesh_dimensions': [4, 4, 1]},
            'orchestration': {'num_processes': 2},
        },
    )


class ActivateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(simdevice_backend, '_active_config', None)
    )
    self.enter_context(
        mock.patch.dict(os.environ, {'JAX_PLATFORMS': _SIMDEVICE_PLATFORMS})
    )
    os.environ.pop('SIMDEVICE_FABRICATED_FILL_SEED', None)
    # Stands in for the plugin, so `import simdevice` does not load it.
    self.enter_context(
        mock.patch.dict(sys.modules, {'simdevice': types.ModuleType('fake')})
    )
    self.mock_config_update = self.enter_context(
        mock.patch.object(jax.config, 'update', autospec=True)
    )
    self.mock_atexit_register = self.enter_context(
        mock.patch.object(atexit, 'register', autospec=True)
    )

  def _activate(self, config: configs.SimDeviceConfig) -> str:
    """Activates SimDevice and returns the topology file path."""
    try:
      simdevice_backend.activate(config)
    finally:
      if 'SIMDEVICE_CONFIG_PATH' in os.environ:
        self.addCleanup(os.remove, os.environ['SIMDEVICE_CONFIG_PATH'])
    return os.environ['SIMDEVICE_CONFIG_PATH']

  def test_writes_topology_file(self):
    topology_path = self._activate(_CONFIG)

    with open(topology_path) as f:
      self.assertEqual(
          f.read(), simdevice_backend.render_topology_yaml(_CONFIG)
      )
    self.mock_atexit_register.assert_called_once_with(os.remove, topology_path)

  def test_selects_simdevice_platform(self):
    self._activate(_CONFIG)

    self.mock_config_update.assert_called_once_with(
        'jax_platforms', _SIMDEVICE_PLATFORMS
    )

  def test_sets_fill_seed(self):
    self._activate(dataclasses.replace(_CONFIG, fabricated_fill_seed=7))

    self.assertEqual(os.environ['SIMDEVICE_FABRICATED_FILL_SEED'], '7')

  def test_no_fill_seed(self):
    self._activate(_CONFIG)

    self.assertNotIn('SIMDEVICE_FABRICATED_FILL_SEED', os.environ)

  def test_activate_twice_raises(self):
    self._activate(_CONFIG)

    with self.assertRaisesRegex(RuntimeError, 'already active'):
      simdevice_backend.activate(_CONFIG)

  def test_failed_import_raises(self):
    # A None entry makes `import simdevice` raise ImportError.
    with mock.patch.dict(sys.modules, {'simdevice': None}):
      with self.assertRaises(ImportError):
        self._activate(_CONFIG)

    self.mock_config_update.assert_not_called()
    # A second attempt is not blocked by the failed one.
    self._activate(_CONFIG)


if __name__ == '__main__':
  absltest.main()
