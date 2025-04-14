# Copyright 2024 The Orbax Authors.
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

import contextlib
import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import sharding
from jax.experimental import mesh_utils
import jax.numpy as jnp
from orbax.export import constants
from orbax.export import jax_module
from orbax.export import obm_configs
from orbax.export import obm_export
from orbax.export import oex_orchestration
from orbax.export import serving_config as osc
from orbax.export import utils
from orbax.export.oex_orchestration import oex_orchestration_pb2
import tensorflow as tf

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'


# TODO(b/363033166): Remove this function once TF isolation is done.
def _package_jax_module(m: jax_module.JaxModule):
  result = tf.Module()
  result.computation_module = m
  return result


def _assert(b):
  assert b


class ObmExportTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._output_dir = self.create_tempdir().full_path

  def test_incorrect_export_version(self):
    pass


if __name__ == '__main__':
  absltest.main()
