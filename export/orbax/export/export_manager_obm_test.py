# Copyright 2025 The Orbax Authors.
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

"""Tests for the `version==ORBAX_MODEL` path of `ExportManager`."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import sharding
from jax.experimental import mesh_utils
import jax.numpy as jnp
from orbax.export import constants
from orbax.export import export_manager
from orbax.export import jax_module
from orbax.export import oex_orchestration
from orbax.export import oex_orchestration_pb2
from orbax.export import serving_config as osc
import tensorflow as tf

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

_VERSIONS = (
    constants.ExportModelType.TF_SAVEDMODEL,
    constants.ExportModelType.ORBAX_MODEL,
)


class ExportManagerObmTest(parameterized.TestCase, tf.test.TestCase):

  # Dummy test to make copybara happy, will be removed once all the obm
  # dependencies are OSSed.
  def test_dummy(self):
    assert True


if __name__ == '__main__':
  absltest.main()
