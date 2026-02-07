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

"""Tests for the `version==ORBAX_MODEL` path of `ExportManager`."""

import os

from absl.testing import absltest
from absl.testing import parameterized
from orbax.export import constants
from orbax.export import export_manager
from orbax.export import export_testing_utils
from orbax.export import oex_orchestration
from orbax.export import serving_config as sc
import tensorflow as tf


_VERSIONS = (
    constants.ExportModelType.TF_SAVEDMODEL,
    constants.ExportModelType.ORBAX_MODEL,
)


# TODO: b/380323586 - Add a e2e test for default xla compile options.
class ExportManagerObmTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    base_path = os.path.dirname(os.path.abspath(__file__))
    self._testdata_dir = os.path.join(base_path, 'testdata')

  # Dummy test to make copybara happy, will be removed once all the obm
  # dependencies are OSSed.
  def test_dummy(self):
    assert True


if __name__ == '__main__':
  absltest.main()
