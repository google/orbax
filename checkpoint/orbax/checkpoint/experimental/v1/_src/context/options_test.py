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

"""Tests for checkpoint file options."""

from absl.testing import absltest
from absl.testing import parameterized
from orbax.checkpoint import options as v0_options_lib
from orbax.checkpoint.experimental.v1._src.context import options as ocp_options



class FileOptionsTest(parameterized.TestCase):

  def test_v0_conversion_with_none_options(self):
    opts = ocp_options.FileOptions()
    v0_opts = opts.v0()
    self.assertIsInstance(v0_opts, v0_options_lib.FileOptions)
    self.assertIsNone(v0_opts.path_permission_mode)

  def test_v0_conversion_with_all_options(self):

    opts = ocp_options.FileOptions(
        path_permission_mode=0o777,
    )
    v0_opts = opts.v0()
    self.assertIsInstance(v0_opts, v0_options_lib.FileOptions)
    self.assertEqual(v0_opts.path_permission_mode, 0o777)


if __name__ == '__main__':
  absltest.main()
