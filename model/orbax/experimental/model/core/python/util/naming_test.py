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

"""Tests for naming utils."""

from orbax.experimental.model.core.python.util import naming
from absl.testing import absltest


class NamingTest(absltest.TestCase):

  def test_validate_node_name(self):
    self.assertTrue(naming.is_valid_node_name('correct_name_1534'))
    self.assertTrue(naming.is_valid_node_name('.A1B1C23'))
    self.assertTrue(naming.is_valid_node_name('5432/_0'))

    self.assertFalse(naming.is_valid_node_name('_bad_name'))
    self.assertFalse(naming.is_valid_node_name('spēcial/chærs.??'))


if __name__ == '__main__':
  absltest.main()
