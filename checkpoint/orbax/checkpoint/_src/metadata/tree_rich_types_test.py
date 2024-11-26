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

"""Tests tree_rich_types.py module."""

from absl.testing import absltest
from absl.testing import parameterized
from orbax.checkpoint._src.metadata import tree_rich_types
from orbax.checkpoint._src.testing import test_tree_utils


class TreeRichTypesTest(parameterized.TestCase):

  @parameterized.product(
      bad_tree=[
          {"a": test_tree_utils.MyClass()},
      ]
  )
  def test_invalid_pytree_leaf(self, bad_tree):
    with self.assertRaisesRegex(
        ValueError, "Expected ValueMetadataEntry, got metadata pytree leaf"
    ):
      tree_rich_types.value_metadata_tree_to_json_str(bad_tree)


if __name__ == "__main__":
  absltest.main()
