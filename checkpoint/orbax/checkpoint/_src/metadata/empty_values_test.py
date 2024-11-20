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

"""Tests for empty values as leafs in the checkpoint tree."""

from absl.testing import absltest
from absl.testing import parameterized
from orbax.checkpoint._src.metadata import empty_values
from orbax.checkpoint._src.testing import test_tree_utils


class EmptyValuesTest(parameterized.TestCase):

  @parameterized.parameters(
      (1, False),
      (dict(), True),
      ({}, True),
      ({"a": {}}, False),
      ([], True),
      ([[]], False),
      (None, True),
      ((1, 2), False),
      (test_tree_utils.EmptyNamedTuple(), False),
      (test_tree_utils.MuNu(mu=None, nu=None), False),
      (test_tree_utils.NamedTupleWithNestedAttributes(), False),
      (test_tree_utils.NamedTupleWithNestedAttributes(nested_dict={}), False),
  )
  def test_is_supported_empty_value(self, value, expected):
    self.assertEqual(expected, empty_values.is_supported_empty_value(value))


if __name__ == "__main__":
  absltest.main()
