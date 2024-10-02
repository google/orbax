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

"""Tests for utils module."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from orbax.checkpoint.tree import utils


class UtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='empty_leafs',
          input_tree={'a': 1, 'b': [], 'c': (), 'd': {}},
          expected_output={'a': 1},
      ),
      dict(
          testcase_name='singleton_leafs',
          input_tree={
              'b': [np.array([0])],
              'c': (np.array([0]),),
              'd': {'e': np.array([0])},
          },
          expected_output={
              'b': [np.array([0])],
              'c': [np.array([0])],
              'd': {'e': np.array([0])},
          },
      ),
      dict(
          testcase_name='multiple_leafs',
          input_tree={
              'b': [np.array([0]), np.array([1])],
              'c': (np.array([0]), np.array([1])),
              'd': {'e': np.array([0])},
          },
          expected_output={
              'b': [np.array([0]), np.array([1])],
              'c': [np.array([0]), np.array([1])],
              'd': {'e': np.array([0])},
          },
      ),
      dict(
          testcase_name='nested_containers',
          input_tree={
              'b': [
                  [np.array([0])],
                  {'a': np.array([1])},
                  (np.array([1]),),
              ],
              'c': (
                  (np.array([0]),),
                  {'a': np.array([1])},
                  [np.array([1])],
              ),
              'd': {
                  'e': {'f': np.array([0])},
                  'f': [np.array([0])],
                  'g': (np.array([0]),),
              },
              'e': [
                  np.array([0]),
                  [
                      np.array([1]),
                      [
                          np.array([2]),
                          (np.array([3]),),
                          (
                              np.array([3]),
                              (
                                  np.array([4]),
                                  (np.array([5]),),
                              ),
                          ),
                      ],
                  ],
              ],
          },
          expected_output={
              'b': [
                  [np.array([0])],
                  {'a': np.array([1])},
                  [np.array([1])],
              ],
              'c': [
                  [np.array([0])],
                  {'a': np.array([1])},
                  [np.array([1])],
              ],
              'd': {
                  'e': {'f': np.array([0])},
                  'f': [np.array([0])],
                  'g': [np.array([0])],
              },
              'e': [
                  np.array([0]),
                  [
                      np.array([1]),
                      [
                          np.array([2]),
                          [np.array([3])],
                          [
                              np.array([3]),
                              [
                                  np.array([4]),
                                  [np.array([5])],
                              ],
                          ],
                      ],
                  ],
              ],
          },
      ),
  )
  def test_serialize_tree_drop_empty_nodes(self, input_tree, expected_output):
    self.assertEqual(utils.serialize_tree(input_tree), expected_output)


if __name__ == '__main__':
  absltest.main()
