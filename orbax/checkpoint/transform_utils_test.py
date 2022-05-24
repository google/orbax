# Copyright 2022 The Orbax Authors.
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

"""Tests for transform_utils."""
from typing import List, Mapping

from absl.testing import absltest
import flax
from flax import serialization
from flax import traverse_util
import flax.linen as nn
import jax
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint import transform_utils

Transform = transform_utils.Transform
apply_transformations = transform_utils.apply_transformations


class TransformUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.original = {
        'a': 0,
        'b': 1,
        'c': {
            'a': 2,
            'e': 3,
        }
    }

  def test_empty(self):
    self.assertDictEqual({}, apply_transformations({}, {}))
    self.assertDictEqual({}, apply_transformations({'a': 0, 'b': 1}, {}))

  def test_no_transform(self):
    transforms = jax.tree_map(lambda _: Transform(), self.original)
    self.assertDictEqual(self.original,
                         apply_transformations(self.original, transforms))

  def test_transform_missing_in_original(self):
    original = {'a': 1}
    transforms = {'b': Transform()}
    with self.assertRaises(ValueError):
      apply_transformations(original, transforms)

  def test_rename(self):
    transforms = {
        'a1': Transform(original_key='a'),  # originally named "a"
        'c': {
            'a': Transform(),  # unchanged
        },
        # moved from being inside "c"
        'e1': Transform(original_key=('c', 'e')),
        'f': Transform(in_checkpoint=False),  # newly added
        'g': Transform(in_checkpoint=False, init_value=100),  # newly added
        # note: dropped "d"
        # copied c/a and moved up
        'ca1': Transform(original_key=('c', 'a')),
        'ca2': Transform(original_key=('c', 'a')),
    }
    expected = {
        'a1': 0,
        'c': {
            'a': 2,
        },
        'e1': 3,
        'f': None,
        'g': 100,
        'ca1': 2,
        'ca2': 2,
    }
    self.assertDictEqual(expected,
                         apply_transformations(self.original, transforms))

  def test_partial_restore(self):
    transforms = {
        'a': Transform(),
        'c': {
            'e': Transform(),
        },
    }
    expected = {
        'a': 0,
        'c': {
            'e': 3,
        },
    }
    self.assertDictEqual(expected,
                         apply_transformations(self.original, transforms))

  def test_function(self):
    transforms = {
        'a': Transform(value_fn=lambda kv: kv['a'] * 2 + 20),
        # dropped b
        'c': {
            # added together two keys, leaving one remaining
            'a': Transform(value_fn=lambda kv: kv['c']['a'] + kv['c']['e']),
        },
        # many to many transformation: input two keys -> output two new keys
        'w': Transform(value_fn=lambda kv: kv['a'] + kv['b']),
        'x': Transform(value_fn=lambda kv: kv['a'] + kv['b'] * 2),
        # copied a single key into multiple
        'y': Transform(value_fn=lambda kv: kv['a']),
        'z': Transform(value_fn=lambda kv: kv['a']),
    }
    expected = {
        'a': 20,
        'c': {
            'a': 5,
        },
        'w': 1,
        'x': 2,
        'y': 0,
        'z': 0,
    }
    self.assertDictEqual(expected,
                         apply_transformations(self.original, transforms))

  def test_non_dict_tree(self):

    @flax.struct.dataclass
    class SubTree:
      x: Mapping[str, int]
      y: List[int]

    @flax.struct.dataclass
    class Tree:
      a: int
      b: np.ndarray
      c: SubTree

    tree = Tree(a=5, b=np.arange(3), c=SubTree(x={'i': 0, 'j': 1}, y=[4, 5, 6]))

    @flax.struct.dataclass
    class NewTree:
      a1: int  # a
      b: np.ndarray  # times 2
      c: SubTree  # same
      d: float  # new
      e: int  # from a.y
      f: List[int]  # from a.y

    transforms = NewTree(
        a1=Transform(original_key='a'),
        b=Transform(value_fn=lambda t: t.b * 2),
        c=jax.tree_map(lambda _: Transform(), tree.c),
        d=Transform(in_checkpoint=False),
        e=Transform(value_fn=lambda t: t.c.y[0]),
        f=Transform(value_fn=lambda t: t.c.y[1:]))
    expected_tree = NewTree(
        a1=5,
        b=np.arange(3) * 2,
        c=SubTree(x={
            'i': 0,
            'j': 1
        }, y=[4, 5, 6]),
        d=None,
        e=4,
        f=[5, 6])

    def assert_equal(a, b):
      if isinstance(a, np.ndarray):
        np.testing.assert_equal(a, b)
      elif isinstance(a, list):
        self.assertListEqual(a, b)
      elif isinstance(a, dict):
        self.assertDictEqual(a, b)
      else:
        self.assertEqual(a, b)

    jax.tree_multimap(assert_equal, expected_tree,
                      apply_transformations(tree, transforms))

  def test_construct_transformations_from_fallback(self):
    fallback = dict(self.original)
    fallback.update({
        'x': 5,
        'y': {
            'z': 10,
        },
    })
    expected = {
        'a': Transform(),
        'b': Transform(),
        'c': {
            'a': Transform(),
            'e': Transform(),
        },
        'x': Transform(in_checkpoint=False, init_value=5),
        'y': {
            'z': Transform(in_checkpoint=False, init_value=10),
        }
    }
    actual = transform_utils.construct_transformations_from_fallback(
        self.original, fallback)
    self.assertDictEqual(expected, actual)

  def test_flax_train_state_fallback(self):

    class SmallModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=8)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(features=8)(x)
        return x

    old_state = test_utils.init_flax_model(SmallModel())

    class LargeModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=8)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(features=8)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(features=4)(x)
        return x

    new_state = test_utils.init_flax_model(LargeModel())

    transforms = transform_utils.construct_transformations_from_fallback(
        old_state, new_state)
    restored_state = transform_utils.apply_transformations(
        old_state, transforms)

    old_state_dict = traverse_util.flatten_dict(
        serialization.to_state_dict(old_state), keep_empty_nodes=True)
    new_state_dict = traverse_util.flatten_dict(
        serialization.to_state_dict(new_state), keep_empty_nodes=True)

    expected_state_dict = {}
    for k, v in new_state_dict.items():
      if k in old_state_dict:
        expected_state_dict[k] = old_state_dict[k]
      else:
        expected_state_dict[k] = v

    expected_state = serialization.from_state_dict(
        new_state, traverse_util.unflatten_dict(expected_state_dict))

    test_utils.assert_tree_equal(self, expected_state, restored_state)


if __name__ == '__main__':
  absltest.main()
