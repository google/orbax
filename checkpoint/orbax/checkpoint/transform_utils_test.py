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

"""Tests for transform_utils."""

from typing import List, Mapping

from absl.testing import absltest
# TODO(b/275613424): Eliminate flax dependency in Orbax test suite.
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
from jax import numpy as jnp
import numpy as np
import optax
from orbax.checkpoint import test_utils
from orbax.checkpoint import transform_utils
from orbax.checkpoint import tree as tree_utils


Transform = transform_utils.Transform
apply_transformations = transform_utils.apply_transformations


def empty_pytree(tree):
  return jax.tree.map(lambda x: object(), tree)


class EmptyNode:
  pass


# Not in common util because we need to eliminate OSS dependency on flax.
def init_flax_model(model):
  params = model.init(jax.random.PRNGKey(0), jnp.ones([8, 8]))
  tx = optax.adamw(learning_rate=0.001)
  state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
  return jax.tree.map(np.asarray, state)


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
    self.assertDictEqual({}, apply_transformations({}, {}, {}))
    self.assertDictEqual({}, apply_transformations({'a': 0, 'b': 1}, {}, {}))
    self.assertDictEqual({'a': 1}, apply_transformations({}, {}, {'a': 1}))

  def test_no_transform(self):
    transforms = jax.tree.map(lambda _: Transform(), self.original)
    self.assertDictEqual(
        self.original,
        apply_transformations(self.original, transforms,
                              empty_pytree(self.original)))

  def test_transform_missing_in_original(self):
    original = {'a': 1}
    transforms = {'b': Transform()}
    with self.assertRaises(ValueError):
      apply_transformations(original, transforms, {'b': ...})

  def test_rename(self):
    transforms = {
        'a1': Transform(original_key='a'),  # originally named "a"
        'c': {
            'a': Transform(),  # unchanged
        },
        # moved from being inside "c"
        'e1': Transform(original_key='c/e'),
        'f': Transform(use_fallback=True),  # newly added
        # note: dropped "b"
        # copied c/a and moved up
        'ca1': Transform(original_key='c/a'),
        'ca2': Transform(original_key='c/a'),
    }
    fallback = {
        'a1': ...,
        'c': {
            'a': ...,
        },
        'e1': ...,
        'f': None,
        'g': 100,  # newly added
        'ca1': ...,
        'ca2': ...,
    }
    expected = {
        'a1': 0,
        'c': {
            'a': 2,
        },
        'e1': 3,
        'f': None,
        'g': 100,  # newly added
        'ca1': 2,
        'ca2': 2,
    }
    self.assertDictEqual(
        expected, apply_transformations(self.original, transforms, fallback))

  def test_partial_transformation(self):
    transforms = {
        'a1': Transform(original_key='a'),  # originally named "a"
        # implicit "a" also gets preserved
        # implicit copy over "c" subtree
        # moved from being inside "c"
        'e1': Transform(original_key='c/e'),
        'e2': Transform(original_key='c/e'),
        # implicit add "f" and "g"
    }
    fallback = {
        'a': ...,
        'a1': ...,
        'c': {
            'a': ...,
            'e': ...,
        },
        'e1': ...,
        'e2': ...,
        'f': None,
        'g': 2,
    }
    expected = {
        'a': 0,
        'a1': 0,
        'c': {
            'a': 2,
            'e': 3,
        },
        'e1': 3,
        'e2': 3,
        'f': None,
        'g': 2,
    }
    self.assertDictEqual(
        expected, apply_transformations(self.original, transforms, fallback))

  def test_default_new(self):
    transforms = {
        'a': Transform(use_fallback=True),  # use value from original
        # implicit drop "b"
        # implicit retain "c/a", "c/a"
        'b1': Transform(original_key='b'),
        # implicit add "f" and "g"
    }
    new = {
        'a': ...,
        'c': {
            'a': 10,
            'e': 11,
        },
        'b1': ...,
        'f': None,
        'g': 2,
    }
    expected = {
        'a': 0,
        'c': {
            'a': 10,
            'e': 11,
        },
        'b1': 1,
        'f': None,
        'g': 2,
    }
    self.assertDictEqual(
        expected,
        apply_transformations(
            self.original, transforms, new, default_to_original=False))

  def test_missing_key_default(self):
    transforms = {'f': Transform(use_fallback=True)}
    new = {
        'a': 2,
        'b': 3,
        'c': {
            'a': 7,
            'e': 8,
        },
        'f': 20,
    }
    with self.assertRaises(ValueError):
      apply_transformations(
          self.original, transforms, new, default_to_original=False)

    expected = {'a': 0, 'b': 1, 'c': {'a': 2, 'e': 3}, 'f': 20}
    self.assertDictEqual(
        expected,
        apply_transformations(
            self.original, transforms, new, default_to_original=True))

  def test_regex(self):
    original = {
        'a1': 1,
        'c': {
            'a2': {
                'd': 2,
            },
            'a3': 3,
        },
        'd': {
            'e1': 2,
            'e2': 4,
        },
        'f': {
            'e3': 6,
            'e4': {  # note doesn't get matched
                'a': 1
            },
        },
    }
    transforms = {
        r'(.*)x(\d.*)':
            Transform(original_key=r'\1a\2'),
        r'(.*)y(\d)':
            Transform(original_key=r'\1e\2', value_fn=lambda val: val * 2),
    }
    expected = {
        'x1': 1,
        'c': {
            'x2': {
                'd': 2,
            },
            'x3': 3,
        },
        'd': {
            'y1': 4,
            'y2': 8,
        },
        'f': {
            'y3': 12,
        },
    }
    self.assertDictEqual(
        expected,
        apply_transformations(original, transforms, empty_pytree(expected)))

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
    self.assertDictEqual(
        expected,
        apply_transformations(self.original, transforms,
                              empty_pytree(expected)))

  def test_function(self):
    transforms = {
        'a': Transform(multi_value_fn=lambda _, kv: kv['a'] * 2 + 20),
        # dropped b
        'c': {
            # added together two keys, leaving one remaining
            'a': Transform(
                multi_value_fn=lambda _, kv: kv['c']['a'] + kv['c']['e']
            ),
        },
        # many to many transformation: input two keys -> output two new keys
        'w': Transform(multi_value_fn=lambda _, kv: kv['a'] + kv['b']),
        'x': Transform(multi_value_fn=lambda _, kv: kv['a'] + kv['b'] * 2),
        # copied a single key into multiple
        'y': Transform(multi_value_fn=lambda _, kv: kv['a']),
        'z': Transform(multi_value_fn=lambda _, kv: kv['a']),
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
    self.assertDictEqual(
        expected,
        apply_transformations(self.original, transforms,
                              empty_pytree(expected)))

  def test_non_dict_tree(self):

    @flax.struct.dataclass
    class SubTree(flax.struct.PyTreeNode):
      x: Mapping[str, int]
      y: List[int]

    @flax.struct.dataclass
    class Tree(flax.struct.PyTreeNode):
      a: int
      b: np.ndarray
      c: SubTree

    tree = Tree(
        a=10, b=np.arange(3), c=SubTree(x={
            'i': 0,
            'j': 1
        }, y=[4, 5, 6]))

    @flax.struct.dataclass
    class NewTree(flax.struct.PyTreeNode):
      a1: int  # a
      b: np.ndarray  # times 2
      c: SubTree  # same
      d: float  # new
      e: int  # from a.y
      f: List[int]  # from a.y

    transforms = NewTree(
        a1=Transform(original_key='a'),
        b=Transform(multi_value_fn=lambda _, t: t.b * 2),
        c=jax.tree.map(lambda _: Transform(), tree.c),
        d=Transform(use_fallback=True),
        e=Transform(multi_value_fn=lambda _, t: t.c.y[0]),
        f=[
            Transform(multi_value_fn=lambda _, t: t.c.y[1]),
            Transform(multi_value_fn=lambda _, t: t.c.y[2]),
        ],
    )
    fallback_tree = NewTree(
        a1=EmptyNode(),
        b=EmptyNode(),
        c=SubTree(
            x={'i': EmptyNode(), 'j': EmptyNode()},
            y=[EmptyNode(), EmptyNode(), EmptyNode()],
        ),
        d=7,
        e=EmptyNode(),
        f=[EmptyNode(), EmptyNode()],
    )
    expected_tree = NewTree(
        a1=10,
        b=np.arange(3) * 2,
        c=SubTree(x={
            'i': 0,
            'j': 1
        }, y=[4, 5, 6]),
        d=7,
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

    jax.tree.map(
        assert_equal, expected_tree,
        apply_transformations(tree, transforms, fallback_tree))

  def test_flax_train_state(self):

    class SmallModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=8)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(features=8)(x)
        return x

    old_state = init_flax_model(SmallModel())

    class LargeModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=16)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(features=8)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(features=8)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(features=4)(x)
        return x

    new_state = init_flax_model(LargeModel())

    transformations = {
        # LargeModel layer 0 is a newly inserted layer, thus use_fallback=True.
        r'(.*)Dense_0(.*)': Transform(use_fallback=True),
        # SmallModel layer 0 maps to LargeModel layer 1
        r'(.*)Dense_1(.*)': Transform(original_key=r'\1Dense_0\2'),
        # SmallModel layer 1 maps to LargeModel layer 2
        r'(.*)Dense_2(.*)': Transform(original_key=r'\1Dense_1\2')
    }  # Note: LargeModel layer 3 is newly added.
    restored_state = apply_transformations(old_state, transformations,
                                           new_state)

    # Construct expected tree
    old_flat_dict = tree_utils.to_flat_dict(old_state, sep='/')
    new_flat_dict = tree_utils.to_flat_dict(new_state, sep='/')
    expected_flat_dict = {}
    for k, v in new_flat_dict.items():
      if 'Dense_1' in k:
        expected_flat_dict[k] = old_flat_dict[k.replace('Dense_1', 'Dense_0')]
      elif 'Dense_2' in k:
        expected_flat_dict[k] = old_flat_dict[k.replace('Dense_2', 'Dense_1')]
      elif 'Dense_' in k:  # layers in new, but not old.
        expected_flat_dict[k] = v
      else:  # extra keys in both, expected is the old value
        expected_flat_dict[k] = old_flat_dict[k]

    expected_state = tree_utils.from_flat_dict(
        expected_flat_dict, target=new_state, sep='/'
    )
    test_utils.assert_tree_equal(self, expected_state, restored_state)

  def test_flax_train_state_default_new(self):

    class Model(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=16)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(features=8)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(features=8)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(features=4)(x)
        return x

    old_state = init_flax_model(Model())
    new_state = init_flax_model(Model())

    transformations = {
        # values default to new_state, use_fallback=True instructs the Transform
        # to fall back on old_state for this key.
        r'(.*)Dense_1(.*)': Transform(use_fallback=True),
    }
    restored_state = apply_transformations(
        old_state, transformations, new_state, default_to_original=False)

    # Construct expected tree
    old_flat_dict = tree_utils.to_flat_dict(old_state, sep='/')
    new_flat_dict = tree_utils.to_flat_dict(new_state, sep='/')
    expected_flat_dict = {}
    for k, v in new_flat_dict.items():
      if 'Dense_1' in k:
        expected_flat_dict[k] = old_flat_dict[k]
      else:
        expected_flat_dict[k] = v

    expected_state = tree_utils.from_flat_dict(
        expected_flat_dict, target=new_state, sep='/'
    )
    test_utils.assert_tree_equal(self, expected_state, restored_state)

  def test_merge_trees(self):
    one = {
        'a': 1,
        'b': {
            'c': 2,
            'd': 3,
        },
    }
    two = {
        'a': 2,
        'b': {
            'c': 4,
            'e': 6,
        },
    }
    three = {}
    four = {
        'f': 7,
        'g': 8,
    }
    expected = {
        'a': 2,
        'b': {
            'c': 4,
            'd': 3,
            'e': 6,
        },
        'f': 7,
        'g': 8,
    }
    self.assertDictEqual(
        expected, transform_utils.merge_trees(one, two, three, four)
    )

  def test_merge_trees_target(self):
    one = {
        'a': 1,
        'b': {
            'c': 2,
            'd': 3,
        },
    }
    two = {
        'a': 2,
        'b': {
            'c': 4,
            'e': 6,
        },
    }
    three = {}
    four = {
        'f': 7,
        'g': 8,
    }

    @flax.struct.dataclass
    class Expected(flax.struct.PyTreeNode):
      a: int
      b: Mapping[str, int]
      f: int
      g: int

      def __eq__(self, other):
        return (
            self.a == other.a
            and self.b == other.b
            and self.f == other.f
            and self.g == other.g
        )

    expected = Expected(
        a=2,
        b={
            'c': 4,
            'd': 3,
            'e': 6,
        },
        f=7,
        g=8,
    )

    self.assertEqual(
        expected,
        transform_utils.merge_trees(
            one,
            two,
            three,
            four,
            target=jax.tree.map(lambda x: 0, expected),
        ),
    )

  def test_merge_trees_empty(self):
    self.assertDictEqual({}, transform_utils.merge_trees({}, {}))


if __name__ == '__main__':
  absltest.main()
