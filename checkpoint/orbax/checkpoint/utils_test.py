# Copyright 2023 The Orbax Authors.
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

"""Test for utils.py."""

from typing import Mapping, Sequence

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
# TODO(b/275613424): Eliminate flax dependency in Orbax test suite.
import flax
import jax
import optax
from orbax.checkpoint import test_utils
from orbax.checkpoint import utils


class UtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='checkpointing_test').full_path
    )

  @parameterized.parameters(
      (3, 'dir', None, None, None, 'dir/3'),
      (3, 'dir', 'params', None, None, 'dir/3/params'),
      (3, 'dir', 'params', 'checkpoint', None, 'dir/checkpoint_3/params'),
      (3, 'dir', None, None, 2, 'dir/03'),
      (4000, 'dir', 'params', None, 5, 'dir/04000/params'),
      (555, 'dir', 'params', 'foo', 8, 'dir/foo_00000555/params'),
      (1234567890, 'dir', 'params', 'foo', 12, 'dir/foo_001234567890/params'),
  )
  def test_get_save_directory(
      self,
      step,
      directory,
      name,
      step_prefix,
      step_format_fixed_length,
      result,
  ):
    self.assertEqual(
        utils.get_save_directory(
            step,
            directory,
            name=name,
            step_prefix=step_prefix,
            step_format_fixed_length=step_format_fixed_length,
        ),
        epath.Path(result),
    )

  def test_get_save_directory_tmp_dir_override(self):
    self.assertEqual(
        utils.get_save_directory(
            42,
            'path/to/my/dir',
            name='params',
            step_prefix='foobar_',
            override_directory='a/different/dir/path',
        ),
        epath.Path('a/different/dir/path/params'),
    )

  @parameterized.parameters((None,), ('checkpoint_',), ('foobar_',))
  def test_is_tmp_checkpoint(self, step_prefix):
    step_dir = utils.get_save_directory(
        5, self.directory, step_prefix=step_prefix
    )
    step_dir.mkdir(parents=True)
    self.assertFalse(utils.is_tmp_checkpoint(step_dir))
    tmp_step_dir = utils.create_tmp_directory(step_dir)
    self.assertTrue(utils.is_tmp_checkpoint(tmp_step_dir))

    item_dir = utils.get_save_directory(
        10, self.directory, name='params', step_prefix=step_prefix
    )
    item_dir.mkdir(parents=True)
    self.assertFalse(utils.is_tmp_checkpoint(item_dir))
    tmp_item_dir = utils.create_tmp_directory(item_dir)
    self.assertTrue(utils.is_tmp_checkpoint(tmp_item_dir))

  @parameterized.parameters(
      ('0', 0),
      ('0000', 0),
      ('1000', 1000),
      ('checkpoint_0', 0),
      ('checkpoint_0000', 0),
      ('checkpoint_003400', 3400),
      ('foobar_1000', 1000),
      ('0.orbax-checkpoint-tmp-1010101', 0),
      ('0000.orbax-checkpoint-tmp-12323232', 0),
      ('foobar_1.orbax-checkpoint-tmp-12424424', 1),
      ('foobar_000505.orbax-checkpoint-tmp-13124', 505),
      ('checkpoint_16.orbax-checkpoint-tmp-123214324', 16),
  )
  def test_step_from_checkpoint_name(self, name, step):
    self.assertEqual(utils.step_from_checkpoint_name(name), step)

  @parameterized.parameters(
      ('abc',),
      ('checkpoint_',),
      ('checkpoint_1010_',),
      ('_191',),
      ('.orbax-checkpoint-tmp-191913',),
      ('0.orbax-checkpoint-tmp-',),
      ('checkpoint_.orbax-checkpoint-tmp-191913',),
  )
  def test_step_from_checkpoint_name_invalid(self, name):
    with self.assertRaises(ValueError):
      utils.step_from_checkpoint_name(name)

  @parameterized.parameters(
      ({'a': 1, 'b': {'c': {}, 'd': 2}}, {('a',): 1, ('b', 'd'): 2}),
      ({'x': ['foo', 'bar']}, {('x', '0'): 'foo', ('x', '1'): 'bar'}),
  )
  def test_to_flat_dict(self, tree, expected):
    self.assertDictEqual(expected, utils.to_flat_dict(tree))

  @parameterized.parameters(
      ({'a': 1, 'b': {'d': 2}}, {('a',): 1, ('b', 'd'): 2}),
      ({'x': ['foo', 'bar']}, {('x', '0'): 'foo', ('x', '1'): 'bar'}),
      ({'a': 1, 'b': 2}, {('b',): 2, ('a',): 1}),
  )
  def test_from_flat_dict(self, expected, flat_dict):
    empty = jax.tree_util.tree_map(lambda _: 0, expected)
    self.assertDictEqual(
        expected, utils.from_flat_dict(flat_dict, target=empty)
    )

  def test_serialize(self):
    tree = {'a': 1, 'b': {'c': {'d': 2}}, 'e': [1, {'x': 5, 'y': 7}, [9, 10]]}
    serialized = utils.serialize_tree(tree, keep_empty_nodes=True)
    self.assertDictEqual(tree, serialized)
    deserialized = utils.deserialize_tree(
        serialized, target=tree, keep_empty_nodes=True
    )
    test_utils.assert_tree_equal(self, tree, deserialized)

  def test_serialize_list(self):
    tree = [1, {'a': 2}, [3, 4]]
    serialized = utils.serialize_tree(tree, keep_empty_nodes=True)
    self.assertListEqual(tree, serialized)
    deserialized = utils.deserialize_tree(
        serialized, target=tree, keep_empty_nodes=True
    )
    test_utils.assert_tree_equal(self, tree, deserialized)

  def test_serialize_filters_empty(self):
    tree = {'a': 1, 'b': None, 'c': {}, 'd': [], 'e': optax.EmptyState()}
    serialized = utils.serialize_tree(tree, keep_empty_nodes=False)
    self.assertDictEqual({'a': 1}, serialized)
    deserialized = utils.deserialize_tree(
        serialized, target=tree, keep_empty_nodes=False
    )
    test_utils.assert_tree_equal(self, tree, deserialized)

  def test_serialize_class(self):
    @flax.struct.dataclass
    class Foo:
      a: int
      b: Mapping[str, str]
      c: Sequence[optax.EmptyState]
      d: Sequence[Mapping[str, str]]

    foo = Foo(
        1,
        {'a': 'b', 'c': 'd'},
        [optax.EmptyState(), optax.EmptyState()],
        [{}, {'x': 'y'}, None],
    )
    serialized = utils.serialize_tree(foo, keep_empty_nodes=True)
    expected = {
        'a': 1,
        'b': {'a': 'b', 'c': 'd'},
        'c': [optax.EmptyState(), optax.EmptyState()],
        'd': [{}, {'x': 'y'}, None],
    }
    self.assertDictEqual(expected, serialized)
    deserialized = utils.deserialize_tree(
        serialized, target=foo, keep_empty_nodes=False
    )
    test_utils.assert_tree_equal(self, foo, deserialized)

  def test_serialize_nested_class(self):
    @flax.struct.dataclass
    class Foo:
      a: int

    nested = {
        'x': Foo(a=1),
        'y': {'z': Foo(a=2)},
    }
    serialized = utils.serialize_tree(nested, keep_empty_nodes=True)
    expected = {
        'x': dict(a=1),
        'y': {'z': dict(a=2)},
    }
    self.assertDictEqual(expected, serialized)

  def test_checkpoint_steps_paths_nonexistent_directory_fails(self):
    with self.assertRaisesRegex(ValueError, 'does not exist'):
      utils.checkpoint_steps_paths('/non/existent/dir')

  def test_checkpoint_steps_paths_returns_finalized_paths(self):
    digit_only_path = epath.Path(self.directory / '2')
    digit_only_path.mkdir()
    prefix_path = epath.Path(self.directory / 'checkpoint_01')
    prefix_path.mkdir()
    epath.Path(self.directory / 'checkpoint').mkdir()
    epath.Path(self.directory / '1000.orbax-checkpoint-tmp-1010101').mkdir()

    self.assertCountEqual(
        utils.checkpoint_steps_paths(self.directory),
        [digit_only_path, prefix_path],
    )

  def test_checkpoint_steps_returns_steps_of_finalized_paths(self):
    epath.Path(self.directory / '2').mkdir()
    epath.Path(self.directory / 'checkpoint_01').mkdir()
    epath.Path(self.directory / 'checkpoint').mkdir()
    epath.Path(self.directory / '1000.orbax-checkpoint-tmp-1010101').mkdir()

    self.assertSameElements(
        [1, 2],
        utils.checkpoint_steps(self.directory),
    )


if __name__ == '__main__':
  absltest.main()
