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

"""Tests for RandomKeyCheckpointHandler."""

from absl.testing import absltest
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import args as args_lib
from orbax.checkpoint._src.handlers import composite_checkpoint_handler
from orbax.checkpoint._src.handlers import random_key_checkpoint_handler

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

JaxRandomKeyCheckpointHandler = (
    random_key_checkpoint_handler.JaxRandomKeyCheckpointHandler
)

NumpyRandomKeyCheckpointHandler = (
    random_key_checkpoint_handler.NumpyRandomKeyCheckpointHandler
)

CompositeCheckpointHandler = (
    composite_checkpoint_handler.CompositeCheckpointHandler
)


class RandomKeyCheckpointHandlerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='checkpointing_test').full_path
    )

  def assert_dict_equal(self, left, right):
    """Asserts that two dicts are equal and allow np.ndarray as elements."""
    assert isinstance(left, type(right))
    for k, v in left.items():
      if k not in right:
        raise ValueError(f'Missing key:{k} in `right`')

      if isinstance(v, dict):
        self.assert_dict_equal(v, right[k])
      elif isinstance(v, np.ndarray):
        np.testing.assert_array_equal(v, right[k])
      else:
        self.assertEqual(v, right[k])

  def assert_tuple_equal(self, left, right):
    """Asserts that two tuples are equal and allow np.ndarray as elements."""
    self.assertIsInstance(left, type(right))
    self.assertLen(left, len(right))

    for i, v in enumerate(left):
      if isinstance(v, np.ndarray):
        np.testing.assert_array_equal(v, right[i])
      else:
        self.assertEqual(v, right[i])

  def test_save_and_restore_jax_random_key_typed(self):
    typed_key = jax.random.key(0)
    handler = JaxRandomKeyCheckpointHandler('typed_key')
    handler.save(self.directory, args_lib.JaxRandomKeySave(item=typed_key))
    handler.finalize(self.directory)

    restore_handler = JaxRandomKeyCheckpointHandler('typed_key')
    restore_typed_keys = restore_handler.restore(
        directory=self.directory, args=args_lib.JaxRandomKeyRestore()
    )
    self.assertTrue(jax.numpy.array_equal(typed_key, restore_typed_keys))

  def test_save_and_restore_jax_random_key_untyped(self):
    typed_key = jax.random.key(0)
    untyped_key = jax.random.PRNGKey(0)
    handler = JaxRandomKeyCheckpointHandler('untyped_key')
    handler.save(
        self.directory,
        args_lib.JaxRandomKeySave(item=untyped_key),
    )
    handler.finalize(self.directory)

    restore_handler = JaxRandomKeyCheckpointHandler('untyped_key')
    restore_untyped_keys = restore_handler.restore(
        directory=self.directory, args=args_lib.JaxRandomKeyRestore()
    )
    self.assertTrue(jax.numpy.array_equal(untyped_key, restore_untyped_keys))

    self.assertFalse(jax.numpy.array_equal(typed_key, untyped_key))

  def test_save_and_restore_numpy_random_key_legacy(self):
    # np random state
    random_state = np.random.get_state()
    handler = NumpyRandomKeyCheckpointHandler('legacy')
    handler.save(self.directory, args_lib.NumpyRandomKeySave(item=random_state))
    handler.finalize(self.directory)

    restore_handler = NumpyRandomKeyCheckpointHandler('legacy')
    restored_random_state = restore_handler.restore(
        directory=self.directory, args=args_lib.NumpyRandomKeyRestore()
    )

    self.assert_tuple_equal(random_state, restored_random_state)

  def test_save_and_restore_numpy_random_key_nonlegacy(self):
    # np random state
    random_state = np.random.get_state(legacy=False)
    handler = NumpyRandomKeyCheckpointHandler('nonlegacy')
    handler.save(
        self.directory, args=args_lib.NumpyRandomKeySave(item=random_state)
    )
    handler.finalize(self.directory)

    restore_handler = NumpyRandomKeyCheckpointHandler('nonlegacy')
    restored_random_state = restore_handler.restore(
        directory=self.directory, args=args_lib.NumpyRandomKeyRestore()
    )

    # dictionary
    self.assert_dict_equal(random_state, restored_random_state)

  def test_save_and_restore_random_keys_with_new_api(self):
    handler = CompositeCheckpointHandler(
        'some_pytree',
        'jax_typed_key',
        'jax_untyped_key',
        'numpy_legacy_key',
        'numpy_nonlegacy_key',
    )

    jax_typed_key = jax.random.key(123)
    jax_untyped_key = jax.random.PRNGKey(256)
    pytree = {'train_state': jax.random.uniform(jax_typed_key, shape=(1, 2, 3))}

    np.random.seed(123)
    np_legacy_key = np.random.get_state(legacy=True)
    np_nonlegacy_key = np.random.get_state(legacy=False)

    handler.save(
        self.directory,
        args=args_lib.Composite(
            some_pytree=args_lib.PyTreeSave(pytree),
            jax_typed_key=args_lib.JaxRandomKeySave(jax_typed_key),
            jax_untyped_key=args_lib.JaxRandomKeySave(jax_untyped_key),
            numpy_legacy_key=args_lib.NumpyRandomKeySave(np_legacy_key),
            numpy_nonlegacy_key=args_lib.NumpyRandomKeySave(np_nonlegacy_key),
        ),
    )

    handler.finalize(self.directory)

    restore_handler = CompositeCheckpointHandler(
        'some_pytree',
        'jax_typed_key',
        'jax_untyped_key',
        'numpy_legacy_key',
        'numpy_nonlegacy_key',
    )

    restored = restore_handler.restore(
        directory=self.directory,
        args=args_lib.Composite(
            some_pytree=args_lib.PyTreeRestore(),
            jax_typed_key=args_lib.JaxRandomKeyRestore(),
            jax_untyped_key=args_lib.JaxRandomKeyRestore(),
            numpy_legacy_key=args_lib.NumpyRandomKeyRestore(),
            numpy_nonlegacy_key=args_lib.NumpyRandomKeyRestore(),
        ),
    )

    self.assertTrue(
        jax.numpy.array_equal(
            pytree['train_state'], restored['some_pytree']['train_state']
        )
    )
    self.assertTrue(
        jax.numpy.array_equal(jax_typed_key, restored['jax_typed_key'])
    )
    self.assertTrue(
        jax.numpy.array_equal(jax_untyped_key, restored['jax_untyped_key'])
    )
    self.assert_tuple_equal(np_legacy_key, restored['numpy_legacy_key'])
    self.assert_dict_equal(np_nonlegacy_key, restored['numpy_nonlegacy_key'])


if __name__ == '__main__':
  absltest.main()
