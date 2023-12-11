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

"""Tests for CompositeHandler."""

from unittest import mock
from absl.testing import absltest
from etils import epath
from jax import numpy as jnp
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import composite_checkpoint_handler
from orbax.checkpoint import json_checkpoint_handler
from orbax.checkpoint import standard_checkpoint_handler
from orbax.checkpoint import value_metadata

CompositeArgs = composite_checkpoint_handler.CompositeArgs
JsonCheckpointHandler = json_checkpoint_handler.JsonCheckpointHandler
StandardCheckpointHandler = (
    standard_checkpoint_handler.StandardCheckpointHandler
)
CompositeCheckpointHandler = (
    composite_checkpoint_handler.CompositeCheckpointHandler
)


class CompositeArgsTest(absltest.TestCase):

  def test_args(self):
    args = CompositeArgs(a=1, b=2, d=4)
    self.assertEqual(1, args.a)
    self.assertEqual(2, args.b)
    self.assertEqual({'a', 'b', 'd'}, args.keys())
    self.assertEqual({1, 2, 4}, set(args.values()))
    self.assertEqual(1, args['a'])
    self.assertLen(args, 3)

    with self.assertRaises(KeyError):
      _ = args['c']

    self.assertIsNone(args.get('c'))
    self.assertEqual(4, args.get('c', 4))

  def test_invalid_key(self):
    with self.assertRaisesRegex(ValueError, 'cannot start with'):
      CompositeArgs(__invalid_name=2)

  def test_reserved_keys_are_unchanged(self):
    # To avoid breaking future users, make sure that CompositeArgs
    # only reserves the following attributes:
    self.assertEqual(
        set([x for x in dir(CompositeArgs) if not x.startswith('__')]),
        {'get', 'items', 'keys', 'values'},
    )

  def test_use_reserved_keys(self):
    args = CompositeArgs(keys=3, values=4)

    self.assertNotEqual(3, args.keys)
    self.assertNotEqual(4, args.values)

    self.assertEqual(3, args['keys'])
    self.assertEqual(4, args['values'])

    self.assertEqual({'keys', 'values'}, args.keys())
    self.assertEqual({3, 4}, set(args.values()))

  def test_special_character_key(self):
    args = CompositeArgs(**{
        '.invalid_attribute_but_valid_key': 15,
        'many special characters!': 16})
    self.assertEqual(15, args['.invalid_attribute_but_valid_key'])
    self.assertEqual(16, args['many special characters!'])
    self.assertLen(args, 2)


class CompositeCheckpointHandlerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir(name='test_dir'))

  def save(self, handler, *args, **kwargs):
    handler.save(*args, **kwargs)
    handler.finalize(self.directory)

  def test_init(self):
    handler = CompositeCheckpointHandler('state', 'dataset')
    self.assertDictEqual(
        handler._known_handlers, {'state': None, 'dataset': None}
    )
    handler = CompositeCheckpointHandler(state=StandardCheckpointHandler())
    self.assertSameElements(handler._known_handlers.keys(), {'state'})
    self.assertIsInstance(
        handler._known_handlers['state'], StandardCheckpointHandler
    )
    handler = CompositeCheckpointHandler(
        'tree', 'dataset', state=StandardCheckpointHandler()
    )
    self.assertSameElements(
        handler._known_handlers.keys(), {'tree', 'state', 'dataset'}
    )
    self.assertIsInstance(
        handler._known_handlers['state'], StandardCheckpointHandler
    )
    self.assertIsNone(handler._known_handlers['tree'])
    self.assertIsNone(handler._known_handlers['dataset'])

  def test_save_restore(self):
    handler = CompositeCheckpointHandler('state', 'metadata')
    state = {'a': 1, 'b': 2}
    dummy_state = {'a': 0, 'b': 0}
    metadata = {'lang': 'en', 'version': 1.0}
    self.save(
        handler,
        self.directory,
        CompositeArgs(
            state=args_lib.StandardSave(state),
            metadata=args_lib.JsonSave(metadata),
        ),
    )
    self.assertTrue((self.directory / 'state').exists())
    self.assertTrue((self.directory / 'metadata').exists())
    restored = handler.restore(
        self.directory,
        CompositeArgs(
            state=args_lib.StandardRestore(dummy_state),
            metadata=args_lib.JsonRestore(),
        ),
    )
    self.assertDictEqual(restored.state, state)
    self.assertDictEqual(restored.metadata, metadata)

  def test_save_restore_partial(self):
    handler = CompositeCheckpointHandler('state', 'opt_state', 'metadata')
    state = {'a': 1, 'b': 2}
    dummy_state = {'a': 0, 'b': 0}
    opt_state = {'x': 1, 'y': 2}
    dummy_opt_state = {'x': 0, 'y': 0}
    metadata = {'lang': 'en', 'version': 1.0}
    self.save(
        handler,
        self.directory,
        CompositeArgs(
            state=args_lib.StandardSave(state),
            metadata=args_lib.JsonSave(metadata),
        ),
    )
    self.assertTrue((self.directory / 'state').exists())
    self.assertTrue((self.directory / 'metadata').exists())
    self.assertFalse((self.directory / 'opt_state').exists())
    self.save(
        handler,
        self.directory,
        CompositeArgs(
            opt_state=args_lib.StandardSave(opt_state),
        ),
    )
    self.assertTrue((self.directory / 'opt_state').exists())
    restored = handler.restore(
        self.directory,
        CompositeArgs(
            state=args_lib.StandardRestore, metadata=args_lib.JsonRestore()
        ),
    )
    self.assertDictEqual(restored.state, state)
    self.assertDictEqual(restored.metadata, metadata)
    restored = handler.restore(
        self.directory,
        CompositeArgs(opt_state=args_lib.StandardRestore(dummy_opt_state)),
    )
    self.assertDictEqual(restored.opt_state, opt_state)

  def test_incorrect_args(self):
    dir1 = epath.Path(self.create_tempdir(name='dir1'))
    dir2 = epath.Path(self.create_tempdir(name='dir2'))
    handler = CompositeCheckpointHandler('state')
    state = {'a': 1, 'b': 2}
    handler.save(dir1, CompositeArgs(state=args_lib.StandardSave(state)))
    self.save(
        handler,
        dir1,
        CompositeArgs(
            state=args_lib.StandardSave(state),
        ),
    )
    with self.assertRaises(ValueError):
      self.save(
          handler,
          dir2,
          CompositeArgs(
              state=args_lib.JsonSave(state),
          ),
      )
    with self.assertRaises(ValueError):
      handler.restore(dir1, CompositeArgs(state=args_lib.JsonRestore(state)))

  def test_no_restore_args(self):
    handler = CompositeCheckpointHandler('state', 'metadata')
    state = {'a': 1, 'b': 2}
    dummy_state = {'a': 0, 'b': 0}
    metadata = {'lang': 'en', 'version': 1.0}
    self.save(
        handler,
        self.directory,
        CompositeArgs(
            state=args_lib.StandardSave(state),
            metadata=args_lib.JsonSave(metadata),
        ),
    )
    self.assertTrue((self.directory / 'state').exists())
    self.assertTrue((self.directory / 'metadata').exists())

    restored = handler.restore(self.directory)
    self.assertDictEqual(restored.state, state)
    self.assertDictEqual(restored.metadata, metadata)

    restored = handler.restore(
        self.directory,
        CompositeArgs(),
    )
    self.assertDictEqual(restored.state, state)
    self.assertDictEqual(restored.metadata, metadata)

  def test_no_restore_args_partial_save(self):
    handler = CompositeCheckpointHandler(
        'state', metadata=JsonCheckpointHandler()
    )
    state = {'a': 1, 'b': 2}
    dummy_state = {'a': 0, 'b': 0}
    self.save(
        handler,
        self.directory,
        CompositeArgs(
            state=args_lib.StandardSave(state),
        ),
    )
    self.assertTrue((self.directory / 'state').exists())
    self.assertFalse((self.directory / 'metadata').exists())

    restored = handler.restore(self.directory)
    self.assertDictEqual(restored.state, state)
    self.assertIsNone(restored.metadata)

    restored = handler.restore(
        self.directory,
        CompositeArgs(),
    )
    self.assertDictEqual(restored.state, state)
    self.assertIsNone(restored.metadata)

  def test_no_restore_args_handler_unspecified(self):
    handler = CompositeCheckpointHandler('state', 'metadata')
    state = {'a': 1, 'b': 2}
    dummy_state = {'a': 0, 'b': 0}
    metadata = {'lang': 'en', 'version': 1.0}
    self.save(
        handler,
        self.directory,
        CompositeArgs(
            state=args_lib.StandardSave(state),
            metadata=args_lib.JsonSave(metadata),
        ),
    )
    self.assertTrue((self.directory / 'state').exists())
    self.assertTrue((self.directory / 'metadata').exists())

    handler = CompositeCheckpointHandler('state', 'metadata')
    with self.assertRaises(ValueError):
      handler.restore(self.directory)
    with self.assertRaises(ValueError):
      handler.restore(
          self.directory,
          CompositeArgs(),
      )

  def test_metadata(self):
    handler = CompositeCheckpointHandler(
        'extra',
        state=StandardCheckpointHandler(),
        metadata=JsonCheckpointHandler(),
    )
    metadata = handler.metadata(self.directory)
    self.assertIsNone(metadata.state)
    self.assertIsNone(metadata.metadata)
    self.assertNotIn('extra', metadata.items())

    state = {'a': 1, 'b': 2}
    self.save(
        handler,
        self.directory,
        CompositeArgs(
            state=args_lib.StandardSave(state),
        ),
    )
    metadata = handler.metadata(self.directory)
    self.assertDictEqual(
        metadata.state,
        {
            'a': value_metadata.ScalarMetadata(
                name='a', directory=self.directory / 'state', dtype=jnp.int64
            ),
            'b': value_metadata.ScalarMetadata(
                name='b', directory=self.directory / 'state', dtype=jnp.int64
            ),
        },
    )
    self.assertIsNone(metadata.metadata)
    self.assertNotIn('extra', metadata.items())

  def test_finalize(self):
    state_handler = mock.create_autospec(StandardCheckpointHandler)
    metadata_handler = mock.create_autospec(JsonCheckpointHandler)
    handler = CompositeCheckpointHandler(
        'extra', state=state_handler, metadata=metadata_handler
    )
    handler.finalize(self.directory)
    state_handler.finalize.assert_called_once()
    metadata_handler.finalize.assert_called_once()

  def test_close(self):
    state_handler = mock.create_autospec(StandardCheckpointHandler)
    metadata_handler = mock.create_autospec(JsonCheckpointHandler)
    handler = CompositeCheckpointHandler(
        'extra', state=state_handler, metadata=metadata_handler
    )
    handler.close()
    state_handler.close.assert_called_once()
    metadata_handler.close.assert_called_once()


if __name__ == '__main__':
  absltest.main()
