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

"""Tests for CompositeHandler."""

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from jax import numpy as jnp
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import multihost
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.handlers import checkpoint_handler
from orbax.checkpoint._src.handlers import composite_checkpoint_handler
from orbax.checkpoint._src.handlers import handler_registration
from orbax.checkpoint._src.handlers import json_checkpoint_handler
from orbax.checkpoint._src.handlers import proto_checkpoint_handler
from orbax.checkpoint._src.handlers import standard_checkpoint_handler
from orbax.checkpoint.metadata import value as value_metadata
from orbax.checkpoint.path import step

CompositeArgs = composite_checkpoint_handler.CompositeArgs
JsonCheckpointHandler = json_checkpoint_handler.JsonCheckpointHandler
StandardCheckpointHandler = (
    standard_checkpoint_handler.StandardCheckpointHandler
)
CompositeCheckpointHandler = (
    composite_checkpoint_handler.CompositeCheckpointHandler
)
ProtoCheckpointHandler = proto_checkpoint_handler.ProtoCheckpointHandler
CompositeOptions = composite_checkpoint_handler.CompositeOptions
CheckpointHandler = checkpoint_handler.CheckpointHandler


# Test save and restore args that wrap the standard save and restore args which
# have both been globally registered. This allows the standard args to be used
# to test the handler registry without falling back to the global registries.
class _TestSaveArgs(standard_checkpoint_handler.StandardSaveArgs):
  ...


class _TestRestoreArgs(standard_checkpoint_handler.StandardRestoreArgs):
  ...


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
        'many special characters!': 16,
    })
    self.assertEqual(15, args['.invalid_attribute_but_valid_key'])
    self.assertEqual(16, args['many special characters!'])
    self.assertLen(args, 2)


class CompositeCheckpointHandlerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir(name='test_dir'))

  def save(self, handler, directory, *args, **kwargs):
    handler.save(directory, *args, **kwargs)
    if multihost.process_index() == 0:
      handler.finalize(directory)
    test_utils.sync_global_processes('CCHTest:finalize_after_save')

  def test_init(self):
    handler = CompositeCheckpointHandler('state', 'dataset')
    self.assertContainsSubset(
        {'state', 'dataset'}, handler._item_names_without_registered_handlers
    )

    handler = CompositeCheckpointHandler(state=StandardCheckpointHandler())
    self.assertIsInstance(
        handler._handler_registry.get(
            'state', standard_checkpoint_handler.StandardSaveArgs
        ),
        StandardCheckpointHandler,
    )

    handler = CompositeCheckpointHandler(
        'tree', 'dataset', state=StandardCheckpointHandler()
    )
    self.assertContainsSubset(
        {'tree', 'dataset'}, handler._item_names_without_registered_handlers
    )
    self.assertIsInstance(
        handler._handler_registry.get(
            'state', standard_checkpoint_handler.StandardSaveArgs
        ),
        StandardCheckpointHandler,
    )

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

  def test_save_restore_no_handler_args(self):
    handler = CompositeCheckpointHandler()
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

  def test_save_restore_handler_registry(self):
    handler_registry = handler_registration.DefaultCheckpointHandlerRegistry()
    state_handler = StandardCheckpointHandler()
    handler_registry.add(
        'state',
        _TestSaveArgs,
        state_handler,
    )
    handler_registry.add(
        'state',
        _TestRestoreArgs,
        state_handler,
    )
    metadata_handler = JsonCheckpointHandler()
    handler_registry.add(
        'metadata',
        json_checkpoint_handler.JsonSaveArgs,
        metadata_handler,
    )
    handler_registry.add(
        'metadata',
        json_checkpoint_handler.JsonRestoreArgs,
        metadata_handler,
    )

    handler = CompositeCheckpointHandler(
        handler_registry=handler_registry,
    )

    state = {'a': 1, 'b': 2}
    dummy_state = {'a': 0, 'b': 0}
    metadata = {'lang': 'en', 'version': 1.0}
    self.save(
        handler,
        self.directory,
        CompositeArgs(
            state=_TestSaveArgs(state),
            metadata=args_lib.JsonSave(metadata),
        ),
    )
    self.assertTrue((self.directory / 'state').exists())
    self.assertTrue((self.directory / 'metadata').exists())
    restored = handler.restore(
        self.directory,
        CompositeArgs(
            state=_TestRestoreArgs(dummy_state),
            metadata=args_lib.JsonRestore(),
        ),
    )
    self.assertDictEqual(restored.state, state)
    self.assertDictEqual(restored.metadata, metadata)

  def test_save_restore_handler_registry_with_default_registry_fallback(self):
    handler_registry = handler_registration.DefaultCheckpointHandlerRegistry()
    metadata_handler = JsonCheckpointHandler()
    handler_registry.add(
        'metadata',
        json_checkpoint_handler.JsonSaveArgs,
        metadata_handler,
    )
    handler_registry.add(
        'metadata',
        json_checkpoint_handler.JsonRestoreArgs,
        metadata_handler,
    )

    handler = CompositeCheckpointHandler(
        handler_registry=handler_registry,
    )

    state = {'a': 1, 'b': 2}
    dummy_state = {'a': 0, 'b': 0}
    metadata = {'lang': 'en', 'version': 1.0}
    self.save(
        handler,
        self.directory,
        CompositeArgs(
            state=standard_checkpoint_handler.StandardSaveArgs(state),
            metadata=args_lib.JsonSave(metadata),
        ),
    )
    self.assertTrue((self.directory / 'state').exists())
    self.assertTrue((self.directory / 'metadata').exists())
    restored = handler.restore(
        self.directory,
        CompositeArgs(
            state=standard_checkpoint_handler.StandardRestoreArgs(dummy_state),
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
    directory_one = self.directory / 'one'
    directory_one.mkdir(exist_ok=True)
    self.save(
        handler,
        directory_one,
        CompositeArgs(
            state=args_lib.StandardSave(state),
            metadata=args_lib.JsonSave(metadata),
        ),
    )
    self.assertTrue((directory_one / 'state').exists())
    self.assertTrue((directory_one / 'metadata').exists())
    self.assertFalse((directory_one / 'opt_state').exists())

    directory_two = self.directory / 'two'
    directory_two.mkdir(exist_ok=True)
    self.save(
        handler,
        directory_two,
        CompositeArgs(
            opt_state=args_lib.StandardSave(opt_state),
        ),
    )
    self.assertTrue((directory_two / 'opt_state').exists())
    self.assertFalse((directory_two / 'state').exists())

    restored = handler.restore(
        directory_one,
        CompositeArgs(
            state=args_lib.StandardRestore(), metadata=args_lib.JsonRestore()
        ),
    )
    self.assertDictEqual(restored.state, state)
    self.assertDictEqual(restored.metadata, metadata)
    restored = handler.restore(
        directory_two,
        CompositeArgs(opt_state=args_lib.StandardRestore(dummy_opt_state)),
    )
    self.assertDictEqual(restored.opt_state, opt_state)

    # Knows to use JSON restore.
    restored = handler.restore(
        directory_one,
        CompositeArgs(metadata=None),
    )
    self.assertSameElements(restored.keys(), {'metadata'})
    self.assertDictEqual(restored.metadata, metadata)

  @parameterized.parameters(('state',), ())
  def test_incorrect_args(self, *item_names: str):
    dir1 = epath.Path(self.create_tempdir(name='dir1'))
    dir2 = epath.Path(self.create_tempdir(name='dir2'))
    handler = CompositeCheckpointHandler(*item_names)
    state = {'a': 1, 'b': 2}
    self.save(handler, dir1, CompositeArgs(state=args_lib.StandardSave(state)))
    with self.assertRaisesRegex(
        ValueError, r'does not match with any registered handler'
    ):
      self.save(
          handler,
          dir2,
          CompositeArgs(
              state=args_lib.JsonSave(state),
          ),
      )
    with self.assertRaisesRegex(
        ValueError, r'does not match with any registered handler'
    ):
      handler.restore(dir1, CompositeArgs(state=args_lib.JsonRestore(state)))

  def test_incorrect_args_handler_registry(self):
    handler_registry = handler_registration.DefaultCheckpointHandlerRegistry()
    state_handler = StandardCheckpointHandler()
    handler_registry.add(
        'state',
        standard_checkpoint_handler.StandardSaveArgs,
        state_handler,
    )
    handler_registry.add(
        'state',
        standard_checkpoint_handler.StandardRestoreArgs,
        state_handler,
    )

    handler = CompositeCheckpointHandler(handler_registry=handler_registry)
    state = {'a': 1, 'b': 2}

    self.save(
        handler,
        self.directory,
        CompositeArgs(state=args_lib.StandardSave(state)),
    )

    with self.assertRaisesRegex(
        ValueError, r'does not match with any registered handler'
    ):
      self.save(
          handler,
          self.directory,
          CompositeArgs(
              state=args_lib.JsonSave(state),
          ),
      )
    with self.assertRaisesRegex(
        ValueError, r'does not match with any registered handler'
    ):
      handler.restore(
          self.directory, CompositeArgs(state=args_lib.JsonRestore(state))
      )

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

  def test_save_and_restore_with_handler_registry_with_default_params(self):
    handler_registry = handler_registration.DefaultCheckpointHandlerRegistry()
    handler = standard_checkpoint_handler.StandardCheckpointHandler()
    handler_registry.add(
        None,
        _TestSaveArgs,
        handler,
    )
    handler_registry.add(
        None,
        _TestRestoreArgs,
        handler,
    )
    save_handler = CompositeCheckpointHandler(handler_registry=handler_registry)

    state = {'a': 1, 'b': 2}

    self.save(
        save_handler,
        self.directory,
        CompositeArgs(
            state=args_lib.StandardSave(state),
        ),
    )

    restore_handler = CompositeCheckpointHandler(
        handler_registry=handler_registry,
    )
    restored = restore_handler.restore(
        self.directory,
        CompositeArgs(
            state=args_lib.StandardRestore(state),
        ),
    )
    self.assertDictEqual(restored.state, state)

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

  def test_no_restore_args_partial_save_handler_registry(self):
    handler_registry = handler_registration.DefaultCheckpointHandlerRegistry()
    handler_registry.add(
        'metadata',
        standard_checkpoint_handler.StandardSaveArgs,
        JsonCheckpointHandler(),
    )
    handler = CompositeCheckpointHandler(handler_registry=handler_registry)

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
    with self.assertRaisesRegex(
        ValueError, r'Providing `None` for the item is valid when restoring'
    ):
      handler.restore(self.directory)
    with self.assertRaisesRegex(
        ValueError, r'Providing `None` for the item is valid when restoring'
    ):
      handler.restore(
          self.directory,
          CompositeArgs(),
      )

  def test_no_restore_args_handler_registry(self):
    handler_registry = handler_registration.DefaultCheckpointHandlerRegistry()
    state_handler = StandardCheckpointHandler()
    handler_registry.add('state', args_lib.StandardSave, state_handler)
    handler_registry.add('state', args_lib.StandardRestore, state_handler)
    metadata_handler = JsonCheckpointHandler()
    handler_registry.add('metadata', args_lib.JsonSave, metadata_handler)
    handler_registry.add('metadata', args_lib.JsonRestore, metadata_handler)

    save_handler = CompositeCheckpointHandler(
        handler_registry=handler_registry,
    )
    state = {'a': 1, 'b': 2}
    dummy_state = {'a': 0, 'b': 0}
    metadata = {'lang': 'en', 'version': 1.0}
    self.save(
        save_handler,
        self.directory,
        CompositeArgs(
            state=args_lib.StandardSave(state),
            metadata=args_lib.JsonSave(metadata),
        ),
    )
    self.assertTrue((self.directory / 'state').exists())
    self.assertTrue((self.directory / 'metadata').exists())

    restore_handler_without_registry = CompositeCheckpointHandler()
    self.assertFalse(restore_handler_without_registry.restore(self.directory))
    self.assertFalse(
        restore_handler_without_registry.restore(
            self.directory,
            CompositeArgs(),
        )
    )

    restore_handler_with_registry = CompositeCheckpointHandler(
        handler_registry=handler_registry,
    )
    restored = restore_handler_with_registry.restore(self.directory)
    self.assertDictEqual(restored.state, state)
    self.assertDictEqual(restored.metadata, metadata)

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

  def test_metadata_handler_registry(self):
    registry = handler_registration.DefaultCheckpointHandlerRegistry()
    state_handler = StandardCheckpointHandler()
    metadata_handler = JsonCheckpointHandler()
    registry.add('state', args_lib.StandardSave, state_handler)
    registry.add('state', args_lib.StandardRestore, state_handler)
    registry.add(
        'metadata',
        args_lib.JsonSave,
        metadata_handler,
    )
    registry.add('metadata', args_lib.JsonRestore, metadata_handler)

    handler = CompositeCheckpointHandler(handler_registry=registry)
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
    with mock.patch.object(
        handler,
        '_get_or_set_handler',
        return_value=state_handler,
        autospec=True,
    ):
      self.save(
          handler,
          self.directory,
          CompositeArgs(
              state=args_lib.StandardSave({'a': 1, 'b': 2}),
          ),
      )
      # Finalize only called for items that are actually present.
      state_handler.finalize.assert_called_once()
      metadata_handler.finalize.assert_not_called()
      self.assertFalse(
          (self.directory / 'state' / step._COMMIT_SUCCESS_FILE).exists()
      )

  @mock.patch.object(step, 'is_gcs_path', autospec=True, return_value=True)
  def test_finalize_gcs(self, is_gcs_path):
    del is_gcs_path
    state_handler = mock.create_autospec(StandardCheckpointHandler)
    handler = CompositeCheckpointHandler(state=state_handler)
    with mock.patch.object(
        handler,
        '_get_or_set_handler',
        return_value=state_handler,
        autospec=True,
    ):
      self.save(
          handler,
          self.directory,
          CompositeArgs(
              state=args_lib.StandardSave({'a': 1, 'b': 2}),
          ),
      )
      state_handler.finalize.assert_called_once()
      self.assertTrue(
          (self.directory / 'state' / step._COMMIT_SUCCESS_FILE).exists()
      )

  def test_close(self):
    state_handler = mock.create_autospec(StandardCheckpointHandler)
    metadata_handler = mock.create_autospec(JsonCheckpointHandler)
    handler = CompositeCheckpointHandler(
        'extra', state=state_handler, metadata=metadata_handler
    )
    handler.close()
    state_handler.close.assert_called_once()
    metadata_handler.close.assert_called_once()


  def test_items_exist_final(self):
    handler = CompositeCheckpointHandler('state', 'metadata')
    state = {'a': 1, 'b': 2}
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
    self.assertDictEqual(
        {'state': True, 'metadata': True, 'blob': False},
        handler._items_exist(
            self.directory,
            ['state', 'metadata', 'blob'],
        ),
    )

  def test_items_exist_temp(self):
    handler = CompositeCheckpointHandler('state', 'metadata')
    state = {'a': 1, 'b': 2}
    metadata = {'lang': 'en', 'version': 1.0}
    handler.save(
        self.directory,
        CompositeArgs(
            state=args_lib.StandardSave(state),
            metadata=args_lib.JsonSave(metadata),
        ),
    )
    self.assertFalse((self.directory / 'state').exists())
    self.assertFalse((self.directory / 'metadata').exists())
    self.assertDictEqual(
        {'state': False, 'metadata': False, 'blob': False},
        handler._items_exist(
            self.directory,
            ['state', 'metadata', 'blob'],
        ),
    )
    tmp_dirs = handler._current_temporary_paths
    self.assertIn('state', tmp_dirs.keys())
    self.assertIn('metadata', tmp_dirs.keys())
    self.assertIn(
        (self.directory / 'state.orbax-checkpoint-tmp-').as_posix(),
        tmp_dirs['state'].get().as_posix(),
    )
    self.assertIn(
        (self.directory / 'metadata.orbax-checkpoint-tmp-').as_posix(),
        tmp_dirs['metadata'].get().as_posix(),
    )

  def test_handler_registry_and_items_and_handlers_raises_error(self):

    with self.assertRaisesRegex(ValueError, 'items_and_handlers'):
      handler_registry = handler_registration.DefaultCheckpointHandlerRegistry()
      CompositeCheckpointHandler(
          handler_registry=handler_registry,
          items_and_handlers={
              'state': StandardCheckpointHandler(),
              'metadata': JsonCheckpointHandler(),
          },
      )

  def test_handler_registry_and_items_names_raises_error(self):

    with self.assertRaisesRegex(ValueError, 'item_names'):
      handler_registry = handler_registration.DefaultCheckpointHandlerRegistry()
      CompositeCheckpointHandler(
          'state',
          'metadata',
          handler_registry=handler_registry,
      )


if __name__ == '__main__':
  absltest.main()
