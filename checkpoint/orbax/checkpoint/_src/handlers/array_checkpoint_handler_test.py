# Copyright 2026 The Orbax Authors.
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

"""Tests for ArrayCheckpointHandler."""

from absl import flags
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.handlers import array_checkpoint_handler
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.testing import multiprocess_test


SaveArgs = type_handlers.SaveArgs
ArraySaveArgs = array_checkpoint_handler.ArraySaveArgs
ArrayRestoreArgs = array_checkpoint_handler.ArrayRestoreArgs


FLAGS = flags.FLAGS


class ArrayCheckpointHandler(array_checkpoint_handler.ArrayCheckpointHandler):

  def save(self, directory, *args, **kwargs):
    super().save(directory, *args, **kwargs)
    test_utils.sync_global_processes('ArrayCheckpointHandler:save')
    if multihost.process_index() == 0:
      self.finalize(directory)
    test_utils.sync_global_processes('ArrayCheckpointHandler:finalize')


class ArrayCheckpointHandlerTest(
    parameterized.TestCase, multiprocess_test.MultiProcessTest
):

  def setUp(self):
    super().setUp()
    self.devices = np.asarray(jax.devices())
    self.directory = epath.Path(
        self.create_tempdir(name='checkpointing_test').full_path
    )

    test_utils.sync_global_processes(
        'ArrayCheckpointHandlerTest:setup_complete'
    )

  def tearDown(self):
    test_utils.sync_global_processes(
        'ArrayCheckpointHandlerTest:tests_complete'
    )
    super().tearDown()

  def validate_save(self):
    path = self.directory / array_checkpoint_handler.PYTREE_METADATA_FILE
    self.assertTrue(path.exists())

  def test_array(self):
    checkpoint_handler = ArrayCheckpointHandler()
    mesh = jax.sharding.Mesh(self.devices, ('x',))
    mesh_axes = jax.sharding.PartitionSpec(
        'x',
    )
    arr = test_utils.create_sharded_array(np.arange(16), mesh, mesh_axes)
    save_args = SaveArgs()
    checkpoint_handler.save(self.directory, args=ArraySaveArgs(arr, save_args))
    self.validate_save()
    restored = checkpoint_handler.restore(
        self.directory,
        args=ArrayRestoreArgs(
            restore_args=type_handlers.ArrayRestoreArgs(
                restore_type=jax.Array, mesh=mesh, mesh_axes=mesh_axes
            )
        ),
    )
    test_utils.assert_tree_equal(self, [arr], [restored])
    checkpoint_handler.close()

  def test_numpy_array(self):
    checkpoint_handler = ArrayCheckpointHandler()
    arr = np.arange(16)
    save_args = SaveArgs()
    checkpoint_handler.save(self.directory, args=ArraySaveArgs(arr, save_args))
    self.validate_save()
    restored = checkpoint_handler.restore(
        self.directory,
        args=ArrayRestoreArgs(
            restore_args=type_handlers.RestoreArgs(restore_type=np.ndarray)
        ),
    )
    test_utils.assert_tree_equal(self, [arr], [restored])
    checkpoint_handler.close()

  def test_scalar(self):
    checkpoint_handler = ArrayCheckpointHandler()
    save_args = SaveArgs()
    checkpoint_handler.save(self.directory, args=ArraySaveArgs(5, save_args))
    self.validate_save()
    restored = checkpoint_handler.restore(
        self.directory,
        args=ArrayRestoreArgs(
            restore_args=type_handlers.RestoreArgs(restore_type=int)
        ),
    )
    self.assertEqual(5, restored)
    checkpoint_handler.close()

  def test_invalid_type(self):
    checkpoint_handler = ArrayCheckpointHandler()
    with self.assertRaises(TypeError):
      checkpoint_handler.save(self.directory, args=ArraySaveArgs('hi'))
    checkpoint_handler.close()

  def test_different_name(self):
    checkpoint_name = 'my_array'
    checkpoint_handler = ArrayCheckpointHandler(checkpoint_name=checkpoint_name)
    arr = np.arange(16)
    save_args = SaveArgs()
    checkpoint_handler.save(self.directory, args=ArraySaveArgs(arr, save_args))
    self.validate_save()
    restored = checkpoint_handler.restore(
        self.directory,
        args=ArrayRestoreArgs(
            restore_args=type_handlers.RestoreArgs(restore_type=np.ndarray)
        ),
    )
    test_utils.assert_tree_equal(self, [arr], [restored])
    checkpoint_handler.close()

  def test_restore_type(self):
    pytree = 5
    checkpoint_handler = ArrayCheckpointHandler()

    checkpoint_handler.save(self.directory, args=ArraySaveArgs(pytree))
    restored = checkpoint_handler.restore(
        self.directory,
        args=ArrayRestoreArgs(
            restore_args=type_handlers.RestoreArgs(restore_type=np.ndarray)
        ),
    )
    self.assertIsInstance(restored, np.ndarray)


if __name__ == '__main__':
  multiprocess_test.main()
