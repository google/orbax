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

import unittest

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
import orbax.checkpoint.experimental.v1 as ocp
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import roc_layout
from orbax.checkpoint.experimental.v1._src.testing import path_utils

from .learning.deepmind.jax.roc import guess
from .learning.deepmind.jax.roc import roc

RocLayout = roc_layout.RocLayout
InvalidLayoutError = checkpoint_layout.InvalidLayoutError

jax.config.update('jax_enable_x64', True)


def _save_roc_checkpoint(
    path: epath.Path,
    state: dict[str, np.ndarray | jax.Array],
    checkpoint_format: roc.checkpoint.FormatEnum = roc.checkpoint.FormatEnum.ORBAX_TENSOR_STORE,
):
  coordinator = roc.coordination.OneShotWriterCoordinator(
      roc.checkpoint.Path(path.as_posix()),
      checkpoint_format=checkpoint_format,
  )
  with coordinator.sole_checkpoint_host_context() as host_context:
    roc.save(
        host_context,
        state,
        shard_desc_fn=roc.sharding.identity_shard_desc_fn,
    )


class RocLayoutTest(unittest.IsolatedAsyncioTestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = epath.Path(self.create_tempdir().full_path)
    self.checkpoint_path_with_state_subdir = self.test_dir / 'checkpoint_nested'
    self.checkpoint_path = self.test_dir / 'checkpoint_flat'

    self.state_to_save = {
        'a': jax.device_put(np.array([1, 2, 3], dtype=np.int32)),
        'b': jax.device_put(np.array(42, dtype=np.int32)),
        'c': jax.device_put(np.array(3.14, dtype=np.float64)),
        'd': jax.device_put(np.array(7, dtype=np.int64)),
    }
    devices = jax.local_devices()
    sharding = jax.sharding.SingleDeviceSharding(devices[0])
    self.abstract_state = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding),
        self.state_to_save,
    )

    # Save nested checkpoint (under 'state' subdirectory)
    _save_roc_checkpoint(
        self.checkpoint_path_with_state_subdir / 'state',
        self.state_to_save,
    )

    # Save flat checkpoint (directly under checkpoint_path)
    _save_roc_checkpoint(self.checkpoint_path, self.state_to_save)

  @parameterized.named_parameters(
      ('state', 'state', None),
      ('none', None, InvalidLayoutError),
      ('invalid_name', 'invalid_name', InvalidLayoutError),
  )
  async def test_validate_state_subdir(
      self, checkpointable_name, expected_exception
  ):
    layout = RocLayout()
    if expected_exception:
      with self.assertRaises(expected_exception):
        await layout.validate(
            self.checkpoint_path_with_state_subdir, checkpointable_name
        )
    else:
      await layout.validate(
          self.checkpoint_path_with_state_subdir, checkpointable_name
      )

  @parameterized.named_parameters(
      ('state', 'state', InvalidLayoutError),
      ('none', None, None),
      ('invalid_name', 'invalid_name', InvalidLayoutError),
  )
  async def test_validate_checkpoint_path(
      self, checkpointable_name, expected_exception
  ):
    layout = RocLayout()
    if expected_exception:
      with self.assertRaises(expected_exception):
        await layout.validate(self.checkpoint_path, checkpointable_name)
    else:
      await layout.validate(self.checkpoint_path, checkpointable_name)

  @parameterized.named_parameters(
      ('state', 'state', None),
      ('none', None, guess.FormatNotFoundError),
      ('auto', checkpoint_layout.AUTO_CHECKPOINTABLE_KEY, InvalidLayoutError),
  )
  async def test_load_state_subdir(
      self, checkpointable_name, expected_exception
  ):
    layout = RocLayout()

    if expected_exception:
      with self.assertRaises(expected_exception):
        restore_fn = await layout.load(
            self.checkpoint_path_with_state_subdir,
            checkpointable_name=checkpointable_name,
            abstract_state=self.abstract_state,
        )
        await restore_fn
    else:
      restore_fn = await layout.load(
          self.checkpoint_path_with_state_subdir,
          checkpointable_name=checkpointable_name,
          abstract_state=self.abstract_state,
      )
      result = await restore_fn
      jax.tree.map(np.testing.assert_array_equal, result, self.state_to_save)

  @parameterized.named_parameters(
      ('state', 'state', ValueError),
      ('none', None, None),
      ('auto', checkpoint_layout.AUTO_CHECKPOINTABLE_KEY, InvalidLayoutError),
  )
  async def test_load_checkpoint_path(
      self, checkpointable_name, expected_exception
  ):
    layout = RocLayout()

    if expected_exception:
      with self.assertRaises(expected_exception):
        restore_fn = await layout.load(
            self.checkpoint_path,
            checkpointable_name=checkpointable_name,
            abstract_state=self.abstract_state,
        )
        await restore_fn
    else:
      restore_fn = await layout.load(
          self.checkpoint_path,
          checkpointable_name=checkpointable_name,
          abstract_state=self.abstract_state,
      )
      result = await restore_fn
      jax.tree.map(np.testing.assert_array_equal, result, self.state_to_save)

  async def test_load_as_jax_array(self):
    layout = RocLayout()
    restore_fn = await layout.load(
        self.checkpoint_path,
        checkpointable_name=None,
        abstract_state=self.abstract_state,
    )
    result = await restore_fn
    jax.tree.map(lambda x: self.assertIsInstance(x, jax.Array), result)
    jax.tree.map(np.testing.assert_array_equal, result, self.state_to_save)

  async def test_load_validation_error(self):
    layout = RocLayout()
    with self.assertRaises(ValueError):
      await layout.load(
          self.checkpoint_path,
          checkpointable_name=None,
          abstract_state={'a': np.zeros((3,), dtype=np.int32)},
      )
    with self.assertRaises(ValueError):
      await layout.load(
          self.checkpoint_path,
          checkpointable_name=None,
          abstract_state=None,
      )
    with self.assertRaises(ValueError):
      await layout.load(
          self.checkpoint_path,
          checkpointable_name=None,
          abstract_state={
              'a': jax.ShapeDtypeStruct(shape=(3,), dtype=np.int32)
          },
      )
    with self.assertRaises(ValueError):
      await layout.load(
          self.checkpoint_path,
          checkpointable_name=None,
          abstract_state={'a': jax.device_put(np.array([1, 2, 3]))},
      )

  async def test_validate_checkpointables_success_state_subdir(self):
    layout = RocLayout()
    await layout.validate_checkpointables(
        self.checkpoint_path_with_state_subdir
    )

  async def test_validate_checkpointables_success_checkpoint_path(self):
    layout = RocLayout()
    await layout.validate_checkpointables(self.checkpoint_path)

  async def test_validate_checkpointables_fail(self):
    layout = RocLayout()
    empty_dir = self.test_dir / 'empty_dir'
    empty_dir.mkdir()
    with self.assertRaises(InvalidLayoutError):
      await layout.validate_checkpointables(empty_dir)

  async def test_get_checkpointable_names_state_subdir(self):
    layout = RocLayout()
    names = await layout.get_checkpointable_names(
        self.checkpoint_path_with_state_subdir
    )
    self.assertEqual(names, ['state'])

  async def test_get_checkpointable_names_checkpoint_path(self):
    layout = RocLayout()
    names = await layout.get_checkpointable_names(self.checkpoint_path)
    self.assertEqual(names, [None])

  async def test_not_implemented_methods(self):
    layout = RocLayout()
    with self.assertRaises(NotImplementedError):
      await layout.checkpointables_metadata(
          self.checkpoint_path_with_state_subdir
      )

    with self.assertRaises(NotImplementedError):
      await layout.load_checkpointables(self.checkpoint_path_with_state_subdir)

    with self.assertRaises(NotImplementedError):
      await layout.save_checkpointables(
          path_utils.PathAwaitingCreationWrapper(
              self.checkpoint_path_with_state_subdir
          ),
          checkpointables={},
      )

  async def test_metadata_state_subdir(self):
    layout = RocLayout()
    metadata = await layout.metadata(
        self.checkpoint_path_with_state_subdir, 'state'
    )
    self.assertEqual(metadata.path, self.checkpoint_path_with_state_subdir)
    jax.tree.map(
        lambda x, y: self.assertEqual(x.shape, y.shape)
        or self.assertEqual(x.dtype, y.dtype)
        or self.assertIsInstance(x, roc.sharding.ArrayDesc),
        metadata.metadata,
        self.abstract_state,
    )

  async def test_metadata_checkpoint_path(self):
    layout = RocLayout()
    metadata = await layout.metadata(self.checkpoint_path, None)
    self.assertEqual(metadata.path, self.checkpoint_path)
    jax.tree.map(
        lambda x, y: self.assertEqual(x.shape, y.shape)
        or self.assertEqual(x.dtype, y.dtype)
        or self.assertIsInstance(x, roc.sharding.ArrayDesc),
        metadata.metadata,
        self.abstract_state,
    )

  async def test_metadata_fail_invalid_name(self):
    layout = RocLayout()
    with self.assertRaises(InvalidLayoutError):
      await layout.metadata(self.checkpoint_path, 'invalid_name')


class RocLayoutPublicInterfaceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = epath.Path(self.create_tempdir().full_path)
    self.checkpoint_path_with_state_subdir = self.test_dir / 'checkpoint_nested'
    self.checkpoint_path = self.test_dir / 'checkpoint_flat'

    self.state_to_save = {'a': np.array([1, 2, 3], dtype=np.int32)}
    devices = jax.local_devices()
    sharding = jax.sharding.SingleDeviceSharding(devices[0])
    self.abstract_state = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding),
        self.state_to_save,
    )

    # Save nested checkpoint (under 'state' subdirectory)
    _save_roc_checkpoint(
        self.checkpoint_path_with_state_subdir / 'state',
        self.state_to_save,
    )

    # Save flat checkpoint (directly under checkpoint_path)
    _save_roc_checkpoint(self.checkpoint_path, self.state_to_save)

  @parameterized.named_parameters(
      ('state', 'state', None),
      ('none', None, InvalidLayoutError),
      ('auto', checkpoint_layout.AUTO_CHECKPOINTABLE_KEY, None),
  )
  def test_load_checkpoint_path_with_state_subdir(
      self, checkpointable_name, expected_exception
  ):
    with ocp.Context(checkpoint_layout=ocp.options.CheckpointLayout.ROC):
      if expected_exception:
        with self.assertRaises(expected_exception):
          ocp.load(
              self.checkpoint_path_with_state_subdir,
              abstract_state=self.abstract_state,
              checkpointable_name=checkpointable_name,
          )
      else:
        result = ocp.load(
            self.checkpoint_path_with_state_subdir,
            abstract_state=self.abstract_state,
            checkpointable_name=checkpointable_name,
        )
        jax.tree.map(np.testing.assert_array_equal, result, self.state_to_save)

  @parameterized.named_parameters(
      ('state', 'state', InvalidLayoutError),
      ('none', None, None),
      ('auto', checkpoint_layout.AUTO_CHECKPOINTABLE_KEY, None),
  )
  def test_load_checkpoint_path(self, checkpointable_name, expected_exception):
    with ocp.Context(checkpoint_layout=ocp.options.CheckpointLayout.ROC):
      if expected_exception:
        with self.assertRaises(expected_exception):
          ocp.load(
              self.checkpoint_path,
              abstract_state=self.abstract_state,
              checkpointable_name=checkpointable_name,
          )
      else:
        result = ocp.load(
            self.checkpoint_path,
            abstract_state=self.abstract_state,
            checkpointable_name=checkpointable_name,
        )
        jax.tree.map(np.testing.assert_array_equal, result, self.state_to_save)

  @parameterized.named_parameters(
      ('state_subdir', 'state_subdir', 'state'),
      ('checkpoint_path', 'checkpoint_path', None),
  )
  def test_metadata(self, checkpoint_type, checkpointable_name):
    with ocp.Context(checkpoint_layout=ocp.options.CheckpointLayout.ROC):
      path = (
          self.checkpoint_path_with_state_subdir
          if checkpoint_type == 'state_subdir'
          else self.checkpoint_path
      )
      metadata = ocp.metadata(path, checkpointable_name)
      self.assertEqual(metadata.path, path)
      leaves = jax.tree_util.tree_leaves(metadata.metadata)
      self.assertLen(leaves, 1)
      self.assertIsInstance(leaves[0], roc.sharding.ArrayDesc)
      self.assertEqual(leaves[0].shape, (3,))
      self.assertEqual(leaves[0].dtype, np.dtype(np.int32))

  @parameterized.named_parameters(
      ('orbax_tensor_store', roc.checkpoint.FormatEnum.ORBAX_TENSOR_STORE),
      ('einshape_numpy_proto', roc.checkpoint.FormatEnum.EINSHAPE_NUMPY_PROTO),
      (
          'einshape_numpy_pickle',
          roc.checkpoint.FormatEnum.EINSHAPE_NUMPY_PICKLE,
      ),
  )
  def test_load_formats(self, checkpoint_format):
    path = self.test_dir / f'checkpoint_{checkpoint_format.name}'
    _save_roc_checkpoint(path, self.state_to_save, checkpoint_format)

    with ocp.Context(checkpoint_layout=ocp.options.CheckpointLayout.ROC):
      result = ocp.load(
          path,
          abstract_state=self.abstract_state,
          checkpointable_name=None,
      )
      jax.tree.map(np.testing.assert_array_equal, result, self.state_to_save)


if __name__ == '__main__':
  absltest.main()
