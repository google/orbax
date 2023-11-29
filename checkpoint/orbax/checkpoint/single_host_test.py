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

"""To test Orbax in single-host setup."""

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import pytree_checkpoint_handler
from orbax.checkpoint import utils


class PyTreeCheckpointHandler(
    pytree_checkpoint_handler.PyTreeCheckpointHandler
):

  def save(self, directory, *args, **kwargs):
    super().save(directory, *args, **kwargs)
    if jax.process_index() == 0:
      self.finalize(directory)
    utils.sync_global_devices('PyTreeCheckpointHandler:finalize')


class SingleHostTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt_dir = epath.Path(self.create_tempdir('ckpt').full_path)

  @parameterized.parameters([False, True])
  def test_save_and_restore_a_single_device_sharded_jax_array(
      self, write_tree_metadata
  ):
    handler = PyTreeCheckpointHandler(write_tree_metadata=write_tree_metadata)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10,))
    assert isinstance(x.sharding, jax.sharding.SingleDeviceSharding)
    handler.save(
        self.ckpt_dir,
        args=pytree_checkpoint_handler.PyTreeSaveArgs({'array_x': x}),
    )

    restored_tree = handler.restore(self.ckpt_dir)
    np.testing.assert_array_equal(x, restored_tree['array_x'])

    if write_tree_metadata:
      self.assertIsInstance(restored_tree['array_x'], jax.Array)
      self.assertEqual(x.sharding, restored_tree['array_x'].sharding)
    else:
      # previously, Orbax just return numpyarray even the saved
      # the array is a SigmleDeviceSharded Jax Array.
      self.assertIsInstance(restored_tree['array_x'], np.ndarray)


if __name__ == '__main__':
  absltest.main()
