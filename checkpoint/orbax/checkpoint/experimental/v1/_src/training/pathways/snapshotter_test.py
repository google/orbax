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

"""Tests for Snapshotter.

This module tests the Snapshot class, which manages asynchronous backups of JAX
array states to pinned host memory.
"""

import jax
from orbax.checkpoint import test_utils
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils
from orbax.checkpoint.experimental.v1._src.training.pathways import snapshotter

from .pyglib.contrib.g3_multiprocessing import g3_multiprocessing
from absl.testing import absltest
from .testing.pybase import parameterized


class SnapshotterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    print('jax.devices():', jax.devices())
    slices = len(set([d.slice_index for d in jax.devices()]))
    self.pytree, self.abstract_pytree = array_test_utils.create_sharded_pytree(
        replicated_arrays=True,
        slices=slices,
    )

    available_slices = [0, 1]
    available_devices = [
        d for d in jax.devices() if d.slice_index in available_slices
    ]
    self.pytree_2_slices, self.abstract_pytree_2_slices = (
        array_test_utils.create_sharded_pytree(
            replicated_arrays=True,
            devices=available_devices,
            slices=len(available_slices),
        )
    )

  def test_save_restore_pytree(self):
    snapshot_manager = snapshotter.Snapshotter()

    snapshot_manager.save_pytree(0, self.pytree)
    snapshot_manager.save_pytree(1, self.pytree)
    self.assertLen(snapshot_manager._snapshots, 2)
    test_utils.assert_tree_equal(
        self, self.pytree, snapshot_manager._snapshots[-1][0]
    )

    restored_pytree = snapshot_manager.load_pytree(self.abstract_pytree)
    # Check that restored array is on default memory (not pinned_host).
    # Since we restored to self.abstract_pytree, it should match self.pytree.
    test_utils.assert_tree_equal(self, self.pytree, restored_pytree)
    jax.tree.map(
        lambda x: self.assertNotEqual(x.sharding.memory_kind, 'pinned_host'),
        restored_pytree,
    )

  def test_scale_down(self):
    snapshot_manager = snapshotter.Snapshotter()

    snapshot_manager.save_pytree(0, self.pytree)
    restored_pytree = snapshot_manager.load_pytree(
        self.abstract_pytree_2_slices
    )
    expected_pytree = jax.tree.map(
        lambda x, y: jax.device_put(x, y.sharding),
        self.pytree,
        self.abstract_pytree_2_slices,
    )
    test_utils.assert_tree_equal(self, expected_pytree, restored_pytree)

  def test_scale_up(self):
    snapshot_manager = snapshotter.Snapshotter()

    snapshot_manager.save_pytree(0, self.pytree_2_slices)
    restored_pytree = snapshot_manager.load_pytree(self.abstract_pytree)
    expected_pytree = jax.tree.map(
        lambda x, y: jax.device_put(x, y.sharding),
        self.pytree_2_slices,
        self.abstract_pytree,
    )
    test_utils.assert_tree_equal(self, expected_pytree, restored_pytree)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  g3_multiprocessing.handle_test_main(googletest.main)
