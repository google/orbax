# Copyright 2025 The Orbax Authors.
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

from unittest import mock

from absl.testing import absltest
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint.experimental.emergency.p2p import local

Mesh = jax.sharding.Mesh


class LocalCheckpointManagerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path)
    self.mesh = Mesh(np.array(jax.devices()), axis_names=('x',))
    self.options = checkpoint_manager.CheckpointManagerOptions()

  @mock.patch(
      'orbax.checkpoint._src.multihost.multihost.process_index', return_value=0
  )
  def test_init(self, unused_process_index):
    manager = local.LocalCheckpointManager(
        self.directory, self.mesh, options=self.options
    )
    self.assertEqual(manager.directory, self.directory)
    self.assertEqual(manager._process_index, 0)
    self.assertIsNotNone(manager._manager)
    manager.close()

  @mock.patch(
      'orbax.checkpoint._src.multihost.multihost.process_index', return_value=0
  )
  def test_scan_stored_steps_empty(self, unused_process_index):
    manager = local.LocalCheckpointManager(
        self.directory, self.mesh, options=self.options
    )
    detected_index, steps = manager.scan_stored_steps()
    self.assertIsNone(detected_index)
    self.assertEmpty(steps)
    manager.close()

  @mock.patch(
      'orbax.checkpoint._src.multihost.multihost.process_index', return_value=0
  )
  def test_detect_process_index(self, unused_process_index):
    manager = local.LocalCheckpointManager(
        self.directory, self.mesh, options=self.options
    )
    step_dir = self.directory / '1'
    step_dir.mkdir()
    (step_dir / 'state' / 'ocdbt.process_42').mkdir(parents=True)

    self.assertEqual(manager._detect_process_index(1), 42)
    self.assertIsNone(manager._detect_process_index(2))
    manager.close()

  @mock.patch(
      'orbax.checkpoint._src.multihost.multihost.process_index', return_value=0
  )
  def test_restore_process_mismatch_raises_error(self, unused_process_index):
    manager = local.LocalCheckpointManager(
        self.directory, self.mesh, options=self.options
    )
    step_dir = self.directory / '1'
    step_dir.mkdir()
    (step_dir / 'state' / 'ocdbt.process_1').mkdir(
        parents=True
    )  # Stored by process 1

    with self.assertRaisesRegex(ValueError, 'Process Mismatch'):
      manager.restore(1)
    manager.close()


if __name__ == '__main__':
  absltest.main()
