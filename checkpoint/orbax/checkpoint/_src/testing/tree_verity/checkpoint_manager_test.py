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

"""Tests covering the tree verity functionality of the checkpoint manager."""

from absl.testing import parameterized
from etils import epath
import jax
from orbax.checkpoint import args
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import step as step_lib
from orbax.checkpoint._src.testing import multiprocess_test

ArrayRestoreArgs = pytree_checkpoint_handler.ArrayRestoreArgs
CheckpointManager = checkpoint_manager.CheckpointManager
CheckpointManagerOptions = checkpoint_manager.CheckpointManagerOptions


@test_utils.barrier_compatible_test
class CheckpointManagerTreeVerityTest(
    parameterized.TestCase, multiprocess_test.MultiProcessTest
):
  """Checkpoint manager tests that run using the GFile tensorstore driver."""

  def setUp(self):
    super().setUp()

    if not multihost.is_runtime_to_distributed_ids_initialized():
      multihost.initialize_runtime_to_distributed_ids()

    self.assertEqual(jax.device_count(), 8)
    self.assertEqual(jax.process_count(), 4)
    self.assertEqual(jax.local_device_count(), 2)
    pytree, _, _ = test_utils.setup_sharded_pytree()

    self.pytree = pytree
    self.save_params = args.Composite(params=args.PyTreeSave(self.pytree))
    self.directory = epath.Path(
        self.create_tempdir(name='checkpoint_manager_test').full_path
    )

    test_utils.sync_global_processes(
        'CheckpointManagerTreeVerityTest:setup_complete'
    )

  def tearDown(self):
    test_utils.sync_global_processes(
        'CheckpointManagerTreeVerityTest:tests_complete'
    )
    super().tearDown()

  def assert_tree_verity_present(self, directory: epath.Path):
    steps = step_lib.checkpoint_steps_paths(directory)
    messages = []
    for step_path in steps:
      if not step_path.joinpath('TREE_VERITY').exists():
        messages.append('TREE_VERITY directory not found at %s' % step_path)
    if messages:
      self.fail('\n'.join(messages))

  @parameterized.named_parameters(
      dict(
          testcase_name='with_hash_on_restore',
          hash_on_restore=True,
          enable_async_checkpointing=False,
      ),
      dict(
          testcase_name='with_hash_on_restore_and_async',
          hash_on_restore=True,
          enable_async_checkpointing=True,
      ),
      dict(
          testcase_name='without_hash_on_restore',
          hash_on_restore=False,
          enable_async_checkpointing=False,
      ),
  )
  def test_verity_present_when_signing_enabled(
      self, hash_on_restore: bool, enable_async_checkpointing: bool
  ):
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            enable_async_checkpointing=enable_async_checkpointing,
            signing_options=checkpoint_manager.SigningOptions(
                sign_on_save=True, hash_on_restore=hash_on_restore
            ),
        ),
    ) as manager:
      self.assertTrue(manager.save(0, args=self.save_params, force=False))
      manager.wait_until_finished()
      self.assert_tree_verity_present(manager.directory)


if __name__ == '__main__':
  multiprocess_test.main()
