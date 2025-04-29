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

"""Base class for v0/v1 compatibility save/load tests."""

# pylint: disable=missing-class-docstring,protected-access,missing-function-docstring

from __future__ import annotations

from absl.testing import parameterized
from etils import epath
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.checkpointers import checkpointer as v0_checkpointer
from orbax.checkpoint._src.checkpointers import standard_checkpointer
from orbax.checkpoint._src.handlers import composite_checkpoint_handler
import orbax.checkpoint.experimental.v1 as ocp
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


PyTree = tree_types.PyTree
Path = path_types.Path

create_numpy_pytree = array_test_utils.create_numpy_pytree
create_sharded_pytree = array_test_utils.create_sharded_pytree


class CompatibilitySaveLoadTestBase:

  class Test(parameterized.TestCase):

    def setUp(self):
      super().setUp()

      self.directory = (
          epath.Path(self.create_tempdir(name='compat_test').full_path) / 'ckpt'
      )
      self.pytree, self.abstract_pytree = create_sharded_pytree()
      self.numpy_pytree, self.abstract_numpy_pytree = create_numpy_pytree()

      test_utils.set_tensorstore_driver_for_test()
      test_utils.sync_global_processes('CompatibilityTest:setup_complete')

    def tearDown(self):
      super().tearDown()
      test_utils.sync_global_processes('CompatibilityTest:teardown_complete')

    def save_v0_checkpoint(
        self,
        directory: Path,
        checkpointable_name: str,
        pytree: PyTree,
        to_checkpointable_subdir: bool = False,
    ):
      if to_checkpointable_subdir:
        with standard_checkpointer.StandardCheckpointer() as checkpointer:
          checkpointer.save(directory / checkpointable_name, pytree)
      else:
        with v0_checkpointer.Checkpointer(
            composite_checkpoint_handler.CompositeCheckpointHandler()
        ) as checkpointer:
          checkpointer.save(
              directory,
              args_lib.Composite(
                  **{checkpointable_name: args_lib.StandardSave(pytree)}
              ),
          )

    @parameterized.product(
        with_abstract_pytree=[True, False],
    )
    def test_load_v0_checkpoint_with_v1_load_pytree(
        self, with_abstract_pytree: bool
    ):
      self.save_v0_checkpoint(
          self.directory,
          'pytree',
          self.pytree,
          to_checkpointable_subdir=False,
      )

      loaded = ocp.load_pytree(
          self.directory,
          self.abstract_pytree if with_abstract_pytree else None,
      )
      test_utils.assert_tree_equal(self, self.pytree, loaded)

    @parameterized.product(
        checkpointable_name=['default', 'state'],
        load_async=[True, False],
        with_abstract_pytree=[True, False],
        to_checkpointable_subdir=[True, False],
    )
    def test_load_v0_checkpoint_with_v1_load_pytree_failures(
        self,
        checkpointable_name: str,
        load_async: bool,
        with_abstract_pytree: bool,
        to_checkpointable_subdir: bool,
    ):
      self.save_v0_checkpoint(
          self.directory,
          checkpointable_name,
          self.pytree,
          to_checkpointable_subdir,
      )

      if load_async:
        with self.assertRaises(NotImplementedError):
          ocp.load_pytree_async(
              self.directory,
              self.abstract_pytree if with_abstract_pytree else None,
          )
      else:
        with self.assertRaisesRegex(
            FileNotFoundError, 'must contain a subdirectory named "pytree"'
        ):
          ocp.load_pytree(
              self.directory,
              self.abstract_pytree if with_abstract_pytree else None,
          )
        with self.assertRaisesRegex(
            FileNotFoundError, 'must contain a subdirectory named "pytree"'
        ):
          ocp.load_pytree(
              self.directory / checkpointable_name,
              self.abstract_pytree if with_abstract_pytree else None,
          )

    @parameterized.product(
        checkpointable_name=['default', 'state'],
        load_async=[True, False],
        with_abstract_pytree=[True, False],
        to_checkpointable_subdir=[True, False],
    )
    def test_load_v0_checkpoint_with_v1_load_checkpointables(
        self,
        checkpointable_name: str,
        load_async: bool,
        with_abstract_pytree: bool,
        to_checkpointable_subdir: bool,
    ):
      self.save_v0_checkpoint(
          self.directory,
          checkpointable_name,
          self.pytree,
          to_checkpointable_subdir,
      )

      abstract_checkpointables = (
          {checkpointable_name: self.abstract_pytree}
          if with_abstract_pytree
          else None
      )

      if load_async:
        with self.assertRaises(NotImplementedError):
          ocp.load_checkpointables_async(
              self.directory, abstract_checkpointables
          )
      else:
        with self.subTest('with_context'):
          checkpointables_options = (
              ocp.options.CheckpointablesOptions.create_with_handlers(
                  **{checkpointable_name: ocp.handlers.PyTreeHandler}
              )
          )
          with ocp.Context(checkpointables_options=checkpointables_options):
            loaded = ocp.load_checkpointables(
                self.directory, abstract_checkpointables
            )
            test_utils.assert_tree_equal(
                self, self.pytree, loaded[checkpointable_name]
            )
        with self.subTest('without_context'):
          loaded = ocp.load_checkpointables(
              self.directory, abstract_checkpointables
          )
          test_utils.assert_tree_equal(
              self, self.pytree, loaded[checkpointable_name]
          )
