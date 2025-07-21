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
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


PyTree = tree_types.PyTree
Path = path_types.Path
InvalidLayoutError = checkpoint_layout.InvalidLayoutError

create_sharded_pytree = array_test_utils.create_sharded_pytree

_V0_ERROR_SUBSTR = (
    'If your checkpoint was saved with the Orbax V0 API, please follow the'
    ' instructions at'
)


class CompatibilitySaveLoadTestBase:

  class Test(parameterized.TestCase):

    def setUp(self):
      super().setUp()

      self.root_directory = epath.Path(
          self.create_tempdir(name='root').full_path
      )
      self.ckpt_directory = (
          epath.Path(self.create_tempdir(name='direct').full_path) / 'ckpt'
      )
      self.pytree, self.abstract_pytree = create_sharded_pytree()

      test_utils.set_tensorstore_driver_for_test()
      test_utils.sync_global_processes('CompatibilityTest:setup_complete')

    def tearDown(self):
      super().tearDown()
      test_utils.sync_global_processes('CompatibilityTest:teardown_complete')

    def save_v0_checkpoint(self, directory: Path):
      with standard_checkpointer.StandardCheckpointer() as checkpointer:
        checkpointer.save(directory, self.pytree)

    def save_v0_checkpoints(
        self, base_dir: Path, *, checkpointable_names: list[str]
    ):
      args = args_lib.Composite(**{
          checkpointable_name: args_lib.StandardSave(self.pytree)
          for checkpointable_name in checkpointable_names
      })
      with v0_checkpointer.Checkpointer(
          composite_checkpoint_handler.CompositeCheckpointHandler()
      ) as checkpointer:
        checkpointer.save(base_dir, args)

    def test_async_load(self):
      with self.assertRaises(NotImplementedError):
        ocp.load_pytree_async(
            self.root_directory,
        )
      with self.assertRaises(NotImplementedError):
        ocp.load_checkpointables_async(
            self.root_directory,
        )

    @parameterized.product(
        with_abstract_pytree=[True, False],
    )
    def test_load_v0_checkpoint_with_v1_load_pytree(
        self, with_abstract_pytree: bool
    ):

      checkpointable_names = ['default', 'state', 'pytree']
      step_dir = self.root_directory / 'load_pytree_0'
      self.save_v0_checkpoints(
          step_dir,
          checkpointable_names=checkpointable_names,
      )
      self.save_v0_checkpoint(self.ckpt_directory)

      with self.subTest('no_checkpointable_name'):
        loaded = ocp.load_pytree(
            step_dir,
            self.abstract_pytree if with_abstract_pytree else None,
        )
        test_utils.assert_tree_equal(self, self.pytree, loaded)

      with self.subTest('no_checkpointable_name_error'):
        with self.assertRaisesRegex(InvalidLayoutError, _V0_ERROR_SUBSTR):
          ocp.load_pytree(
              self.ckpt_directory,
              self.abstract_pytree if with_abstract_pytree else None,
          )
        with self.assertRaisesRegex(InvalidLayoutError, _V0_ERROR_SUBSTR):
          ocp.load_pytree(
              self.root_directory,
              self.abstract_pytree if with_abstract_pytree else None,
          )

      for checkpointable_name in checkpointable_names:
        with self.subTest(f'pass_{checkpointable_name}'):
          loaded = ocp.load_pytree(
              step_dir,
              self.abstract_pytree if with_abstract_pytree else None,
              checkpointable_name=checkpointable_name,
          )
          test_utils.assert_tree_equal(self, self.pytree, loaded)

        with self.subTest(f'pass_{checkpointable_name}_error'):
          with self.assertRaisesRegex(InvalidLayoutError, _V0_ERROR_SUBSTR):
            ocp.load_pytree(
                self.ckpt_directory,
                self.abstract_pytree if with_abstract_pytree else None,
                checkpointable_name=checkpointable_name,
            )
          with self.assertRaisesRegex(InvalidLayoutError, _V0_ERROR_SUBSTR):
            ocp.load_pytree(
                self.root_directory,
                self.abstract_pytree if with_abstract_pytree else None,
                checkpointable_name=checkpointable_name,
            )

      with self.subTest('pass_none'):
        loaded = ocp.load_pytree(
            self.ckpt_directory,
            self.abstract_pytree if with_abstract_pytree else None,
            checkpointable_name=None,
        )
        test_utils.assert_tree_equal(self, self.pytree, loaded)

      with self.subTest('pass_none_error'):
        with self.assertRaisesRegex(InvalidLayoutError, _V0_ERROR_SUBSTR):
          ocp.load_pytree(
              step_dir,
              self.abstract_pytree if with_abstract_pytree else None,
              checkpointable_name=None,
          )
        with self.assertRaisesRegex(InvalidLayoutError, _V0_ERROR_SUBSTR):
          ocp.load_pytree(
              self.root_directory,
              self.abstract_pytree if with_abstract_pytree else None,
              checkpointable_name=None,
          )

    @parameterized.product(
        checkpointable_name=['default', 'state'],
        with_abstract_pytree=[True, False],
    )
    def test_load_v0_checkpoint_with_v1_load_checkpointables(
        self,
        checkpointable_name: str,
        with_abstract_pytree: bool,
    ):

      checkpointable_names = ['default', 'state']
      step_dir = self.root_directory / 'load_checkpointables_0'
      self.save_v0_checkpoints(
          step_dir,
          checkpointable_names=checkpointable_names,
      )
      self.save_v0_checkpoint(self.ckpt_directory)

      abstract_checkpointables = (
          {checkpointable_name: self.abstract_pytree}
          if with_abstract_pytree
          else None
      )

      with self.subTest('with_context'):
        checkpointables_options = (
            ocp.options.CheckpointablesOptions.create_with_handlers(
                **{checkpointable_name: ocp.handlers.PyTreeHandler}
            )
        )
        with ocp.Context(checkpointables_options=checkpointables_options):
          loaded = ocp.load_checkpointables(step_dir, abstract_checkpointables)
          test_utils.assert_tree_equal(
              self, self.pytree, loaded[checkpointable_name]
          )

      with self.subTest('without_context'):
        loaded = ocp.load_checkpointables(step_dir, abstract_checkpointables)
        test_utils.assert_tree_equal(
            self, self.pytree, loaded[checkpointable_name]
        )

      with self.subTest('error_with_checkpoint_path'):
        with self.assertRaisesRegex(
            ValueError,
            'which are expected to match the keys given by the'
            ' _CHECKPOINT_METADATA file',
        ):
          ocp.load_checkpointables(
              self.ckpt_directory, abstract_checkpointables
          )
      with self.subTest('error_with_root_path'):
        with self.assertRaisesRegex(
            ValueError,
            'which are expected to match the keys given by the'
            ' _CHECKPOINT_METADATA file',
        ):
          ocp.load_checkpointables(
              self.root_directory, abstract_checkpointables
          )
