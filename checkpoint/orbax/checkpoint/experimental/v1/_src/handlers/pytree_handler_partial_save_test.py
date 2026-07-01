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

"""Common test cases for PyTreeHandler."""

# pylint: disable=protected-access, missing-function-docstring

from __future__ import annotations

import asyncio
import contextlib
import copy

from absl import flags
from absl.testing import parameterized
from etils import epath
import jax
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.path.snapshot import snapshot
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint._src.tree import structure_utils as tree_structure_utils
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import pytree_handler
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.partial import path as partial_path_lib
from orbax.checkpoint.experimental.v1._src.partial import saving as partial_saving
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils
from orbax.checkpoint.experimental.v1._src.testing import handler_utils as handler_test_utils
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


FLAGS = flags.FLAGS
PyTree = tree_types.PyTree
Path = path_types.Path
PENDING_DIR_SUFFIX = snapshot.PENDING_DIR_SUFFIX
STATE_CHECKPOINTABLE_KEY = checkpoint_layout.STATE_CHECKPOINTABLE_KEY

create_sharded_pytree = array_test_utils.create_sharded_pytree
as_abstract_type = array_test_utils.as_abstract_type


@contextlib.contextmanager
def handler_with_options(
    *,
    use_ocdbt: bool = True,
):
  """Registers handlers with OCDBT support and resets when done."""
  context = context_lib.Context()
  context.array.saving.use_ocdbt = use_ocdbt

  handler = handler_test_utils.create_test_handler(
      pytree_handler.PyTreeHandler, context=context
  )

  try:
    yield handler
  finally:
    pass


@test_utils.barrier_compatible_test
class PyTreeHandlerPartialSaveTest(
    parameterized.TestCase,
    multiprocess_test.MultiProcessTest,
):
  """Test cases for partial saving with PyTreeCheckpointHandler."""

  def setUp(self):
    super().setUp()

    self.pytree, _ = create_sharded_pytree()

    self.directory = epath.Path(
        self.multiprocess_create_tempdir(name='checkpointing_test')
    )

    # default to use_ocdbt=False, so we can test non-ocdbt handler first
    self.handler = self.enter_context(handler_with_options(use_ocdbt=False))
    test_utils.set_tensorstore_driver_for_test()

    test_utils.sync_global_processes(
        'PyTreeCheckpointHandlerTest:setup_complete'
    )

  def tearDown(self):
    test_utils.sync_global_processes(
        'PyTreeCheckpointHandlerTest:tests_complete'
    )
    super().tearDown()

  @parameterized.product(
      use_ocdbt=(True, False),
      finalize_with_partial_path=(True, False),
  )
  def test_partial_save_and_finalize(
      self, use_ocdbt: bool, finalize_with_partial_path: bool
  ):
    final_path = self.directory / 'save_pytree'
    final_pytree_path = final_path / STATE_CHECKPOINTABLE_KEY
    partial_path = self.directory / partial_path_lib.add_partial_save_suffix(
        epath.Path('save_pytree')
    )

    # self.pytree = {
    #     'a': np.arange(...),
    #     'b': np.arange(...),
    #     'c': {
    #         'a': np.arange(...),
    #         'e': np.arange(...),
    #     },
    # }
    first_save_pytree = copy.deepcopy(self.pytree)
    # Dict elements are the only way to partial save lists.
    first_save_pytree['a_list'] = [
        {
            'a1': copy.deepcopy(self.pytree['a']),
            'b1': copy.deepcopy(self.pytree['b']),
        },
        {'a2': copy.deepcopy(self.pytree['a'])},
    ]
    second_save_pytree = {
        'new_node': copy.deepcopy(self.pytree['a']),
        'c': {'new_nested_node': copy.deepcopy(self.pytree['b'])},
        'a_list': [{}, {'b2': copy.deepcopy(self.pytree['b'])}],
    }
    expected_pytree = tree_structure_utils.merge_trees(
        first_save_pytree, second_save_pytree
    )
    abstract_pytree = jax.tree.map(as_abstract_type, expected_pytree)

    # Force rebuild again.
    partial_saving.save(final_path, first_save_pytree)
    pending_dirs = asyncio.run(snapshot.list_pending_dirs(partial_path))
    self.assertLen(pending_dirs, 1)
    self.assertTrue((pending_dirs[0] / STATE_CHECKPOINTABLE_KEY).exists())

    partial_saving.save(final_path, second_save_pytree)  # pyrefly: ignore[bad-argument-type]
    pending_dirs = asyncio.run(snapshot.list_pending_dirs(partial_path))
    self.assertLen(pending_dirs, 2)
    for d in pending_dirs:
      self.assertTrue((d / STATE_CHECKPOINTABLE_KEY).exists())

    partial_saving.finalize(
        partial_path if finalize_with_partial_path else final_path
    )
    self.assertTrue(final_path.exists())

    with handler_with_options(use_ocdbt=use_ocdbt) as load_handler:
      restored_pytree = load_handler.load(final_pytree_path, abstract_pytree)
      test_utils.assert_tree_equal(self, restored_pytree, expected_pytree)

  def test_partial_save_replacement_raises_error(self):
    final_path = self.directory / 'save_pytree_overwrite'

    first_save_pytree = copy.deepcopy(self.pytree)

    second_save_pytree = {
        'a': jax.tree.map(
            lambda x: x + 100, copy.deepcopy(first_save_pytree['a'])
        ),
        'new_node': copy.deepcopy(first_save_pytree['b']),
    }

    partial_saving.save(final_path, first_save_pytree)
    with self.assertRaisesRegex(
        pytree_handler.PartialSaveReplacementError,
        'Partial saving currently does not support REPLACEMENT.',
    ):
      partial_saving.save(final_path, second_save_pytree)  # pyrefly: ignore[bad-argument-type]

  @parameterized.product(first_save_leaf_is_subtree=(True, False))
  def test_partial_save_subtree_replacement_raises_error(
      self, first_save_leaf_is_subtree: bool
  ):
    final_path = self.directory / 'save_pytree_subtree_overwrite'

    if first_save_leaf_is_subtree:
      first_save_pytree = {'a': {'b': 1}}
      second_save_pytree = {'a': 2}
    else:
      first_save_pytree = {'a': 2}
      second_save_pytree = {'a': {'b': 1}}

    partial_saving.save(final_path, first_save_pytree)  # pyrefly: ignore[bad-argument-type]
    with self.assertRaisesRegex(
        pytree_handler.PartialSaveReplacementError,
        'Partial saving currently does not support REPLACEMENT.',
    ):
      partial_saving.save(final_path, second_save_pytree)  # pyrefly: ignore[bad-argument-type]


if __name__ == '__main__':
  multiprocess_test.main()
