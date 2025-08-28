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

"""Common test cases for PyTreeHandler."""

# pylint: disable=protected-access, missing-function-docstring

from __future__ import annotations

import asyncio
import contextlib
import copy
from typing import Any, Awaitable

from absl import flags
from absl.testing import parameterized
from etils import epath
import jax
from orbax.checkpoint import test_utils
from orbax.checkpoint import utils
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint._src.tree import structure_utils as tree_structure_utils
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.handlers import pytree_handler
from orbax.checkpoint.experimental.v1._src.partial import path as partial_path_lib
from orbax.checkpoint.experimental.v1._src.partial import saving as partial_saving
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.serialization import types as serialization_types
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils
from orbax.checkpoint.experimental.v1._src.testing import path_utils as path_test_utils
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


FLAGS = flags.FLAGS
PyTree = tree_types.PyTree

ARRAY_METADATA_STORE = array_metadata_store_lib.Store()

create_sharded_pytree = array_test_utils.create_sharded_pytree
as_abstract_type = array_test_utils.as_abstract_type


Path = path_types.Path


async def _run_awaitable(awaitable: Awaitable[Any]) -> Any:
  return await awaitable


class PyTreeHandler:
  """Wrapper around PyTreeHandler that can block on save and load."""

  def __init__(self, **kwargs):
    self._handler = pytree_handler.PyTreeHandler(**kwargs)

  def save(self, path: Path, checkpointable: PyTree):
    awaitable = self.save_async(path, checkpointable)
    return asyncio.run(_run_awaitable(awaitable))

  def save_async(self, path: Path, checkpointable: PyTree):
    path = path_test_utils.PathAwaitingCreationWrapper(path)
    return asyncio.run(self._handler.save(path, checkpointable))

  def load(self, path: Path, abstract_checkpointable: PyTree | None = None):
    awaitable = self.load_async(path, abstract_checkpointable)
    return asyncio.run(_run_awaitable(awaitable))

  def load_async(
      self, path: Path, abstract_checkpointable: PyTree | None = None
  ):
    return asyncio.run(self._handler.load(path, abstract_checkpointable))

  def metadata(self, path: Path):
    return asyncio.run(self._handler.metadata(path))


@contextlib.contextmanager
def handler_with_options(
    *,
    create_array_storage_options_fn: (
        options_lib.PyTreeOptions.Saving.CreateArrayStorageOptionsFn | None
    ) = None,
    save_concurrent_bytes: int | None = None,
    restore_concurrent_bytes: int | None = None,
    use_ocdbt: bool = True,
    use_zarr3: bool = False,
    enable_padding_and_truncation: bool = True,
    ocdbt_target_data_file_size: int | None = None,
    enable_pinned_host_transfer: bool | None = None,
    pytree_metadata_options: tree_metadata.PyTreeMetadataOptions = (
        tree_metadata.PYTREE_METADATA_OPTIONS
    ),
    array_metadata_store: array_metadata_store_lib.Store | None = (
        ARRAY_METADATA_STORE
    ),
    enable_write_sharding_file: bool = True,
    partial_load: bool = False,
    leaf_handler_registry: (
        serialization_types.LeafHandlerRegistry | None
    ) = None,
):
  """Registers handlers with OCDBT support and resets when done."""
  context = context_lib.Context(
      array_options=options_lib.ArrayOptions(
          saving=options_lib.ArrayOptions.Saving(
              concurrent_bytes=save_concurrent_bytes,
              use_ocdbt=use_ocdbt,
              use_zarr3=use_zarr3,
              ocdbt_target_data_file_size=ocdbt_target_data_file_size,
              enable_pinned_host_transfer=enable_pinned_host_transfer,
              array_metadata_store=array_metadata_store,
              enable_write_sharding_file=enable_write_sharding_file,
              use_replica_parallel=not utils.is_pathways_backend(),
          ),
          loading=options_lib.ArrayOptions.Loading(
              concurrent_bytes=restore_concurrent_bytes,
              enable_padding_and_truncation=enable_padding_and_truncation,
          ),
      ),
      pytree_options=options_lib.PyTreeOptions(
          saving=options_lib.PyTreeOptions.Saving(
              create_array_storage_options_fn=create_array_storage_options_fn,
              pytree_metadata_options=pytree_metadata_options,
          ),
          loading=options_lib.PyTreeOptions.Loading(
              partial_load=partial_load,
          ),
          leaf_handler_registry=leaf_handler_registry,
      ),
  )

  handler = PyTreeHandler(
      context=context,
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
        self.create_tempdir(name='checkpointing_test').full_path
    )

    # default to use_ocdbt=False, so we can test non-ocdbt handler first
    self.handler = self.enter_context(
        handler_with_options(
            use_ocdbt=False, array_metadata_store=ARRAY_METADATA_STORE
        )
    )
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
    final_pytree_path = final_path / 'pytree_checkpointable'
    partial_path = self.directory / partial_path_lib.add_partial_save_suffix(
        'save_pytree'
    )
    partial_pytree_path = partial_path / 'pytree_checkpointable'

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

    with handler_with_options(use_ocdbt=use_ocdbt) as save_handler:
      save_handler.save(partial_pytree_path, first_save_pytree)
      self.assertTrue(partial_pytree_path.exists())

      save_handler.save(partial_pytree_path, second_save_pytree)
      self.assertTrue(partial_pytree_path.exists())

      restored_pytree = save_handler.load(partial_pytree_path, abstract_pytree)
      test_utils.assert_tree_equal(self, restored_pytree, expected_pytree)

    partial_saving.finalize(
        partial_path if finalize_with_partial_path else final_path
    )
    self.assertTrue(final_path.exists())

    with handler_with_options(use_ocdbt=use_ocdbt) as load_handler:
      restored_pytree = load_handler.load(final_pytree_path, abstract_pytree)
      test_utils.assert_tree_equal(self, restored_pytree, expected_pytree)

  def test_partial_save_replacement_raises_error(self):
    partial_path = (
        partial_path_lib.add_partial_save_suffix(self.directory)
        / 'save_pytree_overwrite'
    )

    first_save_pytree = copy.deepcopy(self.pytree)

    second_save_pytree = {
        'a': jax.tree.map(
            lambda x: x + 100, copy.deepcopy(first_save_pytree['a'])
        ),
        'new_node': copy.deepcopy(first_save_pytree['b']),
    }

    with handler_with_options() as save_handler:
      save_handler.save(partial_path, first_save_pytree)
      with self.assertRaisesRegex(
          ValueError,
          'Partial saving currently does not support REPLACEMENT.',
      ):
        save_handler.save(partial_path, second_save_pytree)

  @parameterized.product(first_save_leaf_is_subtree=(True, False))
  def test_partial_save_subtree_replacement_raises_error(
      self, first_save_leaf_is_subtree: bool
  ):
    partial_path = (
        partial_path_lib.add_partial_save_suffix(self.directory)
        / 'save_pytree_subtree_overwrite'
    )

    if first_save_leaf_is_subtree:
      first_save_pytree = {'a': {'b': 1}}
      second_save_pytree = {'a': 2}
    else:
      first_save_pytree = {'a': 2}
      second_save_pytree = {'a': {'b': 1}}

    with handler_with_options() as save_handler:
      save_handler.save(partial_path, first_save_pytree)
      with self.assertRaisesRegex(
          ValueError, 'Partial saving currently does not support REPLACEMENT.'
      ):
        save_handler.save(partial_path, second_save_pytree)


if __name__ == '__main__':
  multiprocess_test.main()
