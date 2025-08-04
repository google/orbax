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
import dataclasses
import datetime
import functools
import json
import threading
from typing import Any, Awaitable, Iterator, List, Sequence, Type
from unittest import mock

from absl.testing import parameterized
import aiofiles
from etils import epath
import flax
import flax.training.train_state
import jax
from jax import numpy as jnp
from jax.experimental import mesh_utils
import numpy as np
import optax
from orbax.checkpoint import test_utils
from orbax.checkpoint import utils
from orbax.checkpoint._src.arrays import abstract_arrays
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.metadata import array_metadata
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.metadata import empty_values
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.serialization import replica_slices
from orbax.checkpoint._src.serialization import serialization
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.tree import utils as tree_utils
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.handlers import pytree_handler
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.serialization import array_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import numpy_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import registry
from orbax.checkpoint.experimental.v1._src.serialization import scalar_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import types as serialization_types
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils
from orbax.checkpoint.experimental.v1._src.testing import path_utils as path_test_utils
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


PyTree = tree_types.PyTree
ParamInfo = pytree_checkpoint_handler.ParamInfo

_SHARDING = '_sharding'
PYTREE_METADATA_FILE = pytree_checkpoint_handler.PYTREE_METADATA_FILE
ARRAY_METADATA_STORE = array_metadata_store_lib.Store()
PLACEHOLDER = type_handlers.PLACEHOLDER

create_sharded_array = array_test_utils.create_sharded_array
create_numpy_pytree = array_test_utils.create_numpy_pytree
create_sharded_pytree = array_test_utils.create_sharded_pytree
as_abstract_type = array_test_utils.as_abstract_type


PathAwaitingCreation = path_types.PathAwaitingCreation
PathLike = path_types.PathLike
Path = path_types.Path


async def _run_awaitable(awaitable: Awaitable[Any]) -> Any:
  return await awaitable


# Custom dataclasses for testing custom leaf handlers.  PyType check requires
# these defines outside of the test.
@dataclasses.dataclass
class Point:
  x: int
  y: float


@dataclasses.dataclass
class AbstractPoint:
  x: Type[int] = int
  y: Type[float] = float


class PointLeafHandler(serialization_types.LeafHandler[Point, AbstractPoint]):
  """A custom leaf handler for testing."""

  def __init__(self, context: context_lib.Context | None = None):
    del context

  async def serialize(
      self,
      params: Sequence[serialization_types.SerializationParam[Point]],
      serialization_context: serialization_types.SerializationContext,
  ) -> Awaitable[None]:

    async def _background_serialize():
      if multihost.is_primary_host(0):
        # make sure the parent directory is created
        await serialization_context.parent_dir.await_creation()

        for param in params:
          async with aiofiles.open(
              serialization_context.parent_dir.path / f'{param.name}.txt',
              'w',
          ) as f:
            await f.write(json.dumps(dataclasses.asdict(param.value)))

    return _background_serialize()

  async def deserialize(
      self,
      params: Sequence[serialization_types.DeserializationParam[AbstractPoint]],
      deserialization_context: serialization_types.DeserializationContext,
  ) -> Awaitable[Sequence[Point]]:

    async def _background_deserialize():
      ret = []
      for param in params:
        async with aiofiles.open(
            deserialization_context.parent_dir / f'{param.name}.txt',
            'r',
        ) as f:
          ret.append(Point(**json.loads(await f.read())))

      return ret

    return _background_deserialize()

  async def metadata(
      self,
      params: Sequence[serialization_types.DeserializationParam[None]],
      deserialization_context: serialization_types.DeserializationContext,
  ) -> Sequence[AbstractPoint]:
    return [AbstractPoint()] * len(params)


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


def create_mixed_format_pytree(
    *,
    add: int = 0,
    strings: bool = False,
    parent_key: str | None = None,
    include_scalars: bool = True,
) -> PyTree:
  """Creates a PyTree with different leaf types for testing.

  Args:
    add: Adds the specified value to numeric leafs.
    strings: If true, adds string leaves to the tree.
    parent_key: If provided, keys will be contained within a dictionary under
      this key.
    include_scalars: If true, adds scalar leaves to the tree.

  Returns:
    PyTree
  """
  numpy_pytree, abstract_numpy_pytree = create_numpy_pytree(
      add=add, include_scalars=include_scalars
  )
  sharded_pytree, abstract_sharded_pytree = create_sharded_pytree(
      add=add, include_scalars=include_scalars
  )
  if parent_key:
    numpy_pytree = {parent_key: numpy_pytree}
    sharded_pytree = {parent_key: sharded_pytree}
    abstract_numpy_pytree = {parent_key: abstract_numpy_pytree}
    abstract_sharded_pytree = {parent_key: abstract_sharded_pytree}
  mixed_pytree = {
      'numpy': numpy_pytree,
      'sharded': sharded_pytree,
  }
  abstract_mixed_pytree = {
      'numpy': abstract_numpy_pytree,
      'sharded': abstract_sharded_pytree,
  }
  if strings:
    mixed_pytree['foo'] = 'foo_val'
    mixed_pytree['bar'] = 'bar_val'
    abstract_mixed_pytree['foo'] = ''
    abstract_mixed_pytree['bar'] = ''
  return mixed_pytree, abstract_mixed_pytree


def _raise_file_not_found_error(*args, **kwargs):
  del args, kwargs
  raise FileNotFoundError()


# Not in common util because we need to eliminate OSS dependency on flax.
def init_flax_model(model):
  params = model.init(jax.random.PRNGKey(0), jnp.ones([8, 8]))
  tx = optax.adamw(learning_rate=0.001)
  state = flax.training.train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx
  )
  return jax.tree.map(np.asarray, state)


def get_d_files(path: Path) -> list[Path]:
  files = []
  for idx in range(multihost.process_count()):
    d_path = path / f'ocdbt.process_{idx}' / 'd'
    if not d_path.exists():
      continue
    files.extend(list(d_path.iterdir()))
  return files


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


class PyTreeHandlerTestBase:
  """Base test cases for PyTreeCheckpointHandler."""

  class Test(parameterized.TestCase):
    """Test class."""

    def setUp(self):
      super().setUp()

      self.pytree, self.abstract_pytree = create_sharded_pytree()
      self.numpy_pytree, self.abstract_numpy_pytree = create_numpy_pytree()

      self.directory = epath.Path(
          self.create_tempdir(name='checkpointing_test').full_path
      )
      # TODO: b/365169723 - Add tests for support_rich_types=True.
      self.pytree_metadata_options = tree_metadata.PyTreeMetadataOptions(
          support_rich_types=False
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

    def validate_save(
        self,
        path: epath.Path,
        abstract_pytree: PyTree | None,
        expected: PyTree,
        checkpoint_handler: PyTreeHandler,
    ):
      """Validate save was performed correctly."""
      actual = checkpoint_handler.load(path, abstract_pytree)
      test_utils.assert_tree_equal(self, expected, actual)

    def validate_metadata(
        self,
        *,
        expected_reference_metadata_tree: PyTree,
        actual_metadata: PyTree,
        pytree_metadata_options: tree_metadata.PyTreeMetadataOptions,
        array_metadata_store: array_metadata_store_lib.Store | None,
    ):
      """Validate metadata, provided the original tree that was saved."""
      expected_reference_metadata_tree = tree_metadata.serialize_tree(
          expected_reference_metadata_tree, pytree_metadata_options
      )

      def _metadata(value):
        if empty_values.is_supported_empty_value(
            value, pytree_metadata_options
        ):
          return value
        if isinstance(value, np.ndarray):
          return numpy_leaf_handler.NumpyMetadata(
              shape=value.shape,
              dtype=value.dtype,
              storage_metadata=value_metadata.StorageMetadata(
                  chunk_shape=value.shape,
              ),
          )
        if isinstance(value, jax.Array):
          expected_sharding = sharding_metadata.from_jax_sharding(
              value.sharding
          )
          expected_chunk_shape = test_utils.get_expected_chunk_shape(value)
          return array_leaf_handler.ArrayMetadata(
              shape=value.shape,
              sharding_metadata=expected_sharding,
              dtype=value.dtype,
              storage_metadata=value_metadata.StorageMetadata(
                  chunk_shape=expected_chunk_shape,
                  write_shape=(
                      expected_chunk_shape
                      if array_metadata_store is not None
                      else None
                  ),
              ),
          )
        if isinstance(value, (float, int)):
          return np.float64 if isinstance(value, float) else np.int64
        if isinstance(value, str):
          return str
        if isinstance(value, optax.EmptyState):
          return None
        if isinstance(value, Point):
          return AbstractPoint()
        raise ValueError(f'Unrecognized type: {type(value)}.')

      expected_metadata = jax.tree.map(
          _metadata,
          expected_reference_metadata_tree,
          is_leaf=tree_utils.is_empty_or_leaf,
      )
      test_utils.assert_tree_equal(self, expected_metadata, actual_metadata)

    def test_get_param_names(self):
      param_names = pytree_checkpoint_handler.get_param_names(self.pytree)
      expected = {
          'a': 'a',
          'b': 'b',
          'c': {
              'a': 'c.a',
              'e': 'c.e',
          },
          'x': 'x',
          'y': 'y',
      }
      test_utils.assert_tree_equal(self, expected, param_names)

    def test_save_format(self):
      pytree = {'a': 0, 'c': {'d': np.arange(3), 'e': {'f': 5}}, 'g': 10}
      self.handler.save(self.directory, pytree)
      fnames = ['a', 'c.d', 'c.e.f', 'g']
      paths = [self.directory / name for name in fnames]
      for p in paths:
        self.assertTrue(p.exists())
        self.assertTrue((p / '.zarray').exists())

    @parameterized.product(use_ocdbt=(True, False))
    def test_save_sharding(self, use_ocdbt: bool):
      if multihost.is_pathways_backend():
        self.skipTest('Sharding metadata not present on Pathways.')
      with handler_with_options(use_ocdbt=use_ocdbt) as checkpoint_handler:
        pytree = {
            'mlp/~/linear_0': {
                'a': self.pytree['a'],
                'b': self.pytree['b'],
                'c': {'a': self.pytree['c']['a'], 'e': self.pytree['c']['e']},
            }
        }
        abstract_pytree = jax.tree.map(
            array_test_utils.as_abstract_type, pytree
        )
        checkpoint_handler.save(self.directory, pytree)

        self.validate_save(
            self.directory,
            abstract_pytree,
            pytree,
            checkpoint_handler,
        )

      self.assertTrue((self.directory / _SHARDING).exists())
      with open(self.directory / _SHARDING, 'r') as file:
        data = json.load(file)
        self.assertCountEqual(
            data.keys(),
            {
                'bWxwL34vbGluZWFyXzAuYQ==',  # mlp/~/linear_0.a
                'bWxwL34vbGluZWFyXzAuYg==',  # mlp/~/linear_0.b
                'bWxwL34vbGluZWFyXzAuYy5h',  # mlp/~/linear_0.c.a
                'bWxwL34vbGluZWFyXzAuYy5l',  # mlp/~/linear_0.c.e
            },
        )
        # mlp/~/linear_0.a
        self.assertEqual(
            sharding_metadata.NamedShardingMetadata.from_deserialized_dict(
                json.loads(data['bWxwL34vbGluZWFyXzAuYQ=='])
            ),
            sharding_metadata.NamedShardingMetadata.from_jax_sharding(
                pytree['mlp/~/linear_0']['a'].sharding
            ),
        )
        # mlp/~/linear_0.b
        self.assertEqual(
            sharding_metadata.NamedShardingMetadata.from_deserialized_dict(
                json.loads(data['bWxwL34vbGluZWFyXzAuYg=='])
            ),
            sharding_metadata.NamedShardingMetadata.from_jax_sharding(
                pytree['mlp/~/linear_0']['b'].sharding
            ),
        )
        # mlp/~/linear_0.c.a
        self.assertEqual(
            sharding_metadata.NamedShardingMetadata.from_deserialized_dict(
                json.loads(data['bWxwL34vbGluZWFyXzAuYy5h'])
            ),
            sharding_metadata.NamedShardingMetadata.from_jax_sharding(
                pytree['mlp/~/linear_0']['c']['a'].sharding
            ),
        )
        # mlp/~/linear_0.c.e
        self.assertEqual(
            sharding_metadata.NamedShardingMetadata.from_deserialized_dict(
                json.loads(data['bWxwL34vbGluZWFyXzAuYy5l'])
            ),
            sharding_metadata.NamedShardingMetadata.from_jax_sharding(
                pytree['mlp/~/linear_0']['c']['e'].sharding
            ),
        )

    @parameterized.product(
        use_ocdbt=(True, False),
        array_metadata_store=(None, ARRAY_METADATA_STORE),
    )
    def test_disable_write_sharding_file(
        self,
        use_ocdbt: bool,
        array_metadata_store: array_metadata_store_lib.Store | None,
    ):
      pytree, abstract_pytree = create_mixed_format_pytree()
      with handler_with_options(
          use_ocdbt=use_ocdbt,
          array_metadata_store=array_metadata_store,
          enable_write_sharding_file=False,
      ) as checkpoint_handler:
        checkpoint_handler.save(self.directory, pytree)
        self.validate_save(
            self.directory,
            abstract_pytree,
            pytree,
            checkpoint_handler,
        )
      self.assertFalse((self.directory / _SHARDING).exists())

    def test_sharding_variable_devices(self):
      if multihost.is_pathways_backend():
        self.skipTest('Sharding metadata not present on Pathways.')
      mesh_axes = jax.sharding.PartitionSpec(
          'x',
      )
      devices_subset = []
      for idx in range(jax.process_count()):
        for d in jax.devices():
          if d.process_index == idx:
            devices_subset.append(d)
            break
      pytree = {
          'a': test_utils.create_sharded_array(
              np.arange(16),
              jax.sharding.Mesh(devices_subset, ('x',)),
              mesh_axes,
          ),
          'b': test_utils.create_sharded_array(
              np.arange(16), jax.sharding.Mesh(jax.devices(), ('x',)), mesh_axes
          ),
      }

      self.handler.save(self.directory, pytree)
      self.assertTrue((self.directory / _SHARDING).exists())
      a_sharding_metadata = sharding_metadata.NamedShardingMetadata(
          shape=np.array([2]),
          axis_names=['x'],
          partition_spec=('x',),
          device_mesh=sharding_metadata.DeviceMetadataMesh.from_jax_mesh(
              jax.sharding.Mesh(devices_subset, ('x',))
          ),
      )
      b_sharding_metadata = sharding_metadata.NamedShardingMetadata(
          shape=np.array([8]),
          axis_names=['x'],
          partition_spec=('x',),
          device_mesh=sharding_metadata.DeviceMetadataMesh.from_jax_mesh(
              jax.sharding.Mesh(jax.devices(), ('x',))
          ),
      )

      restored_metadata = self.handler.metadata(self.directory)
      self.assertEqual(
          a_sharding_metadata,
          restored_metadata['a'].sharding_metadata,
      )
      self.assertEqual(
          b_sharding_metadata,
          restored_metadata['b'].sharding_metadata,
      )
      self.assertEqual(
          pytree['a'].sharding,
          restored_metadata['a'].sharding,
      )
      self.assertEqual(
          pytree['b'].sharding,
          restored_metadata['b'].sharding,
      )

    @parameterized.product(use_ocdbt=(True, False))
    def test_save_main(self, use_ocdbt: bool):
      with handler_with_options(use_ocdbt=use_ocdbt) as checkpoint_handler:
        checkpoint_handler.save(self.directory, self.pytree)
        self.validate_save(
            self.directory,
            self.abstract_pytree,
            self.pytree,
            checkpoint_handler,
        )
        self.assertEqual(
            type_handlers.is_ocdbt_checkpoint(self.directory), use_ocdbt
        )

    @parameterized.product(use_ocdbt=(True, False))
    def test_save_keys_with_slashes(self, use_ocdbt: bool):
      with handler_with_options(use_ocdbt=use_ocdbt) as checkpoint_handler:
        pytree = {
            'a': np.arange(2),
            'b/c': np.arange(4),
        }
        checkpoint_handler.save(self.directory, pytree)
        self.validate_save(
            self.directory,
            None,
            pytree,
            checkpoint_handler,
        )

    def test_save_non_sharded(self):
      self.handler.save(self.directory, self.numpy_pytree)
      self.validate_save(
          self.directory,
          None,
          self.numpy_pytree,
          self.handler,
      )

    @parameterized.product(
        use_ocdbt=(True, False),
        array_metadata_store=(None, ARRAY_METADATA_STORE),
    )
    def test_save_mixed(
        self,
        use_ocdbt: bool,
        array_metadata_store: array_metadata_store_lib.Store | None,
    ):
      with handler_with_options(
          use_ocdbt=use_ocdbt, array_metadata_store=array_metadata_store
      ) as checkpoint_handler:
        pytree, abstract_pytree = create_mixed_format_pytree(strings=True)
        checkpoint_handler.save(self.directory, pytree)
        self.validate_save(
            self.directory,
            abstract_pytree,
            pytree,
            checkpoint_handler,
        )
        if use_ocdbt:
          expected_files_and_directories = [
              '_strings.json',
              'manifest.ocdbt',
              'ocdbt.process_0',
          ]
        else:
          expected_files_and_directories = [
              '_strings.json',
              'numpy.a',
              'numpy.b',
              'numpy.c.a',
              'numpy.c.e',
          ]
        self.assertContainsSubset(
            expected_files_and_directories,
            [f.name for f in self.directory.iterdir()],
        )
        self.validate_metadata(
            expected_reference_metadata_tree=pytree,
            actual_metadata=checkpoint_handler.metadata(self.directory),
            pytree_metadata_options=self.pytree_metadata_options,
            array_metadata_store=array_metadata_store,
        )

    @parameterized.product(
        use_ocdbt=(True, False),
        array_metadata_store=(None, ARRAY_METADATA_STORE),
    )
    def test_save_strings(
        self,
        use_ocdbt: bool,
        array_metadata_store: array_metadata_store_lib.Store | None,
    ):
      if use_ocdbt and multihost.is_pathways_backend():
        self.skipTest('Pathways + OCDBT not supported.')

      with handler_with_options(
          use_ocdbt=use_ocdbt, array_metadata_store=array_metadata_store
      ) as checkpoint_handler:
        pytree, abstract_pytree = create_mixed_format_pytree(strings=True)

        checkpoint_handler.save(self.directory, pytree)
        self.validate_save(
            self.directory,
            abstract_pytree,
            pytree,
            checkpoint_handler,
        )
        self.validate_metadata(
            expected_reference_metadata_tree=pytree,
            actual_metadata=checkpoint_handler.metadata(self.directory),
            pytree_metadata_options=self.pytree_metadata_options,
            array_metadata_store=array_metadata_store,
        )
        self.assertTrue((self.directory / '_strings.json').exists())
        with open(self.directory / '_strings.json') as file:
          data = json.load(file)
          self.assertCountEqual(
              data.keys(),
              {'foo', 'bar'},
              None,
          )
          self.assertEqual(data['foo'], 'foo_val')
          self.assertEqual(data['bar'], 'bar_val')

    def test_cast(self):
      pytree, abstract_pytree = create_mixed_format_pytree(
          include_scalars=False
      )
      origin_dtype = np.int64
      save_dtype = np.uint32
      restore_dtype = np.float64

      def check_dtype(x, dtype):
        if not utils.is_scalar(x):
          self.assertEqual(x.dtype, dtype)

      def set_dtype(v, dtype):
        if hasattr(v, 'dtype'):
          if isinstance(v, jax.ShapeDtypeStruct):
            v = v.update(dtype=dtype)
          else:
            setattr(v, 'dtype', dtype)
        return v

      with self.subTest('check_origin_dtype'):
        jax.tree.map(functools.partial(check_dtype, dtype=origin_dtype), pytree)
        jax.tree.map(
            functools.partial(check_dtype, dtype=origin_dtype), abstract_pytree
        )

      with handler_with_options(
          use_ocdbt=False,
          create_array_storage_options_fn=lambda k, v: options_lib.ArrayOptions.Saving.StorageOptions(
              dtype=save_dtype
          ),
      ) as checkpoint_handler:
        checkpoint_handler.save(self.directory, pytree)

      with self.subTest('check_restore_dtype'):
        abstract_pytree = jax.tree.map(
            functools.partial(set_dtype, dtype=restore_dtype), abstract_pytree
        )
        restored = self.handler.load(self.directory, abstract_pytree)
        jax.tree.map(
            functools.partial(check_dtype, dtype=restore_dtype), restored
        )

      with self.subTest('check_save_dtype'):
        restored = self.handler.load(self.directory)
        jax.tree.map(functools.partial(check_dtype, dtype=save_dtype), restored)

    @parameterized.product(cast_to=(int, float, 0, 0.0))
    def test_cast_scalar_types(self, cast_to):
      pytree = {'a': 5, 'b': 6.1}
      abstract_pytree = {
          'a': cast_to,
          'b': cast_to,
      }

      self.handler.save(self.directory, pytree)
      restored = self.handler.load(self.directory, abstract_pytree)
      expected_type = cast_to if isinstance(cast_to, type) else type(cast_to)
      self.assertIsInstance(restored['a'], expected_type)
      self.assertIsInstance(restored['b'], expected_type)

    @parameterized.product(
        use_ocdbt=(True, False),
        use_zarr3=(True, False),
        array_metadata_store=(None, ARRAY_METADATA_STORE),
    )
    def test_save_restore(
        self,
        use_ocdbt: bool,
        use_zarr3: bool,
        array_metadata_store: array_metadata_store_lib.Store | None,
    ):
      with handler_with_options(
          use_ocdbt=use_ocdbt,
          use_zarr3=use_zarr3,
          array_metadata_store=array_metadata_store,
      ) as checkpoint_handler:
        checkpoint_handler.save(self.directory, self.pytree)
        restored = checkpoint_handler.load(
            self.directory,
            self.abstract_pytree,
        )
        test_utils.assert_tree_equal(self, self.pytree, restored)
        self.validate_metadata(
            expected_reference_metadata_tree=self.pytree,
            actual_metadata=checkpoint_handler.metadata(self.directory),
            pytree_metadata_options=self.pytree_metadata_options,
            array_metadata_store=array_metadata_store,
        )

    def test_save_async(self):
      # The pytree must be larger so that saving doesn't complete too quickly.
      mesh = jax.sharding.Mesh(jax.devices(), 'x')
      np.random.seed(42)
      pytree = {
          'a': array_test_utils.create_sharded_array(
              np.arange(2**20),
              sharding=jax.sharding.NamedSharding(
                  mesh, jax.sharding.PartitionSpec('x')
              ),
          ),
          'b': array_test_utils.create_sharded_array(
              np.random.uniform(size=2**15),
              sharding=jax.sharding.NamedSharding(
                  mesh, jax.sharding.PartitionSpec(None)
              ),
          ),
      }
      abstract_pytree = jax.tree.map(array_test_utils.as_abstract_type, pytree)

      start_serialize = threading.Event()
      original_serialize = serialization.async_serialize_from_host

      def mock_serialize(*args, **kwargs):
        start_serialize.wait()  # Wait for explicit signal before proceeding.
        return original_serialize(*args, **kwargs)

      def is_save_complete(directory):
        return (directory / 'manifest.ocdbt').exists()

      # Serialization to disk does not start until receiving an explicit signal.
      self.enter_context(
          mock.patch.object(
              serialization, 'async_serialize_from_host', new=mock_serialize
          )
      )

      with handler_with_options() as checkpoint_handler:
        awaitable = checkpoint_handler.save_async(self.directory, pytree)
        initial_d_files = get_d_files(self.directory)
        self.assertFalse(is_save_complete(self.directory))
        start_serialize.set()

        asyncio.run(_run_awaitable(awaitable))
        final_d_files = get_d_files(self.directory)
        self.assertNotEmpty(final_d_files)
        self.assertNotEqual(len(initial_d_files), len(final_d_files))
        self.assertTrue(is_save_complete(self.directory))

        restored = checkpoint_handler.load(
            self.directory,
            abstract_pytree,
        )
        test_utils.assert_tree_equal(self, pytree, restored)

    def test_load_async(self):
      with handler_with_options() as checkpoint_handler:
        checkpoint_handler.save(self.directory, self.pytree)
        load_awaitable = checkpoint_handler.load_async(
            self.directory,
            self.abstract_pytree,
        )
        restored = asyncio.run(_run_awaitable(load_awaitable))
        test_utils.assert_tree_equal(self, self.pytree, restored)

    @parameterized.product(use_ocdbt=(True, False))
    def test_load_reverse_mesh(self, use_ocdbt: bool):
      if use_ocdbt and multihost.is_pathways_backend():
        self.skipTest('Pathways + OCDBT not supported.')
      with handler_with_options(use_ocdbt=use_ocdbt) as checkpoint_handler:
        pytree, abstract_pytree = array_test_utils.create_sharded_pytree(
            reverse_devices=True
        )
        checkpoint_handler.save(self.directory, pytree)
        restored = checkpoint_handler.load(self.directory, abstract_pytree)
        test_utils.assert_tree_equal(self, pytree, restored)

    def test_load_multiple_steps(self):
      for step in [0, 1]:
        directory = self.directory / str(step)
        if multihost.process_index() == 0:
          directory.mkdir()
        test_utils.sync_global_processes(
            'PyTreeCheckpointHandlerTest:test_load_different_mkdir'
        )

        pytree, abstract_pytree = create_mixed_format_pytree(add=step)
        self.handler.save(directory, pytree)

        restored = self.handler.load(directory, abstract_pytree)
        test_utils.assert_tree_equal(self, pytree, restored)

    def test_load_missing_checkpoint(self):
      directory = self.directory / 'nothing'
      with self.assertRaises(FileNotFoundError):
        self.handler.load(directory)

    @parameterized.product(
        use_ocdbt=(True, False),
        array_metadata_store=(None, ARRAY_METADATA_STORE),
    )
    def test_flax_model(
        self,
        use_ocdbt: bool,
        array_metadata_store: array_metadata_store_lib.Store | None,
    ):

      @flax.struct.dataclass
      class Params(flax.struct.PyTreeNode):
        params: PyTree
        opt_state: PyTree

      def make_state_with_optax():
        return Params(
            params=self.numpy_pytree,
            opt_state=(optax.EmptyState(), optax.EmptyState()),
        )

      def make_state_with_nones():
        return Params(
            params=self.numpy_pytree,
            opt_state=(None, None),
        )

      state = make_state_with_optax()

      with handler_with_options(
          use_ocdbt=use_ocdbt, array_metadata_store=array_metadata_store
      ) as checkpoint_handler:
        checkpoint_handler.save(self.directory, state)

        with self.subTest('with_abstract_state'):
          abstract_state = jax.tree.map(
              array_test_utils.as_abstract_type, state
          )
          restored = checkpoint_handler.load(self.directory, abstract_state)
          expected_state = state
          test_utils.assert_tree_equal(self, expected_state, restored)
          self.validate_metadata(
              expected_reference_metadata_tree=expected_state,
              actual_metadata=checkpoint_handler.metadata(self.directory),
              pytree_metadata_options=self.pytree_metadata_options,
              array_metadata_store=array_metadata_store,
          )

        with self.subTest('without_abstract_state'):
          if multihost.is_pathways_backend():
            self.skipTest('Must provide abstract_pytree for Pathways.')
          restored = checkpoint_handler.load(self.directory)
          expected_state = tree_utils.serialize_tree(
              make_state_with_nones(),
              keep_empty_nodes=True,
          )
          test_utils.assert_tree_equal(self, expected_state, restored)
          self.validate_metadata(
              expected_reference_metadata_tree=expected_state,
              actual_metadata=checkpoint_handler.metadata(self.directory),
              pytree_metadata_options=self.pytree_metadata_options,
              array_metadata_store=array_metadata_store,
          )

    @parameterized.product(
        use_ocdbt=(
            True,
            False,
        ),
        data=(
            {},
            {'a': {}, 'b': 3},
            [1, {}, 2],
            None,
            {'a': None, 'b': 3},
            [1, None, 2],
            [],
            [1, [], 2],
            {'a': [], 'b': 3},
        ),
        array_metadata_store=(None, ARRAY_METADATA_STORE),
    )
    def test_empty_data(
        self,
        use_ocdbt: bool,
        data: Any,
        array_metadata_store: array_metadata_store_lib.Store | None,
    ):
      with handler_with_options(
          use_ocdbt=use_ocdbt, array_metadata_store=array_metadata_store
      ) as checkpoint_handler:
        if not data:
          with self.assertRaisesRegex(ValueError, 'Found empty item'):
            checkpoint_handler.save(
                self.directory,
                data,
            )
          return

        checkpoint_handler.save(self.directory, data)
        restored = checkpoint_handler.load(self.directory)
        self.assertEqual(restored, data)

        self.validate_metadata(
            expected_reference_metadata_tree=data,
            actual_metadata=checkpoint_handler.metadata(self.directory),
            pytree_metadata_options=self.pytree_metadata_options,
            array_metadata_store=array_metadata_store,
        )

    @parameterized.product(
        use_ocdbt=(True, False),
        array_metadata_store=(None, ARRAY_METADATA_STORE),
    )
    def test_list(
        self,
        use_ocdbt: bool,
        array_metadata_store: array_metadata_store_lib.Store | None,
    ):
      item = [1, 2, 5, 6]
      with handler_with_options(
          use_ocdbt=use_ocdbt, array_metadata_store=array_metadata_store
      ) as checkpoint_handler:
        checkpoint_handler.save(self.directory, item)
        abstract_item = [0, 0, 0, 0]
        restored = checkpoint_handler.load(self.directory, abstract_item)
        self.assertListEqual(restored, item)
        self.validate_metadata(
            expected_reference_metadata_tree=[0, 0, 0, 0],
            actual_metadata=checkpoint_handler.metadata(self.directory),
            pytree_metadata_options=self.pytree_metadata_options,
            array_metadata_store=array_metadata_store,
        )

        restored = checkpoint_handler.load(self.directory)
        self.assertListEqual(
            restored,
            [
                np.asarray([1]),
                np.asarray([2]),
                np.asarray([5]),
                np.asarray([6]),
            ],
        )

    def test_no_metadata_file(self):
      self.handler.save(self.directory, self.pytree)
      metadata_file = self.directory / PYTREE_METADATA_FILE
      if multihost.process_index() == 0:
        self.assertTrue(metadata_file.exists())
        metadata_file.unlink()
      test_utils.sync_global_processes('delete_metadata_file')
      self.assertFalse(metadata_file.exists())
      with self.assertRaises(FileNotFoundError):
        self.handler.metadata(self.directory)

    @parameterized.parameters((True,), (False,))
    def test_reshape_padding(self, enable_padding_and_truncation: bool):
      mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('x',))
      axes = jax.sharding.PartitionSpec(
          'x',
      )
      dtype = np.float32
      pytree = {
          'x': test_utils.create_sharded_array(
              np.arange(8, dtype=dtype), mesh, axes
          )
      }
      abstract_pytree = {
          'x': jax.ShapeDtypeStruct(
              shape=(16,), dtype=dtype, sharding=pytree['x'].sharding
          )
      }
      with handler_with_options(
          enable_padding_and_truncation=enable_padding_and_truncation
      ) as checkpoint_handler:
        checkpoint_handler.save(self.directory, pytree)
        if enable_padding_and_truncation:
          restored = checkpoint_handler.load(self.directory, abstract_pytree)
          expected = {
              'x': test_utils.create_sharded_array(
                  np.concatenate(
                      (np.arange(8, dtype=dtype), np.zeros(8, dtype=dtype))
                  ),
                  mesh,
                  axes,
              )
          }
          test_utils.assert_tree_equal(self, expected, restored)
        else:
          with self.assertRaises(BaseException):
            checkpoint_handler.load(self.directory, abstract_pytree)

    @parameterized.parameters((True,), (False,))
    def test_reshape_truncate(self, enable_padding_and_truncation: bool):
      mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('x',))
      axes = jax.sharding.PartitionSpec(
          'x',
      )
      dtype = np.float32
      pytree = {
          'x': test_utils.create_sharded_array(
              np.arange(16, dtype=dtype), mesh, axes
          )
      }
      abstract_pytree = {
          'x': jax.ShapeDtypeStruct(
              shape=(8,), dtype=dtype, sharding=pytree['x'].sharding
          )
      }

      with handler_with_options(
          enable_padding_and_truncation=enable_padding_and_truncation
      ) as checkpoint_handler:
        checkpoint_handler.save(self.directory, pytree)
        if enable_padding_and_truncation:
          restored = checkpoint_handler.load(self.directory, abstract_pytree)
          expected = {
              'x': test_utils.create_sharded_array(
                  np.arange(8, dtype=dtype), mesh, axes
              )
          }
          test_utils.assert_tree_equal(self, expected, restored)
        else:
          with self.assertRaises(BaseException):
            checkpoint_handler.load(self.directory, abstract_pytree)

    @parameterized.parameters(
        (jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec(('x', 'y'))),
        (jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec(('y', 'x'))),
        (jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec(('x',))),
        (jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec(('y',))),
        (jax.sharding.PartitionSpec(('x', 'y')), jax.sharding.PartitionSpec()),
        (
            jax.sharding.PartitionSpec(('x', 'y')),
            jax.sharding.PartitionSpec(('x',)),
        ),
        (
            jax.sharding.PartitionSpec(('x', 'y')),
            jax.sharding.PartitionSpec(('y',)),
        ),
        (
            jax.sharding.PartitionSpec(('x', 'y')),
            jax.sharding.PartitionSpec(('y', 'x')),
        ),
        (
            jax.sharding.PartitionSpec(('x',)),
            jax.sharding.PartitionSpec(('y',)),
        ),
    )
    def test_reshard(self, save_spec, restore_spec):
      devices = jax.devices()
      len_devices = len(devices)
      self.assertGreaterEqual(len_devices, 4)

      mesh = jax.sharding.Mesh(
          mesh_utils.create_device_mesh((4, len_devices // 4)), ('x', 'y')
      )
      dtype = np.int32
      pytree = {
          'x': test_utils.create_sharded_array(
              np.arange(len_devices, dtype=dtype), mesh, save_spec
          )
      }
      abstract_pytree = {
          'x': jax.ShapeDtypeStruct(
              shape=(len_devices,),
              dtype=dtype,
              sharding=jax.sharding.NamedSharding(mesh, restore_spec),
          )
      }

      self.handler.save(self.directory, pytree)
      restored = self.handler.load(self.directory, abstract_pytree)
      expected = {
          'x': test_utils.create_sharded_array(
              np.arange(len_devices, dtype=dtype), mesh, restore_spec
          )
      }
      test_utils.assert_tree_equal(self, expected, restored)

    def test_load_non_ocdbt(self):
      with handler_with_options(use_ocdbt=False) as checkpoint_handler:
        checkpoint_handler.save(self.directory, self.pytree)
        self.assertFalse(type_handlers.is_ocdbt_checkpoint(self.directory))
      with handler_with_options(use_ocdbt=True) as checkpoint_handler:
        restored = checkpoint_handler.load(
            self.directory,
            self.abstract_pytree,
        )
        test_utils.assert_tree_equal(self, self.pytree, restored)

    def test_load_non_ocdbt_mixed(self):
      pytree, abstract_pytree = create_mixed_format_pytree(strings=True)
      with handler_with_options(use_ocdbt=False) as checkpoint_handler:
        checkpoint_handler.save(self.directory, pytree)
        self.assertFalse(type_handlers.is_ocdbt_checkpoint(self.directory))
      with handler_with_options(use_ocdbt=True) as checkpoint_handler:
        restored = checkpoint_handler.load(self.directory, abstract_pytree)
        test_utils.assert_tree_equal(self, pytree, restored)

    def test_check_zarray(self):
      self.handler.save(self.directory, self.pytree)
      zarr_path = self.directory / 'a' / '.zarray'
      zarr_path.unlink(missing_ok=True)
      test_utils.sync_global_processes(
          'PyTreeCheckpointHandlerTest:delete_zarray'
      )
      self.assertFalse(zarr_path.exists())
      with self.assertRaises(FileNotFoundError):
        self.handler.load(
            self.directory,
            self.abstract_pytree,
        )

    def test_without_abstract_pytree(self):
      arr = test_utils.create_sharded_array(
          np.arange(8),
          jax.sharding.Mesh(jax.devices(), ('x',)),
          jax.sharding.PartitionSpec('x'),
      )
      pytree = [arr]
      self.handler.save(self.directory, pytree)
      restored = self.handler.load(self.directory)
      test_utils.assert_tree_equal(self, pytree, restored)

    @parameterized.product(use_ocdbt=(True, False))
    def test_masked_shape_dtype_struct(self, use_ocdbt: bool):

      def _should_mask(keypath):
        return keypath[0].key == 'a' or (
            keypath[0].key == 'c' and keypath[1].key == 'e'
        )

      def _mask(keypath, x):
        return optax.MaskedNode() if _should_mask(keypath) else x

      def _none(keypath, x):
        return None if _should_mask(keypath) else x

      masked_tree = jax.tree_util.tree_map_with_path(_mask, self.pytree)
      expected = jax.tree_util.tree_map_with_path(_none, self.pytree)

      with handler_with_options(use_ocdbt=use_ocdbt) as handler:
        handler.save(self.directory, masked_tree)
        if use_ocdbt:
          self.assertTrue(type_handlers.is_ocdbt_checkpoint(self.directory))

        # Restore it with state which was given before applying masking.
        restored = handler.load(
            self.directory,
            jax.tree.map(abstract_arrays.to_shape_dtype_struct, self.pytree),
        )
        test_utils.assert_tree_equal(self, expected, restored)

        # Restore it with state after applying masking to it.
        restored = handler.load(
            self.directory,
            jax.tree.map(abstract_arrays.to_shape_dtype_struct, masked_tree),
        )
        test_utils.assert_tree_equal(self, expected, restored)

        # Restore it without any state.
        restored = handler.load(
            self.directory,
            self.abstract_pytree,
        )
        test_utils.assert_tree_equal(self, expected, restored)

    def test_finalize(self):
      with handler_with_options(use_ocdbt=True) as checkpoint_handler:
        checkpoint_handler.save(self.directory, self.pytree)
        process_index = multihost.process_index()
        process_dir = (
            self.directory / f'{ts_utils.PROCESS_SUBDIR_PREFIX}{process_index}'
        )
        self.assertTrue(process_dir.exists())
        self.assertTrue(process_dir.is_dir())
        self.assertTrue(type_handlers.is_ocdbt_checkpoint(self.directory))

    @parameterized.product(use_ocdbt=(True, False))
    def test_unregistered_types(self, use_ocdbt: bool):
      data = {'uncheckpointable_field': datetime.timedelta(seconds=5)}
      with handler_with_options(use_ocdbt=use_ocdbt) as checkpoint_handler:
        with self.assertRaisesRegex(
            ValueError, 'The following leaf types are not registered'
        ):
          checkpoint_handler.save(
              self.directory,
              data,
          )

    @parameterized.product(
        target_data_file_size=[
            50 * 1024,  # 50KB
            10 * 1024,  # 10KB
            0,
            None,
        ],
        chunk_byte_size=[
            None,  # unspecified
            5 * 1024,  # 5KB
            100 * 1024,  # greater than target_data_file_size
        ],
        use_zarr3=[True, False],
        patch_default_ocdbt_data_file_size=[True, False],
    )
    def test_ocdbt_target_data_file_size(
        self,
        target_data_file_size,
        chunk_byte_size,
        use_zarr3,
        patch_default_ocdbt_data_file_size,
    ):
      """Test ocdbt_target_data_file_size."""
      array_len = 16 * 1024  # ~ 64KB of float data
      custom_pytree = {
          'a': np.arange(array_len, dtype=np.int32),
          'b': np.arange(array_len * 2, dtype=np.float32),
          'c': {
              'a': (
                  np.arange(array_len, dtype=np.int32).reshape(
                      2, array_len // 2
                  )
              ),
              'e': (
                  np.arange(array_len * 2, dtype=np.float32).reshape(
                      2, array_len
                  )
              ),
          },
      }
      shardings = {
          'a': self.abstract_pytree['a'].sharding,
          'b': self.abstract_pytree['b'].sharding,
          'c': {
              'a': self.abstract_pytree['c']['a'].sharding,
              'e': self.abstract_pytree['c']['e'].sharding,
          },
      }
      pytree = jax.tree.map(create_sharded_array, custom_pytree, shardings)
      abstract_pytree = jax.tree.map(as_abstract_type, pytree)

      create_array_storage_options_fn = (
          lambda key, value: options_lib.ArrayOptions.Saving.StorageOptions(
              chunk_byte_size=chunk_byte_size
          )
      )

      new_ocdbt_target_data_file_size = (
          1024
          if patch_default_ocdbt_data_file_size
          else ts_utils._DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
      )
      with mock.patch.object(
          ts_utils,
          '_DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE',
          new_ocdbt_target_data_file_size,
      ):
        if patch_default_ocdbt_data_file_size:
          assert ts_utils._DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE == 1024
        with handler_with_options(
            use_ocdbt=True,
            use_zarr3=use_zarr3,
            ocdbt_target_data_file_size=target_data_file_size,
            create_array_storage_options_fn=create_array_storage_options_fn,
        ) as checkpoint_handler:
          checkpoint_handler.save(self.directory, pytree)

          data_dir = self.directory / 'd'
          self.assertTrue(data_dir.exists())
          self.assertTrue(data_dir.is_dir())

          for f in data_dir.iterdir():
            if f.is_file():
              if target_data_file_size not in (0, None):
                # it's expected the resulting file sizes can be larger than
                # the target_data_file_size, so we give some buffer here
                self.assertLessEqual(
                    f.stat().length,
                    target_data_file_size * 2.0,
                )  # some room
                if patch_default_ocdbt_data_file_size:
                  self.assertLessEqual(
                      f.stat().length,
                      (
                          new_ocdbt_target_data_file_size * 4.0
                      ),  # TODO(niketkb): revisit culprit cl/786790774.
                  )

          restored = checkpoint_handler.load(self.directory, abstract_pytree)

          test_utils.assert_tree_equal(self, pytree, restored)

    def test_local_registry(self):

      if multihost.is_pathways_backend():
        # This does not test anything on the pathways backend
        # TODO(b/333114195): add proper pathways testing.
        return

      class PlusOneHandler(scalar_leaf_handler.ScalarLeafHandler):
        """A custom handler that adds one to all scalar values."""

        def __init__(self, context: context_lib.Context | None = None):
          super().__init__(context=context)

        async def serialize(
            self,
            params: Sequence[scalar_leaf_handler.ScalarSerializationParam],
            serialization_context: serialization_types.SerializationContext,
        ) -> Awaitable[None]:
          updated_params = [
              scalar_leaf_handler.ScalarSerializationParam(
                  keypath=param.keypath, value=param.value + 1
              )
              for param in params
          ]

          return await super().serialize(updated_params, serialization_context)

      leaf_registry = registry.BaseLeafHandlerRegistry()
      leaf_registry.add(int, int, PlusOneHandler)

      with handler_with_options(
          leaf_handler_registry=leaf_registry,
          array_metadata_store=None,
          use_zarr3=True,
      ) as handler:
        # TODO(b/430598877) Return V1 Registry error message.
        with self.assertRaisesRegex(
            ValueError, 'The following leaf types are not registered'
        ):
          handler.save(self.directory, {'a': 3, 'b': 1.0})

        handler.save(self.directory, {'a': 3})

        with self.assertRaisesRegex(
            ValueError, 'The following abstract leaf types are not registered'
        ):
          handler.load(self.directory, {'a': 3.0})

        restored = handler.load(self.directory)
        expected = {'a': 4}
        self.assertEqual(restored, expected)

    def test_empty_custom_node(self):

      class PyTreeDict(dict):
        pass

      jax.tree_util.register_pytree_node(
          PyTreeDict,
          lambda d: (tuple(d.values()), tuple(d.keys())),
          lambda keys, values: PyTreeDict(dict(zip(keys, values))),
      )

      with self.assertRaisesRegex(ValueError, 'Found empty item'):
        self.handler.save(self.directory, PyTreeDict())

      self.handler.save(self.directory, {'a': PyTreeDict()})
      restored = self.handler.load(self.directory)
      self.assertDictEqual({'a': {}}, restored)

      restored = self.handler.load(self.directory, {'a': PyTreeDict()})
      test_utils.assert_tree_equal(self, {'a': PyTreeDict()}, restored)

    @parameterized.parameters((5,), (9,))
    def test_concurrent_gb_save(self, limit_bytes):
      # TODO(b/346811105): Enable for Pathways.
      if multihost.is_pathways_backend():
        self.skipTest(
            'Disabled on Pathways because completion_times cannot updated by'
            ' reference outside remote Python.'
        )
      sleep_time = 1.0
      sharding = jax.sharding.NamedSharding(
          jax.sharding.Mesh(
              jax.devices(),
              ('x',),
          ),
          jax.sharding.PartitionSpec(
              None,
          ),
      )
      # 4 arrays, each has a single chunk, with 4 bytes each.
      tree = jax.tree.map(
          functools.partial(
              array_test_utils.create_sharded_array, sharding=sharding
          ),
          {
              'a': np.arange(1, dtype=np.int32),
              'b': np.arange(1, dtype=np.int32),
              'c': np.arange(1, dtype=np.int32),
              'd': np.arange(1, dtype=np.int32),
          },
      )
      byte_limiter = test_utils.get_byte_limiter(limit_bytes, sleep_time)
      with mock.patch.object(
          serialization,
          'get_byte_limiter',
          new=lambda _: byte_limiter,
      ), handler_with_options(
          save_concurrent_bytes=limit_bytes,
      ) as handler:
        handler.save(self.directory, tree)
      # Replicated shards are handled within the _write_array_shard function.
      # Since shards are only saved once per replica, we only have to check
      # the primary process.
      completion_times = byte_limiter.completion_times
      if multihost.process_index() == 0:
        self.assertLen(completion_times, len(jax.tree.leaves(tree)))
        test_utils.assert_every_n_is_x_apart(
            self,
            completion_times,
            limit_bytes // np.int32().itemsize,
            sleep_time,
        )

    @parameterized.parameters((5,), (9,))
    def test_concurrent_gb_restore(self, limit_bytes):
      # TODO(b/346811105): Enable for Pathways.
      if multihost.is_pathways_backend():
        self.skipTest(
            'Disabled on Pathways because completion_times cannot updated by'
            ' reference outside remote Python.'
        )
      sleep_time = 1.0
      sharding = jax.sharding.NamedSharding(
          jax.sharding.Mesh(
              jax.devices(),
              ('x',),
          ),
          jax.sharding.PartitionSpec(
              None,
          ),
      )
      # 4 arrays, each has a single chunk, with 4 bytes each.
      tree = jax.tree.map(
          functools.partial(
              array_test_utils.create_sharded_array, sharding=sharding
          ),
          {
              'a': np.arange(1, dtype=np.int32),
              'b': np.arange(1, dtype=np.int32),
              'c': np.arange(1, dtype=np.int32),
              'd': np.arange(1, dtype=np.int32),
          },
      )
      self.handler.save(self.directory, tree)

      byte_limiter = test_utils.get_byte_limiter(limit_bytes, sleep_time)
      with mock.patch.object(
          serialization,
          'get_byte_limiter',
          new=lambda _,: byte_limiter,
      ), handler_with_options(restore_concurrent_bytes=limit_bytes) as handler:
        restored = handler.load(self.directory)
      test_utils.assert_tree_equal(self, tree, restored)
      completion_times = byte_limiter.completion_times
      self.assertLen(
          completion_times,
          len(jax.tree.leaves(tree)),
      )
      test_utils.assert_every_n_is_x_apart(
          self,
          completion_times,
          limit_bytes // np.int32().itemsize,
          sleep_time,
      )

    @parameterized.product(enable_pinned_host_transfer=(True, False))
    def test_enable_pinned_host_transfer(self, enable_pinned_host_transfer):
      if multihost.is_pathways_backend():
        self.skipTest(
            'Disabled on Pathways because local variables cannot updated by'
            ' reference outside remote Python.'
        )
      true_count = 0
      false_count = 0

      original_transfer_arrays_to_host = replica_slices.transfer_arrays_to_host

      def _transfer_arrays_to_host(
          arrays,
          replica_id,
          use_replica_parallel,
          min_slice_bytes_for_replica_parallel,
          max_replicas_for_replica_parallel,
          enable_pinned_host_transfer,
      ):
        nonlocal true_count, false_count
        if enable_pinned_host_transfer:
          true_count += 1
        else:
          false_count += 1
        return original_transfer_arrays_to_host(
            arrays,
            replica_id,
            use_replica_parallel=use_replica_parallel,
            min_slice_bytes_for_replica_parallel=min_slice_bytes_for_replica_parallel,
            max_replicas_for_replica_parallel=max_replicas_for_replica_parallel,
            enable_pinned_host_transfer=enable_pinned_host_transfer,
        )

      with mock.patch.object(
          replica_slices,
          'transfer_arrays_to_host',
          new=_transfer_arrays_to_host,
      ), handler_with_options(
          enable_pinned_host_transfer=enable_pinned_host_transfer,
      ) as handler:
        handler.save(self.directory, self.pytree)

      if enable_pinned_host_transfer:
        self.assertGreater(true_count, 0)
        self.assertEqual(false_count, 0)
      else:
        self.assertEqual(true_count, 0)
        self.assertGreater(false_count, 0)

    @parameterized.product(
        use_ocdbt=(True, False),
        pytree_metadata_options=(
            tree_metadata.PyTreeMetadataOptions(support_rich_types=False),
            tree_metadata.PyTreeMetadataOptions(support_rich_types=True),
        ),
    )
    def test_write_shape_metadata_missing_for_all_types_other_than_jax_array(
        self,
        use_ocdbt: bool,
        pytree_metadata_options: tree_metadata.PyTreeMetadataOptions,
    ):
      checkpoint = {
          'a': 1,
          'b': np.array([2]),
          'c': 'hello',
      }
      expected_metadata = {
          'a': np.int64,
          'b': numpy_leaf_handler.NumpyMetadata(
              shape=(1,),
              dtype=checkpoint['b'].dtype,
              storage_metadata=value_metadata.StorageMetadata(
                  chunk_shape=(1,), write_shape=None
              ),
          ),
          'c': str,
      }
      with handler_with_options(
          use_ocdbt=use_ocdbt,
          pytree_metadata_options=pytree_metadata_options,
          array_metadata_store=ARRAY_METADATA_STORE,
      ) as checkpoint_handler:
        checkpoint_handler.save(self.directory, checkpoint)

        self.assertFalse((self.directory / 'array_metadatas').exists())
        restored_metadata = checkpoint_handler.metadata(self.directory)
        self.assertEqual(
            expected_metadata,
            restored_metadata,
        )

    @parameterized.product(
        use_ocdbt=(True, False),
        pytree_metadata_options=(
            tree_metadata.PyTreeMetadataOptions(support_rich_types=False),
            tree_metadata.PyTreeMetadataOptions(support_rich_types=True),
        ),
    )
    def test_write_shape_in_metadata_disabled(
        self,
        use_ocdbt: bool,
        pytree_metadata_options: tree_metadata.PyTreeMetadataOptions,
    ):
      with handler_with_options(
          use_ocdbt=use_ocdbt,
          pytree_metadata_options=pytree_metadata_options,
          array_metadata_store=None,
      ) as checkpoint_handler:
        checkpoint_handler.save(self.directory, self.pytree)
        expected_tree_with_write_shapes = {
            'a': {'write_shape': None},
            'b': {'write_shape': None},
            'c': {
                'a': {'write_shape': None},
                'e': {'write_shape': None},
            },
            'x': {'write_shape': None},
            'y': {'write_shape': None},
        }
        metadata = checkpoint_handler.metadata(self.directory)
        tree_with_write_shapes = jax.tree.map(
            lambda m: {'write_shape': m.storage_metadata.write_shape}, metadata
        )
        self.assertDictEqual(
            expected_tree_with_write_shapes, tree_with_write_shapes
        )

    # TODO(b/382230550): Add test for chunk_shape != write_shape.
    @parameterized.product(
        use_ocdbt=(True, False),
        pytree_metadata_options=(
            tree_metadata.PyTreeMetadataOptions(support_rich_types=False),
            tree_metadata.PyTreeMetadataOptions(support_rich_types=True),
        ),
    )
    def test_write_shape_in_metadata(
        self,
        use_ocdbt: bool,
        pytree_metadata_options: tree_metadata.PyTreeMetadataOptions,
    ):
      with handler_with_options(
          use_ocdbt=use_ocdbt, pytree_metadata_options=pytree_metadata_options
      ) as checkpoint_handler:
        checkpoint_handler.save(self.directory, self.pytree)

        expected_tree_with_write_shapes = {
            'a': {
                'write_shape': test_utils.get_expected_chunk_shape(
                    self.pytree['a']
                )
            },
            'b': {'write_shape': (2,)},
            'c': {
                'a': {'write_shape': (1, 1)},
                'e': {'write_shape': (2, 1)},
            },
            'x': {'write_shape': ()},
            'y': {'write_shape': ()},
        }
        metadata = checkpoint_handler.metadata(self.directory)
        tree_with_write_shapes = jax.tree.map(
            lambda m: {'write_shape': m.storage_metadata.write_shape}, metadata
        )
        self.assertDictEqual(
            expected_tree_with_write_shapes, tree_with_write_shapes
        )

    @parameterized.product(use_ocdbt=(True, False))
    def test_array_metadata_disabled(self, use_ocdbt: bool):
      with handler_with_options(
          use_ocdbt=use_ocdbt, array_metadata_store=None
      ) as checkpoint_handler:
        pytree, abstract_pytree = create_mixed_format_pytree()

        checkpoint_handler.save(self.directory, pytree)

        self.validate_save(
            self.directory,
            abstract_pytree,
            pytree,
            checkpoint_handler,
        )

      self.assertFalse((self.directory / 'array_metadatas').exists())

    @parameterized.product(use_ocdbt=(True, False))
    def test_array_metadata(self, use_ocdbt: bool):
      with handler_with_options(use_ocdbt=use_ocdbt) as checkpoint_handler:

        checkpoint_handler.save(self.directory, self.pytree)

        self.validate_save(
            self.directory,
            self.abstract_pytree,
            self.pytree,
            checkpoint_handler,
        )

      self.assertTrue((self.directory / 'array_metadatas').exists())
      if multihost.is_primary_host(0):
        array_metadatas = asyncio.run(ARRAY_METADATA_STORE.read(self.directory))
        self.assertIsInstance(array_metadatas, dict)
        per_process_metadatas = [
            array_metadata.SerializedArrayMetadata(
                param_name='a',
                write_shape=test_utils.get_expected_chunk_shape(
                    self.pytree['a']
                ),
                chunk_shape=test_utils.get_expected_chunk_shape(
                    self.pytree['a']
                ),
            ),
            array_metadata.SerializedArrayMetadata(
                param_name='b',
                write_shape=(2,),
                chunk_shape=(2,),
            ),
            array_metadata.SerializedArrayMetadata(
                param_name='c.a',
                write_shape=(1, 1),
                chunk_shape=(1, 1),
            ),
            array_metadata.SerializedArrayMetadata(
                param_name='c.e',
                write_shape=(2, 1),
                chunk_shape=(2, 1),
            ),
            array_metadata.SerializedArrayMetadata(
                param_name='x',
                write_shape=(),
                chunk_shape=(),
            ),
            array_metadata.SerializedArrayMetadata(
                param_name='y',
                write_shape=(),
                chunk_shape=(),
            ),
        ]
        processes = range(multihost.process_count())
        expected_array_metadatas = {
            idx: per_process_metadatas for idx in processes
        }
        self.assertSameElements(
            expected_array_metadatas.keys(), array_metadatas.keys()
        )
        for process_index in expected_array_metadatas:
          self.assertEqual(  # pylint: disable=g-generic-assert
              sorted(
                  expected_array_metadatas[process_index],
                  key=lambda x: x.param_name,
              ),
              sorted(
                  array_metadatas[process_index], key=lambda x: x.param_name
              ),
          )

    @parameterized.product(use_ocdbt=(True, False))
    def test_save_with_missing_array_metadata_file(self, use_ocdbt: bool):
      if multihost.process_index() != 0:  # only test on primary host
        self.skipTest('Test only for primary host to avoid barrier timeout.')

      class PathResolverReturningNoMetadataFiles(
          array_metadata_store_lib.PathResolver
      ):

        def get_read_file_paths(
            self, checkpoint_dir: epath.Path, process_index: int | None = None
        ) -> Iterator[epath.Path] | epath.Path | None:
          return None

      with handler_with_options(
          use_ocdbt=use_ocdbt,
          array_metadata_store=array_metadata_store_lib.Store(
              path_resolver=PathResolverReturningNoMetadataFiles()
          ),
      ) as checkpoint_handler:
        with self.assertRaisesRegex(
            ValueError, 'No ArrayMetadata found for process_index'
        ):
          checkpoint_handler.save(self.directory, self.pytree)

    @parameterized.product(use_ocdbt=(True, False))
    def test_save_with_missing_array_metadata_for_params(self, use_ocdbt: bool):
      if multihost.process_index() != 0:  # only test on primary host
        self.skipTest('Test only for primary host to avoid barrier timeout.')

      class MissingArrayMetadataSerDeserializer(
          array_metadata_store_lib.SerDeserializer
      ):

        def deserialize(
            self, serialized: str
        ) -> List[array_metadata.SerializedArrayMetadata]:
          true_data = super().deserialize(serialized)
          return [true_data.pop(0)]  # Delete the rest and return partial data.

      with handler_with_options(
          use_ocdbt=use_ocdbt,
          array_metadata_store=array_metadata_store_lib.Store(
              ser_deser=MissingArrayMetadataSerDeserializer()
          ),
      ) as checkpoint_handler:
        with self.assertRaisesRegex(
            ValueError, 'No ArrayMetadata found for param_info'
        ):
          checkpoint_handler.save(self.directory, self.pytree)

    @parameterized.parameters((True,), (False,))
    def test_zero_size_array(self, use_jax_array: bool):
      arr = np.ones(shape=(0,))
      mesh = jax.sharding.Mesh(np.array(jax.devices()), axis_names=('x',))
      pspec = jax.sharding.PartitionSpec()
      if use_jax_array:
        arr = test_utils.create_sharded_array(arr, mesh, pspec)
      pytree = [arr]
      with self.assertRaisesRegex(ValueError, 'zero size'):
        self.handler.save(self.directory, pytree)

    @parameterized.product(use_ocdbt=(True, False))
    def test_save_restore_random_keys(self, use_ocdbt: bool):
      """Test saving and restoring random keys within a pytree."""

      # TODO(b/393160483) investigate Pathways remote Python support for
      # random.keys.
      if multihost.is_pathways_backend():
        self.skipTest(
            'Disabled on Pathways because random keys are not supported by'
            ' remote Python.'
        )

      mesh = jax.sharding.Mesh(jax.devices(), ('x',))
      sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

      pytree = {
          'keys': {
              'kone': jax.random.key(jnp.array(0, device=sharding)),
              'impl_key': {
                  'rbg': jax.random.key(
                      jnp.array(1, device=sharding), impl='rbg'
                  ),
                  'unsafe_rbg': jax.random.key(
                      jnp.array(2, device=sharding), impl='unsafe_rbg'
                  ),
              },
              'split_keys': jax.random.split(
                  jax.random.key(jnp.array(123, device=sharding)), num=10
              ),
          },
          'arrays': self.pytree,
      }

      with handler_with_options(
          use_ocdbt=use_ocdbt,
      ) as save_handler:
        save_handler.save(self.directory, pytree)

      with handler_with_options(
          use_ocdbt=use_ocdbt,
      ) as load_handler:
        restored = load_handler.load(self.directory)
        test_utils.assert_tree_equal(self, pytree, restored)

    def test_pinned_host_loading(self):
      if multihost.is_pathways_backend():
        # TODO(b/404915487): Reenable when possible.
        self.skipTest('Disabled due to b/404915487.')

      mesh = jax.sharding.Mesh(
          np.asarray(jax.devices()).reshape((1, len(jax.devices()))), ('x', 'y')
      )
      sharding = jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec('x', 'y')
      ).with_memory_kind('pinned_host')

      pytree = dict(arr=jnp.ones((1024, 512), device=sharding))
      self.handler.save(self.directory, pytree)

      abstract_pytree = dict(
          arr=jax.ShapeDtypeStruct(
              pytree['arr'].shape, pytree['arr'].dtype, sharding=sharding
          )
      )
      restored = self.handler.load(self.directory, abstract_pytree)
      expected = dict(arr=jax.device_put(np.ones((1024, 512)), sharding))
      test_utils.assert_tree_equal(self, expected, restored)

    @parameterized.product(
        use_ocdbt=(True, False),
        reference_item=(
            {
                'a': 0,
                'b': 0,
                'c': {
                    'e': 0,
                },
            },
            {
                'a': 0,
                'c': {
                    'a': 0,
                    'e': 0,
                },
            },
            {
                'a': 0,
                'b': 0,
            },
        ),
    )
    def test_restore_item_has_missing_leaves(
        self, use_ocdbt: bool, reference_item: dict[str, Any]
    ):
      with handler_with_options(
          use_ocdbt=use_ocdbt,
      ) as handler:
        handler.save(self.directory, self.pytree)

        with self.assertRaisesRegex(
            ValueError, 'User-provided restore item and on-disk value'
        ):
          handler.load(self.directory, reference_item)

    def test_partial_restore_with_placeholder_simple(self):
      original_item = {
          'a': np.arange(8),
          'b': np.arange(8),
          'c': {
              'a': np.arange(8),
              'e': np.arange(8),
          },
      }
      reference_item = jax.tree.map(as_abstract_type, original_item)
      reference_item['b'] = PLACEHOLDER
      reference_item['c']['e'] = PLACEHOLDER
      expected = {
          'a': original_item['a'],
          'b': PLACEHOLDER,
          'c': {
              'a': original_item['c']['a'],
              'e': PLACEHOLDER,
          },
      }

      simple_dir = epath.Path(
          self.create_tempdir(name='simple_placeholder_dir').full_path
      )

      handler = PyTreeHandler()
      handler.save(simple_dir, original_item)
      restored = handler.load(simple_dir, reference_item)
      test_utils.assert_tree_equal(self, expected, restored)

    @parameterized.product(use_ocdbt=(True, False))
    def test_partial_restore_with_placeholder(self, use_ocdbt: bool):
      """Test saving and restoring placeholder."""
      with handler_with_options(
          use_ocdbt=use_ocdbt,
      ) as save_handler:
        save_handler.save(self.directory, self.pytree)

      with self.subTest('success'):
        reference_item = self.abstract_pytree.copy()
        reference_item['b'] = PLACEHOLDER
        reference_item['c']['e'] = PLACEHOLDER

        expected = self.pytree.copy()
        expected['b'] = PLACEHOLDER
        expected['c']['e'] = PLACEHOLDER

        with handler_with_options(
            use_ocdbt=use_ocdbt,
        ) as restore_handler:
          restored = restore_handler.load(self.directory, reference_item)
          test_utils.assert_tree_equal(self, expected, restored)

      with self.subTest('missing_leaf'):
        reference_item = self.abstract_pytree.copy()
        reference_item['b'] = PLACEHOLDER
        reference_item['c']['e'] = PLACEHOLDER
        del reference_item['c']['a']

        with handler_with_options(
            use_ocdbt=use_ocdbt,
        ) as restore_handler:
          with self.assertRaisesRegex(
              ValueError, 'User-provided restore item and on-disk value'
          ):
            restore_handler.load(self.directory, reference_item)

      with self.subTest('non_leaf_placeholder'):
        reference_item = self.abstract_pytree.copy()
        reference_item['c'] = PLACEHOLDER

        with handler_with_options(
            use_ocdbt=use_ocdbt,
        ) as restore_handler:
          with self.assertRaisesRegex(
              ValueError, 'User-provided restore item and on-disk value'
          ):
            restore_handler.load(self.directory, reference_item)

    @parameterized.product(use_ocdbt=(True, False))
    def test_partial_restore_with_omission(self, use_ocdbt: bool):
      """Basic save and restore test."""
      directory = self.directory / 'partial_restore'

      with handler_with_options(
          use_ocdbt=use_ocdbt,
      ) as save_handler:
        save_handler.save(directory, self.pytree)

      with self.subTest('success'):
        with handler_with_options(
            use_ocdbt=use_ocdbt,
            partial_load=True,
        ) as restore_handler:
          # Create a new pytree structure with the same leaves.
          # Leaves (ShapeDtypeStruct) are immutable and can be shared.
          reference_item = jax.tree.map(lambda x: x, self.abstract_pytree)
          # Omit 'b', 'c.e', and 'x' from the reference item.
          del reference_item['b']
          del reference_item['c']['e']
          del reference_item['x']
          expected = {
              'a': self.pytree['a'],
              'c': {
                  'a': self.pytree['c']['a'],
              },
              'y': self.pytree['y'],
          }
          restored = restore_handler.load(directory, reference_item)
          test_utils.assert_tree_equal(self, expected, restored)

      with self.subTest('extra_leaf'):
        with handler_with_options(
            use_ocdbt=use_ocdbt,
            partial_load=True,
        ) as restore_handler:
          # Create a new pytree structure with the same leaves.
          # Leaves (ShapeDtypeStruct) are immutable and can be shared.
          reference_item = jax.tree.map(lambda x: x, self.abstract_pytree)
          del reference_item['b']
          del reference_item['c']['e']
          del reference_item['x']
          # Add an extra leaf to the reference item.
          reference_item['z'] = jax.ShapeDtypeStruct([0], np.int64)
          with self.assertRaisesRegex(
              ValueError,
              r"Missing 1 keys in structure path \(\), including: \['z'\]",
          ):
            restore_handler.load(directory, reference_item)

    @parameterized.product(use_zarr3=(True, False), use_ocdbt=(True, False))
    def test_custom_leaf_handler(self, use_zarr3: bool, use_ocdbt: bool):

      pytree = {
          'point1': Point(1, 2),
          'point2': Point(3, 4),
          'nested': {
              'point3': Point(5, 6),
              'point4': Point(7, 8),
          },
          'string_leaf': 'string_leaf',
          'number': 123,
          'pytree': self.pytree,
      }

      array_metadata_store = ARRAY_METADATA_STORE

      leaf_handler_registry = registry.StandardLeafHandlerRegistry()
      leaf_handler_registry.add(Point, AbstractPoint, PointLeafHandler)

      def _as_abstract_type(x):
        if isinstance(x, Point):
          return AbstractPoint
        return as_abstract_type(x)

      with handler_with_options(
          use_ocdbt=use_ocdbt,
          leaf_handler_registry=leaf_handler_registry,
          array_metadata_store=array_metadata_store,
          use_zarr3=use_zarr3,
      ) as checkpoint_handler:
        checkpoint_handler.save(self.directory, pytree)
        abstract_pytree = jax.tree.map(_as_abstract_type, pytree)
        restored = checkpoint_handler.load(self.directory, abstract_pytree)

        test_utils.assert_tree_equal(self, pytree, restored)

        self.validate_metadata(
            expected_reference_metadata_tree=pytree,
            actual_metadata=checkpoint_handler.metadata(self.directory),
            pytree_metadata_options=self.pytree_metadata_options,
            array_metadata_store=array_metadata_store,
        )
