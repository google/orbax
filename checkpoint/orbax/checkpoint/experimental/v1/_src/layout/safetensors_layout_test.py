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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint._src.serialization import limits
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import safetensors_layout
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types
from orbax.checkpoint.experimental.v1._src.saving import saving
import safetensors.numpy

SafetensorsLayout = safetensors_layout.SafetensorsLayout
np_save_file = safetensors.numpy.save_file
InvalidLayoutError = checkpoint_layout.InvalidLayoutError


class SafetensorsLayoutTest(
    unittest.IsolatedAsyncioTestCase, parameterized.TestCase
):

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir()
    self.orbax_path = epath.Path(self.test_dir.full_path) / 'test_checkpoint'
    self.safetensors_path = (
        epath.Path(self.test_dir.full_path) / 'test_checkpoint.safetensors'
    )

    # Create a mock SafeTensors and Orbax checkpoint
    self.object_to_save = {
        'a': np.array(3 * [1, 2, 3], dtype=np.int32),
        'b': np.array([0, 1, 0.2], dtype=np.float32),
    }
    self.custom_metadata = {'framework': 'JAX', 'version': '1.0'}
    np_save_file(
        self.object_to_save,
        self.safetensors_path,
        metadata=self.custom_metadata,
    )
    saving.save(
        self.orbax_path,
        self.object_to_save,  # pyrefly: ignore[bad-argument-type]
    )

  async def test_valid_safetensors_checkpoint(self):
    layout = SafetensorsLayout()
    await layout.validate_checkpointables(self.safetensors_path)

  async def test_invalid_safetensors_checkpoint_orbax(self):
    layout = SafetensorsLayout()
    with self.assertRaises(InvalidLayoutError):
      await layout.validate_checkpointables(self.orbax_path / '0')

  async def test_validate_fails_wrong_suffix(self):
    wrong_suffix_path = (
        epath.Path(self.test_dir.full_path) / 'test_checkpoint.txt'
    )
    layout = SafetensorsLayout()
    with self.assertRaises(InvalidLayoutError):
      await layout.validate_checkpointables(wrong_suffix_path)

  @parameterized.product(
      dtype=[
          np.int8,
          np.int32,
          np.int64,
          np.float16,
          np.float32,
          np.float64,
          np.bool_,
          jax.numpy.bfloat16,
          jax.numpy.float8_e4m3fn,
      ]
  )
  async def test_load_safetensors_checkpoint(self, dtype: np.dtype):
    """Tests loading a SafeTensors checkpoint with various dtypes."""
    test_path = (
        epath.Path(self.test_dir.full_path)
        / f'test_{dtype.__name__}.safetensors'  # pyrefly: ignore
    )
    if dtype == np.bool_:
      arr = np.array([True, False, True, False])
    else:
      arr = np.arange(8, dtype=dtype)

    obj_to_save = {'x': arr, 'y': np.array([1, 2, 3], dtype=np.int32)}
    np_save_file(obj_to_save, test_path)

    layout = SafetensorsLayout()
    restore_fn = await layout.load(test_path)
    pytree = await restore_fn

    if np.issubdtype(dtype, np.floating):
      np.testing.assert_allclose(pytree['x'], obj_to_save['x'], strict=True)
    else:
      np.testing.assert_array_equal(pytree['x'], obj_to_save['x'], strict=True)
    np.testing.assert_array_equal(pytree['y'], obj_to_save['y'])

  async def test_load_truncated_file_raises(self):
    # A partially-copied checkpoint: drop the last data bytes so the file is
    # shorter than its header declares. The length check runs in load()'s setup
    # phase, so load() itself raises before any byte-range read is issued.
    full = self.safetensors_path.read_bytes()
    self.safetensors_path.write_bytes(full[:-8])
    layout = SafetensorsLayout()
    with self.assertRaisesRegex(ValueError, 'truncated'):
      await layout.load(self.safetensors_path)

  async def test_load_fails_with_incomplete_dtypes(self):
    incomplete_dtypes = {
        'F32': np.float32,
        'BOOL': np.bool_,
        # Intentionally missing I32: int32 for testing, which is used in the
        # test checkpoint.
    }
    layout = SafetensorsLayout()
    with self.assertRaises(ValueError):
      with mock.patch.object(
          safetensors_layout,
          '_get_dtypes',
          return_value=incomplete_dtypes,
          spec=True,
      ):
        awaitable_fn = await layout.load(self.safetensors_path)
        _ = await awaitable_fn

  async def test_metadata(self):
    layout = SafetensorsLayout()
    metadata = await layout.metadata(self.safetensors_path, None)
    self.assertIsInstance(metadata, metadata_types.CheckpointMetadata)
    self.assertEqual(
        metadata.metadata,
        {
            'b': jax.ShapeDtypeStruct(shape=(3,), dtype=np.float32),
            'a': jax.ShapeDtypeStruct(shape=(9,), dtype=np.int32),
        },
    )
    self.assertEqual(metadata.custom_metadata, self.custom_metadata)
    self.assertIsInstance(metadata.commit_timestamp_nsecs, int)
    self.assertGreater(metadata.commit_timestamp_nsecs, 0)

  async def test_checkpointables_metadata_raises(self):
    layout = SafetensorsLayout()
    with self.assertRaisesRegex(
        NotImplementedError,
        'SafetensorsLayout does not support `.checkpointables_metadata`. Use'
        ' `.metadata` instead.',
    ):
      await layout.checkpointables_metadata(self.safetensors_path)

  async def test_load_checkpointables_raises(self):
    layout = SafetensorsLayout()
    with self.assertRaisesRegex(
        NotImplementedError,
        'SafetensorsLayout does not support `.load_checkpointables`. Use'
        ' `.load` instead.',
    ):
      await layout.load_checkpointables(self.safetensors_path)

  async def test_save_raises_not_implemented(self):
    layout = SafetensorsLayout()
    mock_path = mock.Mock(spec=types.PathAwaitingCreation)
    with self.assertRaises(NotImplementedError):
      await layout.save_checkpointables(mock_path, checkpointables={})


class SafetensorsLayoutDirectoryTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir()
    self.checkpoint_dir = epath.Path(self.test_dir.full_path) / 'checkpoint_dir'
    self.checkpoint_dir.mkdir()

    self.file1 = self.checkpoint_dir / 'part1.safetensors'
    self.file2 = self.checkpoint_dir / 'part2.safetensors'

    self.data1 = {
        'a': np.array([1, 2], dtype=np.int32),
        'c': np.array([5, 6], dtype=np.int32),
    }
    self.data2 = {
        'b': np.array([3.0, 4.0], dtype=np.float32),
        'd': np.array([7.0, 8.0], dtype=np.float32),
    }

    np_save_file(self.data1, self.file1)
    np_save_file(self.data2, self.file2)

  async def test_validate_directory(self):
    layout = SafetensorsLayout()
    await layout.validate_checkpointables(self.checkpoint_dir)

  async def test_validate_directory_fails_empty(self):
    empty_dir = epath.Path(self.test_dir.full_path) / 'empty'
    empty_dir.mkdir()
    layout = SafetensorsLayout()
    with self.assertRaisesRegex(InvalidLayoutError, 'does not contain any'):
      await layout.validate_checkpointables(empty_dir)

  async def test_load_directory(self):
    layout = SafetensorsLayout()
    restore_fn = await layout.load(self.checkpoint_dir)
    pytree = await restore_fn
    np.testing.assert_array_equal(pytree['a'], self.data1['a'])
    np.testing.assert_array_equal(pytree['b'], self.data2['b'])
    np.testing.assert_array_equal(pytree['c'], self.data1['c'])
    np.testing.assert_array_equal(pytree['d'], self.data2['d'])

  async def test_checkpointables_metadata_directory(self):
    layout = SafetensorsLayout()
    metadata = await layout.metadata(self.checkpoint_dir, None)
    state_meta = metadata.metadata
    self.assertIn('a', state_meta)
    self.assertIn('b', state_meta)
    self.assertIn('c', state_meta)
    self.assertIn('d', state_meta)
    self.assertEqual(state_meta['a'].shape, (2,))
    self.assertEqual(state_meta['a'].dtype, np.int32)
    self.assertEqual(state_meta['b'].shape, (2,))
    self.assertEqual(state_meta['b'].dtype, np.float32)
    self.assertEqual(state_meta['c'].shape, (2,))
    self.assertEqual(state_meta['c'].dtype, np.int32)
    self.assertEqual(state_meta['d'].shape, (2,))
    self.assertEqual(state_meta['d'].dtype, np.float32)

  async def test_metadata_directory(self):
    layout = SafetensorsLayout()
    metadata = await layout.metadata(self.checkpoint_dir, None)
    state_meta = metadata.metadata
    self.assertIn('a', state_meta)
    self.assertIn('b', state_meta)
    self.assertIn('c', state_meta)
    self.assertIn('d', state_meta)
    self.assertEqual(state_meta['a'].shape, (2,))
    self.assertEqual(state_meta['a'].dtype, np.int32)
    self.assertEqual(state_meta['b'].shape, (2,))
    self.assertEqual(state_meta['b'].dtype, np.float32)
    self.assertEqual(state_meta['c'].shape, (2,))
    self.assertEqual(state_meta['c'].dtype, np.int32)
    self.assertEqual(state_meta['d'].shape, (2,))
    self.assertEqual(state_meta['d'].dtype, np.float32)

  async def test_load_directory_abstract_tree_all_keys(self):
    layout = SafetensorsLayout()
    tree = {
        'a': jax.ShapeDtypeStruct(shape=(2,), dtype=np.int32),
        'b': jax.ShapeDtypeStruct(shape=(2,), dtype=np.float32),
        'c': jax.ShapeDtypeStruct(shape=(2,), dtype=np.int32),
        'd': jax.ShapeDtypeStruct(shape=(2,), dtype=np.float32),
    }
    restore_fn = await layout.load(self.checkpoint_dir, abstract_state=tree)
    pytree = await restore_fn
    self.assertLen(pytree, 4)
    np.testing.assert_array_equal(pytree['a'], self.data1['a'])
    np.testing.assert_array_equal(pytree['b'], self.data2['b'])
    np.testing.assert_array_equal(pytree['c'], self.data1['c'])
    np.testing.assert_array_equal(pytree['d'], self.data2['d'])

  async def test_load_directory_abstract_tree_sharding(self):
    layout = SafetensorsLayout()
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.array(jax.devices()), ('x',)),
        jax.sharding.PartitionSpec(
            None,
        ),
    )
    tree = {
        'a': jax.ShapeDtypeStruct(
            shape=(2,), dtype=np.int32, sharding=sharding
        ),
        'c': jax.ShapeDtypeStruct(
            shape=(2,), dtype=np.int32, sharding=sharding
        ),
    }
    restore_fn = await layout.load(self.checkpoint_dir, abstract_state=tree)
    pytree = await restore_fn
    self.assertLen(pytree, 2)
    np.testing.assert_array_equal(
        pytree['a'], jax.device_put(self.data1['a'], device=sharding)
    )
    np.testing.assert_array_equal(
        pytree['c'], jax.device_put(self.data1['c'], device=sharding)
    )

  async def test_load_directory_abstract_tree_subset_one_file(self):
    layout = SafetensorsLayout()
    tree = {
        'a': jax.ShapeDtypeStruct(shape=(2,), dtype=np.int32),
        'c': jax.ShapeDtypeStruct(shape=(2,), dtype=np.int32),
    }
    restore_fn = await layout.load(self.checkpoint_dir, abstract_state=tree)
    pytree = await restore_fn
    self.assertLen(pytree, 2)
    self.assertIn('a', pytree)
    self.assertIn('c', pytree)
    np.testing.assert_array_equal(pytree['a'], self.data1['a'])
    np.testing.assert_array_equal(pytree['c'], self.data1['c'])

  async def test_load_directory_abstract_tree_subset_many_files(self):
    layout = SafetensorsLayout()
    tree = {
        'a': jax.ShapeDtypeStruct(shape=(2,), dtype=np.int32),
        'b': jax.ShapeDtypeStruct(shape=(2,), dtype=np.float32),
    }
    restore_fn = await layout.load(self.checkpoint_dir, abstract_state=tree)
    pytree = await restore_fn
    self.assertLen(pytree, 2)
    self.assertIn('a', pytree)
    self.assertIn('b', pytree)
    np.testing.assert_array_equal(pytree['a'], self.data1['a'])
    np.testing.assert_array_equal(pytree['b'], self.data2['b'])

  async def test_load_directory_abstract_tree_key_not_found(self):
    layout = SafetensorsLayout()
    tree = {
        'e': jax.ShapeDtypeStruct(shape=(2,), dtype=np.int32),
    }
    restore_fn = await layout.load(self.checkpoint_dir, abstract_state=tree)
    with self.assertRaisesRegex(KeyError, "Tensor 'e' not found"):
      await restore_fn

  async def test_load_directory_duplicate_tensor_raises(self):
    # Save the same tensor name into a second file.
    dup_file = self.checkpoint_dir / 'part_dup.safetensors'
    np_save_file({'a': np.array([99, 99], dtype=np.int32)}, dup_file)
    layout = SafetensorsLayout()
    with self.assertRaisesRegex(ValueError, 'Duplicate tensor a'):
      restore_fn = await layout.load(self.checkpoint_dir)
      await restore_fn


class IndexDomainToByteRunsTest(parameterized.TestCase):

  def test_scalar(self):
    runs = safetensors_layout.index_domain_to_byte_runs(
        bounds=(), global_shape=(), itemsize=4, tensor_base=128
    )
    self.assertEqual(runs, [(128, 4)])

  def test_whole_tensor_is_one_run(self):
    runs = safetensors_layout.index_domain_to_byte_runs(
        bounds=((0, 4), (0, 8)),
        global_shape=(4, 8),
        itemsize=4,
        tensor_base=0,
    )
    self.assertEqual(runs, [(0, 4 * 8 * 4)])

  def test_leading_dimension_shard_is_one_run(self):
    runs = safetensors_layout.index_domain_to_byte_runs(
        bounds=((0, 2), (0, 8)),
        global_shape=(4, 8),
        itemsize=4,
        tensor_base=0,
    )
    self.assertEqual(runs, [(0, 2 * 8 * 4)])

  def test_one_dimensional_shard_is_one_run(self):
    runs = safetensors_layout.index_domain_to_byte_runs(
        bounds=((2, 4),), global_shape=(8,), itemsize=4, tensor_base=64
    )
    self.assertEqual(runs, [(64 + 2 * 4, 2 * 4)])

  def test_inner_dimension_shard_is_strided(self):
    runs = safetensors_layout.index_domain_to_byte_runs(
        bounds=((0, 4), (0, 4)),
        global_shape=(4, 8),
        itemsize=4,
        tensor_base=0,
    )
    # 4 rows, each containing a 4-element slice (16 bytes), stride 32 bytes.
    self.assertEqual(runs, [(0, 16), (32, 16), (64, 16), (96, 16)])

  def test_non_zero_start_leading_shard(self):
    runs = safetensors_layout.index_domain_to_byte_runs(
        bounds=((2, 4), (0, 8)),
        global_shape=(4, 8),
        itemsize=4,
        tensor_base=128,
    )
    self.assertEqual(runs, [(128 + 2 * 32, 2 * 32)])

  def test_partial_in_both_dimensions(self):
    runs = safetensors_layout.index_domain_to_byte_runs(
        bounds=((1, 3), (2, 6)),
        global_shape=(4, 8),
        itemsize=4,
        tensor_base=0,
    )
    # 2 rows (1, 2), each 4 elements wide starting at col 2 (offset 8 bytes).
    self.assertEqual(runs, [(32 + 8, 16), (64 + 8, 16)])

  def test_three_dimensional_middle_dimension_partial(self):
    runs = safetensors_layout.index_domain_to_byte_runs(
        bounds=((0, 2), (1, 3), (0, 4)),
        global_shape=(2, 4, 4),
        itemsize=4,
        tensor_base=0,
    )
    # boundary = 1 (middle dim partial). Per outer i in (0, 1):
    #   run_base + i * stride[0] = (0|1) * 64
    #   + bounds[1][0] * stride[1] = 1 * 16
    #   length = (3-1) * 16 = 32
    self.assertEqual(runs, [(16, 32), (64 + 16, 32)])

  def test_non_four_byte_itemsize(self):
    runs = safetensors_layout.index_domain_to_byte_runs(
        bounds=((0, 4), (0, 4)),
        global_shape=(4, 8),
        itemsize=2,
        tensor_base=0,
    )
    self.assertEqual(runs, [(0, 8), (16, 8), (32, 8), (48, 8)])


class ByteMathHelpersTest(parameterized.TestCase):

  def test_normalize_index_slices(self):
    self.assertEqual(
        safetensors_layout._normalize_index(
            (slice(0, 2), slice(3, 7)), (8, 16)
        ),
        ((0, 2), (3, 7)),
    )

  def test_normalize_index_open_ended(self):
    self.assertEqual(
        safetensors_layout._normalize_index(
            (slice(None, None), slice(4, None)), (8, 16)
        ),
        ((0, 8), (4, 16)),
    )

  def test_normalize_index_integer(self):
    self.assertEqual(
        safetensors_layout._normalize_index(
            (2, slice(None, None)),  # pyrefly: ignore[bad-argument-type]
            (8, 16),
        ),  # pytype: disable=wrong-arg-types
        ((2, 3), (0, 16)),
    )

  def test_normalize_index_strided_raises(self):
    with self.assertRaisesRegex(ValueError, 'Strided shard index'):
      safetensors_layout._normalize_index((slice(0, 4, 2),), (8,))

  @parameterized.named_parameters(
      ('1d_f32', (16,), 4, [4]),
      ('2d_f32', (4, 8), 4, [32, 4]),
      ('3d_f32', (2, 3, 4), 4, [48, 16, 4]),
      ('2d_bf16', (4, 8), 2, [16, 2]),
  )
  def test_byte_strides(self, shape, itemsize, expected):
    self.assertEqual(
        safetensors_layout._byte_strides(shape, itemsize), expected
    )


class PartitionRunsTest(parameterized.TestCase):

  def test_empty(self):
    self.assertEqual(
        safetensors_layout._partition_runs([], max_over_read_ratio=2.0), []
    )

  def test_single_run_is_one_block(self):
    blocks = safetensors_layout._partition_runs(
        [(100, 50)], max_over_read_ratio=2.0
    )
    self.assertEqual(
        blocks,
        [safetensors_layout._CoalescedBlock(100, 150, (0,))],
    )

  def test_adjacent_runs_collapse_into_one_block(self):
    blocks = safetensors_layout._partition_runs(
        [(0, 100), (100, 50), (150, 50)], max_over_read_ratio=1.0
    )
    # Densely packed -> ratio 1.0 -> one block.
    self.assertLen(blocks, 1)
    self.assertEqual((blocks[0].start, blocks[0].end), (0, 200))
    self.assertEqual(blocks[0].members, (0, 1, 2))

  def test_ratio_bound_prevents_excess_over_read(self):
    # Two 1 KB runs separated by 3 KB gap -> block extent 5 KB, needed 2 KB,
    # ratio 2.5 > 2.0 cap -> must split. The gap is below the default page
    # floor, so disable it to exercise the ratio in isolation.
    blocks = safetensors_layout._partition_runs(
        [(0, 1024), (4096, 1024)], max_over_read_ratio=2.0, min_coalesce_gap=0
    )
    self.assertLen(blocks, 2)
    self.assertEqual((blocks[0].start, blocks[0].end), (0, 1024))
    self.assertEqual((blocks[1].start, blocks[1].end), (4096, 4096 + 1024))

  def test_ratio_bound_allows_when_ratio_within_cap(self):
    # Same two runs but cap is 2.5 -> coalesces.
    blocks = safetensors_layout._partition_runs(
        [(0, 1024), (4096, 1024)], max_over_read_ratio=2.5
    )
    self.assertLen(blocks, 1)
    self.assertEqual((blocks[0].start, blocks[0].end), (0, 5120))
    self.assertEqual(blocks[0].members, (0, 1))

  def test_large_dense_block_is_not_capped_by_partition(self):
    # Block size is intentionally unbounded in partition; oversized blocks
    # are split at the read chunk size later. A single 1 GiB run is one
    # block of 1 GiB regardless of any "size cap" -- there isn't one.
    blocks = safetensors_layout._partition_runs(
        [(0, 1 << 30)], max_over_read_ratio=1.0
    )
    self.assertLen(blocks, 1)
    self.assertEqual(blocks[0].end - blocks[0].start, 1 << 30)

  def test_sorts_unsorted_input(self):
    blocks = safetensors_layout._partition_runs(
        [(200, 50), (0, 100), (100, 100)], max_over_read_ratio=1.0
    )
    self.assertLen(blocks, 1)
    # Members are reported in sorted order of offset.
    self.assertEqual(blocks[0].members, (1, 2, 0))

  def test_inner_dim_strided_collapses_when_ratio_allows(self):
    # 8 strided runs of 8 bytes each, stride 16. needed=64, extent=120,
    # ratio=1.875 < 2.0 -> coalesces to 1 block.
    runs = [(i * 16, 8) for i in range(8)]
    blocks = safetensors_layout._partition_runs(runs, max_over_read_ratio=2.0)
    self.assertLen(blocks, 1)
    self.assertEqual(blocks[0].end - blocks[0].start, 120)

  def test_inner_dim_strided_partitions_when_ratio_too_high(self):
    # 4 strided runs, stride 64, length 8. needed=32, extent=200, ratio=6.25.
    # With cap 2.0 and the floor disabled -> single-run blocks. (The 56-byte
    # gaps are below the default page floor; the floor case is covered below.)
    runs = [(i * 64, 8) for i in range(4)]
    blocks = safetensors_layout._partition_runs(
        runs, max_over_read_ratio=2.0, min_coalesce_gap=0
    )
    self.assertLen(blocks, 4)
    for b in blocks:
      self.assertEqual(b.end - b.start, 8)

  def test_small_gap_coalesces_despite_ratio(self):
    # Two 8-byte runs split by a 192-byte gap: ratio 13x >> 2.0, but the gap is
    # well below the page floor, so they coalesce into one read (the worked
    # inner-dim example) rather than shattering into microscopic requests.
    blocks = safetensors_layout._partition_runs(
        [(0, 8), (200, 8)], max_over_read_ratio=2.0
    )
    self.assertLen(blocks, 1)
    self.assertEqual((blocks[0].start, blocks[0].end), (0, 208))

  def test_gap_above_floor_respects_ratio(self):
    # An 8 KB gap is above the page floor, so the ratio governs again and a
    # huge over-read ratio splits the runs into separate reads.
    blocks = safetensors_layout._partition_runs(
        [(0, 8), (8192, 8)], max_over_read_ratio=2.0
    )
    self.assertLen(blocks, 2)

  def test_explicit_larger_floor_coalesces_above_default(self):
    # Raising the floor past the gap coalesces what the default would split.
    blocks = safetensors_layout._partition_runs(
        [(0, 8), (8192, 8)], max_over_read_ratio=2.0, min_coalesce_gap=16384
    )
    self.assertLen(blocks, 1)

  def test_invalid_ratio_raises(self):
    with self.assertRaisesRegex(ValueError, 'max_over_read_ratio'):
      safetensors_layout._partition_runs([(0, 100)], max_over_read_ratio=0.5)


class FileReadTest(unittest.IsolatedAsyncioTestCase, parameterized.TestCase):
  """End-to-end correctness of `_plan_chunk_reads` + `_read_file`.

  The partition policy is exercised separately in `PartitionRunsTest`; here we
  verify the planned chunk reads deliver the right bytes to each destination
  buffer, under both single-chunk and split-block paths.
  """

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir()
    self.file_path = epath.Path(self.test_dir.full_path) / 'data.bin'
    self.payload = bytes(range(256)) * 16  # 4 KB of varied bytes
    self.file_path.write_bytes(self.payload)

  async def _run(self, ratio, chunk_bytes, offsets, lengths):
    """Reads `(offset, length)` runs; returns the bytes each received."""
    # +1 mirrors `_load`: LimitInFlightBytes reserves strictly fewer than its
    # max, so the largest reservable chunk is `chunk_bytes`.
    budget = limits.LimitInFlightBytes(chunk_bytes + 1)
    reads = [
        safetensors_layout._Read(o, l, np.zeros(l, dtype=np.uint8))
        for o, l in zip(offsets, lengths)
    ]
    await safetensors_layout._read_file(
        self.file_path,
        reads,
        budget,
        max_over_read_ratio=ratio,
        chunk_bytes=chunk_bytes,
    )
    return [r.dst.tobytes() for r in reads]

  async def test_delivers_bytes_for_dense_runs(self):
    offsets, lengths = [0, 16, 32, 48], [16, 16, 16, 16]
    received = await self._run(1.0, 1 << 20, offsets, lengths)
    for i, b in enumerate(received):
      self.assertEqual(b, self.payload[i * 16 : (i + 1) * 16])

  async def test_delivers_bytes_for_sparse_runs(self):
    offsets = [0, 1024, 2048]
    received = await self._run(1.5, 1 << 20, offsets, [16] * 3)
    for off, b in zip(offsets, received):
      self.assertEqual(b, self.payload[off : off + 16])

  async def test_delivers_bytes_with_auto_ratio(self):
    offsets = [0, 1024, 2048]
    received = await self._run(None, 1 << 20, offsets, [16] * 3)
    for off, b in zip(offsets, received):
      self.assertEqual(b, self.payload[off : off + 16])

  async def test_delivers_bytes_when_block_splits_into_chunks(self):
    # 5 dense 16-byte runs coalesce into one 80-byte block (ratio 1.0). A
    # 64-byte chunk size splits the block into two reads (64 + 16); the run
    # straddling the boundary is served by both.
    offsets = [0, 16, 32, 48, 64]
    received = await self._run(1.0, 64, offsets, [16] * 5)
    for off, b in zip(offsets, received):
      self.assertEqual(b, self.payload[off : off + 16])

  async def test_delivers_bytes_when_chunk_smaller_than_single_run(self):
    # One 64-byte run with a 16-byte chunk size -> 4 reads, each serving a
    # quarter of the run.
    received = await self._run(1.0, 16, [0], [64])
    self.assertEqual(received[0], self.payload[:64])

  async def test_read_file_returns_read_stats(self):
    # 4 dense 16-byte runs coalesce into one 64-byte block (ratio 1.0); the
    # huge chunk size reads it in a single ranged GET.
    budget = limits.LimitInFlightBytes((1 << 20) + 1)
    reads = [
        safetensors_layout._Read(o, 16, np.zeros(16, dtype=np.uint8))
        for o in (0, 16, 32, 48)
    ]
    stats = await safetensors_layout._read_file(
        self.file_path,
        reads,
        budget,
        max_over_read_ratio=1.0,
        chunk_bytes=1 << 20,
    )
    self.assertEqual(stats.needed_bytes, 64)
    self.assertEqual(stats.read_bytes, 64)
    self.assertEqual(stats.num_reads, 1)
    self.assertEqual(stats.num_gets, 1)  # fits one chunk -> single GET.

  async def test_stats_count_block_and_split_chunks(self):
    # One 64-byte run with a 16-byte chunk size splits into 4 reads. It is
    # still a single coalesced block (num_reads == 1), but four actual GETs
    # are issued (num_gets == 4).
    budget = limits.LimitInFlightBytes(16 + 1)
    reads = [safetensors_layout._Read(0, 64, np.zeros(64, dtype=np.uint8))]
    stats = await safetensors_layout._read_file(
        self.file_path,
        reads,
        budget,
        max_over_read_ratio=1.0,
        chunk_bytes=16,
    )
    self.assertEqual(stats.read_bytes, 64)
    self.assertEqual(stats.num_reads, 1)
    self.assertEqual(stats.num_gets, 4)

  async def test_read_file_with_no_reads_is_noop(self):
    budget = limits.LimitInFlightBytes(1 << 20)
    stats = await safetensors_layout._read_file(
        self.file_path,
        [],
        budget,
        max_over_read_ratio=2.0,
        chunk_bytes=1 << 20,
    )  # must not raise, must not open the file.
    self.assertEqual(stats.num_gets, 0)

  async def test_read_failure_raises(self):
    budget = limits.LimitInFlightBytes(1 << 20)
    reads = [
        safetensors_layout._Read(o, 16, np.zeros(16, dtype=np.uint8))
        for o in (0, 16)
    ]
    with self.assertRaises(Exception):
      await safetensors_layout._read_file(
          epath.Path('/nonexistent/path.bin'),
          reads,
          budget,
          max_over_read_ratio=2.0,
          chunk_bytes=1 << 20,
      )


class RecordReadStatsTest(absltest.TestCase):
  """`_record_read_stats` reports per-host totals via jax.monitoring."""

  def test_emits_per_host_totals(self):
    stats = [
        safetensors_layout._ReadStats(
            needed_bytes=40, read_bytes=64, num_reads=1, num_gets=1
        ),
        safetensors_layout._ReadStats(
            needed_bytes=16, read_bytes=20, num_reads=2, num_gets=5
        ),
    ]
    with mock.patch.object(jax.monitoring, 'record_scalar') as rec:
      safetensors_layout._record_read_stats(stats)
    emitted = {c.args[0]: c.args[1] for c in rec.call_args_list}
    self.assertEqual(emitted['/jax/orbax/read/safetensors/bytes_read'], 84.0)
    self.assertEqual(emitted['/jax/orbax/read/safetensors/num_reads'], 3.0)
    self.assertEqual(emitted['/jax/orbax/read/safetensors/storage_reads'], 6.0)


class AutoOverReadRatioTest(absltest.TestCase):
  """`_auto_over_read_ratio` picks 1.0 or whole-span from the run geometry."""

  # A stride whose gaps stay above the coalesce floor, so every run is its
  # own read at ratio 1.0.
  _STRIDE = 8 + 2 * safetensors_layout._DEFAULT_MIN_COALESCE_GAP

  def test_no_runs(self):
    self.assertEqual(safetensors_layout._auto_over_read_ratio([]), 1.0)

  def test_few_contiguous_runs_pick_no_over_read(self):
    # Leading-dim shards: large contiguous slabs, far apart.
    runs = [(i * (1 << 20), 1 << 18) for i in range(64)]
    self.assertEqual(safetensors_layout._auto_over_read_ratio(runs), 1.0)

  def test_sub_floor_gaps_count_as_merged(self):
    # Many tiny runs whose gaps the floor always coalesces -> few effective
    # reads -> no over-read needed.
    gap = safetensors_layout._DEFAULT_MIN_COALESCE_GAP
    runs = [(i * (8 + gap), 8) for i in range(10_000)]
    self.assertEqual(safetensors_layout._auto_over_read_ratio(runs), 1.0)

  def test_fragmented_runs_pick_whole_span(self):
    n = safetensors_layout._AUTO_MAX_READS_PER_FILE + 1
    runs = [(i * self._STRIDE, 8) for i in range(n)]
    span = (n - 1) * self._STRIDE + 8
    self.assertEqual(
        safetensors_layout._auto_over_read_ratio(runs), span / (8 * n)
    )

  def test_read_count_at_cap_picks_no_over_read(self):
    runs = [
        (i * self._STRIDE, 8)
        for i in range(safetensors_layout._AUTO_MAX_READS_PER_FILE)
    ]
    self.assertEqual(safetensors_layout._auto_over_read_ratio(runs), 1.0)

  def test_unsorted_runs(self):
    n = safetensors_layout._AUTO_MAX_READS_PER_FILE + 1
    runs = [(i * self._STRIDE, 8) for i in reversed(range(n))]
    span = (n - 1) * self._STRIDE + 8
    self.assertEqual(
        safetensors_layout._auto_over_read_ratio(runs), span / (8 * n)
    )

  def test_auto_ratio_collapses_fragmented_plan(self):
    # Through the planner: the picked ratio turns a fragmented plan into a
    # single whole-span block instead of one read per run.
    n = safetensors_layout._AUTO_MAX_READS_PER_FILE + 1
    reads = [
        safetensors_layout._Read(
            i * self._STRIDE, 8, np.zeros(8, dtype=np.uint8)
        )
        for i in range(n)
    ]
    _, stats = safetensors_layout._plan_chunk_reads(
        reads, max_over_read_ratio=None, chunk_bytes=1 << 30
    )
    self.assertEqual(stats.num_reads, 1)
    self.assertEqual(stats.needed_bytes, 8 * n)
    self.assertEqual(stats.read_bytes, (n - 1) * self._STRIDE + 8)

  def test_auto_ratio_keeps_contiguous_plan_exact(self):
    # Leading-dim-style slabs stay unmerged: bytes read == bytes needed.
    reads = [
        safetensors_layout._Read(
            i * (1 << 20), 1 << 18, np.zeros(1 << 18, dtype=np.uint8)
        )
        for i in range(64)
    ]
    _, stats = safetensors_layout._plan_chunk_reads(
        reads, max_over_read_ratio=None, chunk_bytes=1 << 30
    )
    self.assertEqual(stats.num_reads, 64)
    self.assertEqual(stats.read_bytes, stats.needed_bytes)


class OverReadWarningTest(absltest.TestCase):
  """`_warn_if_over_read` fires only on heavy over-read under the auto ratio."""

  def _stats(self, needed, read):
    return safetensors_layout._ReadStats(
        needed_bytes=needed, read_bytes=read, num_reads=1, num_gets=1
    )

  def test_warns_on_heavy_over_read(self):
    # Whole-span reads for a fragmenting sharding: 4x the needed bytes.
    stats = [self._stats(needed=8192, read=32768)]
    with mock.patch.object(safetensors_layout.logging, 'warning') as warn:
      safetensors_layout._warn_if_over_read(
          stats, max_over_read_ratio_is_auto=True
      )
    warn.assert_called_once()

  def test_no_warn_when_reads_match_needs(self):
    stats = [self._stats(needed=8192, read=8192)]
    with mock.patch.object(safetensors_layout.logging, 'warning') as warn:
      safetensors_layout._warn_if_over_read(
          stats, max_over_read_ratio_is_auto=True
      )
    warn.assert_not_called()

  def test_no_warn_below_threshold(self):
    # Gap-floor merges cause mild over-read; not worth a warning.
    stats = [self._stats(needed=8192, read=9000)]
    with mock.patch.object(safetensors_layout.logging, 'warning') as warn:
      safetensors_layout._warn_if_over_read(
          stats, max_over_read_ratio_is_auto=True
      )
    warn.assert_not_called()

  def test_no_warn_when_user_set_ratio(self):
    # User already engaged with the knob -> stay quiet.
    stats = [self._stats(needed=8192, read=32768)]
    with mock.patch.object(safetensors_layout.logging, 'warning') as warn:
      safetensors_layout._warn_if_over_read(
          stats, max_over_read_ratio_is_auto=False
      )
    warn.assert_not_called()

  def test_no_warn_when_nothing_needed(self):
    stats = [self._stats(needed=0, read=0)]
    with mock.patch.object(safetensors_layout.logging, 'warning') as warn:
      safetensors_layout._warn_if_over_read(
          stats, max_over_read_ratio_is_auto=True
      )
    warn.assert_not_called()


class SafetensorsLayoutEdgeCaseTest(
    unittest.IsolatedAsyncioTestCase, parameterized.TestCase
):

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir()
    self.file_path = epath.Path(self.test_dir.full_path) / 'edge.safetensors'

  async def test_shape_mismatch_raises(self):
    np_save_file({'a': np.arange(8, dtype=np.float32)}, self.file_path)
    layout = SafetensorsLayout()
    tree = {'a': jax.ShapeDtypeStruct(shape=(4,), dtype=np.float32)}
    restore_fn = await layout.load(self.file_path, abstract_state=tree)
    with self.assertRaisesRegex(ValueError, 'Shape mismatch'):
      await restore_fn

  async def test_dtype_cast_on_load(self):
    np_save_file({'a': np.arange(8, dtype=np.int32)}, self.file_path)
    layout = SafetensorsLayout()
    # Requesting float32 from an int32 file -> cast applied during assembly.
    # int32 -> float32 stays within JAX's default supported dtypes (no need
    # for jax_enable_x64).
    tree = {'a': jax.ShapeDtypeStruct(shape=(8,), dtype=np.float32)}
    restore_fn = await layout.load(self.file_path, abstract_state=tree)
    pytree = await restore_fn
    self.assertEqual(pytree['a'].dtype, np.float32)
    np.testing.assert_allclose(pytree['a'], np.arange(8, dtype=np.float32))

  async def test_scalar_loads(self):
    np_save_file({'s': np.array(3.14, dtype=np.float32)}, self.file_path)
    layout = SafetensorsLayout()
    restore_fn = await layout.load(self.file_path)
    pytree = await restore_fn
    np.testing.assert_allclose(pytree['s'], 3.14)

  async def test_nested_abstract_state_raises(self):
    np_save_file({'a': np.arange(8, dtype=np.float32)}, self.file_path)
    layout = SafetensorsLayout()
    nested = {
        'group': {'a': jax.ShapeDtypeStruct(shape=(8,), dtype=np.float32)}
    }
    with self.assertRaisesRegex(ValueError, 'not a flat dictionary'):
      await layout.load(self.file_path, abstract_state=nested)

  async def test_empty_abstract_state_returns_empty(self):
    np_save_file({'a': np.arange(8, dtype=np.float32)}, self.file_path)
    layout = SafetensorsLayout()
    restore_fn = await layout.load(self.file_path, abstract_state={})
    pytree = await restore_fn
    self.assertEqual(pytree, {})

  async def test_options_override_partition_defaults(self):
    np_save_file({'a': np.arange(8, dtype=np.float32)}, self.file_path)
    layout = SafetensorsLayout()
    # ratio 1.0 forces per-tensor block isolation; a tiny budget forces each
    # block to split into multiple chunk reads. Correctness must hold under
    # both -- the chunks must reassemble into the full tensor bytes.
    with context_lib.Context(
        memory_options=options_lib.MemoryOptions(read_concurrent_bytes=16),
        safetensors_options=options_lib.SafetensorsOptions(
            max_over_read_ratio=1.0,
        ),
    ):
      restore_fn = await layout.load(self.file_path)
      pytree = await restore_fn
    np.testing.assert_allclose(pytree['a'], np.arange(8, dtype=np.float32))

  async def test_read_chunk_bytes_option_splits_reads(self):
    # One 32-byte tensor with a 16-byte chunk size: still one coalesced
    # block, but two storage GETs. Pins that the option reaches the planner.
    np_save_file({'a': np.arange(8, dtype=np.float32)}, self.file_path)
    layout = SafetensorsLayout()
    with mock.patch.object(jax.monitoring, 'record_scalar') as rec:
      with context_lib.Context(
          safetensors_options=options_lib.SafetensorsOptions(
              read_chunk_bytes=16,
          ),
      ):
        restore_fn = await layout.load(self.file_path)
        pytree = await restore_fn
    emitted = {c.args[0]: c.args[1] for c in rec.call_args_list}
    self.assertEqual(emitted['/jax/orbax/read/safetensors/num_reads'], 1.0)
    self.assertEqual(emitted['/jax/orbax/read/safetensors/storage_reads'], 2.0)
    np.testing.assert_allclose(pytree['a'], np.arange(8, dtype=np.float32))

  async def test_non_positive_read_chunk_bytes_raises(self):
    np_save_file({'a': np.arange(8, dtype=np.float32)}, self.file_path)
    layout = SafetensorsLayout()
    with context_lib.Context(
        safetensors_options=options_lib.SafetensorsOptions(
            read_chunk_bytes=0,
        ),
    ):
      restore_fn = await layout.load(self.file_path)
      with self.assertRaisesRegex(ValueError, 'read_chunk_bytes'):
        await restore_fn

  async def test_broadcast_replicated_is_noop_on_single_process(self):
    # A single process already reads each byte exactly once, so the flag
    # must not change behavior (and must not attempt any broadcast).
    np_save_file({'a': np.arange(8, dtype=np.float32)}, self.file_path)
    layout = SafetensorsLayout()
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.array(jax.devices()), ('x',)),
        jax.sharding.PartitionSpec(),
    )
    tree = {
        'a': jax.ShapeDtypeStruct(
            shape=(8,), dtype=np.float32, sharding=sharding
        )
    }
    with context_lib.Context(
        safetensors_options=options_lib.SafetensorsOptions(
            broadcast_replicated=True,
        ),
    ):
      restore_fn = await layout.load(self.file_path, abstract_state=tree)
      pytree = await restore_fn
    np.testing.assert_allclose(pytree['a'], np.arange(8, dtype=np.float32))


class DedupReadShardingGateTest(absltest.TestCase):
  """Cases where `_dedup_read_sharding` must decline (single-device host).

  The placement logic needs a multi-device mesh and is exercised in the
  multiprocess suite; here we pin the gates that need no replication.
  """

  def test_scalar_returns_none(self):
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.array(jax.devices()), ('x',)),
        jax.sharding.PartitionSpec(),
    )
    self.assertIsNone(safetensors_layout._dedup_read_sharding(sharding, ()))

  def test_non_named_sharding_returns_none(self):
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    self.assertIsNone(
        safetensors_layout._dedup_read_sharding(sharding, (8,))
    )

  def test_no_replicating_axes_returns_none(self):
    # Every mesh axis is used by the spec, so nothing is replicated and
    # there is nothing to de-duplicate.
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.array(jax.devices()), ('x',)),
        jax.sharding.PartitionSpec('x'),
    )
    self.assertIsNone(
        safetensors_layout._dedup_read_sharding(sharding, (8,))
    )


class SingleHostOneReadInvariantTest(
    unittest.IsolatedAsyncioTestCase, parameterized.TestCase
):
  """Verifies the single-host-degenerate-case property: 1 file = 1 read."""

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir()
    self.file_path = epath.Path(self.test_dir.full_path) / 'pack.safetensors'
    # Multiple densely-packed tensors in one file.
    np_save_file(
        {f'tensor_{i}': np.arange(64, dtype=np.float32) for i in range(8)},
        self.file_path,
    )

  async def _count_reads_during_load(self, abstract_state=None):
    read_count = 0
    original = safetensors_layout._read_chunk

    async def counting_read_chunk(path, chunk, byte_budget):
      nonlocal read_count
      read_count += 1
      return await original(path, chunk, byte_budget)

    layout = SafetensorsLayout()
    with mock.patch.object(
        safetensors_layout,
        '_read_chunk',
        new=counting_read_chunk,
    ):
      restore_fn = await layout.load(
          self.file_path, abstract_state=abstract_state
      )
      _ = await restore_fn
    return read_count

  async def test_whole_file_load_is_one_read(self):
    # No abstract_state -> whole tensors -> all densely packed -> 1 read.
    self.assertEqual(await self._count_reads_during_load(), 1)

  async def test_sharded_to_all_local_devices_is_one_read(self):
    # All devices addressable on single host, all tensors needed -> 1 read.
    if jax.local_device_count() < 1:
      self.skipTest('Requires at least one local device.')
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    abstract_state = {
        f'tensor_{i}': jax.ShapeDtypeStruct(
            shape=(64,), dtype=np.float32, sharding=sharding
        )
        for i in range(8)
    }
    self.assertEqual(await self._count_reads_during_load(abstract_state), 1)


if __name__ == '__main__':
  absltest.main()
