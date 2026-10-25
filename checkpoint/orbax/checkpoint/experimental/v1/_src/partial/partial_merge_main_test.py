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

import collections
import random
from unittest import mock

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint._src.tree import utils as tree_utils
from orbax.checkpoint.experimental.v1._src.loading import loading as loading_lib
from orbax.checkpoint.experimental.v1._src.partial import partial_merge_main
from orbax.checkpoint.experimental.v1._src.saving import saving as saving_lib
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils

from .learning.deepmind.jax.roc.experimental import eval_fragments

FLAGS = flags.FLAGS

MEMORY_LIMIT_BYTES = 2**30 * 16
MB = 2**10


def _create_mock_metadata(shape, dtype):
  return value_metadata.ArrayMetadata(
      name='test_array',
      directory=None,
      shape=shape,
      dtype=dtype,
      sharding=None,
  )


def _setup_large_pytree(sharding: jax.sharding.Sharding):
  """Creates a large pytree with arrays of random sizes."""
  # 100 arrays
  array_sizes_mb = [50] * 2 + [10] * 5 + [2] * 13 + [1] * 80
  rng = random.Random(42)
  rng.shuffle(array_sizes_mb)

  pytree = {}
  for i in range(10):
    for j in range(10):
      array_size = array_sizes_mb[i * 10 + j] * MB // 4
      pytree.setdefault(f'param{i}', {})[f'param{j}'] = np.ones(
          array_size, dtype=np.float32
      )
  pytree = jax.device_put(pytree, sharding)
  return pytree


def _permute_pytree(pytree, idx):
  def _permute(x):
    if isinstance(x, np.ndarray):
      rng = np.random.default_rng(seed=idx)
      x *= rng.random(x.shape).astype(x.dtype)
    return x

  return jax.tree.map(_permute, pytree)


def _get_abstract_pytree(pytree):
  return jax.tree.map(array_test_utils.as_abstract_type, pytree)


class PartialMergeTest(
    parameterized.TestCase, multiprocess_test.MultiProcessTest
):

  def setUp(self):
    super().setUp()

    self.directory = epath.Path(
        self.create_tempdir(name='partial_merging_test').full_path
    )
    self.pytree, self.abstract_pytree = array_test_utils.create_sharded_pytree()

    test_utils.set_tensorstore_driver_for_test()
    test_utils.sync_global_processes('PartialMergingTest:setUp:complete')

  def tearDown(self):
    super().tearDown()
    test_utils.sync_global_processes('PartialMergingTest:tearDown:complete')

  def test_batch_fragments_logic(self):
    """Tests the greedy batching logic based on memory limits."""

    # helper to create a mock FragmentInfo with a specific size
    def _mock_info(size_bytes, name):
      m = mock.Mock(spec=partial_merge_main.FragmentInfo)
      type(m).size_bytes = mock.PropertyMock(return_value=size_bytes)
      m.name = name  # Just for debugging/identification in test
      return m

    # 1 GB limit
    limit_gb = 1
    limit_bytes = limit_gb * 1024**3

    # Case 1: Items fit perfectly into one batch
    infos = [
        _mock_info(int(limit_bytes * 0.4), 'A'),
        _mock_info(int(limit_bytes * 0.4), 'B'),
    ]
    batches = list(partial_merge_main.batch_fragments(infos, limit_gb))
    self.assertLen(batches, 1)
    self.assertEqual(batches[0], infos)

    # Case 2: Items spill over to second batch
    infos = [
        _mock_info(int(limit_bytes * 0.6), 'A'),
        _mock_info(int(limit_bytes * 0.5), 'B'),  # 0.6 + 0.5 > 1.0
        _mock_info(int(limit_bytes * 0.2), 'C'),
    ]
    batches = list(partial_merge_main.batch_fragments(infos, limit_gb))
    self.assertLen(batches, 2)
    self.assertEqual(batches[0], [infos[0]])  # A (0.6)
    self.assertEqual(batches[1], [infos[1], infos[2]])  # B (0.5) + C (0.2)

    # Case 3: Single item equals limit
    infos = [_mock_info(limit_bytes, 'A')]
    batches = list(partial_merge_main.batch_fragments(infos, limit_gb))
    self.assertLen(batches, 1)

    # Case 4: Single item exceeds limit (Should raise ValueError)
    infos = [_mock_info(limit_bytes + 1, 'TooBig')]
    with self.assertRaisesRegex(ValueError, 'larger than memory limit'):
      list(partial_merge_main.batch_fragments(infos, limit_gb))

  @mock.patch('random.shuffle')
  def test_create_fragment_infos(self, mock_shuffle):
    """Tests flattening of fragment trees into FragmentInfo objects."""
    # Ensure shuffle does nothing so we can assert order deterministically
    mock_shuffle.side_effect = lambda x: x

    # Mock the input structure: List[PyTree[Fragments]]
    # We simulate 2 checkpoints.

    # Mock Fragment object (from eval_fragments usually)
    MockFragment = collections.namedtuple('MockFragment', ['index'])

    # Mock array_fragments.Fragments
    class MockFragmentsContainer:

      def __init__(self, frags, dtype):
        self.fragments = frags
        self.dtype = dtype

    # Checkpoint 0 structure
    ckpt0_tree = {
        'layer1': MockFragmentsContainer(
            [MockFragment(index=1), MockFragment(index=2)], np.float32
        )
    }

    # Checkpoint 1 structure
    ckpt1_tree = {
        'layer1': MockFragmentsContainer([MockFragment(index=3)], np.float32),
        'layer2': MockFragmentsContainer([MockFragment(index=4)], np.int32),
    }

    required_input_fragments = [ckpt0_tree, ckpt1_tree]

    infos = partial_merge_main.create_fragment_infos(required_input_fragments)

    # We expect:
    # Ckpt0: 2 fragments from layer1
    # Ckpt1: 1 fragment from layer1, 1 fragment from layer2
    # Total = 4 FragmentInfos
    self.assertLen(infos, 4)

    # Verify content of the first info (Ckpt 0, layer 1, first frag)
    self.assertEqual(infos[0].ckpt_idx, 0)
    self.assertEqual(infos[0].keypath, ('layer1',))
    self.assertEqual(infos[0].fragment.index, 1)
    self.assertEqual(infos[0].dtype, np.float32)

    # Verify content of the last info (Ckpt 1, layer 2)
    # Note: dict iteration order is insertion order in modern python,
    # but create_fragment_infos iterates the list of ckpts.

    # Find the int32 fragment
    int_frag = next(x for x in infos if x.dtype == np.int32)
    self.assertEqual(int_frag.ckpt_idx, 1)
    self.assertEqual(int_frag.keypath, ('layer2',))
    self.assertEqual(int_frag.fragment.index, 4)

  def test_fragments_to_arrays_passthrough(self):
    """Tests that non-fragment leaves are passed through unchanged."""
    target = {'a': jax.ShapeDtypeStruct((1,), np.float32)}
    # If the input isn't an eval_fragments.Fragments instance, returns it as is.
    # Happens if tree structure doesn't match perfectly or for non-array leaves.
    inputs = {'a': 123}

    result = partial_merge_main.fragments_to_arrays(inputs, target)
    self.assertEqual(result['a'], 123)

  def test_fragments_to_arrays_conversion(self):
    """Tests conversion of Fragments to jax.Array via callback."""
    shape = (2, 2)
    dtype = np.float32

    local_devices = [
        d for d in jax.devices('cpu') if d.process_index == jax.process_index()
    ]
    if not local_devices:
      self.skipTest('No local CPU devices found')

    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(local_devices[:1], ('x',)),
        jax.sharding.PartitionSpec('x'),
    )

    target_leaf = jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
    target = {'param': target_leaf}

    mock_fragment_data = np.ones(shape, dtype)

    with mock.patch.object(
        partial_merge_main.eval_fragments, '_extract_fragment'
    ) as mock_extract:
      mock_extract.return_value.value = mock_fragment_data

      fragments_input = eval_fragments.ConcreteFragments(
          shape=shape,
          dtype=np.dtype(dtype),
          fragments=[
              eval_fragments.ConcreteFragment(
                  index=(slice(0, 2, 1), slice(0, 2, 1)),
                  value=np.zeros(shape, dtype),
              )
          ],
      )

      inputs = {'param': fragments_input}

      result_tree = partial_merge_main.fragments_to_arrays(inputs, target)

      self.assertIsInstance(result_tree['param'], jax.Array)
      self.assertEqual(result_tree['param'].shape, shape)
      self.assertEqual(result_tree['param'].dtype, dtype)

      # Force materialization to trigger the callback
      result_data = np.array(result_tree['param'])
      np.testing.assert_array_equal(result_data, mock_fragment_data)

  def test_resolve_pytree_path(self):
    """Tests path resolution logic."""
    with self.subTest('valid_path'):
      temp_dir = self.create_tempdir()
      path = epath.Path(temp_dir.full_path)
      (
          path / partial_merge_main.checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY
      ).mkdir()

      result = partial_merge_main.resolve_pytree_path(path)
      self.assertEqual(
          result,
          path / partial_merge_main.checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY,
      )

    with self.subTest('invalid_path'):
      temp_dir = self.create_tempdir('invalid_path')
      path = epath.Path(temp_dir.full_path)

      with self.assertRaisesRegex(ValueError, 'does not contain a pytree'):
        partial_merge_main.resolve_pytree_path(path)

  @flagsaver.flagsaver
  def test_main(self):
    ckpt_paths = [
        self.directory / 'ckpt0',
        self.directory / 'ckpt1',
        self.directory / 'ckpt2',
    ]
    out_path = self.directory / 'out'

    trees = [
        {
            'a': self.pytree['a'],
            'b': self.pytree['b'],
            'c': {
                'a': self.pytree['c']['a'],
                'e': self.pytree['c']['e'],
            },
            # skip 'x' and 'y'
        },
        {
            'a': jax.tree.map(lambda x: x * 2, self.pytree['a']),
            # skip 'b'
            'c': {
                'a': jax.tree.map(lambda x: x * 2, self.pytree['c']['a']),
                # skip 'c.e'
            },
        },
        {
            # skip 'a' and 'b'
            'c': {
                'a': jax.tree.map(lambda x: x * 3, self.pytree['c']['a']),
                # skip 'c.e'
            },
            'x': jax.tree.map(lambda x: x * 3, self.pytree['x']),
            'y': jax.tree.map(lambda x: x * 3, self.pytree['y']),
        },
    ]

    for path, pytree in zip(ckpt_paths, trees):
      saving_lib.save_pytree(path, pytree)

    expected_tree = {
        'a': trees[1]['a'],
        'b': trees[0]['b'],
        'c': {
            'a': trees[2]['c']['a'],
            'e': trees[0]['c']['e'],
        },
        'x': trees[2]['x'],
        'y': trees[2]['y'],
    }
    abstract_expected_tree = jax.tree.map(
        tree_utils.to_shape_dtype_struct, expected_tree
    )

    FLAGS.in_paths = [str(path) for path in ckpt_paths]
    FLAGS.out_path = str(out_path)
    FLAGS.per_host_memory_limit_gb = 1

    partial_merge_main.main()

    # Check that the merged checkpoint exists and has the correct contents.
    merged_ckpt = loading_lib.load_pytree(out_path, abstract_expected_tree)
    test_utils.assert_tree_equal(self, merged_ckpt, expected_tree)


if __name__ == '__main__':
  # Initialize required flags with dummy values before googletest.main()
  # if they are marked as required in the binary.
  FLAGS.in_paths = ['default_in']
  FLAGS.out_path = 'default_out'
  multiprocess_test.main()
