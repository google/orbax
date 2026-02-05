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

import dataclasses
import random
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.arrays import fragments as array_fragments
from orbax.checkpoint.experimental.v1._src.partial import merging


GB = 1024**3


AbstractFragment = array_fragments.AbstractFragment
FragmentInfo = merging.FragmentInfo


def _mock_info(size_bytes: int) -> mock.MagicMock:
  m = mock.create_autospec(merging.FragmentInfo, instance=True)
  type(m).size_bytes = mock.PropertyMock(return_value=size_bytes)
  return m


class PartialMergingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.directory = epath.Path(
        self.create_tempdir(name='partial_merging_test').full_path
    )

    test_utils.set_tensorstore_driver_for_test()

  def test_resolve_pytree_path_valid(self):
    (self.directory / merging.PYTREE_CHECKPOINTABLE_KEY).mkdir()
    result = merging.resolve_pytree_path(self.directory)
    self.assertEqual(result, self.directory / merging.PYTREE_CHECKPOINTABLE_KEY)

  def test_resolve_pytree_path_invalid(self):
    with self.assertRaisesRegex(ValueError, 'does not contain a pytree'):
      merging.resolve_pytree_path(self.directory)

  def test_resolve_target_structure(self):
    """Tests resolve_target_structure."""
    host_cpus = jax.devices('cpu')

    abstract_sources = [
        {
            'a': jax.ShapeDtypeStruct((4,), np.float32),
            'b': jax.ShapeDtypeStruct((2, 2), np.int32),
        },
        {
            'b': jax.ShapeDtypeStruct((2, 2), np.float32),
            'c': jax.ShapeDtypeStruct((1,), np.float32),
        },
    ]
    result = merging.resolve_target_structure(abstract_sources, host_cpus)

    # Check keys are sorted
    self.assertEqual(list(result.keys()), ['a', 'b', 'c'])

    self.assertEqual(result['a'].shape, (4,))
    self.assertEqual(result['a'].dtype, np.float32)
    self.assertIsNotNone(result['a'].sharding)

    # Check 'b' is from source 1 (overwrites source 0)
    self.assertEqual(result['b'].shape, (2, 2))
    self.assertEqual(result['b'].dtype, np.float32)
    self.assertIsNotNone(result['b'].sharding)

    self.assertEqual(result['c'].shape, (1,))
    self.assertEqual(result['c'].dtype, np.float32)
    self.assertIsNotNone(result['c'].sharding)

  def test_get_global_fragments_unsharded(self):
    """Tests get_global_fragments for unsharded leaves."""
    target = jax.ShapeDtypeStruct((4,), np.float32)  # No sharding
    global_frags = merging.get_global_fragments(target)
    self.assertLen(global_frags.fragments, 1)
    self.assertEqual(global_frags.fragments[0].index, (slice(0, 4, 1),))

  @mock.patch.object(random, 'shuffle', autospec=True)
  def test_flatten_fragments(self, mock_shuffle: mock.MagicMock) -> None:
    """Tests flattening of fragment trees into FragmentInfo objects."""
    # Ensure shuffle does nothing so we can assert order deterministically
    mock_shuffle.side_effect = lambda x: x

    @dataclasses.dataclass
    class MockFragmentsContainer:
      fragments: list[AbstractFragment]
      dtype: np.dtype

    ckpt0_tree = {
        'layer1': MockFragmentsContainer(
            [
                AbstractFragment(index=(slice(0, 1),)),
                AbstractFragment(index=(slice(1, 2),)),
            ],
            np.dtype(np.float32),
        )
    }

    ckpt1_tree = {
        'layer1': MockFragmentsContainer(
            [AbstractFragment(index=(slice(2, 3),))], np.dtype(np.float32)
        ),
        'layer2': MockFragmentsContainer(
            [AbstractFragment(index=(slice(3, 4),))], np.dtype(np.int32)
        ),
    }

    required_input_fragments = [ckpt0_tree, ckpt1_tree]

    infos_by_ckpt = merging.flatten_fragments_by_checkpoint(
        required_input_fragments
    )
    fragment_groups = merging.group_and_shuffle_fragment_infos(infos_by_ckpt)

    self.assertEqual(
        fragment_groups,
        [
            [
                FragmentInfo(
                    ckpt_idx=0,
                    keypath=('layer1',),
                    fragment=AbstractFragment(index=(slice(0, 1),)),
                    dtype=np.dtype(np.float32),
                ),
                FragmentInfo(
                    ckpt_idx=0,
                    keypath=('layer1',),
                    fragment=AbstractFragment(index=(slice(1, 2),)),
                    dtype=np.dtype(np.float32),
                ),
                FragmentInfo(
                    ckpt_idx=1,
                    keypath=('layer1',),
                    fragment=AbstractFragment(index=(slice(2, 3),)),
                    dtype=np.dtype(np.float32),
                ),
            ],
            [
                FragmentInfo(
                    ckpt_idx=1,
                    keypath=('layer2',),
                    fragment=AbstractFragment(index=(slice(3, 4),)),
                    dtype=np.dtype(np.int32),
                ),
            ],
        ],
    )

  def test_get_per_host_costs_unsharded(self):
    """Tests get_per_host_costs for unsharded leaf."""
    key = ('p',)
    target_leaf = jax.ShapeDtypeStruct((100,), np.int8)  # No sharding
    group = [
        merging.FragmentInfo(
            0,
            key,
            AbstractFragment(index=(slice(0, 100, 1),)),
            np.dtype(np.int8),
        )
    ]

    with mock.patch.object(jax, 'process_count', return_value=2):
      costs = merging.get_per_host_costs(group, target_leaf)
      # Both hosts load the whole thing
      np.testing.assert_array_equal(costs, [100, 100])

  def test_batch_fragments_empty_infos(self):
    limit_bytes = 1 * GB
    groups = []
    costs = []

    batches = list(merging.batch_fragments(groups, costs, limit_bytes))

    self.assertEmpty(batches)

  @parameterized.named_parameters(
      ('single_group_less_than_limit', [[0.4]]),
      ('single_group_equal_limit', [[1.0]]),
      ('multiple_groups_less_than_limit', [[0.4], [0.4]]),
      ('multiple_groups_equal_limit', [[0.5], [0.5]]),
  )
  def test_batch_fragments_single_batch(self, multipliers):
    limit_bytes = 1 * GB
    groups = []
    costs = []
    for i, group_multipliers in enumerate(multipliers):
      size = int(limit_bytes * group_multipliers[0])
      groups.append([
          merging.FragmentInfo(
              0,
              (f'p{i}',),
              AbstractFragment(index=(slice(0, size, 1),)),
              np.dtype(np.int8),
          )
      ])
      costs.append(np.array([size], dtype=np.int64))

    batches = list(merging.batch_fragments(groups, costs, limit_bytes))

    self.assertLen(batches, 1)
    expected_batch = []
    for group in groups:
      for info in group:
        expected_batch.append(info)
    self.assertEqual(batches[0], expected_batch)

  @parameterized.named_parameters(
      ('split_after_first', [[0.6], [0.5], [0.2]], [[0], [1, 2]]),
      ('split_after_second', [[0.4], [0.4], [0.4]], [[0, 1], [2]]),
      ('split_after_third', [[0.3], [0.3], [0.3], [0.3]], [[0, 1, 2], [3]]),
      ('exact_fit_after_first', [[1.0], [0.5], [0.5]], [[0], [1, 2]]),
      ('exact_fit_after_second', [[0.5], [0.5], [0.5]], [[0, 1], [2]]),
      ('exact_fit_after_third', [[0.4], [0.4], [0.2], [0.1]], [[0, 1, 2], [3]]),
      ('split_all', [[0.9], [0.9], [0.9]], [[0], [1], [2]]),
      ('exact_fit_all', [[1.0], [1.0], [1.0]], [[0], [1], [2]]),
  )
  def test_batch_fragments_multiple_batches(
      self, multipliers, expected_group_indices
  ):
    limit_bytes = 1 * GB
    groups = []
    costs = []
    for i, group_multipliers in enumerate(multipliers):
      size = int(limit_bytes * group_multipliers[0])
      groups.append([
          merging.FragmentInfo(
              0,
              (f'p{i}',),
              AbstractFragment(index=(slice(0, size, 1),)),
              np.dtype(np.int8),
          )
      ])
      costs.append(np.array([size], dtype=np.int64))

    batches = list(merging.batch_fragments(groups, costs, limit_bytes))

    self.assertLen(batches, len(expected_group_indices))
    for i, group_indices in enumerate(expected_group_indices):
      expected_batch = []
      for idx in group_indices:
        expected_batch.extend(groups[idx])
      self.assertEqual(batches[i], expected_batch)

  @parameterized.named_parameters(
      ('single_exceeds', [[1.1]]),
      ('first_exceeds', [[1.1], [0.5]]),
      ('second_exceeds', [[0.5], [1.1]]),
      ('third_exceeds', [[0.5], [0.5], [1.1]]),
  )
  def test_batch_fragments_exceeds_limit(self, multipliers):
    limit_bytes = 1 * GB
    groups = []
    costs = []
    for i, group_multipliers in enumerate(multipliers):
      size = int(limit_bytes * group_multipliers[0])
      groups.append([
          merging.FragmentInfo(
              0,
              (f'p{i}',),
              AbstractFragment(index=(slice(0, size, 1),)),
              np.dtype(np.int8),
          )
      ])
      costs.append(np.array([size], dtype=np.int64))

    with self.assertRaisesRegex(ValueError, 'exceeds the hard limit'):
      list(merging.batch_fragments(groups, costs, limit_bytes))

  def test_load_batch_fragments(self):
    """Tests load_batch_fragments."""
    # Source 0: 'a' requested, 'b' not requested
    # Source 1: 'c' requested
    abstract_sources = [
        {
            'a': jax.ShapeDtypeStruct((4,), np.float32),
            'b': jax.ShapeDtypeStruct((2,), np.float32),
        },
        {
            'c': jax.ShapeDtypeStruct((2, 2), np.int32),
        },
    ]

    frag_a = mock.Mock(spec=AbstractFragment)
    frag_c = mock.Mock(spec=AbstractFragment)

    batch_fragments_map = {
        0: {('a',): [frag_a]},
        1: {('c',): [frag_c]},
    }

    sc0 = mock.Mock()
    sc1 = mock.Mock()
    source_checkpoints = [sc0, sc1]

    memory_limit_gb = 1

    merging.load_batch_fragments(
        abstract_sources,
        batch_fragments_map,
        source_checkpoints,
        memory_limit_gb,
    )

    # Verify Source 0 call
    self.assertEqual(sc0.load_fragments.call_count, 1)
    args0, kwargs0 = sc0.load_fragments.call_args
    self.assertEqual(kwargs0['concurrent_gb'], memory_limit_gb)

    request_tree0 = args0[0]
    # 'a' should have fragments
    self.assertEqual(request_tree0['a'].fragments, [frag_a])
    self.assertEqual(request_tree0['a'].shape, (4,))
    self.assertEqual(request_tree0['a'].dtype, np.float32)
    # 'b' should be empty fragments
    self.assertEqual(request_tree0['b'].fragments, [])
    self.assertEqual(request_tree0['b'].shape, (2,))

    # Verify Source 1 call
    self.assertEqual(sc1.load_fragments.call_count, 1)
    args1, kwargs1 = sc1.load_fragments.call_args
    self.assertEqual(kwargs1['concurrent_gb'], memory_limit_gb)

    request_tree1 = args1[0]
    # 'c' should have fragments
    self.assertEqual(request_tree1['c'].fragments, [frag_c])
    self.assertEqual(request_tree1['c'].shape, (2, 2))
    self.assertEqual(request_tree1['c'].dtype, np.int32)

  def test_load_batch_fragments_sparse_map(self):
    """Tests load_batch_fragments with missing entries in batch_fragments_map."""
    # Source 0: 'a' requested
    # Source 1: Not in batch_fragments_map (nothing requested)
    abstract_sources = [
        {
            'a': jax.ShapeDtypeStruct((4,), np.float32),
        },
        {
            'b': jax.ShapeDtypeStruct((2,), np.float32),
        },
    ]

    frag_a = mock.Mock(spec=AbstractFragment)

    # Only Source 0 has fragments in this batch
    batch_fragments_map = {
        0: {('a',): [frag_a]},
    }

    sc0 = mock.Mock()
    sc1 = mock.Mock()
    source_checkpoints = [sc0, sc1]

    merging.load_batch_fragments(
        abstract_sources,
        batch_fragments_map,
        source_checkpoints,
        memory_limit_gb=1,
    )

    # Verify Source 0 call
    self.assertEqual(sc0.load_fragments.call_count, 1)
    args0, _ = sc0.load_fragments.call_args
    self.assertEqual(args0[0]['a'].fragments, [frag_a])

    # Verify Source 1 call
    self.assertEqual(sc1.load_fragments.call_count, 1)
    args1, _ = sc1.load_fragments.call_args
    # Should request empty fragments for 'b'
    self.assertEqual(args1[0]['b'].fragments, [])

  def test_resolve_merge_plan(self):
    """Tests that resolve_merge_plan correctly identifies source fragments."""
    host_cpus = jax.devices('cpu')

    abstract_sources = [
        {
            'a': jax.ShapeDtypeStruct((4,), np.float32),
            'b': jax.ShapeDtypeStruct((2, 2), np.int32),
        },
        {
            'b': jax.ShapeDtypeStruct((2, 2), np.float32),
            'c': jax.ShapeDtypeStruct((1,), np.float32),
        },
    ]
    sharded_abstract_target = merging.resolve_target_structure(
        abstract_sources, host_cpus
    )

    required_outputs = jax.tree.map(
        array_fragments.abstract_fragments, sharded_abstract_target
    )

    required_input_fragments, _ = merging.resolve_merge_plan(
        required_outputs, abstract_sources
    )

    self.assertLen(required_input_fragments, 2)

    self.assertNotEmpty(required_input_fragments[0]['a'].fragments)
    self.assertNotEmpty(required_input_fragments[1]['b'].fragments)
    self.assertNotEmpty(required_input_fragments[1]['c'].fragments)

    # Overwritten key should have no fragments
    self.assertEmpty(required_input_fragments[0]['b'].fragments)

  def test_fragments_to_arrays_passthrough(self):
    """Tests that non-fragment leaves are passed through unchanged."""
    target = {'a': jax.ShapeDtypeStruct((1,), np.float32)}
    # If the input isn't an eval_fragments.Fragments instance, returns it as is.
    # Happens if tree structure doesn't match perfectly or for non-array leaves.
    inputs = {'a': 123}

    result = merging.fragments_to_arrays(inputs, target)
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
        merging.eval_fragments, '_extract_fragment'
    ) as mock_extract:
      mock_extract.return_value.value = mock_fragment_data

      fragments_input = array_fragments.ConcreteFragments(
          shape=shape,
          dtype=np.dtype(dtype),
          fragments=[
              array_fragments.ConcreteFragment(
                  index=(slice(0, 2, 1), slice(0, 2, 1)),
                  value=np.zeros(shape, dtype),
              )
          ],
      )

      inputs = {'param': fragments_input}

      result_tree = merging.fragments_to_arrays(inputs, target)

      self.assertIsInstance(result_tree['param'], jax.Array)
      self.assertEqual(result_tree['param'].shape, shape)
      self.assertEqual(result_tree['param'].dtype, dtype)

      # Force materialization to trigger the callback
      result_data = np.array(result_tree['param'])
      np.testing.assert_array_equal(result_data, mock_fragment_data)
      self.assertGreater(mock_extract.call_count, 0)

  def test_fragments_to_arrays_filtering(self):
    """Tests that degenerate fragments are filtered out."""
    shape = (2,)
    dtype = np.float32
    local_devices = [
        d for d in jax.devices('cpu') if d.process_index == jax.process_index()
    ]
    if not local_devices:
      self.skipTest('No local CPU devices found')
    sharding = jax.sharding.SingleDeviceSharding(local_devices[0])
    target = {
        'keep': jax.ShapeDtypeStruct(shape, dtype, sharding=sharding),
        'filter': jax.ShapeDtypeStruct(shape, dtype, sharding=sharding),
        'none': jax.ShapeDtypeStruct(shape, dtype, sharding=sharding),
    }

    # Non-degenerate fragments
    fragments_keep = array_fragments.ConcreteFragments(
        shape=shape,
        dtype=np.dtype(dtype),
        fragments=[
            array_fragments.ConcreteFragment(
                index=(slice(0, 2),),
                value=np.zeros(shape, dtype),
            )
        ],
    )
    # Degenerate fragments
    fragments_filter = array_fragments.ConcreteFragments(
        shape=shape,
        dtype=np.dtype(dtype),
        fragments=[
            array_fragments.ConcreteFragment(
                index=(slice(0, 0),),
                value=np.zeros((0,), dtype),
            )
        ],
    )

    inputs = {
        'keep': fragments_keep,
        'filter': fragments_filter,
        'none': 123,
    }

    result = merging.fragments_to_arrays(inputs, target)

    self.assertIn('keep', result)
    self.assertIsInstance(result['keep'], jax.Array)
    self.assertNotIn('filter', result)
    self.assertIn('none', result)
    self.assertEqual(result['none'], 123)

  def test_fragments_to_arrays_filtering_partial(self):
    """Tests that partial fragments (incomplete coverage) are filtered out."""
    shape = (2,)
    dtype = np.float32
    local_devices = [
        d for d in jax.devices('cpu') if d.process_index == jax.process_index()
    ]
    if not local_devices:
      self.skipTest('No local CPU devices found')
    sharding = jax.sharding.SingleDeviceSharding(local_devices[0])
    target = {
        'partial': jax.ShapeDtypeStruct(shape, dtype, sharding=sharding),
    }

    # Partial fragments: slice(0, 1) for a (2,) array
    fragments_partial = array_fragments.ConcreteFragments(
        shape=shape,
        dtype=np.dtype(dtype),
        fragments=[
            array_fragments.ConcreteFragment(
                index=(slice(0, 1),),
                value=np.zeros((1,), dtype),
            )
        ],
    )

    inputs = {
        'partial': fragments_partial,
    }

    with self.assertRaisesRegex(ValueError, 'Incomplete fragment coverage'):
      merging.fragments_to_arrays(inputs, target)


if __name__ == '__main__':
  absltest.main()
