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

from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.arrays import fragments as array_fragments
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.partial import merging
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils


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
    self.pytree, self.abstract_pytree = array_test_utils.create_sharded_pytree()

    self.context = context_lib.Context(
        multiprocessing_options=options_lib.MultiprocessingOptions(
            primary_host=0,
            barrier_sync_key_prefix='PartialSavingTest',
        )
    )

    test_utils.set_tensorstore_driver_for_test()
    test_utils.sync_global_processes('PartialMergingTest:setUp:complete')

  def tearDown(self):
    super().tearDown()
    test_utils.sync_global_processes('PartialMergingTest:tearDown:complete')

  @property
  def primary_host(self):
    return self.context.multiprocessing_options.primary_host

  def test_resolve_pytree_path_valid(self):
    if multihost.is_primary_host(self.primary_host):
      (self.directory / merging.PYTREE_CHECKPOINTABLE_KEY).mkdir()
    test_utils.sync_global_processes(
        'test_resolve_pytree_path_valid:make_pytree_checkpointable'
    )

    result = merging.resolve_pytree_path(self.directory)
    self.assertEqual(result, self.directory / merging.PYTREE_CHECKPOINTABLE_KEY)

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

  def test_get_global_fragments(self):
    """Tests get_global_fragments returns all shards globally."""
    shape = (4,)
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest('Need at least 2 devices')

    mesh = jax.sharding.Mesh(np.array(devices[:2]), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x'))
    target = jax.ShapeDtypeStruct(shape, np.float32, sharding=sharding)

    global_frags = merging.get_global_fragments(target)

    # Should have 2 fragments (one for each device's shard)
    self.assertLen(global_frags.fragments, 2)
    indices = [f.index for f in global_frags.fragments]
    self.assertIn((slice(0, 2, 1),), indices)
    self.assertIn((slice(2, 4, 1),), indices)

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

  def test_resolve_merge_plan_consistency_global(self):
    """Verifies that get_global_fragments produces identical plans across hosts."""
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest('Need at least 2 devices')

    mesh = jax.sharding.Mesh(np.array(devices[:2]), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x'))
    # Use a dict target to match the sources structure more realistically.
    target_tree = {
        'a': jax.ShapeDtypeStruct((4,), np.float32, sharding=sharding)
    }
    sources = [{'a': jax.ShapeDtypeStruct((4,), np.float32)}]

    # Simulate Host 0
    with mock.patch.object(jax, 'process_index', return_value=0):
      # get_global_fragments identifies all shards
      required_outputs_h0 = merging.get_global_fragments(target_tree)
      plan_global_h0, _ = merging.resolve_merge_plan(
          required_outputs_h0, sources
      )

    # Simulate Host 1
    with mock.patch.object(jax, 'process_index', return_value=1):
      required_outputs_h1 = merging.get_global_fragments(target_tree)
      plan_global_h1, _ = merging.resolve_merge_plan(
          required_outputs_h1, sources
      )

    self.assertEqual(plan_global_h0, plan_global_h1)
    # eval_fragments optimizes contiguous fragments into a single range.
    self.assertLen(plan_global_h0[0]['a'].fragments, 1)
    self.assertEqual(
        plan_global_h0[0]['a'].fragments[0].index, (slice(0, 4, 1),)
    )

  def test_resolve_merge_plan_inconsistency_local(self):
    """Verifies that using local shards directly leads to inconsistent plans."""
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest('Need at least 2 devices')

    mesh = jax.sharding.Mesh(np.array(devices[:2]), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x'))
    target_tree = {
        'a': jax.ShapeDtypeStruct((4,), np.float32, sharding=sharding)
    }
    sources = [{'a': jax.ShapeDtypeStruct((4,), np.float32)}]

    # Without get_global_fragments, they would differ if we used local shards
    with mock.patch.object(jax, 'process_index', return_value=0):
      with mock.patch.object(
          array_fragments,
          'addressable_shards',
          return_value=[(slice(0, 2, 1),)],
      ):
        required_outputs_local_h0 = jax.tree.map(
            array_fragments.abstract_fragments, target_tree
        )
        plan_local_h0, _ = merging.resolve_merge_plan(
            required_outputs_local_h0, sources
        )

    with mock.patch.object(jax, 'process_index', return_value=1):
      with mock.patch.object(
          array_fragments,
          'addressable_shards',
          return_value=[(slice(2, 4, 1),)],
      ):
        required_outputs_local_h1 = jax.tree.map(
            array_fragments.abstract_fragments, target_tree
        )
        plan_local_h1, _ = merging.resolve_merge_plan(
            required_outputs_local_h1, sources
        )

    self.assertNotEqual(plan_local_h0, plan_local_h1)

  def test_get_per_host_costs_sharded(self):
    """Tests get_per_host_costs for sharded leaf."""
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest('Need 2 devices')

    mesh = jax.sharding.Mesh(np.array(devices[:2]), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x'))
    # Host 0: [0, 50), Host 1: [50, 100)
    target_leaf = jax.ShapeDtypeStruct((100,), np.int8, sharding=sharding)

    # Fragment overlaps with only Shard 0
    finfo0 = merging.FragmentInfo(
        0, ('p',), AbstractFragment(index=(slice(0, 25, 1),)), np.dtype(np.int8)
    )
    # Fragment overlaps with only Shard 1
    finfo1 = merging.FragmentInfo(
        0,
        ('p',),
        AbstractFragment(index=(slice(75, 100, 1),)),
        np.dtype(np.int8),
    )
    # Fragment overlaps with both
    finfo2 = merging.FragmentInfo(
        0,
        ('p',),
        AbstractFragment(index=(slice(40, 60, 1),)),
        np.dtype(np.int8),
    )

    with mock.patch.object(jax, 'process_count', return_value=2):
      costs0 = merging.get_per_host_costs([finfo0], target_leaf)
      np.testing.assert_array_equal(costs0, [25, 0])

      costs1 = merging.get_per_host_costs([finfo1], target_leaf)
      np.testing.assert_array_equal(costs1, [0, 25])

      costs2 = merging.get_per_host_costs([finfo2], target_leaf)
      np.testing.assert_array_equal(costs2, [20, 20])

  def test_batch_fragments_empty_infos(self):
    limit_bytes = 1 * GB
    groups = []
    costs = []

    batches = list(merging.batch_fragments(groups, costs, limit_bytes))

    self.assertEmpty(batches)

  def test_batch_fragments_non_uniform_sharding(self):
    """Tests batching with non-uniform shards across hosts."""
    limit_bytes = 100
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest('Need 2 devices')

    # Array 0: Heavily skewed to Host 0.
    # Host 0: 80 bytes, Host 1: 20 bytes.
    costs0 = np.array([80, 20], dtype=np.int64)
    groups = [[
        merging.FragmentInfo(
            0,
            ('a',),
            AbstractFragment(index=(slice(0, 100, 1),)),
            np.dtype(np.int8),
        )
    ]]

    # Array 1: Balanced.
    # Host 0: 30 bytes, Host 1: 30 bytes.
    costs1 = np.array([30, 30], dtype=np.int64)
    groups.append([
        merging.FragmentInfo(
            0,
            ('b',),
            AbstractFragment(index=(slice(0, 60, 1),)),
            np.dtype(np.int8),
        )
    ])

    costs = [costs0, costs1]

    # Simulation on 2 hosts.
    with mock.patch.object(jax, 'process_count', return_value=2):
      batches = list(merging.batch_fragments(groups, costs, limit_bytes))

      # Actual Host 0 usage:
      # a: 80. b: 30. Total = 110. (Exceeds 100)
      # So they MUST be in separate batches.
      self.assertLen(batches, 2)
      self.assertEqual(batches[0], groups[0])
      self.assertEqual(batches[1], groups[1])

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

  def test_fragments_to_arrays_missing_local_shard_error(self):
    """Tests that fragments_to_arrays raises error if a local shard is missing."""
    shape = (4,)
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest('Need at least 2 devices')

    mesh = jax.sharding.Mesh(np.array(devices[:2]), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x'))
    target_leaf = jax.ShapeDtypeStruct(shape, np.float32, sharding=sharding)
    target = {'p': target_leaf}

    # If we are Host 0, we need shard 0: slice(0, 2)
    # Provide ONLY shard 1: slice(2, 4)
    frags = array_fragments.ConcreteFragments(
        shape=shape,
        dtype=np.dtype(np.float32),
        fragments=[
            array_fragments.ConcreteFragment(
                index=(slice(2, 4),),
                value=np.zeros((2,), np.float32),
            )
        ],
    )

    with mock.patch.object(jax, 'process_index', return_value=0):
      with mock.patch.object(
          array_fragments,
          'addressable_shards',
          return_value=[(slice(0, 2, 1),)],
      ):
        with self.assertRaisesRegex(
            ValueError, 'Incomplete fragment coverage.*local shard'
        ):
          merging.fragments_to_arrays({'p': frags}, target)


if __name__ == '__main__':
  multiprocess_test.main()
