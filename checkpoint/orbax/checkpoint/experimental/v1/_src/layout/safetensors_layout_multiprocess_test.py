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

"""Tests for sharded loading with SafetensorsLayout."""

import gc
import tracemalloc
import unittest
from unittest import mock

from absl.testing import parameterized
from etils import epath
import jax
import jax.experimental.multihost_utils
import jax.sharding
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.layout import safetensors_layout
import safetensors.numpy

SafetensorsLayout = safetensors_layout.SafetensorsLayout
np_save_file = safetensors.numpy.save_file
Mesh = jax.sharding.Mesh
NamedSharding = jax.sharding.NamedSharding
PartitionSpec = jax.sharding.PartitionSpec
jnp = jax.numpy


def _get_partition_spec(
    mesh_config, array_shape, sharding_type
) -> PartitionSpec | None:
  """Returns the partition spec for a given sharding type, or None if invalid."""
  mesh_shape, mesh_axes = mesh_config["shape"], mesh_config["axes"]
  rank = len(array_shape)
  is_scalar_like = rank == 0 or (rank == 1 and array_shape[0] == 1)
  if is_scalar_like and sharding_type != "fully_replicated":
    return None

  if sharding_type == "fully_replicated":
    pspec = PartitionSpec()
  elif sharding_type == "fully_sharded":
    if rank > len(mesh_axes):
      return None
    pspec = PartitionSpec(*mesh_axes[:rank])
  else:
    pspec_list = [mesh_axes[0]]
    for _ in range(rank - 1):
      pspec_list.append(None)
    pspec = PartitionSpec(*pspec_list)

  # Need to verify that an array dimension is divisible by the size of the
  # mesh axis.
  for i, axis_name in enumerate(pspec):
    if axis_name is not None:
      array_dim_size = array_shape[i]
      mesh_axis_index = mesh_axes.index(axis_name)
      mesh_axis_size = mesh_shape[mesh_axis_index]
      if array_dim_size % mesh_axis_size != 0:
        return None

  # If all checks pass, the combination is possible.
  return pspec


class ShardedSafetensorsLayoutTest(
    unittest.IsolatedAsyncioTestCase,
    parameterized.TestCase,
    multiprocess_test.MultiProcessTest,
):

  def setUp(self):
    super().setUp()
    self.assertEqual(jax.device_count(), 8)
    self.assertEqual(jax.process_count(), 4)
    self.assertEqual(jax.local_device_count(), 2)
    self.test_dir = epath.Path(
        self.multiprocess_create_tempdir(name="test_dir")
    )

    devices = jax.devices()
    mesh_shape = (len(devices) // 2, 2)
    self.mesh = Mesh(np.array(devices).reshape(mesh_shape), ("data", "model"))
    test_utils.sync_global_processes("setUp")

  def tearDown(self):
    super().tearDown()
    test_utils.sync_global_processes("tearDown")

  @parameterized.product(
      mesh_config=[
          {"shape": (4, 2), "axes": ("data", "model")},
          {"shape": (2, 4), "axes": ("data", "model")},
          {"shape": (8, 1), "axes": ("data", "model")},
          {"shape": (1, 8), "axes": ("data", "model")},
          {"shape": (1, 8, 1), "axes": ("d1", "d2", "d3")},
          {"shape": (1, 2, 4, 1), "axes": ("d1", "d2", "d3", "d4")},
      ],
      array_shape=[
          (),
          (1,),
          (16,),
          (8, 8),
          (4, 4, 4),
      ],
      sharding_type=[
          "fully_replicated",
          "fully_sharded",
          "one_axis_sharded",
      ],
  )
  async def test_sharding_scenarios(
      self, mesh_config, array_shape, sharding_type
  ):
    # We are skipping tests that attempt to construct an invalid sharding spec.
    # In the next test, we validate that we get the expected error message.
    mesh_shape, mesh_axes = mesh_config["shape"], mesh_config["axes"]
    sharding_spec = _get_partition_spec(mesh_config, array_shape, sharding_type)
    if sharding_spec is None:
      self.skipTest("Invalid sharding spec.")

    # Create the tensor to save
    if not array_shape:
      tensor_to_save = np.array(1.0, dtype=np.float32)
    else:
      num_elements = np.prod(array_shape)
      tensor_to_save = np.arange(num_elements, dtype=np.float32).reshape(
          array_shape
      )

    tensor_data = {"params.tensor": tensor_to_save}
    mesh = Mesh(np.array(jax.devices()).reshape(mesh_shape), mesh_axes)
    st_path = self.test_dir / f"{self.id()}.safetensors"
    if jax.process_index() == 0:
      np_save_file(tensor_data, st_path)  # pyrefly: ignore[bad-argument-type]
    test_utils.sync_global_processes(self.id())

    abstract_sharding = NamedSharding(mesh, sharding_spec)
    abstract_state = {
        "params.tensor": jax.ShapeDtypeStruct(
            shape=array_shape, dtype=np.float32, sharding=abstract_sharding
        ),
    }
    expected_tensor = jax.device_put(tensor_to_save, abstract_sharding)

    layout = SafetensorsLayout()
    restore_fn = await layout.load(st_path, abstract_state=abstract_state)
    restored_tensor = await restore_fn
    restored_tensor = restored_tensor["params.tensor"]

    self.assertEqual(restored_tensor.sharding, expected_tensor.sharding)
    test_utils.assert_array_equal(self, expected_tensor, restored_tensor)

  async def test_load_multi_host_memory_efficiency(self):
    """Verifies that non-owner hosts don't materialize full zero buffers."""
    num_tensors = 100
    tensor_shape = (1024, 1024)  # 1M elements = 4MB for float32
    # Total logical size for 100 tensors = 400MB.
    num_elements = np.prod(tensor_shape)
    bytes_per_tensor = num_elements * np.dtype(np.float32).itemsize

    abstract_sharding = NamedSharding(self.mesh, PartitionSpec("data", "model"))

    abstract_pytree = {
        f"tensor_{i}": jax.ShapeDtypeStruct(
            shape=tensor_shape, dtype=np.float32, sharding=abstract_sharding
        )
        for i in range(num_tensors)
    }

    file_path = self.test_dir / "dummy.safetensors"

    if jax.process_index() == 0:
      tensors = {
          f"tensor_{i}": np.zeros(tensor_shape, dtype=np.float32)
          for i in range(num_tensors)
      }
      safetensors.numpy.save_file(tensors, file_path)
      del tensors

      gc.collect()

    test_utils.sync_global_processes(self.id())

    layout = SafetensorsLayout()

    tracemalloc.start()

    # Pin a small in-flight budget so the loader streams reads instead of
    # holding every per-host shard concurrently. With the 2 GiB default the
    # whole per-host read set (plus coalescing over-read) is in flight at
    # once and peak runs ~2x destination size; the budget is the knob that
    # makes the peak-near-destination guarantee actually hold and testable.
    with context_lib.Context(
        memory_options=options_lib.MemoryOptions(
            read_concurrent_bytes=8 * 1024 * 1024
        )
    ):
      restore_fn = await layout.load(
          file_path,
          abstract_state=abstract_pytree,
      )
      pytree = await restore_fn

    jax.block_until_ready(pytree)

    unused_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Peak memory is dominated by this host's addressable share of every
    # tensor (~100 MB for tensors_per_host = 25 and bytes_per_tensor = 4 MB).
    # On top of that the loader holds:
    #   * a few coalesced read blocks in flight at any moment (bounded by
    #     `_MAX_CONCURRENT_READS * coalesce_max_merged_bytes`); and
    #   * the usual JAX / NumPy object overhead for hundreds of sharded
    #     `jax.Array`s and per-shard `NamedSharding` references.
    # If non-owner hosts were materializing zero buffers instead -- the
    # regression this test guards against -- the peak would be ~400 MB.
    tensors_per_host = num_tensors // jax.process_count()
    expected_peak = bytes_per_tensor * (tensors_per_host + 1)
    fudge_factor = 1.5  # in-flight reads + JAX object overhead

    self.assertLess(peak, fudge_factor * expected_peak)

  def test_sharding_fails_when_divisibility_check_fails(self):
    """Tests that JAX errors when an array dim is not divisible by a mesh dim."""
    mesh_config = {"shape": (8, 1), "axes": ("data", "model")}
    array_shape = (
        4,
        4,
        4,
    )  # Dimension size 4 is not divisible by mesh axis data (size 8)
    sharding_spec = PartitionSpec("data", None, None)

    mesh = Mesh(
        np.array(jax.devices()).reshape(mesh_config["shape"]),
        mesh_config["axes"],
    )
    abstract_sharding = NamedSharding(mesh, sharding_spec)
    tensor_to_save = np.zeros(array_shape, dtype=np.float32)

    with self.assertRaisesRegex(
        ValueError, "partitioned 8 times, but the dimension size is 4"
    ):
      jax.device_put(tensor_to_save, abstract_sharding)

  def test_sharding_fails_with_scalar(self):
    """Tests that JAX errors when attempting to shard a scalar."""
    mesh_config = {"shape": (4, 2), "axes": ("data", "model")}
    sharding_spec = PartitionSpec("data")

    mesh = Mesh(
        np.array(jax.devices()).reshape(mesh_config["shape"]),
        mesh_config["axes"],
    )
    abstract_sharding = NamedSharding(mesh, sharding_spec)
    tensor_to_save = np.float32(1.0)
    with self.assertRaisesRegex(
        ValueError, "For scalars the PartitionSpec should be P()"
    ):
      jax.device_put(tensor_to_save, abstract_sharding)

  def test_sharding_fails_with_non_existent_axes(self):
    """Tests that JAX errors when the PartitionSpec references non-existent mesh axes."""
    mesh_config = {"shape": (4, 2), "axes": ("data", "model")}
    # d3 does not exist in the mesh axes
    sharding_spec = PartitionSpec("data", "model", "d3")

    mesh = Mesh(
        np.array(jax.devices()).reshape(mesh_config["shape"]),
        mesh_config["axes"],
    )

    with self.assertRaisesRegex(ValueError, "is not found in mesh"):
      _ = NamedSharding(mesh, sharding_spec)

  async def test_load_sharded_fails_with_nested_abstract_pytree(self):
    """Tests that loading fails if the abstract pytree is nested."""
    st_path = self.test_dir / "nested_fail.safetensors"
    if jax.process_index() == 0:
      np_save_file({"a": np.arange(8)}, st_path)
    test_utils.sync_global_processes(
        "test_load_sharded_fails_with_nested_abstract_pytree"
    )
    layout = SafetensorsLayout()

    nested_abstract_pytree = {
        "params": {
            "a": jax.ShapeDtypeStruct(
                shape=(8,),
                dtype=np.int32,
                sharding=NamedSharding(self.mesh, PartitionSpec()),
            ),
        }
    }
    with self.assertRaisesRegex(
        ValueError, "The PyTree is not a flat dictionary."
    ):
      test_awaitable = await layout.load(
          st_path, abstract_state=nested_abstract_pytree
      )
      await test_awaitable

  async def test_load_sharded_fails_with_wrong_key_abstract_pytree(self):
    """Tests that loading fails if a key in the abstract pytree is not in the file."""
    st_path = self.test_dir / "wrong_key_fail.safetensors"
    if jax.process_index() == 0:
      np_save_file({"a": np.arange(8)}, st_path)
    test_utils.sync_global_processes(
        "test_load_sharded_fails_with_wrong_key_abstract_pytree"
    )

    layout = SafetensorsLayout()

    wrong_key_abstract_pytree = {
        "a": jax.ShapeDtypeStruct(
            shape=(8,),
            dtype=np.int32,
            sharding=NamedSharding(self.mesh, PartitionSpec()),
        ),
        "c": jax.ShapeDtypeStruct(shape=(3,), dtype=np.float32),  # Wrong key
    }

    with self.assertRaisesRegex(
        KeyError, "not found in Safetensors checkpoint"
    ):
      test_awaitable = await layout.load(
          st_path, abstract_state=wrong_key_abstract_pytree
      )
      await test_awaitable


class BroadcastReplicatedTest(
    unittest.IsolatedAsyncioTestCase,
    parameterized.TestCase,
    multiprocess_test.MultiProcessTest,
):
  """`SafetensorsOptions.broadcast_replicated` on the 4-process mesh."""

  def setUp(self):
    super().setUp()
    self.test_dir = epath.Path(
        self.multiprocess_create_tempdir(name="test_dir")
    )
    devices = jax.devices()
    self.mesh = Mesh(np.array(devices).reshape(4, 2), ("data", "model"))
    test_utils.sync_global_processes("BroadcastReplicatedTest.setUp")

  def tearDown(self):
    super().tearDown()
    test_utils.sync_global_processes("BroadcastReplicatedTest.tearDown")

  def _assert_domains_cover_once(self, sharding, shape):
    """Every element owned by exactly one unique index domain (no replicas)."""
    domains = {
        safetensors_layout._normalize_index(index, shape)
        for index in sharding.devices_indices_map(shape).values()
    }
    covered = sum(
        np.prod([stop - start for start, stop in bounds]) for bounds in domains
    )
    self.assertEqual(covered, np.prod(shape))

  def test_dedup_fully_replicated_covers_once(self):
    sharding = NamedSharding(self.mesh, PartitionSpec())
    read = safetensors_layout._dedup_read_sharding(sharding, (16, 8))
    self.assertIsNotNone(read)
    self._assert_domains_cover_once(read, (16, 8))

  def test_dedup_partially_replicated_covers_once(self):
    sharding = NamedSharding(self.mesh, PartitionSpec(None, "model"))
    read = safetensors_layout._dedup_read_sharding(sharding, (16, 8))
    self.assertIsNotNone(read)
    self._assert_domains_cover_once(read, (16, 8))

  def test_dedup_places_axis_on_later_divisible_dim(self):
    # dim 0 (size 3) cannot absorb either mesh axis; dim 1 (size 8) takes
    # both. JAX requires even partitions, so placement must skip dim 0.
    sharding = NamedSharding(self.mesh, PartitionSpec())
    read = safetensors_layout._dedup_read_sharding(sharding, (3, 8))
    self.assertIsNotNone(read)
    self._assert_domains_cover_once(read, (3, 8))

  def test_dedup_fully_sharded_returns_none(self):
    sharding = NamedSharding(self.mesh, PartitionSpec("data", "model"))
    self.assertIsNone(
        safetensors_layout._dedup_read_sharding(sharding, (16, 8))
    )

  def test_dedup_nothing_placeable_returns_none(self):
    # No dimension of a (3, 3) tensor divides evenly by either mesh axis.
    sharding = NamedSharding(self.mesh, PartitionSpec())
    self.assertIsNone(
        safetensors_layout._dedup_read_sharding(sharding, (3, 3))
    )

  def test_dedup_partial_placement_still_reduces_replicas(self):
    # Only "model" (size 2) fits a (2, 2) tensor; "data" stays unplaced, so
    # 2 unique domains remain across 8 devices instead of 1 -- a partial
    # de-duplication (4 owners per domain down from 8).
    sharding = NamedSharding(self.mesh, PartitionSpec())
    read = safetensors_layout._dedup_read_sharding(sharding, (2, 2))
    self.assertIsNotNone(read)
    domains = {
        safetensors_layout._normalize_index(index, (2, 2))
        for index in read.devices_indices_map((2, 2)).values()
    }
    self.assertLen(domains, 2)

  @parameterized.named_parameters(
      ("fully_replicated", PartitionSpec(), (16, 512)),
      ("partially_replicated", PartitionSpec(None, "model"), (16, 512)),
      ("indivisible_leading_dim", PartitionSpec(), (3, 8)),
  )
  async def test_load_with_broadcast_matches_direct_device_put(
      self, spec, shape
  ):
    tensor = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    st_path = self.test_dir / f"{self.id()}.safetensors"
    if jax.process_index() == 0:
      np_save_file({"w": tensor}, st_path)
    test_utils.sync_global_processes(self.id())

    target = NamedSharding(self.mesh, spec)
    abstract_state = {
        "w": jax.ShapeDtypeStruct(
            shape=shape, dtype=np.float32, sharding=target
        )
    }
    layout = SafetensorsLayout()
    with context_lib.Context(
        safetensors_options=options_lib.SafetensorsOptions(
            broadcast_replicated=True,
        ),
    ):
      restore_fn = await layout.load(st_path, abstract_state=abstract_state)
      restored = (await restore_fn)["w"]

    expected = jax.device_put(tensor, target)
    self.assertEqual(restored.sharding, expected.sharding)
    test_utils.assert_array_equal(self, expected, restored)

  async def test_broadcast_reads_each_byte_once_per_cluster(self):
    # Fully replicated target: without the flag every process reads the
    # whole tensor (4x cluster egress); with it, the read spreads so each
    # process pulls ~1/4. Each process asserts its own bytes_read metric.
    shape = (16, 4096)
    tensor = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    total_bytes = tensor.nbytes
    st_path = self.test_dir / f"{self.id()}.safetensors"
    if jax.process_index() == 0:
      np_save_file({"w": tensor}, st_path)
    test_utils.sync_global_processes(self.id())

    target = NamedSharding(self.mesh, PartitionSpec())
    abstract_state = {
        "w": jax.ShapeDtypeStruct(
            shape=shape, dtype=np.float32, sharding=target
        )
    }
    layout = SafetensorsLayout()

    async def bytes_read_for(broadcast_replicated):
      with mock.patch.object(jax.monitoring, "record_scalar") as rec:
        with context_lib.Context(
            safetensors_options=options_lib.SafetensorsOptions(
                broadcast_replicated=broadcast_replicated,
            ),
        ):
          restore_fn = await layout.load(
              st_path, abstract_state=abstract_state
          )
          restored = (await restore_fn)["w"]
      test_utils.assert_array_equal(
          self, jax.device_put(tensor, target), restored
      )
      emitted = {c.args[0]: c.args[1] for c in rec.call_args_list}
      return emitted["/jax/orbax/read/safetensors/bytes_read"]

    replicated_bytes = await bytes_read_for(False)
    test_utils.sync_global_processes(f"{self.id()}.baseline")
    deduped_bytes = await bytes_read_for(True)

    self.assertEqual(replicated_bytes, total_bytes)
    # This process's share is 1/4 of the tensor; allow coalescing slack.
    self.assertLessEqual(deduped_bytes, 0.3 * total_bytes)


if __name__ == "__main__":
  multiprocess_test.main()
