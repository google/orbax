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
from absl.testing import parameterized
from etils import epath
import jax
import jax.experimental.multihost_utils
import jax.sharding
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
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
    self.mesh = Mesh(
        np.array(devices).reshape(mesh_shape), ("data", "model")
    )
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
      tensor_to_save = np.float32(1.0)
    else:
      num_elements = np.prod(array_shape)
      tensor_to_save = np.arange(num_elements, dtype=np.float32).reshape(
          array_shape
      )

    tensor_data = {"params.tensor": tensor_to_save}
    mesh = Mesh(np.array(jax.devices()).reshape(mesh_shape), mesh_axes)
    st_path = self.test_dir / f"{self.id()}.safetensors"
    if jax.process_index() == 0:
      np_save_file(tensor_data, st_path)
    test_utils.sync_global_processes(self.id())

    abstract_sharding = NamedSharding(mesh, sharding_spec)
    abstract_state = {
        "params.tensor": jax.ShapeDtypeStruct(
            shape=array_shape, dtype=np.float32, sharding=abstract_sharding
        ),
    }
    expected_tensor = jax.device_put(tensor_to_save, abstract_sharding)

    layout = SafetensorsLayout()
    restore_fn = await layout.load_pytree(
        st_path, abstract_pytree=abstract_state
    )
    restored_tensor = await restore_fn
    restored_tensor = restored_tensor["params.tensor"]

    self.assertEqual(restored_tensor.sharding, expected_tensor.sharding)
    test_utils.assert_array_equal(self, expected_tensor, restored_tensor)

  async def test_load_without_global_reshard_single_tensor(self):
    """Tests loading with ignore_load_sharding=True with a single tensor."""
    array_shape = (4, 4)
    tensor_to_save = np.arange(16, dtype=np.float32).reshape(array_shape)
    tensor_data = {"params.tensor": tensor_to_save}

    st_path = self.test_dir / f"{self.id()}.safetensors"
    if jax.process_index() == 0:
      np_save_file(tensor_data, st_path)
    test_utils.sync_global_processes(self.id())

    abstract_sharding = NamedSharding(self.mesh, PartitionSpec("data", "model"))
    abstract_state = {
        "params.tensor": jax.ShapeDtypeStruct(
            shape=array_shape, dtype=np.float32, sharding=abstract_sharding
        ),
    }

    layout = SafetensorsLayout()
    ctx = context_lib.Context()
    ctx.safetensors.ignore_load_sharding = True
    with ctx:
      restore_fn = await layout.load_pytree(
          st_path, abstract_pytree=abstract_state
      )
      restored_pytree = await restore_fn
    restored_tensor = restored_pytree["params.tensor"]

    self.assertEqual(restored_tensor.shape, array_shape)

    if len(restored_tensor.addressable_shards) == 1:
      np.testing.assert_array_equal(
          restored_tensor.addressable_shards[0].data, tensor_to_save
      )
    else:
      self.assertEmpty(restored_tensor.addressable_shards)

  async def test_load_without_global_reshard_multi_tensor(self):
    """Tests loading with ignore_load_sharding=True with multiple tensors."""
    array_shape = (4, 4)
    tensor_data = {
        f"params.tensor_{i}": (
            np.arange(16, dtype=np.float32).reshape(array_shape) + i
        )
        for i in range(4)
    }

    st_path = self.test_dir / f"{self.id()}.safetensors"
    if jax.process_index() == 0:
      np_save_file(tensor_data, st_path)
    test_utils.sync_global_processes(self.id())

    abstract_sharding = NamedSharding(self.mesh, PartitionSpec("data", "model"))
    abstract_state = {
        f"params.tensor_{i}": jax.ShapeDtypeStruct(
            shape=array_shape, dtype=np.float32, sharding=abstract_sharding
        )
        for i in range(4)
    }

    layout = SafetensorsLayout()
    ctx = context_lib.Context()
    ctx.safetensors.ignore_load_sharding = True
    with ctx:
      restore_fn = await layout.load_pytree(
          st_path, abstract_pytree=abstract_state
      )
      restored_pytree = await restore_fn

    # Tensors are expected to be distributed among hosts.
    # With 4 hosts and 4 equal sized tensors, each host should own one.
    for i in range(4):
      tensor_name = f"params.tensor_{i}"
      restored_tensor = restored_pytree[tensor_name]
      self.assertEqual(restored_tensor.shape, array_shape)

      if len(restored_tensor.addressable_shards) == 1:
        np.testing.assert_array_equal(
            restored_tensor.addressable_shards[0].data, tensor_data[tensor_name]
        )
      else:
        self.assertEmpty(restored_tensor.addressable_shards)

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

    restore_fn = await layout.load_pytree(
        file_path,
        abstract_pytree=abstract_pytree,
    )
    pytree = await restore_fn

    jax.block_until_ready(pytree)

    unused_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Peak memory should be dominated by owned tensors (approx 100MB).
    # Will also contain a single zero buffer of 4MB to be used in place of all
    # non-owned tensors.
    # If non-owned tensors were materialized, it would be 400MB.
    tensors_per_host = num_tensors // jax.process_count()
    expected_peak = bytes_per_tensor * (tensors_per_host + 1)
    fudge_factor = 1.2  # Account for overhead, Python objects, etc.

    self.assertLess(peak, fudge_factor * expected_peak)

  async def test_load_without_global_reshard_memory_efficiency(self):
    """Verifies that non-owner hosts don't materialize full zero buffers when ignore_load_sharding=True."""
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

    file_path = self.test_dir / "dummy_no_reshard.safetensors"

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

    ctx = context_lib.Context()
    ctx.safetensors.ignore_load_sharding = True
    with ctx:
      restore_fn = await layout.load_pytree(
          file_path,
          abstract_pytree=abstract_pytree,
      )
      pytree = await restore_fn

    jax.block_until_ready(pytree)

    unused_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Peak memory should be dominated by owned tensors (approx 100MB).
    # If non-owned tensors were materialized, it would be 400MB.
    tensors_per_host = num_tensors // jax.process_count()
    expected_peak = bytes_per_tensor * (tensors_per_host + 1)
    fudge_factor = 1.2  # Account for overhead, Python objects, etc.

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
      test_awaitable = await layout.load_pytree(
          st_path, abstract_pytree=nested_abstract_pytree
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
      test_awaitable = await layout.load_pytree(
          st_path, abstract_pytree=wrong_key_abstract_pytree
      )
      await test_awaitable


if __name__ == "__main__":
  multiprocess_test.main()
