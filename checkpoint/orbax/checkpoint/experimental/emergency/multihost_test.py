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

"""Multihost test for emergency checkpoint utils.
"""

import copy
import logging
import math
import sys

from absl.testing import absltest
import jax
import numpy as np
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint.experimental.emergency import multihost as emergency_multihost


def create_global_mesh(mesh_shape, axis_names, devices=None):
  devices = devices or sorted(jax.devices(), key=lambda d: d.id)
  size = math.prod(mesh_shape)
  if len(devices) < size:
    raise ValueError(f"Test requires {size} global devices.")

  mesh_devices = np.array(devices[:size]).reshape(mesh_shape)
  global_mesh = jax.sharding.Mesh(mesh_devices, axis_names)
  return global_mesh


def find_start_index_where(array, predicate):
  for i, element in enumerate(array):
    if predicate(element):
      return i
  raise ValueError("No element in array satisfies predicate.")


# Declare as global variable so that it can be overridden.
class MultihostTest(absltest.TestCase):
  pass


def get_test_class(base):
  class MultihostTestConcrete(base):

    def setUp(self):
      super().setUp()
      logging.info("Jax devices: %s", jax.devices())
      logging.info("Jax local devices: %s", jax.local_devices())
      multihost.initialize_distributed_to_device_ids()

    # If all the ids are the same across restarts, the mesh should be the same.
    def test_consistent_restore_mesh_all_ids_same(self):
      input_mesh = create_global_mesh((jax.device_count() // 2, 2), ("x", "y"))
      # All ids are the same across restarts.
      previous_flattened_mesh_device_ids = [
          d.id for d in input_mesh.devices.flatten()
      ]

      transformed_mesh = emergency_multihost.consistent_restore_mesh(
          jax.devices(),
          input_mesh,
          previous_flattened_mesh_device_ids,
          multihost.distributed_to_device_ids(),
          multihost.distributed_to_device_ids(),
      )

      # All ids are the same, so transformed mesh should be the same.
      self.assertEqual(input_mesh, transformed_mesh)

    # Suppose that the processes 0 and 1 are swapped.
    def test_consistent_restore_mesh_simple(self):
      input_mesh = create_global_mesh((jax.device_count() // 2, 2), ("x", "y"))

      # Swap process 0 and 1.
      previous_distributed_to_device_ids = copy.deepcopy(
          multihost.distributed_to_device_ids()
      )
      (
          previous_distributed_to_device_ids[0],
          previous_distributed_to_device_ids[1],
      ) = (
          previous_distributed_to_device_ids[1],
          previous_distributed_to_device_ids[0],
      )

      # The mesh is constructed in the same way.
      previous_flattened_mesh_device_ids = [
          d.id for d in input_mesh.devices.flatten()
      ]
      transformed_mesh = emergency_multihost.consistent_restore_mesh(
          jax.devices(),
          input_mesh,
          previous_flattened_mesh_device_ids,
          previous_distributed_to_device_ids,
          multihost.distributed_to_device_ids(),
      )

      # Process 0 and 1 are swapped.
      current_devices = jax.devices()
      num_local_devices = jax.local_device_count()
      past_devices = current_devices.copy()
      process_0_start_index = find_start_index_where(
          past_devices,
          lambda d: d.id in previous_distributed_to_device_ids[0],
      )
      process_1_start_index = find_start_index_where(
          past_devices,
          lambda d: d.id in previous_distributed_to_device_ids[1],
      )
      for i in range(num_local_devices):
        (
            past_devices[process_0_start_index + i],
            past_devices[process_1_start_index + i],
        ) = (
            past_devices[process_1_start_index + i],
            past_devices[process_0_start_index + i],
        )
      expected_mesh = create_global_mesh(
          (jax.device_count() // 2, 2), ("x", "y"), past_devices
      )
      logging.info("Input mesh devices: %s", input_mesh.devices)
      logging.info("Transformed mesh devices: %s", transformed_mesh.devices)
      logging.info("Expected mesh devices: %s", expected_mesh.devices)

      self.assertEqual(expected_mesh, transformed_mesh)

    # Suppose that the first and last processes are swapped.
    # This is useful for testing multi-slice logic where the first and last
    # processes are usually on different slices.
    def test_consistent_restore_mesh_swap_first_and_last_process(self):
      input_mesh = create_global_mesh((jax.device_count() // 2, 2), ("x", "y"))

      # Swap process 0 and n-1 (last process).
      previous_distributed_to_device_ids = copy.deepcopy(
          multihost.distributed_to_device_ids()
      )
      (
          previous_distributed_to_device_ids[0],
          previous_distributed_to_device_ids[-1],
      ) = (
          previous_distributed_to_device_ids[-1],
          previous_distributed_to_device_ids[0],
      )

      # The mesh is constructed in the same way.
      previous_flattened_mesh_device_ids = [
          d.id for d in input_mesh.devices.flatten()
      ]
      transformed_mesh = emergency_multihost.consistent_restore_mesh(
          jax.devices(),
          input_mesh,
          previous_flattened_mesh_device_ids,
          previous_distributed_to_device_ids,
          multihost.distributed_to_device_ids(),
      )

      # Process 0 and n-1 (last) are swapped.
      current_devices = jax.devices()
      num_local_devices = jax.local_device_count()
      past_devices = current_devices.copy()

      first_process_start_index = find_start_index_where(
          past_devices,
          lambda d: d.id in previous_distributed_to_device_ids[0],
      )
      last_process_start_index = find_start_index_where(
          past_devices,
          lambda d: d.id in previous_distributed_to_device_ids[-1],
      )
      for i in range(num_local_devices):
        (
            past_devices[first_process_start_index + i],
            past_devices[last_process_start_index + i],
        ) = (
            past_devices[last_process_start_index + i],
            past_devices[first_process_start_index + i],
        )
      expected_mesh = create_global_mesh(
          (jax.device_count() // 2, 2), ("x", "y"), past_devices
      )
      logging.info("Input mesh devices: %s", input_mesh.devices)
      logging.info("Transformed mesh devices: %s", transformed_mesh.devices)

      self.assertEqual(expected_mesh, transformed_mesh)

  # Override the declared class.
  global MultihostTest
  MultihostTest = MultihostTestConcrete


# Use multi-process (single slice) test class by default.
get_test_class(multiprocess_test.MultiProcessTest)


if __name__ == "__main__":
  # Use Megascale test driver if megascale flags are set.
  if any(True for _ in filter(lambda x: x.startswith("--megascale"), sys.argv)):
    get_test_class(absltest.TestCase)
    absltest.main()
  else:
    multiprocess_test.main()
