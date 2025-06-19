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

import jax
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint.experimental.emergency.test_utils import test_base


@test_base.barrier_compatible_test
class CheckpointManagerTest(
    test_base.CheckpointManagerTestBase.Test,
    multiprocess_test.MultiProcessTest,
):

  def make_global_mesh(self, replica_axis_index: int = 0) -> jax.sharding.Mesh:
    if replica_axis_index not in [0, 1]:
      raise ValueError(
          'replica_axis_index must be 0 or 1 for this test. Got: %s'
          % replica_axis_index
      )
    self.assertEqual(jax.device_count(), 8)
    self.assertEqual(jax.process_count(), 4)
    self.assertEqual(jax.local_device_count(), 2)

    # setup global mesh info for 2-slice tests
    slice_processes = [{0, 1}, {2, 3}]
    mesh = test_base.get_fake_global_mesh_for_slices(
        slice_processes, replica_axis_index
    )
    if replica_axis_index == 0:
      assert mesh.devices.shape == (2, 4), mesh.devices.shape
    if replica_axis_index == 1:
      assert mesh.devices.shape == (4, 2), mesh.devices.shape
    return mesh


if __name__ == '__main__':
  multiprocess_test.main()
