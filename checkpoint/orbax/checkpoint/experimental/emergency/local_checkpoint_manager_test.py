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

import jax
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint.experimental.emergency.test_utils import test_base


@test_base.barrier_compatible_test
class LocalCheckpointManagerTest(
    test_base.LocalCheckpointManagerTestBase.Test,
    multiprocess_test.MultiProcessTest,
):

  def make_global_mesh(self) -> jax.sharding.Mesh:
    self.assertEqual(jax.device_count(), 8)
    self.assertEqual(jax.process_count(), 4)
    self.assertEqual(jax.local_device_count(), 2)

    # setup global mesh info for 2-slice tests
    slice_processes = [{0, 1}, {2, 3}]
    return test_base.get_fake_global_mesh_for_slices(slice_processes)


if __name__ == '__main__':
  multiprocess_test.main()
