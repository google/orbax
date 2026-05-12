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

"""Multi-process test for replica_slices."""

import jax
from jax import sharding
import numpy as np
from orbax.checkpoint._src.serialization import replica_slices
from orbax.checkpoint._src.testing import multiprocess_test

PartitionSpec = sharding.PartitionSpec
NamedSharding = sharding.NamedSharding


def make_multi_device_array():
  """Creates a replicated array across the multi-host TPU mesh."""
  devices = np.array(jax.devices()).reshape((len(jax.devices()),))
  mesh = jax.sharding.Mesh(devices, axis_names=('x',))
  spec = PartitionSpec()
  shape = (4096,)
  arr = jax.make_array_from_callback(
      shape,
      NamedSharding(mesh, spec),
      lambda idx: np.zeros(
          tuple(len(range(*s.indices(shape[i]))) for i, s in enumerate(idx))
      ),
  )
  return arr


class ReplicaSlicesMultiProcessTest(multiprocess_test.MultiProcessTest):

  def test_replica_parallel_sub_slicing(self):
    arr = make_multi_device_array()

    filtered_arrays, _ = replica_slices.filter_arrays_to_replica(
        [arr],
        replica_id=0,
        use_replica_parallel=True,
    )

    filtered = filtered_arrays[0]
    # Under multi-process execution, each host holds only a fraction of the
    # local shards in the mesh, so the filtered size on this host will be
    # strictly smaller than the whole original array!
    self.assertEqual(filtered.size, arr.size / 2)


if __name__ == '__main__':
  multiprocess_test.main()
