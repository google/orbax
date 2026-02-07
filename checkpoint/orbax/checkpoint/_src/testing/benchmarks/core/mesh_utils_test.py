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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.testing.benchmarks.core import mesh_utils


class MeshUtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(replica_axis_index=0),
      dict(replica_axis_index=1),
  )
  @mock.patch.object(multislice, 'local_replica_devices', autospec=True)
  def test_get_local_replica_mesh_succeeds(
      self, mock_local_replica_devices, replica_axis_index
  ):
    devices = jax.devices()
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape((len(devices), 1)), ('replica', 'data')
    )

    mock_local_replica_devices.return_value = np.array([devices[0]])
    local_replica_mesh = mesh_utils.get_local_replica_mesh(
        mesh, replica_axis_index=replica_axis_index
    )
    self.assertEqual(local_replica_mesh.axis_names, ('replica', 'data'))
    self.assertEqual(local_replica_mesh.devices.shape, (len(devices), 1))
    mock_local_replica_devices.assert_called_with(
        mesh, replica_axis_index=replica_axis_index
    )


if __name__ == '__main__':
  absltest.main()
