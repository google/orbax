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
from jax.experimental.topologies import get_topology_desc
import numpy as np
from orbax.experimental.model import core as obm
from orbax.experimental.model.jax2obm import sharding
from tensorflow.python.util.protobuf import compare
from google.protobuf import text_format
from absl.testing import absltest


class ShardingTest(absltest.TestCase):

  def get_mesh(self):
    def _create_mesh(devices) -> jax.sharding.Mesh:
      return jax.sharding.Mesh(devices, ('x', 'y'))

    return _create_mesh(
        np.array(get_topology_desc('df=4x2').devices).reshape(4, 4)
    )

  def test_jax_mesh_to_obm_device_mesh(self):
    device_mesh = sharding.jax_mesh_to_obm_device_mesh(self.get_mesh())
    expected_device_mesh_text = """
      axis {
        name: "x"
        size: 4
      }
      axis {
        name: "y"
        size: 4
      }
    """
    expected_device_mesh = text_format.Parse(
        expected_device_mesh_text, obm.manifest_pb2.DeviceMesh()
    )
    compare.assertProtoEqual(
        self,
        device_mesh,
        expected_device_mesh,
    )

  def test_jax_mesh_to_obm_device_assignment_by_coords(self):
    device_assignment_by_coords = (
        sharding.jax_mesh_to_obm_device_assignment_by_coords(self.get_mesh())
    )
    expected_device_assignment_by_coords_text = """
        devices {
          coords: 0
          coords: 0
          coords: 0
          core_on_chip: 0
        }
        devices {
          id: 1
          coords: 0
          coords: 0
          coords: 0
          core_on_chip: 1
        }
        devices {
          id: 2
          coords: 1
          coords: 0
          coords: 0
          core_on_chip: 0
        }
        devices {
          id: 3
          coords: 1
          coords: 0
          coords: 0
          core_on_chip: 1
        }
        devices {
          id: 8
          coords: 0
          coords: 1
          coords: 0
          core_on_chip: 0
        }
        devices {
          id: 9
          coords: 0
          coords: 1
          coords: 0
          core_on_chip: 1
        }
        devices {
          id: 10
          coords: 1
          coords: 1
          coords: 0
          core_on_chip: 0
        }
        devices {
          id: 11
          coords: 1
          coords: 1
          coords: 0
          core_on_chip: 1
        }
        devices {
          id: 4
          coords: 2
          coords: 0
          coords: 0
          core_on_chip: 0
        }
        devices {
          id: 5
          coords: 2
          coords: 0
          coords: 0
          core_on_chip: 1
        }
        devices {
          id: 6
          coords: 3
          coords: 0
          coords: 0
          core_on_chip: 0
        }
        devices {
          id: 7
          coords: 3
          coords: 0
          coords: 0
          core_on_chip: 1
        }
        devices {
          id: 12
          coords: 2
          coords: 1
          coords: 0
          core_on_chip: 0
        }
        devices {
          id: 13
          coords: 2
          coords: 1
          coords: 0
          core_on_chip: 1
        }
        devices {
          id: 14
          coords: 3
          coords: 1
          coords: 0
          core_on_chip: 0
        }
        devices {
          id: 15
          coords: 3
          coords: 1
          coords: 0
          core_on_chip: 1
        }
    """
    expected_device_assignment_by_coords = text_format.Parse(
        expected_device_assignment_by_coords_text,
        obm.manifest_pb2.DeviceAssignmentByCoords(),
    )
    compare.assertProtoEqual(
        self,
        device_assignment_by_coords,
        expected_device_assignment_by_coords,
    )


if __name__ == '__main__':
  absltest.main()
