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

"""Sharding related utils."""

import jax
from jax.lib import xla_client
from orbax.experimental.model import core as obm


def hlo_sharding_to_op_sharding(
    hlo_sharding: xla_client.HloSharding | None,
) -> obm.OpSharding | None:
  """Converts `HloSharding` to proto `OpSharding`.

  `to_proto` is defined
  [here](

  Args:
    hlo_sharding: an `HloSharding`.

  Returns:
    An `OpSharding` proto.
  """
  if hlo_sharding is None:
    return None
  output = obm.OpSharding()
  # Note: `to_proto(self) -> xla_extension.OpSharding` so we need to serialize
  # and then parse.
  # TODO(b/364954510): See if
  # help us share messages between Python and C++.
  output.ParseFromString(hlo_sharding.to_proto().SerializeToString())
  return output


def jax_mesh_to_obm_device_mesh(
    jax_mesh: jax.sharding.Mesh,
) -> obm.manifest_pb2.DeviceMesh:
  """Converts `jax.sharding.Mesh` to proto `DeviceMesh`."""
  output = obm.manifest_pb2.DeviceMesh()
  for axis_name, axis_size in jax_mesh.shape.items():
    output.axis.append(
        obm.manifest_pb2.DeviceMesh.Axis(name=axis_name, size=axis_size)
    )
  return output


def jax_mesh_to_obm_device_assignment_by_coords(
    jax_mesh: jax.sharding.Mesh,
) -> obm.manifest_pb2.DeviceAssignmentByCoords:
  """Converts `jax.sharding.Mesh` to proto `DeviceAssignmentByCoords`."""
  def spec_for_device(d):
    if d.platform == "tpu":
      return obm.manifest_pb2.DeviceAssignmentByCoords.Device(
          id=d.id, coords=d.coords, core_on_chip=d.core_on_chip
      )
    return obm.manifest_pb2.DeviceAssignmentByCoords.Device(id=d.id)

  return obm.manifest_pb2.DeviceAssignmentByCoords(
      devices=[spec_for_device(d) for d in jax_mesh.devices.flat]
  )
