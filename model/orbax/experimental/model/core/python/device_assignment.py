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

"""DeviceAssignment and its utilities."""

from collections.abc import Sequence
import dataclasses
import jax

@dataclasses.dataclass
class DeviceAssignment:
  """DeviceAssignment info to be saved in the manifest.pb.

  Attributes:
    id: The device id.
    coords: The device coordinates.
    core_on_chip: The core on chip.
  """

  id: int
  coords: Sequence[int] | None = None
  core_on_chip: int | None = None

  def __post_init__(self):
    if (self.coords is None) != (self.core_on_chip is None):
      raise ValueError(
          "coords and core_on_chip should be either both set or both None,  but"
          f" got coords: {self.coords} and core_on_chip: {self.core_on_chip}."
      )


def mesh_to_device_assignment(
    mesh: jax.sharding.Mesh,
) -> Sequence[DeviceAssignment]:
  """Returns a list of DeviceAssignment objects for each device in the mesh.

  Args:
    mesh: A jax.sharding.Mesh object.

  Returns:
    A list of DeviceAssignment objects, one for each device in the mesh.
  """

  def _to_assignment(d):
    if d.platform == "tpu":
      return DeviceAssignment(
          id=d.id, coords=d.coords, core_on_chip=d.core_on_chip
      )
    return DeviceAssignment(id=d.id)

  return [_to_assignment(d) for d in mesh.devices.flat]
