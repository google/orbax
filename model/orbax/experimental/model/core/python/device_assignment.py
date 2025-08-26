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
    core_on_chip: The core on chip. If not set, the device is assumed to be a
      single-core device.
  """

  id: int
  coords: Sequence[int] | None = None
  core_on_chip: int | None = None


def jax_mesh_to_device_assignment(
    jax_mesh: jax.sharding.Mesh,
) -> Sequence[DeviceAssignment]:
  """Converts `jax.sharding.Mesh` to a sequence of `DeviceAssignment`s."""

  def spec_for_device(d):
    if d.platform == "tpu":
      return DeviceAssignment(
          id=d.id, coords=d.coords, core_on_chip=d.core_on_chip
      )
    return DeviceAssignment(id=d.id)

  return [spec_for_device(d) for d in jax_mesh.devices.flat]
