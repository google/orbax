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

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper functions for Cloud Pathways persistence."""

import base64
from collections.abc import Mapping, Sequence
import concurrent.futures
import datetime
import json
from typing import Any

import jax
from jax import core
import numpy as np
from orbax.checkpoint._src.serialization import cloud_pathways_plugin_executable


def dtype_to_xla_primitive_type_str(dtype: np.dtype) -> str:
  """Converts a numpy dtype to an xla PrimitiveType."""
  if dtype == np.dtype("bfloat16"):
    return "BF16"
  elif dtype == np.dtype("float32"):
    return "F32"
  elif dtype == np.dtype("float64"):
    return "F64"
  elif dtype == np.dtype("int8"):
    return "S8"
  elif dtype == np.dtype("int16"):
    return "S16"
  elif dtype == np.dtype("int32"):
    return "S32"
  elif dtype == np.dtype("int64"):
    return "S64"
  elif dtype == np.dtype("uint8"):
    return "U8"
  elif dtype == np.dtype("uint16"):
    return "U16"
  elif dtype == np.dtype("uint32"):
    return "U32"
  elif dtype == np.dtype("uint64"):
    return "U64"
  else:
    raise ValueError(f"Unsupported dtype: {dtype}")


def base64_utf8_stringify(bs: bytes) -> str:
  """Converts bytes to a base64-encoded utf-8 string.

  Args:
    bs: The bytes to convert.

  Returns:
    The base64-encoded utf-8 string.
  """
  return base64.b64encode(bs).decode("utf-8")


def string_to_base64(text: str) -> str:
  """Encodes a string to base64 format.

  Args:
    text: The string to encode.

  Returns:
    The base64-encoded string.
  """
  return base64_utf8_stringify(text.encode("utf-8"))


def get_hlo_sharding_string(
    sharding: jax.sharding.Sharding,
    num_dimensions: int,
) -> str:
  """Serializes the sharding to an hlo-sharding, encodes it to base64 and returns the base-64 as an utf-8 string."""
  return base64_utf8_stringify(
      # pylint:disable=protected-access
      sharding._to_xla_hlo_sharding(num_dimensions)  # pytype: disable=attribute-error
      # pylint:enable=protected-access
      .to_proto().SerializeToString()
  )


def get_shape_info(
    dtype: np.dtype,
    dimensions: Sequence[int],
) -> Mapping[str, Sequence[int] | str]:
  """Returns shape info in the format expected by read requests."""
  return {
      "xla_primitive_type_str": dtype_to_xla_primitive_type_str(dtype),
      "dimensions": dimensions,
  }


def get_write_request(
    location_path: str,
    name: str,
    jax_array: jax.Array,
    timeout: datetime.timedelta,
    return_dict: bool = False,
) -> str | Mapping[str, Any]:
  """Returns a string representation of the plugin program which writes the given jax_array to the given location."""
  sharding = jax_array.sharding
  assert isinstance(sharding, jax.sharding.Sharding), sharding

  timeout_seconds, timeout_fractional_seconds = divmod(
      timeout.total_seconds(), 1
  )
  timeout_nanoseconds = timeout_fractional_seconds * 1e9
  d = {
      "persistenceWriteRequest": {
          "b64_location": string_to_base64(location_path),
          "b64_name": string_to_base64(name),
          "b64_hlo_sharding_string": get_hlo_sharding_string(
              jax_array.sharding, len(jax_array.shape)
          ),
          "shape": jax_array.shape,
          "devices": {
              "device_ids": [
                  # pylint:disable=protected-access
                  device.id
                  for device in sharding._device_assignment
                  # pylint:enable=protected-access
              ],
          },
          "timeout": {
              "seconds": int(timeout_seconds),
              "nanos": int(timeout_nanoseconds),
          },
      }
  }

  if return_dict:
    return d
  return json.dumps(d)


def get_bulk_write_request(
    location_path: str,
    names: Sequence[str],
    jax_arrays: Sequence[jax.Array],
    timeout: datetime.timedelta,
) -> str:
  """Returns a string representation of a bulk write request, writes multiple arrays with one call."""
  write_requests = [
      get_write_request(location_path, name, jax_array, timeout, True)[
          "persistenceWriteRequest"
      ]
      for name, jax_array in zip(names, jax_arrays)
  ]
  return json.dumps(
      {"bulk_persistence_write_request": {"write_requests": write_requests}}
  )


def get_read_request(
    location_path: str,
    name: str,
    dtype: np.dtype,
    shape: Sequence[int],
    sharding: jax.sharding.Sharding,
    devices: Sequence[jax.Device],
    timeout: datetime.timedelta,
    return_dict: bool = False,
) -> str | Mapping[str, Any]:
  """Returns a string representation of the plugin program which reads the given array from the given location into the provided sharding."""
  if not isinstance(devices, np.ndarray):
    devices = np.array(devices)

  timeout_seconds, timeout_fractional_seconds = divmod(
      timeout.total_seconds(), 1
  )
  timeout_nanoseconds = timeout_fractional_seconds * 1e9
  d = {
      "persistenceReadRequest": {
          "b64_location": string_to_base64(location_path),
          "shape": get_shape_info(dtype, shape),
          "b64_name": string_to_base64(name),
          "b64_hlo_sharding_string": get_hlo_sharding_string(
              sharding, len(shape)
          ),
          "devices": {
              "device_ids": [device.id for device in devices.flatten()]
          },
          "timeout": {
              "seconds": int(timeout_seconds),
              "nanos": int(timeout_nanoseconds),
          },
      }
  }

  if return_dict:
    return d
  return json.dumps(d)


def get_bulk_read_request(
    location_path: str,
    names: Sequence[str],
    dtypes: Sequence[np.dtype],
    shapes: Sequence[Sequence[int]],
    shardings: Sequence[jax.sharding.Sharding],
    devices: Sequence[jax.Device],
    timeout: datetime.timedelta,
) -> str:
  """Returns a string representation of a bulk read request, reads multiple arrays with one call."""
  read_requests = [
      get_read_request(
          location_path, name, dtype, shape, sharding, devices, timeout, True
      )["persistenceReadRequest"]
      for name, dtype, shape, sharding in zip(names, dtypes, shapes, shardings)
  ]
  return json.dumps(
      {"bulk_persistence_read_request": {"read_requests": read_requests}}
  )


def write_one_array(
    location: str,
    name: str,
    value: jax.Array,
    timeout: datetime.timedelta,
):
  """Creates the write array plugin program string, compiles it to an executable, calls it and returns an awaitable future."""
  write_request = get_write_request(location, name, value, timeout)
  write_executable = cloud_pathways_plugin_executable.PluginExecutable(
      write_request
  )
  _, write_future = write_executable.call([value])
  return write_future


def write_arrays(
    location: str,
    names: Sequence[str],
    values: Sequence[jax.Array],
    timeout: datetime.timedelta,
) -> concurrent.futures.Future[None]:
  """Creates the write array plugin program string, compiles it to an executable, calls it and returns an awaitable future."""
  bulk_write_request = get_bulk_write_request(location, names, values, timeout)
  bulk_write_executable = cloud_pathways_plugin_executable.PluginExecutable(
      bulk_write_request
  )
  _, bulk_write_future = bulk_write_executable.call(values)
  return bulk_write_future


def read_arrays(
    location: str,
    names: Sequence[str],
    dtypes: Sequence[np.dtype],
    shapes: Sequence[Sequence[int]],
    shardings: Sequence[jax.sharding.Sharding],
    devices: Sequence[jax.Device] | np.ndarray,
    timeout: datetime.timedelta,
) -> tuple[Sequence[jax.Array], concurrent.futures.Future[None]]:
  """Creates the read array plugin program string, compiles it to an executable, calls it and returns the result."""

  bulk_read_request = get_bulk_read_request(
      location, names, dtypes, shapes, shardings, devices, timeout
  )
  bulk_read_executable = cloud_pathways_plugin_executable.PluginExecutable(
      bulk_read_request
  )
  out_avals = [
      core.ShapedArray(shape, dtype) for shape, dtype in zip(shapes, dtypes)
  ]
  arrays, read_future = bulk_read_executable.call(
      out_shardings=shardings, out_avals=out_avals
  )
  return (arrays, read_future)
