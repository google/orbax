# Copyright 2024 The Orbax Authors.
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

"""UnstructuredData and its utilities."""

import dataclasses
import os

from orbax.experimental.model.core.protos.manifest_pb2 import UnstructuredData  # pylint: disable=g-importing-member


@dataclasses.dataclass
class UnstructuredDataWithExtName:
  """An `UnstructuredData` proto and a file extension name.

  A receiver of `UnstructuredData` sometimes wants to save the data to
  a file, in which case a file extension name is needed to make up a sensible
  file name.

  Attributes:
    proto: an `UnstructuredData` proto.
    ext_name: an optional file extension name appropriate for the data. If it's
      `None`, the data won't be saved to a file. If it's `""`, the full file
      name will be just the base file name (no dot). If it's not empty, the full
      file name will be `base_name + "." + ext_name` (therefore `ext_name`
      shouldn't contain a leading ".").
  """
  proto: UnstructuredData
  ext_name: str | None


def build_filename_from_extension(base_name, ext):
  suffix = "." + ext if ext else ""
  return base_name + suffix


def write_inlined_data_to_file(
    proto: UnstructuredData, dirname: str, filename: str
) -> UnstructuredData:
  """Writes an inlined `UnstructuredData` to a file.

  `proto.data` must be set to either `inlined_string` or `inlined_bytes`.

  Args:
    proto: The `UnstructuredData` to write.
    dirname: The directory name of the file to write to.
    filename: The name of the file to write to. The full name of the file will
      be `os.path.join(dirname, filename)`.

  Returns:
    A new `UnstructuredData` which is the same as `proto` except that its `data`
    attribute becomes `file_system_location` with value `filename`.

  Raises:
    ValueError: If `proto` is not inlined data (i.e. it's already a
    `file_system_location` data).
  """
  case_name = proto.WhichOneof("data")
  if case_name == "file_system_location":
    raise ValueError("Can only write inlined data (string or bytes) to a file.")
  elif case_name is None:
    raise ValueError("The 'data' field of this UnstructuredData is not set.")
  if case_name == "inlined_string":
    data_to_write = proto.inlined_string
    io_mode = "w"
  else:
    data_to_write = proto.inlined_bytes
    io_mode = "wb"
  with open(os.path.join(dirname, filename), io_mode) as f:
    f.write(data_to_write)
  result = UnstructuredData()
  result.file_system_location.string_path = filename
  result.mime_type = proto.mime_type
  result.version = proto.version
  return result


def maybe_write_inlined_data_to_file(
    proto: UnstructuredData, dirname: str, filename: str
) -> UnstructuredData:
  """Similar to `write_inlined_data_to_file`.

  But simply returns `proto` if it is already a `file_system_location`.

  Args:
    proto: See `write_inlined_data_to_file`.
    dirname: See `write_inlined_data_to_file`.
    filename: See `write_inlined_data_to_file`.

  Returns:
    See `write_inlined_data_to_file`.
  """
  case_name = proto.WhichOneof("data")
  if case_name == "file_system_location":
    return proto
  return write_inlined_data_to_file(proto, dirname, filename)
