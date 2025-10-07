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

import os

from absl.testing import absltest
from orbax.experimental.model.core.python import unstructured_data


class UnstructuredDataTest(absltest.TestCase):

  def test_build_relative_filepath_from_extension(self):
    self.assertEqual(
        unstructured_data.build_relative_filepath_from_extension("foo", "txt"),
        "foo.txt",
    )
    self.assertEqual(
        unstructured_data.build_relative_filepath_from_extension(
            "foo", "txt", subfolder="bar"
        ),
        "bar/foo.txt",
    )

  def test_write_inlined_string_to_file(self):
    proto = unstructured_data.UnstructuredData()
    proto.inlined_string = "hello"
    proto.mime_type = "text/plain"
    proto.version = "1"
    dirname = self.create_tempdir().full_path
    filename = "data.txt"
    new_proto = unstructured_data.write_inlined_data_to_file(
        proto, dirname, filename
    )
    self.assertEqual(new_proto.WhichOneof("data"), "file_system_location")
    self.assertEqual(new_proto.file_system_location.string_path, filename)
    self.assertEqual(new_proto.mime_type, "text/plain")
    self.assertEqual(new_proto.version, "1")
    with open(os.path.join(dirname, filename), "r") as f:
      self.assertEqual(f.read(), "hello")

  def test_write_inlined_bytes_to_file(self):
    proto = unstructured_data.UnstructuredData()
    proto.inlined_bytes = b"hello"
    proto.mime_type = "application/octet-stream"
    proto.version = "1"
    dirname = self.create_tempdir().full_path
    filename = "data.bin"
    new_proto = unstructured_data.write_inlined_data_to_file(
        proto, dirname, filename
    )
    self.assertEqual(new_proto.WhichOneof("data"), "file_system_location")
    self.assertEqual(new_proto.file_system_location.string_path, filename)
    self.assertEqual(new_proto.mime_type, "application/octet-stream")
    self.assertEqual(new_proto.version, "1")
    with open(os.path.join(dirname, filename), "rb") as f:
      self.assertEqual(f.read(), b"hello")

  def test_write_location_pointer_to_file(self):
    proto = unstructured_data.UnstructuredData()
    proto.file_system_location.string_path = "path/to/file"
    proto.mime_type = "text/plain"
    proto.version = "1"
    dirname = self.create_tempdir().full_path
    filename = "data.txt"
    with self.assertRaisesRegex(
        ValueError, "Can only write inlined data .*to a file"
    ):
      unstructured_data.write_inlined_data_to_file(proto, dirname, filename)

  def test_maybe_write_inlined_string_to_file(self):
    proto = unstructured_data.UnstructuredData()
    proto.inlined_string = "hello"
    proto.mime_type = "text/plain"
    proto.version = "1"
    dirname = self.create_tempdir().full_path
    filename = "data.txt"
    new_proto = unstructured_data.maybe_write_inlined_data_to_file(
        proto, dirname, filename
    )
    self.assertEqual(new_proto.WhichOneof("data"), "file_system_location")
    self.assertEqual(new_proto.file_system_location.string_path, filename)
    self.assertEqual(new_proto.mime_type, "text/plain")
    self.assertEqual(new_proto.version, "1")
    with open(os.path.join(dirname, filename), "r") as f:
      self.assertEqual(f.read(), "hello")

  def test_maybe_write_inlined_bytes_to_file(self):
    proto = unstructured_data.UnstructuredData()
    proto.inlined_bytes = b"hello"
    proto.mime_type = "application/octet-stream"
    proto.version = "1"
    dirname = self.create_tempdir().full_path
    filename = "data.bin"
    new_proto = unstructured_data.maybe_write_inlined_data_to_file(
        proto, dirname, filename
    )
    self.assertEqual(new_proto.WhichOneof("data"), "file_system_location")
    self.assertEqual(new_proto.file_system_location.string_path, filename)
    self.assertEqual(new_proto.mime_type, "application/octet-stream")
    self.assertEqual(new_proto.version, "1")
    with open(os.path.join(dirname, filename), "rb") as f:
      self.assertEqual(f.read(), b"hello")

  def test_maybe_write_location_pointer_to_file(self):
    proto = unstructured_data.UnstructuredData()
    proto.file_system_location.string_path = "path/to/file"
    proto.mime_type = "text/plain"
    proto.version = "1"
    dirname = self.create_tempdir().full_path
    filename = "data.txt"
    new_proto = unstructured_data.maybe_write_inlined_data_to_file(
        proto, dirname, filename
    )
    self.assertEqual(new_proto.WhichOneof("data"), "file_system_location")
    self.assertEqual(new_proto.file_system_location.string_path, "path/to/file")
    with self.assertRaises(FileNotFoundError):
      with open(os.path.join(dirname, filename), "r") as f:
        f.read()


if __name__ == "__main__":
  absltest.main()
