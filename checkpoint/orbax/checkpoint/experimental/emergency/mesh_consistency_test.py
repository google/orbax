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

"""Tests for mesh consistency metadata helpers."""

import asyncio
import json
from unittest import mock

from absl.testing import absltest
from etils import epath
import numpy as np
from orbax.checkpoint.experimental.emergency import mesh_consistency


class MeshConsistencyTest(absltest.TestCase):

  def test_read_process_metadata_returns_legacy_device_ids(self):
    directory = epath.Path(self.create_tempdir().full_path)
    metadata_dir = mesh_consistency.process_metadata_folder(directory)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    (
        metadata_dir / mesh_consistency._GLOBAL_PROCESS_METADATA_FILE_NAME
    ).write_text(json.dumps([[0, 1], [2, 3]]))
    (metadata_dir / mesh_consistency._MESH_METADATA_FILE_NAME).write_text(
        json.dumps([10, 11, 12, 13])
    )

    distributed_to_device_ids, device_ids = (
        mesh_consistency.read_process_metadata(metadata_dir)
    )

    self.assertEqual(distributed_to_device_ids, [[0, 1], [2, 3]])
    self.assertEqual(device_ids, [10, 11, 12, 13])

  def test_save_process_metadata_creates_directory_and_writes_device_ids(self):
    directory = (
        epath.Path(self.create_tempdir().full_path) / 'nested' / 'metadata'
    )
    global_mesh = mock.Mock()
    global_mesh.device_ids = np.array([[10, 11]])

    asyncio.run(
        mesh_consistency.save_process_metadata(  # pytype: disable=wrong-arg-types
            directory,
            global_mesh,
            [[10, 11]],
        )
    )

    self.assertTrue(directory.exists())
    distributed_to_device_ids, device_ids = (
        mesh_consistency.read_process_metadata(directory)
    )
    self.assertEqual(distributed_to_device_ids, [[10, 11]])
    self.assertEqual(device_ids, [10, 11])


if __name__ == '__main__':
  absltest.main()
