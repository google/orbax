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

"""Tests for checkpoint storage backend base implementations."""

from absl.testing import absltest
from etils import epath
from orbax.checkpoint._src.path import storage_backend


class LocalStorageBackendTest(absltest.TestCase):

  def test_name_to_path_component_is_identity(self):
    backend = storage_backend.LocalStorageBackend()
    self.assertEqual(backend.name_to_path_component('step_1'), 'step_1')

  def test_path_component_to_name_is_identity(self):
    backend = storage_backend.LocalStorageBackend()
    self.assertEqual(backend.path_component_to_name('step_1'), 'step_1')

  def test_list_checkpoints_returns_children(self):
    tmpdir = self.create_tempdir()
    base = epath.Path(tmpdir.full_path)
    (base / 'step_0').mkdir()
    (base / 'step_1').mkdir()
    backend = storage_backend.LocalStorageBackend()
    assets = backend.list_checkpoints(str(base))
    self.assertLen(assets, 2)
    paths = sorted([a.path for a in assets])
    self.assertEqual(
        paths,
        sorted([
            str(base / 'step_0'),
            str(base / 'step_1'),
        ]),
    )
    for asset in assets:
      self.assertEqual(
          asset.status,
          storage_backend.CheckpointPathMetadata.Status.COMMITTED,
      )
      self.assertIsNone(asset.version)

  def test_list_checkpoints_non_existent_path_returns_empty(self):
    tmpdir = self.create_tempdir()
    base = epath.Path(tmpdir.full_path) / 'non_existent'
    backend = storage_backend.LocalStorageBackend()
    assets = backend.list_checkpoints(str(base))
    self.assertEmpty(assets)

  def test_list_checkpoints_empty_directory(self):
    tmpdir = self.create_tempdir()
    base = epath.Path(tmpdir.full_path)
    backend = storage_backend.LocalStorageBackend()
    assets = backend.list_checkpoints(str(base))
    self.assertEmpty(assets)

  def test_resolve_storage_backend_returns_local(self):
    backend = storage_backend.resolve_storage_backend('/tmp/some/path')
    self.assertIsInstance(backend, storage_backend.LocalStorageBackend)


if __name__ == '__main__':
  absltest.main()
