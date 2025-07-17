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

"""test cases for colossus snapshot."""

import asyncio
from unittest import mock

from absl.testing import absltest
from etils import epath
from orbax.checkpoint._src.path.snapshot import snapshot



class DefaultSnapshotTest(absltest.TestCase):

  def __init__(self, *args, **kwargs):
    super(DefaultSnapshotTest, self).__init__(*args, **kwargs)
    self.root = epath.Path(self.create_tempdir(name='root').full_path)

    self.source_path = self.root / 'path/to/source'
    self.source_path.mkdir(exist_ok=True, parents=True, mode=0o750)
    self.source_file = self.source_path / 'data.txt'
    self.source_file.write_text('data')

    self.dest_path = self.root / 'path/to/dest'

  def test_create_snapshot(self):
    default_snapshot = snapshot._DefaultSnapshot(
        self.source_path, self.dest_path
    )
    self.assertFalse(self.dest_path.exists())
    asyncio.run(default_snapshot.create_snapshot())
    self.assertTrue(self.dest_path.exists())
    self.assertEqual('data', (self.dest_path / 'data.txt').read_text())

  def test_release_snapshot(self):
    default_snapshot = snapshot._DefaultSnapshot(
        self.source_path, self.dest_path
    )
    asyncio.run(default_snapshot.create_snapshot())
    self.assertTrue(self.dest_path.exists())
    self.assertTrue(asyncio.run(default_snapshot.release_snapshot()))
    self.assertFalse(self.dest_path.exists())

  def test_create_snapshot_with_relative_dest_path_fails(self):
    default_snapshot = snapshot._DefaultSnapshot(
        self.source_path, epath.Path('relative/path')
    )
    with self.assertRaisesRegex(
        ValueError, 'Snapshot destination must be absolute'
    ):
      asyncio.run(default_snapshot.create_snapshot())

  def test_create_snapshot_with_non_existent_source_fails(self):
    non_existent_source = self.root / 'non/existent'
    self.assertFalse(non_existent_source.exists())
    default_snapshot = snapshot._DefaultSnapshot(
        non_existent_source, self.dest_path
    )
    with self.assertRaisesRegex(ValueError, 'Snapshot source does not exist'):
      asyncio.run(default_snapshot.create_snapshot())

  def test_release_non_existent_snapshot(self):
    default_snapshot = snapshot._DefaultSnapshot(
        self.source_path, self.dest_path
    )
    self.assertFalse(self.dest_path.exists())
    self.assertFalse(asyncio.run(default_snapshot.release_snapshot()))

  def test_release_snapshot_fails_on_rmtree_error(self):
    default_snapshot = snapshot._DefaultSnapshot(
        self.source_path, self.dest_path
    )
    asyncio.run(default_snapshot.create_snapshot())
    self.assertTrue(self.dest_path.exists())
    mock_rmtree = mock.MagicMock()
    mock_rmtree.side_effect = OSError('fake error')
    with epath.testing.mock_epath(rmtree=mock_rmtree):
      self.assertFalse(asyncio.run(default_snapshot.release_snapshot()))
    mock_rmtree.assert_called_once()


if __name__ == '__main__':
  absltest.main()
