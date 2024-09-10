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

"""test cases for colossus snapshot."""

import os
from unittest import mock

from absl.testing import absltest
from etils import epath
from orbax.checkpoint._src.path.snapshot import snapshot



class DefaultSnapshotTest(absltest.TestCase):

  def __init__(self, *args, **kwargs):
    super(DefaultSnapshotTest, self).__init__(*args, **kwargs)
    self.source_dir = epath.Path(self.create_tempdir(name='source').full_path)
    # self.source_dir = '/tmp/test/path/to/source/'
    os.makedirs(self.source_dir, exist_ok=True)
    with open(os.path.join(self.source_dir, 'data.txt'), 'w') as f:
      f.write('data')

  def test_create_snapshot(self):
    dst_dir = '/tmp/test/path/to/dest/'

    # dst_dir = epath.Path(self.create_tempdir(name='dest').full_path)
    self.assertFalse(os.path.exists(dst_dir))
    snapshot.DefaultSnapshot.create_snapshot(str(self.source_dir), str(dst_dir))
    self.assertTrue(os.path.exists(dst_dir))
    with open(os.path.join(dst_dir, 'data.txt')) as f:
      self.assertEqual('data', f.read())
      f.close()

  def test_release_snapshot(self):
    self.assertTrue(os.path.exists(self.source_dir))
    self.assertTrue(snapshot.DefaultSnapshot.release_snapshot(self.source_dir))
    self.assertFalse(os.path.exists(self.source_dir))


if __name__ == '__main__':
  absltest.main()
