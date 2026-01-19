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

from absl.testing import absltest
from etils import epath
from orbax.checkpoint.experimental.emergency.p2p import utils


class UtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path)

  def test_detect_process_index(self):
    step_dir = self.directory / '1'
    step_dir.mkdir()
    (step_dir / 'state' / 'ocdbt.process_42').mkdir(parents=True)

    self.assertEqual(utils.detect_process_index(self.directory, 1), 42)
    self.assertIsNone(utils.detect_process_index(self.directory, 2))


if __name__ == '__main__':
  absltest.main()
