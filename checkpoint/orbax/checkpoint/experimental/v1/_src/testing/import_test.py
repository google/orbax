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

import unittest


class ImportTest(unittest.TestCase):

  def test_import_orbax_checkpoint(self):
    try:
      import orbax.checkpoint as ocp  # pylint: disable=unused-import, g-import-not-at-top
    except ImportError as e:
      self.fail(f"Failed to import orbax.checkpoint: {e}")

  def test_import_orbax_checkpoint_experimental_v1(self):
    try:
      import orbax.checkpoint.experimental.v1 as ocp  # pylint: disable=unused-import, g-import-not-at-top
    except ImportError as e:
      self.fail(f"Failed to import orbax.checkpoint.experimental.v1: {e}")


if __name__ == "__main__":
  unittest.main()
