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

import logging

from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint._src.testing.benchmarks import lustre_benchmark


class LustreBenchmarkTest(multiprocess_test.MultiProcessTest):

  def test_xid(self):
    xid = lustre_benchmark._get_xid()
    self.assertIsInstance(xid, int)
    logging.info('XID: %s', xid)


if __name__ == '__main__':
  multiprocess_test.main()
