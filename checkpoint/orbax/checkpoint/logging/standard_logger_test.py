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

from absl.testing import absltest
from orbax.checkpoint.logging import standard_logger


class StandardLoggerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.logger = standard_logger.StandardLogger()

  def test_log_entry(self):
    with self.assertLogs(level='INFO') as log_output:
      entry = {'test-step': 'test-log-entry'}
      expected_message = str(entry)
      self.logger.log_entry(entry)
      self.assertEqual(log_output[0][0].message, expected_message)
      self.assertEqual(log_output[0][0].levelname, 'INFO')

if __name__ == '__main__':
  absltest.main()
