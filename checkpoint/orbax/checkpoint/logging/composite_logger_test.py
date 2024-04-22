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

from absl import logging
from absl.testing import absltest
import google.cloud.logging as google_cloud_logging
import mock
from orbax.checkpoint.logging import abstract_logger
from orbax.checkpoint.logging import cloud_logger
from orbax.checkpoint.logging import composite_logger
from orbax.checkpoint.logging import standard_logger


_PROJECT = 'test-project'
_JOB_NAME = 'test-run'
_LOGGER_NAME = 'test-log'


class TestLogger(abstract_logger.AbstractLogger):
  def log_entry(self, entry):
    """Test logger."""

    logging.info(entry)


class CompositeLoggerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_gcloud_client = mock.create_autospec(google_cloud_logging.Client)
    self.standard_logger = standard_logger.StandardLogger()
    options = cloud_logger.CloudLoggerOptions()
    options.client = self.mock_gcloud_client
    self.cloud_logger = cloud_logger.CloudLogger(options)
    self.test_logger = TestLogger()

  def test_create_standard_logger(self):
    logger = composite_logger.CompositeLogger(self.standard_logger)
    with self.assertLogs(level='INFO') as log_output:
      entry = {'test-step': 'test-log-entry'}
      expected_message = str(entry)
      logger.log_entry(entry)
      self.assertEqual(log_output[0][0].message, expected_message)
      self.assertEqual(log_output[0][0].levelname, 'INFO')

  def test_create_cloud_logger(self):
    logger = composite_logger.CompositeLogger(self.cloud_logger)
    with mock.patch.object(
        self.cloud_logger,
        'log_entry',
        autospec=True,
    ) as mock_log_entry:
      entry = {'test-step': 'test-log-entry'}
      logger.log_entry(entry)
      mock_log_entry.assert_called_once_with(entry)

  def test_create_standard_and_test_logger(self):
    logger = composite_logger.CompositeLogger(
        self.standard_logger, self.test_logger
    )
    with self.assertLogs(level='INFO') as log_output:
      entry = {'test-step': 'test-log-entry'}
      expected_message = str(entry)
      logger.log_entry(entry)

      # Standard Logger
      standard_logger_output = log_output[0][0]
      self.assertEqual(standard_logger_output.message, expected_message)
      self.assertEqual(standard_logger_output.levelname, 'INFO')

      # Test Logger
      test_logger_output = log_output[0][1]
      self.assertEqual(test_logger_output.message, expected_message)
      self.assertEqual(test_logger_output.levelname, 'INFO')

if __name__ == '__main__':
  absltest.main()
