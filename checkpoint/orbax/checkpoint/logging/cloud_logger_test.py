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
import google.cloud.logging as google_cloud_logging
import mock
from orbax.checkpoint.logging import cloud_logger


_PROJECT = 'test-project'
_JOB_NAME = 'test-run'
_LOGGER_NAME = 'test-log'


class CloudLoggerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_gcloud_client = mock.create_autospec(
        google_cloud_logging.Client
    )
    self.cloud_logger = cloud_logger.CloudLogger(
        _JOB_NAME,
        _LOGGER_NAME,
        mock_gcloud_client
    )

  def test_log_entry(self):
    entry = {'test-step': 'test-log-entry'}
    self.cloud_logger.log_entry(entry)


if __name__ == '__main__':
  absltest.main()
