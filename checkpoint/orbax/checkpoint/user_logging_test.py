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

"""Tests for logging from a user's perspective."""

import logging
from absl.testing import absltest


from orbax import checkpoint as ocp  # pylint: disable=unused-import

logger = logging.getLogger(__name__)

root_has_handlers = len(logging.root.handlers)
logger_has_handlers = len(logger.handlers)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)


class LoggingTest(absltest.TestCase):

  def test_handlers_before_user_basic_config(self):
    self.assertFalse(root_has_handlers)
    self.assertFalse(logger_has_handlers)

  def test_handlers_after_user_basic_config(self):
    self.assertTrue(logging.root.handlers)
    self.assertFalse(logger.handlers)


if __name__ == '__main__':
  absltest.main()
