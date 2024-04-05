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

"""Cloud logging implementation for checkpointing."""

from typing import Any, Optional

import google.cloud.logging as google_cloud_logging
from orbax.checkpoint.logging import abstract_logger


class CloudLogger(abstract_logger.AbstractLogger):
  """Logging implementation utilizing a cloud logging API."""

  def __init__(
      self,
      job_name: str,
      log_name: str,
      client: Optional[google_cloud_logging.Client] = None,
  ):
    """CloudLogger constructor.

    Args:
      job_name: Name of the job the CheckpointLogger is for.
      log_name: Name of the log being written.
      client: Optional client to use for logging.
    """
    self.job_name = job_name
    if client is None:
      self.logging_client = google_cloud_logging.Client()
    else:
      self.logging_client = client
    self.logger = self.logging_client.logger(log_name)

  def log_entry(self, entry: Any):
    """Logs an informational message.

    Args:
        entry: Additional structured data to be included in the log record.
    """

    pass
