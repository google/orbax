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

"""Shared wrapper functions for logging.

TODO(b/362726102) Move to a final resting place and eliminate utils. DO NOT
make these symbols publicly available.
"""

import threading
from typing import Protocol
from absl import logging
from orbax.checkpoint._src.multihost import multihost


class _ObjectWithShutdown(Protocol):
  """Protocol for objects with a shutdown method."""

  def shutdown(self):
    """Shuts down the object."""
    ...


def shutdown_and_log(obj: _ObjectWithShutdown, name: str):
  """Logs shutdown of object with source name."""
  logging.info(
      '[process=%s][thread=%s] Shutting down executor in %s.',
      multihost.process_index(),
      threading.current_thread().name,
      name,
  )
  obj.shutdown()
