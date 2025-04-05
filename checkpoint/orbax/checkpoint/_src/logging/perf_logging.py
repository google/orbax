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

"""Performance logging."""

from absl import logging
import jax
import psutil


def log_memory_info(label: str | None = None):
  """Logs memory usage of the current os process."""
  jax_process_index = jax.process_index()
  process = psutil.Process()
  memory_info = process.memory_info()
  logging.info(
      '[process=%s] %s memory_info: %s',
      jax_process_index,
      label or '',
      memory_info,
  )
