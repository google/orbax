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

"""Multihost utilities for benchmarks."""

import threading

from absl import logging
from orbax.checkpoint._src.multihost import multihost


def get_process_index() -> int:
  """Returns process index from torch if available, else from multihost."""
  try:
    import torch.distributed as dist  # pylint: disable=g-import-not-at-top

    if dist.is_initialized():
      return dist.get_rank()
  except ImportError:
    pass
  return multihost.process_index()


def sync_global_processes(
    name: str,
) -> None:
  """Syncs global processes using torch if available, else multihost."""
  try:
    import torch.distributed as dist  # pylint: disable=g-import-not-at-top

    if dist.is_initialized():
      logging.vlog(
          1,
          "[process=%s][thread=%s] sync_global_processes with torch"
          " barrier: %s",
          dist.get_rank(),
          threading.current_thread().name,
          name,
      )
      dist.barrier()
      return
  except ImportError:
    pass
  multihost.sync_global_processes(name)
