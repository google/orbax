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

import json
import threading
from typing import Any

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


def gather_on_primary(name: str, value: Any) -> list[Any] | None:
  """Collects every process's JSON-serializable `value` on process 0.

  Collective: every process must call it with the same `name`, which must be
  unique in the job. Each process stores its value in the jax.distributed
  key-value store and waits at barrier `name`; then process 0 alone reads all
  values with one directory read. No process blocks on a key, since many such
  calls at once (one per process) can overload the coordination service.

  Args:
    name: Names the exchange and its barrier.
    value: This process's JSON-serializable value.

  Returns:
    On process 0, every process's value in process order; elsewhere, None.
  """
  if multihost.process_count() == 1:
    return [value]
  client = multihost.get_jax_distributed_client()
  prefix = f"benchmark_gather/{name}"
  client.key_value_set(f"{prefix}/{get_process_index()}", json.dumps(value))
  sync_global_processes(name)
  if get_process_index() != 0:
    return None
  values = {
      int(key.rsplit("/", 1)[1]): json.loads(v)
      for key, v in client.key_value_dir_get(prefix)
  }
  client.key_value_delete(prefix)
  return [values[i] for i in sorted(values)]
