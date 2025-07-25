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

"""Utils for path constructs."""

import asyncio
from collections.abc import AsyncIterable
import os
import time
from typing import Iterable, Optional

from absl import logging
from etils import epath



def get_storage_type(path: epath.Path | str) -> str:
    return 'other'


class Timer(object):
  """A simple timer to measure the time it takes to run a function."""

  def __init__(self):
    self._start = time.time()

  def get_duration(self):
    return time.time() - self._start




def recursively_copy_files(
    src: epath.PathLike,
    dst: epath.PathLike,
    *,
    skip_paths: Iterable[str] | None = None,
) -> None:
  """Recursively copies files from src to dst.

  Args:
    src: The source directory to copy from.
    dst: The destination directory to copy to.
    skip_paths: An optional iterable of relative paths to skip.
  """
  src_path = epath.Path(src)
  dst_path = epath.Path(dst)
  skip_paths_set = set(skip_paths) if skip_paths is not None else set()

  for root, dirs, files in os.walk(src_path):
    relative_path = str(root)[len(str(src_path)) :].lstrip(os.sep)
    if relative_path in skip_paths_set:
      continue
    # Prune dirs that are in skip_paths to prevent traversal.
    dirs[:] = [
        d for d in dirs if os.path.join(relative_path, d) not in skip_paths_set
    ]

    dst_root = dst_path / relative_path
    dst_root.mkdir(parents=True, exist_ok=True)

    for file in files:
      relative_file_path = os.path.join(relative_path, file)
      if relative_file_path in skip_paths_set:
        continue

      src_file = epath.Path(root) / file
      dst_file = epath.Path(dst_root) / file
      src_file.copy(dst_file)
