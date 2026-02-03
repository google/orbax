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

"""File utilities."""

import contextlib
import os
import shutil


_file_opener = open
_mkdir_p = lambda path: os.makedirs(path, exist_ok=True)
_copy = shutil.copyfile



@contextlib.contextmanager
def open_file(filename: str, mode: str):
  """Opens a file with the given filename and mode."""
  f = _file_opener(filename, mode)
  try:
    yield f
  finally:
    f.close()


def mkdir_p(path: str) -> None:
  """Creates a directory, creating parent directories as needed."""
  _mkdir_p(path)


def copy(source: str, dest: str) -> None:
  _copy(source, dest)
