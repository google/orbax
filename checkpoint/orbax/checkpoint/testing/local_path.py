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

"""Test class for process-local paths."""

from __future__ import annotations

import os
import pathlib
from typing import Iterator

from etils import epath
from orbax.checkpoint._src.multihost import multihost


@epath.register_path_cls
class LocalPath(pathlib.PurePosixPath):
  """A Path implementation for testing process-local paths.

  In the future, this class may more completely provide all functions and
  properties of a pathlib Path, but for now, it only provides the minimum
  needed to support relevant tests.

  This class is intended to receive a base path, and append a process-specific
  suffix to it when path operations are performed (the appending should be
  delayed as much as possible).

  Operations that combine two LocalPaths should not re-append the suffix.

  One may ask - why not just construct a path initially with some
  process-specific suffix appended, and use that for testing? This works for
  multi-controller, but not for single-controller, where paths are typically
  constructed in the
  controller (single-process) and passed to workers (multi-process). The
  process index must be appended when path operations are performed.
  """

  def __init__(self, *parts: epath.PathLike):
    super().__init__(*parts)
    self._path = epath.Path('/'.join(os.fspath(p) for p in parts))

  def __repr__(self) -> str:
    return f'{self.__class__.__name__}({self.path})'

  @property
  def base_path(self) -> epath.Path:
    return self._path

  @property
  def path(self) -> epath.Path:
    return self.base_path / str(f'local_{multihost.process_index()}')

  def exists(self) -> bool:
    """Returns True if self exists."""
    return self.path.exists()

  def is_dir(self) -> bool:
    """Returns True if self is a dir."""
    return self.path.is_dir()

  def is_file(self) -> bool:
    """Returns True if self is a file."""
    return self.path.is_file()

  def iterdir(self) -> Iterator[LocalPath]:
    """Iterates over the directory."""
    return (LocalPath(p) for p in self.path.iterdir())

  def glob(self, pattern: str) -> Iterator[LocalPath]:
    """Yields all matching files (of any kind)."""
    return (LocalPath(p) for p in self.path.glob(pattern))

  def read_bytes(self) -> bytes:
    """Reads contents of self as bytes."""
    return self.path.read_bytes()

  def read_text(self, encoding: str | None = None) -> str:
    """Reads contents of self as a string."""
    return self.path.read_text(encoding=encoding)

  def mkdir(
      self,
      mode: int | None = None,
      parents: bool = False,
      exist_ok: bool = False,
  ) -> None:
    """Create a new directory at this given path."""
    del mode  # mode is not supported by epath.Path.mkdir
    self.path.mkdir(parents=parents, exist_ok=exist_ok)

  def rmdir(self) -> None:
    """Remove the empty directory at this given path."""
    self.path.rmdir()

  def rmtree(self, missing_ok: bool = False) -> None:
    """Remove the directory, including all sub-files."""
    self.path.rmtree(missing_ok=missing_ok)

  def unlink(self, missing_ok: bool = False) -> None:
    """Remove this file or symbolic link."""
    self.path.unlink(missing_ok=missing_ok)

  def write_bytes(self, data: bytes) -> int:
    """Writes content as bytes."""
    return self.path.write_bytes(data)

  def write_text(
      self,
      data: str,
      encoding: str | None = None,
      errors: str | None = None,
  ) -> int:
    """Writes content as str."""
    return self.path.write_text(data, encoding=encoding, errors=errors)

  def as_posix(self) -> str:
    return self.path.as_posix()

  def __truediv__(self, key: epath.PathLike) -> epath.Path:
    return self.path / key

  @property
  def name(self) -> str:
    return self.path.name

  @property
  def parent(self) -> epath.Path:
    return self.path.parent
