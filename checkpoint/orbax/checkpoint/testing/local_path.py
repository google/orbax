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

"""Test class for process-local paths."""

from __future__ import annotations

import os
import typing
from typing import Iterator, Sequence

from etils import epath
from orbax.checkpoint._src.multihost import multihost


_LOCAL_PATH_BASE_NAME = '_local_path_base'
_LOCAL_PART_PREFIX = 'local'


# The following is a hack to pass the type checker.
if typing.TYPE_CHECKING:
  _BasePath = epath.Path
else:
  _BasePath = object


def create_local_path_base(testclass) -> epath.Path:
  return epath.Path(
      testclass.create_tempdir(name=_LOCAL_PATH_BASE_NAME).full_path
  )


def _get_local_part_index(parts: Sequence[str]) -> int:
  for i, part in enumerate(parts):
    if part.startswith(_LOCAL_PART_PREFIX):
      return i
  raise ValueError(
      f'Did not find a local part ({_LOCAL_PART_PREFIX}) in parts: {parts}'
  )


class LocalPath(_BasePath):
  """A Path implementation for testing process-local paths.

  IMPORTANT: Use `create_local_path_base` to create the base path for test
  cases.

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
    self._path = epath.Path('/'.join(os.fspath(p) for p in parts))
    # Assumes this class will always be constructed on the controller first
    # (otherwise this check will return the wrong value).
    self._is_pathways_backend = multihost.is_pathways_backend()

  def __repr__(self) -> str:
    return f'LocalPath({self.path})'

  def __str__(self) -> str:
    return str(self.path)

  @property
  def base_path(self) -> epath.Path:
    return self._path

  @property
  def path(self) -> epath.Path:
    parts = list(self.base_path.parts)

    # Fail if the path is not properly configured. The local part should be
    # immediately following the base name.
    try:
      base_idx = parts.index(_LOCAL_PATH_BASE_NAME)
    except ValueError as e:
      raise ValueError(
          f'Base path for LocalPath must contain {_LOCAL_PATH_BASE_NAME}. Got:'
          f' {self.base_path}'
      ) from e

    if multihost.is_pathways_controller():
      local_part = f'{_LOCAL_PART_PREFIX}_controller'
    else:
      local_part = f'{_LOCAL_PART_PREFIX}_{multihost.process_index()}'

    try:
      # If the local part is already present, potentially replace it with the
      # correct local part (e.g. controller vs worker).
      local_part_idx = _get_local_part_index(parts)
      assert local_part_idx == base_idx + 1
      parts[local_part_idx] = local_part
      return epath.Path(*parts)
    except ValueError:
      pass

    # Otherwise, insert following the base part.
    parts.insert(base_idx + 1, local_part)
    return epath.Path(*parts)

  @property
  def parts(self) -> tuple[str, ...]:
    return self.path.parts

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

  def touch(self, mode: int = 0o666, exist_ok: bool = False) -> None:
    """Creates the file at this path."""
    self.path.touch(exist_ok=exist_ok)

  def rename(self, new_path: epath.PathLike) -> None:
    """Renames this file or directory to the given path."""
    self.path.rename(new_path)

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

  def __truediv__(self, key: epath.PathLike) -> LocalPath:
    return LocalPath(self.path / key)

  @property
  def name(self) -> str:
    return self.path.name

  @property
  def parent(self) -> LocalPath:
    return LocalPath(self.path.parent)

  def __fspath__(self) -> str:
    return os.fspath(self.path)
