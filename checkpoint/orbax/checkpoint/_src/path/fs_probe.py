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

"""Batched filesystem probing utilities for checkpoint inspection.

A shared path utility library providing low-latency directory indexing and
concurrent probe primitives, especially used by checkpoint layout and format
detection routines.

Probing individual marker files sequentially on distributed or cloud
filesystems incurs significant RPC round-trip latency. This module minimizes
filesystem round trips through two core primitives:

1. **Ask once, answer many:** A single non-recursive directory scan answers all
   marker existence checks directly under a directory with one round trip.
   `DirectoryIndex` snapshots that listing for fast membership lookups.
2. **Ask in parallel:** When multiple subdirectories or paths must be checked
   concurrently, `exists_many`, `is_dir_many`, and `index_directories` issue
   checks concurrently via asyncio rather than serially.

Note: Directory listings are strictly non-recursive to avoid traversing large
tensor or chunk subtrees.
"""

from __future__ import annotations

import asyncio

from etils import epath
from orbax.checkpoint._src.path import async_path


class DirectoryIndex:
  """The immediate contents of one directory, fetched in a single round trip."""

  def __init__(
      self,
      path: epath.Path,
      exists: bool = False,
      is_directory: bool = False,
      names: frozenset[str] = frozenset(),
  ):
    """Initializes the directory index."""
    self._path = path
    self._exists = exists
    self._is_directory = is_directory
    self._names = names

  def path(self) -> epath.Path:
    """Returns the directory path that was indexed."""
    return self._path

  def exists(self) -> bool:
    """Returns whether the path exists."""
    return self._exists

  def is_directory(self) -> bool:
    """Returns whether the path is a directory."""
    return self._is_directory

  def listable(self) -> bool:
    """Returns whether the path exists and is a directory."""
    return self._exists and self._is_directory

  def names(self) -> frozenset[str]:
    """Returns immediate child names."""
    return self._names

  def has(self, name: str) -> bool:
    """Returns whether a child with exactly this name exists."""
    return name in self._names

  def has_any(self, *candidates: str) -> bool:
    """Returns whether any of the named children exist."""
    return any(candidate in self._names for candidate in candidates)

  def matching(self, *prefixes: str) -> list[str]:
    """Returns sorted child names starting with any of the given prefixes."""
    return sorted(name for name in self._names if name.startswith(prefixes))

  def with_suffix(self, suffix: str) -> list[str]:
    """Returns sorted child names ending with the given suffix."""
    return sorted(name for name in self._names if name.endswith(suffix))

  def present(self, candidates: tuple[str, ...]) -> list[str]:
    """Returns the candidates that exist, preserving candidate order."""
    return [candidate for candidate in candidates if candidate in self._names]

  def __repr__(self) -> str:
    return (
        f"DirectoryIndex(path={self._path!r}, exists={self._exists!r}, "
        f"is_directory={self._is_directory!r}, names={self._names!r})"
    )

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, DirectoryIndex):
      return False
    return (
        self._path == other._path
        and self._exists == other._exists
        and self._is_directory == other._is_directory
        and self._names == other._names
    )


_MISSING = frozenset()


async def index_directory(path: epath.Path) -> DirectoryIndex:
  """Lists one directory asynchronously, discovering child names."""
  try:
    entries = await async_path.iterdir(path)
    names = frozenset(entry.name for entry in entries)
    return DirectoryIndex(
        path=path, exists=True, is_directory=True, names=names
    )
  except FileNotFoundError:
    return DirectoryIndex(
        path=path, exists=False, is_directory=False, names=_MISSING
    )
  except NotADirectoryError:
    return DirectoryIndex(
        path=path, exists=True, is_directory=False, names=_MISSING
    )
  except Exception:  # pylint: disable=broad-exception-caught
    # Probing remote/cloud filesystems may raise driver-specific exceptions.
    try:
      exists = await async_path.exists(path)
      is_dir = await async_path.is_dir(path) if exists else False
      return DirectoryIndex(
          path=path, exists=exists, is_directory=is_dir, names=_MISSING
      )
    except Exception:  # pylint: disable=broad-exception-caught
      # Fall back to missing index if exists/is_dir probe fails.
      return DirectoryIndex(
          path=path, exists=False, is_directory=False, names=_MISSING
      )


async def index_directories(
    paths: tuple[epath.Path, ...],
) -> tuple[DirectoryIndex, ...]:
  """Lists several directories concurrently.

  Args:
    paths: Directories to list.

  Returns:
    Indexes positionally aligned with `paths`.
  """
  return tuple(await asyncio.gather(*(index_directory(p) for p in paths)))


async def exists_many(
    paths: tuple[epath.Path, ...],
) -> tuple[bool, ...]:
  """Checks several paths for existence concurrently.

  Prefer answering from a `DirectoryIndex` when the paths share a parent; use
  this only when they do not.

  Args:
    paths: Paths to check.

  Returns:
    Booleans positionally aligned with `paths`.
  """

  async def _safe_exists(target_path: epath.Path) -> bool:
    try:
      return await async_path.exists(target_path)
    except Exception:  # pylint: disable=broad-exception-caught
      # Remote/cloud filesystem driver exceptions treat path as nonexistent.
      return False

  return tuple(await asyncio.gather(*(_safe_exists(p) for p in paths)))


async def is_dir_many(
    paths: tuple[epath.Path, ...],
) -> tuple[bool, ...]:
  """Checks several paths for directory-ness concurrently.

  Args:
    paths: Paths to check.

  Returns:
    Booleans positionally aligned with `paths`.
  """

  async def _safe_is_dir(target_path: epath.Path) -> bool:
    try:
      return await async_path.is_dir(target_path)
    except Exception:  # pylint: disable=broad-exception-caught
      # Remote/cloud filesystem driver exceptions treat path as non-directory.
      return False

  return tuple(await asyncio.gather(*(_safe_is_dir(p) for p in paths)))
