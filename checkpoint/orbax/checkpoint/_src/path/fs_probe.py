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

1. **Ask once, answer many:** A single non-recursive `iterdir` answers all
   marker existence checks directly under a directory with one round trip.
   `DirectoryIndex` snapshots that listing for fast membership lookups.
2. **Ask in parallel:** When multiple subdirectories or paths must be checked
   concurrently, `probe_paths` issues checks in parallel rather than serially.

Note: Directory listings are strictly non-recursive to avoid traversing large
tensor or chunk subtrees.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
from typing import Callable, Iterable, TypeVar

from etils import epath

_T = TypeVar("_T")
_R = TypeVar("_R")


@dataclasses.dataclass(frozen=True)
class DirectoryIndex:
  """The immediate contents of one directory, fetched in a single round trip.

  Attributes:
    path: The directory that was listed.
    listable: Whether the listing succeeded. False covers both "does not
      exist" and "exists but is not a directory"; callers that must tell those
      apart need an explicit stat.
    names: Immediate child names. Empty when `listable` is False.
  """

  path: epath.Path
  listable: bool
  names: frozenset[str]

  def has(self, name: str) -> bool:
    """Returns whether a child with exactly this name exists."""
    return name in self.names

  def has_any(self, *candidates: str) -> bool:
    """Returns whether any of the named children exist."""
    return any(c in self.names for c in candidates)

  def matching(self, *prefixes: str) -> list[str]:
    """Returns sorted child names starting with any of the given prefixes."""
    return sorted(n for n in self.names if n.startswith(prefixes))

  def with_suffix(self, suffix: str) -> list[str]:
    """Returns sorted child names ending with the given suffix."""
    return sorted(n for n in self.names if n.endswith(suffix))

  def present(self, candidates: tuple[str, ...]) -> list[str]:
    """Returns the candidates that exist, preserving the candidate order."""
    return [c for c in candidates if c in self.names]


_MISSING = frozenset()


def index_directory(path: epath.Path) -> DirectoryIndex:
  """Lists one directory, tolerating missing paths and permission errors."""
  try:
    names = frozenset(entry.name for entry in path.iterdir())
  except Exception:  # pylint: disable=broad-exception-caught
    # Missing, not a directory, or unreadable: all mean "nothing to see here"
    # for format detection, which must degrade rather than fail.
    return DirectoryIndex(path=path, listable=False, names=_MISSING)
  return DirectoryIndex(path=path, listable=True, names=names)


def _map(
    fn: Callable[[_T], _R],
    items: Iterable[_T],
    pool: concurrent.futures.Executor | None = None,
) -> list[_R]:
  """Applies fn across items concurrently using pool or local executor."""
  if pool is not None:
    return list(pool.map(fn, items))
  with concurrent.futures.ThreadPoolExecutor() as local_pool:
    return list(local_pool.map(fn, items))


def index_directories(
    paths: tuple[epath.Path, ...],
    pool: concurrent.futures.Executor | None = None,
) -> tuple[DirectoryIndex, ...]:
  """Lists several directories concurrently.

  Args:
    paths: Directories to list.
    pool: Optional thread pool executor to run listings on.

  Returns:
    Indexes positionally aligned with `paths`. A directory that could not be
    listed yields an index with `listable=False`.
  """
  return tuple(_map(index_directory, paths, pool))


def exists_many(
    paths: tuple[epath.Path, ...],
    pool: concurrent.futures.Executor | None = None,
) -> tuple[bool, ...]:
  """Checks several paths for existence concurrently.

  Prefer answering from a `DirectoryIndex` when the paths share a parent; use
  this only when they do not.

  Args:
    paths: Paths to check.
    pool: Optional thread pool executor to run checks on.

  Returns:
    Booleans positionally aligned with `paths`.
  """

  def _exists(p: epath.Path) -> bool:
    try:
      return p.exists()
    except Exception:  # pylint: disable=broad-exception-caught
      return False

  return tuple(_map(_exists, paths, pool))


def is_dir_many(
    paths: tuple[epath.Path, ...],
    pool: concurrent.futures.Executor | None = None,
) -> tuple[bool, ...]:
  """Checks several paths for directory-ness concurrently.

  Args:
    paths: Paths to check.
    pool: Optional thread pool executor to run checks on.

  Returns:
    Booleans positionally aligned with `paths`.
  """

  def _is_dir(p: epath.Path) -> bool:
    try:
      return p.is_dir()
    except Exception:  # pylint: disable=broad-exception-caught
      return False

  return tuple(_map(_is_dir, paths, pool))
