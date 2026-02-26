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

"""Async wrappers for path operations.

Any function conforming to::

  epath.Path.foo(*args, **kwargs)

Should be implemented as::

  async def foo(path: epath.Path, *args, **kwargs):
    return await asyncio.to_thread(path.foo, *args, **kwargs)
"""

import asyncio
import contextlib
import os
from typing import Any, AsyncIterator, Iterator

from etils import epath



async def mkdir(
    path: epath.Path,
    parents: bool = False,
    exist_ok: bool = False,
    mode: int | None = None,
):
  """Creates a directory asynchronously."""

  def _mkdir_sync(**thread_kwargs):
    """Synchronously creates a directory."""
    path.mkdir(parents=parents, exist_ok=exist_ok, mode=mode)

  thread_kwargs = {}
  return await asyncio.to_thread(_mkdir_sync, **thread_kwargs)


async def write_bytes(path: epath.Path, data: Any) -> int:

  def _write():
    try:
      path.unlink()
    except OSError:
      pass
    return path.write_bytes(data)

  return await asyncio.to_thread(_write)


async def read_bytes(path: epath.Path) -> bytes:
  return await asyncio.to_thread(path.read_bytes)


async def write_text(path: epath.Path, text: str) -> int:

  def _write():
    try:
      path.unlink()
    except OSError:
      pass
    return path.write_text(text)

  return await asyncio.to_thread(_write)


async def read_text(path: epath.Path) -> str:
  return await asyncio.to_thread(path.read_text)


async def exists(path: epath.Path):
  return await asyncio.to_thread(path.exists)


async def async_stat(path: epath.Path):
  return await asyncio.to_thread(path.stat)


async def async_rmtree(path: epath.Path):
  return await asyncio.to_thread(path.rmtree)


async def rmtree(path: epath.Path):
  return await asyncio.to_thread(path.rmtree)


async def touch(path: epath.Path, *, exist_ok: bool = False):
  return await asyncio.to_thread(path.touch, exist_ok=exist_ok)


async def rename(src: epath.Path, dst: epath.Path):
  return await asyncio.to_thread(src.rename, dst)


async def is_dir(path: epath.Path):
  return await asyncio.to_thread(path.is_dir)


async def is_file(path: epath.Path):
  return await asyncio.to_thread(path.is_file)


async def is_link(path: epath.Path):
  return await asyncio.to_thread(os.path.islink, path)


async def iterdir(path: epath.Path):
  return await asyncio.to_thread(path.iterdir)


async def glob(path: epath.Path, pattern: str) -> Iterator[epath.Path]:
  return await asyncio.to_thread(path.glob, pattern)


async def unlink(path: epath.Path, missing_ok: bool = False):
  return await asyncio.to_thread(path.unlink, missing_ok=missing_ok)


class AsyncFile:
  """Async wrapper for file operations."""

  def __init__(self, f: Any):
    self._f = f

  async def read(self, n: int = -1) -> bytes | str:
    return await asyncio.to_thread(self._f.read, n)

  async def seek(self, offset: int, whence: int = 0) -> int:
    return await asyncio.to_thread(self._f.seek, offset, whence)

  async def tell(self) -> int:
    return await asyncio.to_thread(self._f.tell)

  async def write(self, data: Any) -> int:
    return await asyncio.to_thread(self._f.write, data)

  async def flush(self) -> None:
    return await asyncio.to_thread(self._f.flush)

  async def close(self) -> None:
    return await asyncio.to_thread(self._f.close)


@contextlib.asynccontextmanager
async def open_file(
    path: epath.Path, mode: str = 'rb'
) -> AsyncIterator[AsyncFile]:
  """Async context manager for opening files."""
  f = await asyncio.to_thread(path.open, mode=mode)
  try:
    yield AsyncFile(f)
  finally:
    await asyncio.to_thread(f.close)
