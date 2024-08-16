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

"""Path utility functions for Orbax."""

# TODO(b/337137764): Add unit tests.
# TODO(b/337137764): If needed, export the functions from
# third_party/py/orbax/checkpoint/path/__init__.py. Currently, they are not used
# outside Orbax (ignoring OSS).

import asyncio
from typing import List, Optional, Tuple

from etils import epath
from orbax.checkpoint.path import async_utils
from orbax.checkpoint.path import step as step_lib


_LOCK_ITEM_NAME = 'LOCKED'


def lockdir(directory: epath.Path) -> epath.Path:
  """Constructs a directory used to indicate that a checkpoint step is `locked`."""
  return directory / _LOCK_ITEM_NAME


async def _async_is_locked(directory: epath.Path) -> bool:
  """(Async) determines whether a checkpoint step is considered `locked`."""
  parent_dir_exists = await async_utils.async_exists(directory)
  if not parent_dir_exists:
    raise ValueError(f'Parent directory {directory} does not exist.')
  return await async_utils.async_exists(lockdir(directory))


def is_locked(directory: epath.Path) -> bool:
  """Determines whether a checkpoint step is considered `locked`."""
  return asyncio.run(_async_is_locked(directory))


def are_locked(
    directory: epath.Path,
    steps: Tuple[int, ...],
    step_prefix: Optional[str] = None,
    step_format_fixed_length: Optional[int] = None,
    step_name_format: Optional[step_lib.NameFormat[step_lib.Metadata]] = None,
) -> List[bool]:
  """In parallel, determines whether the steps are considered `locked`."""
  step_name_format = step_name_format or step_lib.standard_name_format(
      step_prefix=step_prefix,
      step_format_fixed_length=step_format_fixed_length,
  )
  assert step_name_format is not None

  async def _run_in_parallel(ops):
    return await asyncio.gather(*ops)

  ops = [
      _async_is_locked(step_name_format.find_step(directory, step).path)
      for step in steps
  ]
  return asyncio.run(_run_in_parallel(ops))
