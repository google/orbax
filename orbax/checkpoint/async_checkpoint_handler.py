# Copyright 2022 The Orbax Authors.
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

"""AsyncCheckpointHandler interface."""

import abc
import asyncio
from typing import Any, Optional, Sequence


class AsyncCheckpointHandler(abc.ABC):
  """An interface providing async methods that can be used with CheckpointHandler."""

  @abc.abstractmethod
  async def async_save(self, directory: str, item: Any, *args,
                       **kwargs) -> Optional[Sequence[asyncio.Future]]:
    """Constructs a save operation.

    Synchronously awaits a copy of the item, before returning commit futures
    necessary to save the item.

    Args:
      directory: the directory to save to.
      item: the item to be saved.
      *args: additional arguments for save.
      **kwargs: additional arguments for save.
    """
    pass
