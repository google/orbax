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

"""Shorthand for `Checkpointer(StandardCheckpointHandler())`."""

from typing import Optional
from orbax.checkpoint import checkpointer
from orbax.checkpoint import standard_checkpoint_handler


class StandardCheckpointer(checkpointer.Checkpointer):
  """Shorthand class.

  Instead of::
    ckptr = Checkpointer(StandardCheckpointHandler())

  we can use::
    ckptr = StandardCheckpointer()
  """

  def __init__(self, primary_host: Optional[int] = 0):
    super().__init__(
        standard_checkpoint_handler.StandardCheckpointHandler(
            primary_host=primary_host
        ),
        primary_host=primary_host,
    )
