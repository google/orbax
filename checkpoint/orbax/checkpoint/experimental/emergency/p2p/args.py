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

"""P2P composite checkpoint argument."""

from typing import final
from orbax.checkpoint import args as args_lib
from orbax.checkpoint.experimental.emergency.p2p import constants


@final
class Composite(args_lib.Composite):
  """Composite argument that only supports 'state' key."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if constants.STATE_SUBDIR not in self or len(self) > 1:
      raise ValueError(
          f'Composite must contain "{constants.STATE_SUBDIR}" key and no other'
          f' keys: {list(self.keys())}'
      )
