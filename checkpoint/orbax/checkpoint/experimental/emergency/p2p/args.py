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

from typing import Any, final

from orbax.checkpoint import args as args_lib
from orbax.checkpoint.experimental.emergency.p2p import constants
from orbax.checkpoint.experimental.emergency.p2p import utils


def _check_data_iter(value: Any):
  """Checks if data_iter is valid."""
  if utils.pygrain() is None:
    raise ImportError(
        'grain library is not available. Please install grain to use data_iter.'
    )
  if not isinstance(
      value,
      (
          utils.pygrain().PyGrainCheckpointSave,
          utils.pygrain().PyGrainCheckpointRestore,
      ),
  ):
    raise TypeError(f'Unsupported type for data_iter: {type(value)}')


@final
class Composite(args_lib.Composite):
  """Composite argument that supports 'state' and 'data_iter' keys."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if constants.STATE_SUBDIR not in self:
      raise ValueError(
          f'Composite must contain "{constants.STATE_SUBDIR}" key:'
          f' {list(self.keys())}'
      )
    for key in self:
      if key not in [constants.STATE_SUBDIR, constants.DATA_ITER_KEY]:
        raise ValueError(f'Unsupported key in Composite: {key}')
      if key == constants.DATA_ITER_KEY:
        _check_data_iter(self[key])
