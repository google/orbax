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

"""Utils for P2P checkpointing."""

from absl import logging
from etils import epath
from orbax.checkpoint.experimental.emergency.p2p import constants


def detect_process_index(directory: epath.Path, step: int) -> int | None:
  """Inspects the disk to find which process index created this step."""
  step_path = directory / str(step)
  if not step_path.exists():
    return None

  # Check for standard Orbax/OCDBT structure
  # P2P checkpoint requires 'state' item in CompositeArgs
  try:
    item_path = step_path / constants.STATE_SUBDIR
    if item_path.exists():
      for path in item_path.glob(f'{constants.PROCESS_SUBDIR_PREFIX}*'):
        if path.is_dir():
          # Format: ocdbt.process_0, ocdbt.process_12, etc.
          return int(path.name.split('_')[-1])
  except (ValueError, IndexError, OSError) as e:
    logging.warning('Could not detect process index for step %d: %s', step, e)

  return None
