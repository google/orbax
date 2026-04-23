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

"""Validation functions involved in saving."""

from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout

RESERVED_CHECKPOINTABLE_KEYS = checkpoint_layout.RESERVED_CHECKPOINTABLE_KEYS
EMPTY_CHECKPOINTABLE_KEY = checkpoint_layout.EMPTY_CHECKPOINTABLE_KEY


def validate_save_checkpointables(checkpointables):
  """Validates the checkpointables dictionary.

  Args:
    checkpointables: A dictionary of checkpointables.

  Raises:
    ValueError: If any of the keys in checkpointables are reserved.
  """
  if not checkpointables or not isinstance(
      checkpointables, dict
  ):
    raise ValueError(
        '`checkpointables` must be a valid mapping of checkpointable names to'
        ' desired checkpointables to save, but got'
        f' {type(checkpointables)}'
    )

  if EMPTY_CHECKPOINTABLE_KEY in checkpointables:
    raise ValueError(
        'Empty string is not supported as a checkpointable name in'
        ' `save_checkpointables`. Each checkpointable name must be a valid'
        ' non-empty string name.'
    )
  if (
      provided_reserved_keys := checkpointables.keys()
      & RESERVED_CHECKPOINTABLE_KEYS
  ):
    raise ValueError(
        f'Provided reserved checkpointable keys: {provided_reserved_keys}.'
    )
