# Copyright 2025 The Orbax Authors.
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

"""Validation functions involved in loading."""

from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout


def validate_abstract_checkpointables(abstract_checkpointables):
  """Validates the abstract_checkpointables dictionary.

  Args:
    abstract_checkpointables: A dictionary of abstract checkpointables.

  Raises:
    ValueError: If any of the keys in abstract_checkpointables are reserved.
  """
  if abstract_checkpointables is None:
    return
  if (
      provided_reserved_keys := abstract_checkpointables.keys()
      & checkpoint_layout.RESERVED_CHECKPOINTABLE_KEYS
  ):
    raise ValueError(
        f'Provided reserved checkpointable keys: {provided_reserved_keys}.'
    )
