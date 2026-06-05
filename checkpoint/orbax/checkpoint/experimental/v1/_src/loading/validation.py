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

"""Validation functions involved in loading."""

import jax
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types
from orbax.checkpoint.experimental.v1._src.tree import types_validation as tree_types_validation

RESERVED_CHECKPOINTABLE_KEYS = checkpoint_layout.RESERVED_CHECKPOINTABLE_KEYS
EMPTY_CHECKPOINTABLE_KEY = checkpoint_layout.EMPTY_CHECKPOINTABLE_KEY


def validate_state_checkpointable_name(
    checkpointable_name: str | None,
) -> None:
  """Validates the checkpointable name.

  Args:
    checkpointable_name: The name of the checkpointable.

  Raises:
    ValueError: If the checkpointable name is reserved.
  """
  if (
      checkpointable_name is None
      or checkpointable_name == checkpoint_layout.AUTO_CHECKPOINTABLE_KEY
  ):
    return
  if checkpointable_name == EMPTY_CHECKPOINTABLE_KEY:
    raise ValueError(
        'Empty string is not supported as a checkpointable name in'
        ' `load`. Checkpointable name must be a valid non-empty string'
        ' name or None if loading a legacy V0 direct pytree checkpoint.'
    )
  if checkpointable_name in RESERVED_CHECKPOINTABLE_KEYS:
    raise ValueError(
        f'Provided reserved checkpointable key: {checkpointable_name}.'
    )


def validate_abstract_checkpointables(
    abstract_checkpointables: (
        dict[str, handler_types.AbstractCheckpointable] | None
    ),
) -> None:
  """Validates the abstract_checkpointables dictionary.

  Args:
    abstract_checkpointables: A dictionary of abstract checkpointables.

  Raises:
    ValueError: If any of the keys in abstract_checkpointables are reserved.
  """
  if abstract_checkpointables is None:
    return
  if not isinstance(abstract_checkpointables, dict):
    raise ValueError(
        '`abstract_checkpointables` must be a valid mapping of checkpointable'
        ' names to abstract checkpointables to load, but got'
        f' {type(abstract_checkpointables)}'
    )
  if EMPTY_CHECKPOINTABLE_KEY in abstract_checkpointables:
    raise ValueError(
        'Empty string is not supported as a checkpointable name in'
        ' `load_checkpointables`. Each checkpointable name must be a valid'
        ' non-empty string name.'
    )
  if (
      provided_reserved_keys := abstract_checkpointables.keys()
      & RESERVED_CHECKPOINTABLE_KEYS
  ):
    raise ValueError(
        f'Provided reserved checkpointable keys: {provided_reserved_keys}.'
    )


def validate_abstract_state(
    abstract_state: tree_types.PyTreeOf[tree_types.AbstractLeaf] | None,
) -> None:
  """Validates an abstract PyTree state before loading."""
  if abstract_state is None:
    return
  leaves = jax.tree.leaves(abstract_state)
  for leaf in leaves:
    if not tree_types_validation.is_supported_abstract_leaf(leaf):
      raise TypeError(
          f'Unsupported abstract leaf type for loading: `{type(leaf)}`.'
          f' Supported abstract leaf types are: {tree_types.AbstractLeaf} (or'
          f' concrete values of {tree_types.Leaf}).'
      )
