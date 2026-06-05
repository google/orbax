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

"""Validation utilities for tree types."""

import typing
from typing import Any

from orbax.checkpoint.experimental.v1._src.arrays import types as array_types
from orbax.checkpoint.experimental.v1._src.serialization import protocol_utils
from orbax.checkpoint.experimental.v1._src.tree import types


def is_supported_leaf(x: Any) -> bool:
  """Returns True if the given object is a supported concrete Leaf."""
  return isinstance(x, types.Leaf)


def is_supported_abstract_leaf(x: Any) -> bool:
  """Returns True if the given object is a supported AbstractLeaf."""
  if x is types.PLACEHOLDER:
    return True

  if isinstance(x, type):
    if protocol_utils.is_subclass_protocol(x, array_types.AbstractArray):
      return True
    if protocol_utils.is_subclass_protocol(x, array_types.AbstractShardedArray):
      return True
    return issubclass(x, typing.get_args(array_types.AbstractScalar) + (str,))

  if protocol_utils.is_subclass_protocol(type(x), array_types.AbstractArray):
    return True
  if protocol_utils.is_subclass_protocol(
      type(x), array_types.AbstractShardedArray
  ):
    return True
  return isinstance(x, (array_types.AbstractScalar, str))
