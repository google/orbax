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

"""Renders V1 metadata trees as JSON-compatible values."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from orbax.checkpoint._src.tree import types as tree_types

JsonValue = tree_types.JsonValue

_VALUE_TYPE = 'value_type'
# Scalar and string leaf handlers report a type witness, e.g. `0` or
# `'string'`, instead of the saved value. Mirrors `arrays.types.Scalar`.
_SCALAR_TYPES = (bool, int, float, bytes, np.number)


@runtime_checkable
class JsonSerializable(Protocol):
  """Metadata that renders itself as a JSON-compatible dict."""

  def to_json_dict(self) -> dict[str, JsonValue]:
    """Returns this metadata as a JSON-compatible dict."""
    ...


def shape_to_json(shape: Sequence[int] | None) -> list[JsonValue] | None:
  """Returns `shape` as a list of Python ints, or None."""
  if shape is None:
    return None
  return [int(d) for d in shape]


def dtype_to_json(dtype: np.typing.DTypeLike | None) -> str | None:
  """Returns the canonical name of `dtype`, e.g. `'bfloat16'`, or None."""
  if dtype is None:
    return None
  try:
    return np.dtype(dtype).name
  except TypeError:
    return str(dtype)


def leaf_to_json(leaf: Any) -> JsonValue:
  """Renders one metadata leaf as a JSON-compatible value.

  Args:
    leaf: A metadata tree leaf, e.g. `ArrayMetadata`, or the type witness that a
      scalar or string leaf handler reports.

  Returns:
    `leaf.to_json_dict()` if the leaf provides it, a `value_type`-tagged dict
    for scalar and string witnesses, None for None, and otherwise an
    `unknown` dict that carries the leaf's repr.
  """
  if leaf is None:
    return None
  # Classes may define `to_json_dict` too, but cannot be called unbound.
  if isinstance(leaf, JsonSerializable) and not isinstance(leaf, type):
    return leaf.to_json_dict()
  if isinstance(leaf, str):
    return {_VALUE_TYPE: 'string'}
  if isinstance(leaf, _SCALAR_TYPES):
    return {_VALUE_TYPE: 'scalar', 'python_type': type(leaf).__name__}
  # TODO(dnlng): Render `jax.ShapeDtypeStruct` leaves, e.g. from Safetensors
  # checkpoints, as structured JSON instead of a repr.
  return {_VALUE_TYPE: 'unknown', 'repr': repr(leaf)}


def tree_to_json(
    tree: Any,
    *,
    leaf_fn: Callable[[Any], JsonValue] = leaf_to_json,
) -> JsonValue:
  """Renders a metadata tree as nested JSON-compatible dicts and lists.

  Args:
    tree: A tree of mappings, lists and tuples; anything else is a leaf.
    leaf_fn: Renders each leaf. Defaults to `leaf_to_json`.

  Returns:
    The tree with mapping keys converted to strings, tuples converted to lists
    and each leaf rendered by `leaf_fn`.
  """
  if isinstance(tree, Mapping):
    return {str(k): tree_to_json(v, leaf_fn=leaf_fn) for k, v in tree.items()}
  if isinstance(tree, (list, tuple)):
    return [tree_to_json(v, leaf_fn=leaf_fn) for v in tree]
  return leaf_fn(tree)
