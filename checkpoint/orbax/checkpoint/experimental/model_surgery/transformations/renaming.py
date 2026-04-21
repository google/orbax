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

"""Renaming utilities for model surgery."""

import re
from typing import Sequence, Tuple

import jax

from orbax.checkpoint.experimental.model_surgery.transformations import types

Transformation = types.Transformation


def rename_by_regex(
    rules: Sequence[Tuple[str, str]],
) -> Transformation:
  r"""Renames parameters by applying a series of regex replacement rules.

  Example::

    rules = [
        (r"^params\.", "carried_state."),  # Strip prefix
        (r"\.weight_scale_inv$", ".weight_scale"), # Strip suffix
    ]
    transform = rename_by_regex(rules)
    result = transform(params)

  Args:
      rules: Sequence of (pattern, replacement) tuples. Every rule is applied
        in order to every key, potentially multiple times to the same key.

  Returns:
      A Transformation function that accepts a parameter structure and returns
      a pytree with renamed parameters.
  """
  compiled_rules = [(re.compile(pat), repl) for pat, repl in rules]

  def transform(
      *params: types.PyTreeOf[jax.Array],
  ) -> types.PyTreeOf[jax.Array]:
    if len(params) > 1:
      raise ValueError(
          'Can only rename parameters in a single parameter structure.'
      )
    params = params[0]
    result = {}
    for key, val in params.items():
      new_key = key
      for pattern, replacement in compiled_rules:
        new_key = pattern.sub(replacement, new_key)
      result[new_key] = val
    return result

  return transform
