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

"""Repeating utilities for model surgery."""

import functools
import re
from typing import Sequence

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.model_surgery.transformations import types


Transformation = types.Transformation


def _is_host_array(x) -> bool:
  """True if x is a jax.Array on CPU."""
  return isinstance(x, jax.Array) and next(iter(x.devices())).platform == "cpu"


@functools.partial(jax.jit, backend="cpu", static_argnames=("repeats", "axis"))
def _cpu_repeat(array, *, repeats, axis):
  return jnp.repeat(array, repeats, axis=axis)


def _repeat_val(val, dimension: int, repeat_count: int) -> jax.Array:
  if _is_host_array(val):
    # Ensure that host arrays are repeated on CPU, to avoid unnecessary
    # device transfers.
    return _cpu_repeat(val, repeats=repeat_count, axis=dimension)
  elif isinstance(val, np.ndarray):
    return np.repeat(val, repeat_count, axis=dimension)
  else:
    return jnp.repeat(val, repeat_count, axis=dimension)


def repeat_by_pattern(
    *,
    pattern: str,
    dimension: int,
    repeat_count: int,
) -> Transformation:
  r"""Repeats parameters by finding keys that match a regex pattern.

  Example:
      pattern = r"^(.*)\\.weight$"
      dimension = 1
      repeat_count = 2

        Transforms:
          "model.layers.0.weight": [[1, 2], [3, 4]]
        Into:
          "model.layers.0.weight": [[1, 1, 2, 2], [3, 3, 4, 4]]

  Args:
      pattern: Regex to filter keys that will be repeated.
      dimension: The axis/dimension to repeat along.
      repeat_count: Number of times to repeat elements.

  Returns:
      A Transformation function.
  """
  compiled_pattern = re.compile(pattern)

  def transform(
      *params: types.PyTreeOf[jax.Array],
  ) -> types.PyTreeOf[jax.Array]:
    if len(params) > 1:
      raise ValueError(
          "Can only repeat parameters in a single parameter structure."
      )
    params = params[0]
    result = dict(params)
    del params

    for key in result:
      if compiled_pattern.match(key):
        result[key] = _repeat_val(result[key], dimension, repeat_count)

    return result

  return transform


def repeat_by_keys(
    *,
    target_keys: Sequence[str],
    dimension: int,
    repeat_count: int,
) -> Transformation:
  """Repeats specific target keys.

  Example::
      target_keys = ["layer0.weight"]
      dimension = 1
      repeat_count = 2

      # Transforms:
      #   "layer0.weight": [[1, 2], [3, 4]]
      # Into:
      #   "layer0.weight": [[1, 1, 2, 2], [3, 3, 4, 4]]

  Args:
      target_keys: Sequence of keys to repeat.
      dimension: The axis/dimension to repeat along.
      repeat_count: Number of times to repeat elements.

  Returns:
      A Transformation function.
  """

  def transform(
      *params: types.PyTreeOf[jax.Array],
  ) -> types.PyTreeOf[jax.Array]:
    if len(params) > 1:
      raise ValueError(
          "Can only repeat parameters in a single parameter structure."
      )
    params = params[0]
    result = dict(params)
    del params

    missing_keys = [k for k in target_keys if k not in result]
    if missing_keys:
      logging.warning(
          "Could not repeat keys %s. They were not found in params.",
          missing_keys,
      )

    for k in target_keys:
      if k in result:
        result[k] = _repeat_val(result[k], dimension, repeat_count)

    return result

  return transform
