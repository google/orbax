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

"""Fusing utilities for model surgery."""

import collections
import functools
import re
from typing import Any, Sequence

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.model_surgery.transformations import types


Transformation = types.Transformation


def _is_host_array(x) -> bool:
  """True if x is a jax.Array on CPU."""
  return isinstance(x, jax.Array) and next(iter(x.devices())).platform == "cpu"


@functools.partial(jax.jit, backend="cpu", static_argnames=("axis",))
def _cpu_concat(arrays, *, axis):
  return jnp.concatenate(arrays, axis=axis)


def _fuse_keys(
    params_dict: dict[str, Any],
    keys_to_fuse: Sequence[str],
    fused_key: str,
    axis: int,
) -> None:
  """Fuses values of keys_to_fuse into fused_key in params_dict."""
  vals_to_fuse = [params_dict[k] for k in keys_to_fuse]

  if all(_is_host_array(x) for x in vals_to_fuse):
    logging.info("DEBUG: Fusing %s on CPU", fused_key)
    # Force concatenation on CPU using JAX to avoid touching TPU
    fused_val = _cpu_concat(vals_to_fuse, axis=axis)
  else:
    is_numpy = isinstance(vals_to_fuse[0], np.ndarray)
    concat_fn = np.concatenate if is_numpy else jnp.concatenate
    fused_val = concat_fn(vals_to_fuse, axis=axis)

  for k in keys_to_fuse:
    del params_dict[k]
  params_dict[fused_key] = fused_val


def fuse_by_pattern(
    *,
    pattern: str,
    unique_parts: Sequence[str],
    fused_unique_part: str,
    axis: int = 0,
) -> Transformation:
  r"""Fuses parameters by finding sets that match a pattern.

  Example:
      pattern = r"^(.*)\.(gate_proj|up_proj)\.weight$"
      unique_parts = ["gate_proj", "up_proj"]
      fused_unique_part = "gate_up_proj"

        Transforms:
          "model.layers.0.gate_proj.weight": arr1
          "model.layers.0.up_proj.weight": arr2
        Into:
          "model.layers.0.gate_up_proj.weight": jnp.concatenate([arr1, arr2])

  Args:
      pattern: Regex to filter keys that are candidates for fusing.
      unique_parts: Ordered sequence of unique parts to find and concatenate.
      fused_unique_part: The replacement unique part for the fused key.
      axis: Axis to concatenate along.

  Returns:
      A Transformation function.
  """
  compiled_pattern = re.compile(pattern)
  unique_regex = "|".join([re.escape(p) for p in unique_parts])
  compiled_unique = re.compile(unique_regex)

  def transform(
      *params: types.PyTreeOf[jax.Array],
  ) -> types.PyTreeOf[jax.Array]:
    if len(params) > 1:
      raise ValueError(
          "Can only fuse parameters in a single parameter structure."
      )
    params = params[0]
    groups = collections.defaultdict(dict)

    for key in params:
      if not compiled_pattern.match(key):
        continue
      match_unique = compiled_unique.search(key)
      if not match_unique:
        continue

      fused_key = compiled_unique.sub(lambda _: fused_unique_part, key)
      unique_part = match_unique.group(0)
      if unique_part in unique_parts:
        groups[fused_key][unique_part] = key

    result = dict(params)
    del params
    for fused_key, unique_dict in groups.items():
      if len(unique_dict) == len(unique_parts):
        keys_to_fuse = [unique_dict[p] for p in unique_parts]
        _fuse_keys(result, keys_to_fuse, fused_key, axis)
      else:
        logging.warning(
            "Could not fuse %s. Found parts: %s, expected: %s",
            fused_key,
            list(unique_dict.keys()),
            unique_parts,
        )

    return result

  return transform


def fuse_by_keys(
    *,
    source_keys: Sequence[str],
    target_key: str,
    axis: int = 0,
) -> Transformation:
  """Fuses a specific set of source keys into a single target key.

  Example::
      source_keys = ["layer0.gate", "layer0.up"]
      target_key = "layer0.gate_up"

      # Transforms:
      #   "layer0.gate": arr1
      #   "layer0.up": arr2
      # Into:
      #   "layer0.gate_up": jnp.concatenate([arr1, arr2])

  Args:
      source_keys: Ordered sequence of keys to find and concatenate.
      target_key: The replacement key for the fused key.
      axis: Axis to concatenate along.

  Returns:
      A Transformation function.
  """

  def transform(
      *params: types.PyTreeOf[jax.Array],
  ) -> types.PyTreeOf[jax.Array]:
    if len(params) > 1:
      raise ValueError(
          "Can only fuse parameters in a single parameter structure."
      )
    params = params[0]
    result = dict(params)
    del params
    found_keys = [k for k in source_keys if k in result]
    if len(found_keys) == len(source_keys):
      _fuse_keys(result, source_keys, target_key, axis)
    elif found_keys:
      logging.warning(
          "Could not fuse %s. Found keys: %s, expected: %s",
          target_key,
          found_keys,
          source_keys,
      )

    return result

  return transform
