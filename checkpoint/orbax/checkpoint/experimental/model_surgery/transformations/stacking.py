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

"""Stacking utilities for model surgery."""

import collections
import re
from typing import Callable, Mapping, Sequence

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.model_surgery.transformations import types


Transformation = types.Transformation


def _streaming_stack(
    items: Sequence[jax.Array],
    axis: int,
    sharding: jax.sharding.Sharding | None = None,
) -> jax.Array:
  """Stacks items along a given axis.

  Memory optimized code path for stacking arrays when the source arrays are in
  host memory and the target array sharding is in device memory.

  Args:
    items: Sequence of arrays to stack.
    axis: Axis along which to stack.
    sharding: Optional target sharding for the stacked array.

  Returns:
    The stacked array.
  """
  num_arrays = len(items)
  base_shape = items[0].shape

  # Calculate the final shape, for example with 256 arrays of shape (32, 1024):
  # base=(32, 1024), axis=1 -> final=(32, 256, 1024)
  final_shape = list(base_shape)
  final_shape.insert(axis, num_arrays)
  final_shape = tuple(final_shape)

  def _callback(indices):
    # Extract the slice for the stacking axis
    stack_slice = indices[axis]
    start, stop, _ = stack_slice.indices(num_arrays)
    # Grab the requested chunk of arrays from the host list
    chunk_list = items[start:stop]
    # Stack them locally on the CPU along the correct axis
    stacked_chunk = jnp.stack(chunk_list, axis=axis)
    return stacked_chunk

  return jax.make_array_from_callback(tuple(final_shape), sharding, _callback)


def _is_host_array(x) -> bool:
  """True if x is a jax.Array on CPU."""
  return isinstance(x, jax.Array) and next(iter(x.devices())).platform == "cpu"


def _select_stack_fn(
    items: Sequence[jax.Array],
    sharding: jax.sharding.Sharding | None = None,
) -> Callable[[Sequence[jax.Array | np.ndarray], int], jax.Array | np.ndarray]:
  """Selects the stack function based on the input array sharding.

  * If any of the `items` is a numpy array, use np.stack.
  * If a sharding is specified and all the `items` are jax arrays that live in
    host memory, use _streaming_stack.
  * Otherwise, use jnp.stack.

  Args:
    items: Sequence of arrays to stack.
    sharding: Optional target sharding for the stacked array.

  Returns:
    The stack function to use for the given items.
  """
  if any(isinstance(x, np.ndarray) for x in items):
    return np.stack
  if sharding is not None and all(_is_host_array(x) for x in items):
    return lambda items, axis: _streaming_stack(items, axis, sharding)
  return jnp.stack


def stack(
    pattern: str,
    *,
    expected_count: int | None = None,
    axis: int = 0,
    filler_mapping: Mapping[str, float] | None = None,
    default_filler: float | None = None,
    inplace: bool = False,
    sort_by_size: bool = True,
    target_sharding: jax.sharding.Sharding | None = None,
) -> Transformation:
  r"""Stacks parameters by finding sets that match a pattern.

  The pattern must contain exactly one capture group with a positive integer,
  which is used to extract the index of the parameter in the stack. This capture
  group is removed from parameter names to form the base key for the stacked
  parameter.

  Example:
      pattern = r"mlp\.experts\.(\d+\.)"
      expected_count = 64

        Transforms:
          "layers.0.mlp.experts.0.weight": arr0
          "layers.0.mlp.experts.1.weight": arr1
          ...
          "layers.0.mlp.experts.63.weight": arr63
        Into:
          "layers.0.mlp.experts.weight": stack([arr0, arr1, ..., arr63])

  Args:
      pattern: Regex to filter keys for stacking. Must contain exactly one
        capture group, which is used to extract the index. This capture group is
        removed from parameter names to form the base key.
      expected_count: Expected number of indices to stack for each base key. If
        None, it is inferred as the maximum index found across all base keys
        plus one. If fewer items are found than expected, padding will be
        attempted. Padding requires `default_filler` or a matching entry in
        `filler_mapping` to be provided, otherwise an error will be raised.
      axis: Axis along which to stack.
      filler_mapping: Optional map from base key regex pattern to filler value.
        Used to pad missing indices. If padding is required and no filler is
        provided via this argument or `default_filler`, an error will be raised.
      default_filler: Default filler value if not in mapping. If padding is
        required and no filler is provided via this argument or
        `filler_mapping`, an error will be raised.
      inplace: If True, deletes matched keys from input params to save memory.
        Requires input params to be a dict.
      sort_by_size: If True, stacks largest parameters first to manage peak
        headroom.
      target_sharding: If not None, reshards the stacked parameter to this
        sharding.

  Returns:
      A Transformation function.
  """
  compiled_pattern = re.compile(pattern)

  def transform(
      *params: types.PyTreeOf[jax.Array],
  ) -> types.PyTreeOf[jax.Array]:
    if len(params) > 1:
      raise ValueError(
          "Can only stack parameters in a single parameter structure."
      )
    params = params[0]
    if inplace and not isinstance(params, dict):
      raise ValueError("Inplace operations require parameters to be a dict.")

    params_dict = params if isinstance(params, dict) else {}

    groups = collections.defaultdict(dict)
    unmatched = {}

    keys_to_delete = []

    for key, value in params.items():
      match = compiled_pattern.search(key)
      if not match:
        unmatched[key] = value
        continue

      if len(match.groups()) != 1:
        raise ValueError(
            "pattern must have exactly 1 capture group for the index, "
            f"got {len(match.groups())}: {pattern}."
        )

      idx_str = match.group(1)
      idx_matches = re.findall(r"\d+", idx_str)
      assert len(idx_matches) == 1, (
          "Capture group must contain exactly one single positive integer, "
          f"got {idx_str}."
      )
      idx = int(idx_matches[0])
      # Remove group 1 of pattern from the key to get the base key.
      base_key = key[: match.start(1)] + key[match.end(1) :]
      assert (
          idx not in groups[base_key]
      ), f"Duplicate index {idx} found for base_key {base_key}"
      groups[base_key][idx] = value

      if inplace:
        keys_to_delete.append(key)

    for key in keys_to_delete:
      del params_dict[key]

    if not groups:
      return unmatched

    # Determine types and stack function (use NumPy if inputs are NumPy)
    rep_val = next(iter(next(iter(groups.values())).values()))
    is_numpy = isinstance(rep_val, np.ndarray)
    ones_fn = np.ones if is_numpy else jnp.ones

    # Determine expected_count if not provided
    local_expected_count = expected_count
    if local_expected_count is None:
      local_expected_count = (
          max(max(idx_dict.keys()) for idx_dict in groups.values()) + 1
      )

    # Sort base_keys by size if requested (for peak memory optimization)
    base_keys = list(groups.keys())
    if sort_by_size:

      def _size_bytes(base_key):
        return sum(v.nbytes for v in groups[base_key].values())

      base_keys = sorted(base_keys, key=_size_bytes, reverse=True)

    result = dict(unmatched)
    for base_key in base_keys:
      idx_dict = groups.pop(base_key)

      # Determine filler
      filler_val = default_filler
      if filler_mapping:
        for p, val in filler_mapping.items():
          if re.search(p, base_key):
            filler_val = val
            break

      if filler_val is None:
        if sorted(idx_dict.keys()) != list(range(local_expected_count)):
          raise ValueError(
              f'Stacking "{base_key}": Found keys {sorted(idx_dict.keys())},'
              f" but expected indices 0..{local_expected_count - 1} when"
              " padding is disabled (no filler_mapping or default_filler was"
              " provided)."
          )
        items_to_stack = [idx_dict[i] for i in range(local_expected_count)]
        stack_fn = _select_stack_fn(items_to_stack, target_sharding)
        stacked = stack_fn(items_to_stack, axis)
        if target_sharding is not None and stack_fn in (np.stack, jnp.stack):
          stacked = jax.device_put(stacked, target_sharding)
      else:
        # Find representative shape/dtype
        rep_val = next(iter(idx_dict.values()))
        shape = list(rep_val.shape)
        shape.insert(axis, local_expected_count)
        dtype = rep_val.dtype

        # Initialize stacked array
        stacked = (ones_fn(shape, dtype=dtype) * filler_val).astype(dtype)
        if len(idx_dict) > local_expected_count:
          raise ValueError(
              f"Found {len(idx_dict)} items, but expected maximum"
              f" {local_expected_count} for {base_key}"
          )
        # Fill in values
        for idx, val in idx_dict.items():
          if idx >= local_expected_count:
            logging.warning(
                "Stacking %s: Found %d items, expected %d. Skipping index %d.",
                base_key,
                len(idx_dict),
                local_expected_count,
                idx,
            )
            continue
          slices = [slice(None)] * len(shape)
          slices[axis] = idx
          if is_numpy:
            stacked[tuple(slices)] = val
          else:
            stacked = stacked.at[tuple(slices)].set(val)

        if len(idx_dict) != local_expected_count:
          logging.warning(
              "Stacking %s: Found %d items, expected %d. Padded with %s.",
              base_key,
              len(idx_dict),
              local_expected_count,
              filler_val,
          )

        if target_sharding is not None:
          stacked = jax.device_put(stacked, target_sharding)

      result[base_key] = stacked

    return result

  return transform
