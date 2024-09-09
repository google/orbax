# Copyright 2024 The Orbax Authors.
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

"""Utilities for managing storage layout of arrays in checkpoints."""

import math
from typing import Union

from absl import logging
from jax import numpy as jnp
import numpy as np
from orbax.checkpoint._src.arrays import types


Shape = types.Shape


_MIB = 1024**2  # 1 MiB


def _find_divisors(size: int):
  """Fast-ish method for finding divisors of a number."""
  sqrt_divs = [
      i for i in range(1, math.ceil(math.sqrt(size + 1))) if size % i == 0
  ]
  return sorted(set(sqrt_divs + [size // div for div in sqrt_divs][::-1]))


def choose_chunk_shape(
    global_shape: Shape,
    write_shape: Shape,
    dtype: Union[jnp.dtype, np.dtype],
    target_byte_size: int,
    *,
    shard_axes: tuple[int, ...] = (),
) -> Shape:
  """Chooses a chunk shape that divides the `write_shape`.

  The chunk shape is chosen such that the resulting byte size is less than
  or equal to `target_byte_size`, but is otherwise as large as possible.

  This uses a greedy algorithm that attempts to split the largest and sharded
  dimensions first, unless the `shard_axes` optional parameter is also provided.
  In the latter case, the algorithm will prioritize these explicitly specified
  axes and ensure that array's storage representation is sharded at least once
  on as many of these axes as possible.

  Args:
    global_shape: The global shape of the array.
    write_shape: The local shape being written.
    dtype: The dtype of the array.
    target_byte_size: Desired chunk byte size. Must be >= dtype.itemsize.
    shard_axes: [optional] A list of axes that should be prioritized for
      storage sharding. The implementation will try to shard at least once on as
      many of these axes as possible.

  Returns:
    List of length `len(write_shape)` specifying the chosen chunk shape.
  """
  if len(global_shape) != len(write_shape):
    raise ValueError(
        f'global_shape={global_shape} and write_shape={write_shape} must have'
        ' the same length.'
    )
  if target_byte_size < 1 * _MIB:
    logging.warning(
        'Setting the target_byte_size too small could reduce performance.'
    )

  sharded_dimensions = np.array(global_shape) != np.array(write_shape)
  dtype_size = dtype.itemsize
  target_elements = target_byte_size // dtype_size

  rank = len(write_shape)

  # `dim_factors[i]` is the list of divisors of `write_shape[i]`
  dim_factors = [_find_divisors(size) for size in write_shape]

  # The current chunk shape is:
  # [dim_factors[i][-1] for i in range(rank)]

  total_elements = math.prod(write_shape)

  def reduce_dim(dim_to_reduce: int) -> None:
    """Reduces the given dimension in the current chunk shape."""
    nonlocal total_elements
    current_dim = dim_factors[dim_to_reduce].pop()
    new_dim = dim_factors[dim_to_reduce][-1]
    total_elements = (total_elements // current_dim) * new_dim
    sharded_dimensions[dim_to_reduce] = True

  # First, try to reduce the size of the chunk shape on the `shard_axes`.
  # If some of these specified axes are already sharded, we will skip them on
  # the first iteration which ensures that we shard at least once on each of the
  # `shard_axes`. It might also be the case that the given target_byte_size is
  # too big to shard on all of the requested axes, in which case we will
  # maximize the number of the number of axes that are sharded.
  could_shard = bool(shard_axes)
  first_sharding_iteration = True
  while could_shard and total_elements > target_elements:
    could_shard = False
    # For the first pass, exclude dimensions that are already sharded.
    # We do our best to shard at least once of each of the `shard_axes`.
    if first_sharding_iteration:
      must_shard_dims = (i for i in shard_axes if not sharded_dimensions[i])
      first_sharding_iteration = False
    else:
      must_shard_dims = shard_axes
    # Exclude dimensions that can no longer be sharded.
    must_shard_dims = set(i for i in must_shard_dims if len(dim_factors[i]) > 1)
    # Shard once on each of the remaining dimensions in a round-robin fashion,
    # while we can.
    while must_shard_dims and total_elements > target_elements:
      could_shard = True
      # Find the minimum available divisor among the remaining dimensions.
      dim_idx = min(
          must_shard_dims,
          key=lambda i: dim_factors[i][-1] // dim_factors[i][-2],
      )
      reduce_dim(dim_idx)
      must_shard_dims.remove(dim_idx)

  if shard_axes:
    current_shape = tuple(dim_factors[i][-1] for i in range(rank))
    if current_shape != write_shape:
      logging.vlog(
          1,
          'Reduced write shape using shard_axes=%s: global_shape=%s,'
          ' write_shape=%s, dtype=%s, target_byte_size=%d; reduced shape: %s',
          shard_axes,
          global_shape,
          write_shape,
          dtype,
          target_byte_size,
          current_shape,
      )

  # If we are not within target_byte_size yet, continue to reduce the current
  # chunk shape until the desired number of elements is reached.
  while total_elements > target_elements:
    # Greedily reduce the largest dimension.  This is not guaranteed to bring us
    # the closest to `target_elements`, but is simple to implement and should
    # work well enough.
    dim_to_reduce = -1
    dim_to_reduce_size = 1
    for i in range(rank):
      size = dim_factors[i][-1]
      if sharded_dimensions[i] and size > dim_to_reduce_size:
        dim_to_reduce_size = size
        dim_to_reduce = i

    if dim_to_reduce_size > 1:
      reduce_dim(dim_to_reduce)
    else:
      # We need to start splitting on unsharded dimensions.
      sharded_dimensions = np.ones(len(write_shape))

  chosen_shape = tuple(dim_factors[i][-1] for i in range(rank))

  # TODO: b/363218206 - Consider info logging the storage shape in saving and
  # loading code.
  logging.vlog(
      1,
      'Reduced write shape: global_shape=%s, write_shape=%s, dtype=%s,'
      ' target_byte_size=%d, shard_axes=%s; chosen shape: %s',
      global_shape,
      write_shape,
      dtype,
      target_byte_size,
      shard_axes,
      chosen_shape,
  )

  return chosen_shape


def validate_divisible_shapes(
    divided_shape: Shape,
    dividing_shape: Shape,
) -> bool:
  """Returns True only if dividing_shape is a divisor of divided_shape."""
  try:
    return not np.mod(divided_shape, dividing_shape).any()
  except ValueError:
    # eg. imcompatible shape
    return False
