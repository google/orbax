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
) -> Shape:
  """Chooses a chunk shape that divides the `write_shape`.

  The chunk shape is chosen such that the resulting byte size is less than
  or equal to `target_byte_size`, but is otherwise as large as possible.

  This uses a greedy algorithm that attempts to split the largest and sharded
  dimensions first.

  Args:
    global_shape: the global shape of the array
    write_shape: the local shape being written
    dtype: the dtype of the array
    target_byte_size: Desired chunk byte size.  Must be >= dtype.itemsize.

  Returns:
    List of length `len(write_shape)` specifying the chosen chunk shape.
  """
  if len(global_shape) != len(write_shape):
    raise ValueError(
        f'global_shape={global_shape} and write_shape={write_shape} must have'
        ' the same length.'
    )
  if target_byte_size < 1048576:  # 1 MB
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

  def get_total_elements():
    """Returns the number of elements in the current chunk shape."""
    total_elements = 1
    for i in range(rank):
      total_elements *= dim_factors[i][-1]
    return total_elements

  # Reduce the current chunk shape until the desired number of elements is
  # reached.
  while get_total_elements() > target_elements:
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
      dim_factors[dim_to_reduce].pop()
    else:
      # need to start splitting on unsharded dimension
      sharded_dimensions = np.ones(len(write_shape))

  chosen_shape = tuple(dim_factors[i][-1] for i in range(rank))

  logging.vlog(
      1,
      'global_shape=%s, write_shape=%s, dtype=%s, target_byte_size=%d,'
      ' chosen_shape=%s',
      global_shape,
      write_shape,
      dtype,
      target_byte_size,
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
