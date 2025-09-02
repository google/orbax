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

"""Utility functions for PyTree operations in benchmarks."""

from typing import Any
from absl import logging
import jax
import jax.numpy as jnp
import numpy as np


def assert_pytree_equal(pytree_expected: Any, pytree_actual: Any):
  """Asserts that two pytrees are equal, including their dtypes and shapes.

  Args:
    pytree_expected: The expected pytree.
    pytree_actual: The actual pytree.

  Raises:
    AssertionError: If the pytrees are not equal, or if their leaf types or
    dtypes do not match.
  """

  def _assert_equal(path: str, v_expected: Any, v_actual: Any):
    if not isinstance(v_actual, type(v_expected)):
      raise AssertionError(
          f"Type mismatch at {path}: Expected {type(v_expected)} vs Actual"
          f" {type(v_actual)}"
      )
    if hasattr(v_expected, "dtype") and hasattr(v_actual, "dtype"):
      if v_expected.dtype != v_actual.dtype:
        raise AssertionError(
            f"Dtype mismatch at {path}: Expected {v_expected.dtype} vs Actual"
            f" {v_actual.dtype}"
        )
    if isinstance(v_expected, jax.Array):
      for shard_expected, shard_actual in zip(
          v_expected.addressable_shards, v_actual.addressable_shards
      ):
        np.testing.assert_array_equal(
            shard_expected.data, shard_actual.data, err_msg=f"Error at {path}"
        )
    elif isinstance(v_expected, (np.ndarray, jnp.ndarray)):
      np.testing.assert_array_equal(
          v_expected, v_actual, err_msg=f"Error at {path}"
      )
    else:
      if v_expected != v_actual:
        raise AssertionError(
            f"Value mismatch at {path}: Expected {v_expected} vs Actual"
            f" {v_actual}"
        )

  jax.tree.map_with_path(_assert_equal, pytree_expected, pytree_actual)


def log_pytree(msg: str, pytree: Any):
  """Logs the pytree in a pretty format."""
  jax.tree.map_with_path(
      lambda path, x: logging.info(
          "%r path: %r: %r, spec=%r\nvalue=%r",
          msg,
          path,
          x.shape,
          x.sharding.spec,
          x,
      ),
      pytree,
  )
