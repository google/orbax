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

"""Utility functions for PyTree operations in benchmarks."""

from collections.abc import Callable, Sequence
import hashlib
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
          "%r path: %r: %r, spec=%r, mesh=%r,\nvalue=%r\n dtype=%r",
          msg,
          path,
          x.shape,
          x.sharding.spec,
          x.sharding.mesh,
          [shard.data for shard in x.addressable_shards],
          x.dtype if hasattr(x, "dtype") else None,
      ),
      pytree,
  )


def _leaf_path(keypath: tuple[Any, ...]) -> str:
  """Formats a jax keypath as one string ('<root>' for the empty path)."""
  parts = "".join(jax.tree_util.keystr((entry,)) for entry in keypath)
  return parts or "<root>"


def _leaf_to_numpy(leaf: Any) -> np.ndarray:
  """Materialises a leaf to a contiguous numpy array for hashing.

  Multi-host jax.Arrays span devices the local process can't reach, so we
  hash only this process's addressable shards, concatenated in a stable
  (sorted-by-index) order — each host hashes the slice it owns.

  Args:
    leaf: A pytree leaf — a numpy array, a jax.Array, or a scalar.

  Returns:
    A contiguous numpy array of this process's addressable data.
  """
  if isinstance(leaf, np.ndarray):
    return np.ascontiguousarray(leaf)
  if hasattr(leaf, "addressable_shards"):
    shards = list(leaf.addressable_shards)
    if not shards:
      return np.ascontiguousarray(np.zeros((0,), dtype=np.float32))
    shards.sort(key=lambda s: tuple(s.index or ()))
    return np.ascontiguousarray(
        np.concatenate([np.asarray(s.data).ravel() for s in shards])
    )
  return np.ascontiguousarray(np.asarray(leaf))


def digest_pytree(pytree: Any) -> dict[str, str]:
  """Computes a per-leaf SHA-256 digest keyed by leaf path.

  Useful for load-only benchmarks, where no in-memory reference pytree
  exists to pass to `assert_pytree_equal`: capture digests once, then check
  future loads against them with `assert_digests_match`.

  Args:
    pytree: The pytree to digest.

  Returns:
    Mapping of leaf path to the hex SHA-256 of its (dtype, shape, bytes).
  """
  leaves = jax.tree_util.tree_flatten_with_path(pytree)[0]
  digests: dict[str, str] = {}
  for keypath, leaf in leaves:
    arr = _leaf_to_numpy(leaf)
    h = hashlib.sha256()
    h.update(arr.dtype.str.encode("ascii"))
    h.update(repr(arr.shape).encode("ascii"))
    h.update(arr.tobytes())
    digests[_leaf_path(keypath)] = h.hexdigest()
  return digests


def assert_digests_match(expected_digests: dict[str, str], pytree: Any):
  """Asserts a pytree's per-leaf digests match a previously captured set.

  Args:
    expected_digests: Per-leaf SHA-256 from a prior `digest_pytree` call.
    pytree: The pytree to check.

  Raises:
    AssertionError: If a leaf is missing, unexpected, or its digest differs.
  """
  actual = digest_pytree(pytree)
  for path in sorted(set(expected_digests) ^ set(actual)):
    where = "missing in pytree" if path in expected_digests else "unexpected"
    raise AssertionError(f"Digest key mismatch at {path}: {where}")
  for path, expected in expected_digests.items():
    if actual[path] != expected:
      raise AssertionError(
          f"Digest mismatch at {path}: expected {expected}, got {actual[path]}"
      )


def assert_functional_equivalence(
    model_apply_fn: Callable[[Any, Any], Any],
    reference_inputs: Sequence[Any],
    expected_params: Any,
    actual_params: Any,
    tolerance: float = 1e-6,
):
  """Asserts two parameter sets produce equivalent model outputs.

  Runs `model_apply_fn(params, x)` for each reference input against both
  parameter sets and compares the outputs. A NaN or shape mismatch fails
  closed (the comparison is `not max_abs_diff <= tolerance`).

  Args:
    model_apply_fn: `apply(params, x) -> y`.
    reference_inputs: Non-empty sequence of inputs to run through the model.
    expected_params: Reference parameter set.
    actual_params: Parameter set validated against the reference.
    tolerance: Maximum allowed absolute output difference per input.

  Raises:
    ValueError: If `reference_inputs` is empty.
    AssertionError: If any input's output diverges beyond `tolerance`.
  """
  if not reference_inputs:
    raise ValueError("reference_inputs must be a non-empty sequence")
  for idx, x in enumerate(reference_inputs):
    expected_out = np.asarray(
        model_apply_fn(expected_params, x), dtype=np.float64
    )
    actual_out = np.asarray(model_apply_fn(actual_params, x), dtype=np.float64)
    if expected_out.shape != actual_out.shape:
      raise AssertionError(
          f"Output shape mismatch on input {idx}: "
          f"{expected_out.shape} vs {actual_out.shape}"
      )
    diff = (
        float(np.max(np.abs(expected_out - actual_out)))
        if expected_out.size
        else 0.0
    )
    if not diff <= tolerance:
      raise AssertionError(
          f"Functional divergence on input {idx}: "
          f"max_abs_diff={diff} > tolerance={tolerance}"
      )
