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

"""Provides dispatchers for running functions on multiple workers."""

import abc
from collections.abc import Sequence
from typing import Any, Callable

from absl import logging
import jax
import jax.experimental.colocated_python as cp
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.multihost import multihost


PyTree = Any


def get_dummy_input_array(
    devices: Sequence[jax.Device],
) -> jax.Array:
  """Returns a dummy array with replicated sharding on the given devices."""
  mesh = jax.sharding.Mesh(np.array(devices), ('d',))
  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  return jax.device_put(jnp.array(True, dtype=jnp.bool), sharding)


def _make_dummy_result_array(
    pytree: PyTree, abstract: bool = False
) -> jax.Array | jax.ShapeDtypeStruct:
  """Returns a dummy array with replicated across the devices in the pytree.

  Args:
    pytree: The pytree to use to determine the devices.
    abstract: Whether to return a jax.ShapeDtypeStruct instead of a jax.Array.

  Returns:
    A dummy array with replicated sharding across the devices in the pytree.
    If abstract is True, returns a jax.ShapeDtypeStruct instead.
  """
  devices = set()
  jax.tree.map(lambda x: devices.update(x.sharding.device_set), pytree)

  replicated_sharding = jax.sharding.NamedSharding(
      jax.sharding.Mesh(np.array(list(devices)), ('d',)),
      jax.sharding.PartitionSpec(),
  )
  if abstract:
    return jax.ShapeDtypeStruct((), jnp.bool, sharding=replicated_sharding)
  else:
    return jax.make_array_from_callback(
        (), replicated_sharding, lambda _: True, dtype=jnp.bool
    )


def _vlog_dispatch(fn: Callable[..., Any], dispatcher_name: str):
  if logging.vlog_is_on(1):
    logging.vlog(
        1,
        'Executing function %r via %s on process=%s/%s',
        fn,
        dispatcher_name,
        multihost.process_index(),
        multihost.process_count(),
    )


class Dispatcher(abc.ABC):
  """Dispatches a function to run on workers and returns a future."""

  @abc.abstractmethod
  def dispatch(
      self,
      func: Callable[..., PyTree | None],
      *,
      input_arrays: PyTree,
      result_specs: PyTree | None = None,
      func_args: tuple[Any, ...] = (),
      func_kwargs: dict[str, Any] | None = None,
  ) -> PyTree:
    """Dispatches func with given arrays.

    Args:
      func: Function to dispatch. Must be synchronous and return a PyTree of
        jax.Arrays matching result_specs.
      input_arrays: Input arrays to pass to func.
      result_specs: A PyTree of jax.ShapeDtypeStructs describing fn's output and
        the desired final shardings of the result from func on TPU. If None,
        func output is discarded and a dummy array is returned.
      func_args: Positional arguments for func.
      func_kwargs: Keyword arguments for func.

    Returns:
      A PyTree of jax.Arrays matching result_specs or a dummy array signaling
      completion of func if result_specs is None.
    """
    ...


class ColocatedPythonDispatcher(Dispatcher):
  """Dispatches functions using colocated Python."""

  def _colocated_cpu_sharding(
      self,
      sharding: jax.sharding.Sharding,
  ) -> jax.sharding.Sharding:
    """Returns a CPU sharding colocated with the given device sharding."""
    if isinstance(sharding, jax.sharding.NamedSharding):
      cpu_mesh = cp.colocated_cpu_devices(sharding.mesh)
      return jax.sharding.NamedSharding(cpu_mesh, sharding.spec)
    else:
      raise TypeError(
          f'Sharding type {type(sharding)} not supported in'
          ' to_colocated_python.'
      )

  def to_colocated_python(self, input_tree: PyTree) -> PyTree:
    """Copies a PyTree of arrays to colocated CPU devices."""

    def _get_sharding(x):
      if isinstance(x, jax.Array):
        return self._colocated_cpu_sharding(x.sharding)
      return None

    cpu_sharding_tree = jax.tree.map(_get_sharding, input_tree)
    return jax.device_put(input_tree, cpu_sharding_tree)

  def _transform_pytree_shardings(self, input_tree: PyTree) -> Any:
    """Converts Sharding or ShapeDtypeStruct args to use CPU devices.

    Args:
      input_tree: The input tree to transform.

    Returns:
      The input tree with CPU shardings.
    """

    def _transform_leaf_sharding(leaf: Any):
      if isinstance(leaf, jax.sharding.Sharding):
        return self._colocated_cpu_sharding(leaf)
      if isinstance(leaf, jax.ShapeDtypeStruct) and hasattr(leaf, 'sharding'):
        cpu_sharding = self._colocated_cpu_sharding(leaf.sharding)
        return jax.ShapeDtypeStruct(
            leaf.shape, leaf.dtype, sharding=cpu_sharding
        )
      # Typically, only sharding specs within args/kwargs need conversion.
      return leaf

    return jax.tree.map(_transform_leaf_sharding, input_tree)

  def _to_final_specs(
      self,
      input_tree: PyTree,
      tpu_or_cpu_specs: PyTree,
  ) -> jax.Array:
    """Transfers jax.Arrays to the final sharding specs.

    Args:
      input_tree: The input pytree containing jax.Arrays to transfer.
      tpu_or_cpu_specs: The final sharding specs on TPU or CPU.

    Returns:
      Pytree with jax.Arrays transferred to the final sharding spec.
    """

    def _to_final_spec(leaf, tpu_or_cpu_spec):
      if isinstance(leaf, jax.Array) and hasattr(tpu_or_cpu_spec, 'sharding'):
        logging.vlog(
            1,
            'Transferring array from %s to final sharding %s',
            leaf.sharding,
            tpu_or_cpu_spec.sharding,
        )
      return jax.device_put(leaf, tpu_or_cpu_spec.sharding)

    return jax.tree.map(_to_final_spec, input_tree, tpu_or_cpu_specs)

  def dispatch(
      self,
      func: Callable[..., PyTree | None],
      *,
      input_arrays: PyTree,
      result_specs: PyTree | None = None,
      func_args: tuple[Any, ...] = (),
      func_kwargs: dict[str, Any] | None = None,
  ) -> PyTree:
    """Dispatches func with given arrays using colocated Python.

    Args:
      func: Function to dispatch. Must be synchronous and return a PyTree of
        jax.Arrays matching result_specs.
      input_arrays: Input arrays to pass to func.
      result_specs: A PyTree of jax.ShapeDtypeStructs describing fn's output and
        the desired final shardings of the result from func on TPU. If None,
        func output is discarded and a dummy array is returned.
      func_args: Positional arguments for func.
      func_kwargs: Keyword arguments for func.

    Returns:
      A PyTree of jax.Arrays matching result_specs or a dummy array signaling
      completion of func if result_specs is None.
    """
    is_func_output_discarded = result_specs is None
    if func_kwargs is None:
      func_kwargs = {}

    cpu_args = self._transform_pytree_shardings(func_args)
    cpu_kwargs = self._transform_pytree_shardings(func_kwargs)

    @cp.colocated_python
    def _cp_wrapper(inp: PyTree) -> PyTree:
      _vlog_dispatch(func, 'ColocatedPythonDispatcher')
      if is_func_output_discarded:
        func(inp, *cpu_args, **cpu_kwargs)
        return _make_dummy_result_array(inp)
      else:
        return func(inp, *cpu_args, **cpu_kwargs)

    result_specs = result_specs or _make_dummy_result_array(
        input_arrays, abstract=True
    )
    cpu_result_specs = self._transform_pytree_shardings(result_specs)
    _cp_wrapper.specialize(out_specs_fn=lambda _: cpu_result_specs)

    result = _cp_wrapper(self.to_colocated_python(input_arrays))
    return self._to_final_specs(result, result_specs)


