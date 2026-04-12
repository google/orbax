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

"""Provides dispatchers for running functions on multiple workers."""

import abc
from collections.abc import Sequence, Set
from typing import Any, Callable

from absl import logging
import jax
import jax.experimental.colocated_python as cp
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.multihost import colocated_transport
from orbax.checkpoint._src.multihost import multihost

PyTree = Any


def get_dummy_input_array(
    devices: Sequence[jax.Device],
) -> jax.Array:
  """Returns a dummy array with replicated sharding on the given devices."""
  mesh = jax.sharding.Mesh(np.array(devices), ('d',))
  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  return jax.device_put(jnp.array(True, dtype=jnp.bool), sharding)


def _get_dummy_input_array_from_result_specs(
    result_specs: PyTree,
) -> jax.Array:
  """Returns a dummy array with replicated sharding on a mesh found in specs."""
  device_lists = set()
  devices = set()

  def _collect(x):
    if hasattr(x, 'sharding') and x.sharding is not None:
      devices.update(x.sharding.device_set)
      if isinstance(x.sharding, jax.sharding.NamedSharding):
        device_lists.add(tuple(x.sharding.mesh.devices.flatten()))

  jax.tree.map(_collect, result_specs)

  if not device_lists:
    logging.warning('No mesh found in result_specs, using sorted device set.')
    device_list = sorted(list(devices), key=lambda d: d.id)
    return get_dummy_input_array(device_list)

  # colocated_python requires all args to be on the same device list.
  if len(device_lists) > 1:
    raise ValueError(f'Multiple meshes found in result specs: {device_lists}')
  return get_dummy_input_array(device_lists.pop())


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
  device_list: list[jax.Device] = sorted(list(devices), key=lambda d: d.id)
  replicated_sharding = jax.sharding.NamedSharding(
      jax.sharding.Mesh(device_list, ('d',)),
      jax.sharding.PartitionSpec(),
  )
  if abstract:
    return jax.ShapeDtypeStruct((), jnp.bool, sharding=replicated_sharding)
  else:
    return jax.make_array_from_callback(
        (), replicated_sharding, lambda _: True, dtype=jnp.bool
    )


def _collect_device_lists_from_pytree(
    tree: PyTree,
) -> frozenset[tuple[jax.Device, ...]]:
  """Collects all distinct device lists from NamedShardings in the PyTree."""
  device_lists = set()

  def _collect(x):
    if hasattr(x, 'sharding') and isinstance(
        x.sharding, jax.sharding.NamedSharding
    ):
      device_lists.add(tuple(x.sharding.mesh.devices.flatten()))

  jax.tree.map(_collect, tree)
  return frozenset(device_lists)


def _group_by_device_list(
    tree: PyTree, device_tuples: Set[tuple[jax.Device, ...]]
) -> dict[tuple[jax.Device, ...], PyTree]:
  """Groups the PyTree by device list, keeping only leaves matching each list."""
  groups: dict[tuple[jax.Device, ...], PyTree] = {}

  def _get_device_tuple(x):
    if hasattr(x, 'sharding') and isinstance(
        x.sharding, jax.sharding.NamedSharding
    ):
      return tuple(x.sharding.mesh.devices.flatten())
    return ()

  # Add empty tuple to handle leaves without sharding info.
  for dev_tuple in device_tuples | {()}:

    def _filter(x, dt=dev_tuple):
      device_tuple = _get_device_tuple(x)
      if device_tuple == dt:
        return x
      return None

    grouped = jax.tree.map(_filter, tree)
    if not all(x is None for x in jax.tree.leaves(grouped)):
      groups[dev_tuple] = grouped
  return groups


def _merge_results(results: list[PyTree]) -> PyTree:
  """Merges a list of PyTrees with identical structure, picking non-None leaves."""

  def _merge_leaf(*leaves):
    for leaf in leaves:
      if leaf is not None:
        return leaf
    return None

  return jax.tree.map(_merge_leaf, *results, is_leaf=lambda x: x is None)


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
  def name(self) -> str:
    """Returns the name of type of the dispatcher."""
    ...

  @abc.abstractmethod
  def device_to_host(self, arrays: PyTree) -> PyTree:
    """Performs device to host transfer on the given arrays."""
    ...

  @abc.abstractmethod
  def dispatch(
      self,
      func: Callable[..., PyTree | None],
      *,
      input_arrays: PyTree | None = None,
      result_specs: PyTree | None = None,
      func_args: tuple[Any, ...] = (),
      func_kwargs: dict[str, Any] | None = None,
  ) -> PyTree:
    """Dispatches func with given arrays.

    Args:
      func: Function to dispatch. Must be synchronous and return a PyTree of
        jax.Arrays matching result_specs.
      input_arrays: Input arrays to determine which devices to run on, if
        provided is also passed to func as the first argument.
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

  def name(self) -> str:
    return 'colocated_python'

  def _colocated_cpu_sharding(
      self,
      sharding: jax.sharding.Sharding,
  ) -> jax.sharding.Sharding:
    return colocated_transport.colocated_cpu_sharding(sharding)

  def to_colocated_python(self, input_tree: PyTree) -> PyTree:
    """Copies a PyTree of arrays to colocated CPU devices."""
    return colocated_transport.to_colocated_python(input_tree)

  def _convert_single_replica_restore_args(self, restore_args):
    return colocated_transport.convert_single_replica_restore_args(restore_args)

  def _convert_array_restore_args(self, restore_args):
    return colocated_transport.convert_array_restore_args(restore_args)

  def _transform_pytree_shardings(self, input_tree: PyTree) -> Any:
    """Converts Sharding or ShapeDtypeStruct args to use CPU devices."""
    return colocated_transport.transform_tree_shardings(input_tree)

  def _to_final_specs(
      self,
      input_tree: PyTree,
      tpu_or_cpu_specs: PyTree,
  ) -> jax.Array:
    return colocated_transport.to_final_specs(input_tree, tpu_or_cpu_specs)

  def device_to_host(self, arrays: PyTree) -> PyTree:
    """Performs device to host transfer on the given arrays."""
    return self.to_colocated_python(arrays)

  def dispatch(
      self,
      func: Callable[..., PyTree | None],
      *,
      input_arrays: PyTree | None = None,
      result_specs: PyTree | None = None,
      func_args: tuple[Any, ...] = (),
      func_kwargs: dict[str, Any] | None = None,
  ) -> PyTree:
    """Dispatches func with given arrays using colocated Python.

    Args:
      func: Function to dispatch. Must be synchronous and return a PyTree of
        jax.Arrays matching result_specs.
      input_arrays: Input arrays to determine which devices to run on, if
        provided is also passed to func as the first argument.
      result_specs: A PyTree of jax.ShapeDtypeStructs describing fn's output and
        the desired final shardings of the result from func on TPU. If None,
        func output is discarded and a dummy array is returned.
      func_args: Positional arguments for func.
      func_kwargs: Keyword arguments for func.

    Returns:
      A PyTree of jax.Arrays matching result_specs or a dummy array signaling
      completion of func if result_specs is None.
    """
    is_input_arrays_provided = input_arrays is not None
    is_func_output_discarded = result_specs is None
    if func_kwargs is None:
      func_kwargs = {}

    if input_arrays is None:
      if result_specs is None:
        input_arrays = get_dummy_input_array(jax.devices())
      else:
        # We will handle with grouping by mesh in the 'if
        # handle_multiple_meshes:' block below.
        pass

    cpu_args = self._transform_pytree_shardings(func_args)
    cpu_kwargs = self._transform_pytree_shardings(func_kwargs)

    @cp.colocated_python
    def _cp_wrapper(inp: PyTree) -> PyTree:
      _vlog_dispatch(func, 'ColocatedPythonDispatcher')
      args = (inp,) + cpu_args if is_input_arrays_provided else cpu_args
      if is_func_output_discarded:
        func(*args, **cpu_kwargs)
        return _make_dummy_result_array(inp)
      else:
        return func(*args, **cpu_kwargs)

    # Check for multiple meshes in result_specs
    if input_arrays is None and result_specs is not None:
      device_lists = _collect_device_lists_from_pytree(result_specs)
    else:
      device_lists = frozenset()

    if len(device_lists) > 1:
      device_list_groups = _group_by_device_list(result_specs, device_lists)

      results: list[PyTree] = []
      for grouped_specs in device_list_groups.values():
        placeholder_inp = _get_dummy_input_array_from_result_specs(
            grouped_specs
        )
        cpu_grouped_specs = self._transform_pytree_shardings(grouped_specs)
        specialized_wrapper = _cp_wrapper.specialize(
            out_specs_fn=lambda _, s=cpu_grouped_specs: s
        )
        group_result = specialized_wrapper(
            self.to_colocated_python(placeholder_inp)
        )

        # Filter result to only keep leaves belonging to this mesh group.
        def _filter_output(val, spec):
          return val if spec is not None else None

        filtered_result = jax.tree.map(
            _filter_output, group_result, cpu_grouped_specs
        )
        results.append(filtered_result)

      result = _merge_results(results)
    else:
      if input_arrays is None:
        input_arrays = _get_dummy_input_array_from_result_specs(result_specs)

      result_specs = result_specs or _make_dummy_result_array(
          input_arrays, abstract=True
      )
      cpu_result_specs = self._transform_pytree_shardings(result_specs)
      specialized_wrapper = _cp_wrapper.specialize(
          out_specs_fn=lambda _: cpu_result_specs
      )
      result = specialized_wrapper(self.to_colocated_python(input_arrays))

    return self._to_final_specs(result, result_specs)


