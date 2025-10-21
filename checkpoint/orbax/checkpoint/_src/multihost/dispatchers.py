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
from orbax.checkpoint._src.futures import future as future_lib
from orbax.checkpoint._src.multihost import multihost


PyTree = Any


class Dispatcher(abc.ABC):
  """Dispatches a function to run on workers and returns a future."""

  @abc.abstractmethod
  def dispatch_devices(
      self,
      fn: Callable[..., Any],
      devices: Sequence[jax.Device] | None = None,
      *args,
      **kwargs,
  ) -> future_lib.Future:
    """Dispatches fn to given devices.

    Args:
      fn: Function to dispatch.
      devices: Devices to dispatch to. If None, dispatches to all devices.
      *args: Additional positional arguments for fn.
      **kwargs: Additional keyword arguments for fn.

    Returns:
      A future that blocks until the computation is complete.
    """
    ...

  @abc.abstractmethod
  def dispatch_arrays(
      self,
      fn: Callable[..., Any],
      arrays: Sequence[jax.Array],
      *args,
      **kwargs,
  ) -> future_lib.Future:
    """Dispatches fn with given arrays.

    The arrays are first copied before being passed to fn.

    Args:
      fn: Function to dispatch. It will be called with `fn(arrays, *args,
        **kwargs)`.
      arrays: A sequence of jax.Arrays to be passed to fn.
      *args: Additional positional arguments for fn.
      **kwargs: Additional keyword arguments for fn.

    Returns:
      A future that blocks until the computation is complete.
    """
    ...


def _colocated_cpu_sharding(
    sharding: jax.sharding.Sharding,
) -> jax.sharding.Sharding:
  """Returns a sharding for the colocated CPU devices."""
  if isinstance(sharding, jax.sharding.NamedSharding):
    cpu_mesh = cp.colocated_cpu_devices(sharding.mesh)
    return jax.sharding.NamedSharding(cpu_mesh, sharding.spec)
  else:
    raise TypeError(
        f'Sharding type {type(sharding)} not supported in to_colocated_python.'
    )


def to_colocated_python(input_tree: PyTree) -> PyTree:
  """Copies a pytree of arrays to colocated CPU devices."""

  def _get_sharding(x):
    if isinstance(x, jax.Array):
      return _colocated_cpu_sharding(x.sharding)
    return None

  cpu_sharding_tree = jax.tree.map(_get_sharding, input_tree)
  return jax.device_put(input_tree, cpu_sharding_tree)


def get_abstract_dummy_result(
    arrays: Sequence[jax.Array],
) -> Sequence[jax.ShapeDtypeStruct]:
  """Returns an abstract dummy result for the given arrays."""

  def _get_dummy_array(x):
    sharding = jax.sharding.NamedSharding(
        x.sharding.mesh, jax.sharding.PartitionSpec()
    )
    return jax.ShapeDtypeStruct((), jnp.bool, sharding=sharding)

  return jax.tree.map(_get_dummy_array, arrays)


class ColocatedPythonDispatcher(Dispatcher):
  """Dispatches functions using colocated Python."""

  def _dispatch_devices_wrapper(
      self, fn, devices, sharding, *args, **kwargs
  ) -> Callable[..., jax.Array]:
    """Wraps the given function for dispatching to specific devices.

    Args:
      fn: The function to be wrapped.
      devices: The devices to dispatch to.
      sharding: The sharding to use for the dummy result.
      *args: Additional positional arguments for fn.
      **kwargs: Additional keyword arguments for fn.

    Returns:
      A `cp.colocated_python` wrapped function.
    """

    @cp.colocated_python
    def cp_wrapper():
      logging.vlog(
          1,
          'Executing function %r via ColocatedPythonDispatcher on'
          ' process=%s/%s',
          fn,
          multihost.process_index(),
          multihost.process_count(),
      )
      fn(*args, **kwargs)
      return jax.make_array_from_callback(
          (),
          sharding,
          lambda _: True,
          jnp.bool,
      )

    return cp_wrapper.specialize(devices=devices)

  def dispatch_devices(
      self,
      fn: Callable[..., Any],
      devices: Sequence[jax.Device] | None = None,
      *args,
      **kwargs,
  ) -> future_lib.JaxBlockUntilReadyFuture:
    """Dispatches fn to given devices using colocated Python.

    Args:
      fn: Function to dispatch.
      devices: Devices to dispatch to. If None, dispatches to all devices.
      *args: Positional arguments for fn.
      **kwargs: Keyword arguments for fn.

    Returns:
      A future that blocks until the computation is complete.
    """
    if devices is None:
      devices = jax.devices()

    cpu_devices = cp.colocated_cpu_devices(devices)
    cpu_replicated_sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(cpu_devices, ('device',)),
        jax.sharding.PartitionSpec(),
    )

    cp_wrapper = self._dispatch_devices_wrapper(
        fn, cpu_devices, cpu_replicated_sharding, *args, **kwargs
    )
    result = cp_wrapper()

    return future_lib.JaxBlockUntilReadyFuture(result)

  def _dispatch_arrays_wrapper(
      self, fn, *args, **kwargs
  ) -> Callable[..., Sequence[jax.Array]]:
    """Wraps the given function for dispatching with arrays.

    Args:
      fn: The function to be wrapped.
      *args: Additional positional arguments for fn.
      **kwargs: Additional keyword arguments for fn.

    Returns:
      A `cp.colocated_python` wrapped function.
    """

    @cp.colocated_python
    def cp_wrapper(input_arrays: Sequence[jax.Array]):
      logging.vlog(
          1,
          'Executing function %r via ColocatedPythonDispatcher on'
          ' process=%s/%s',
          fn,
          multihost.process_index(),
          multihost.process_count(),
      )
      fn(input_arrays, *args, **kwargs)
      return jax.tree.map(
          lambda x: jax.make_array_from_callback(
              x.shape, x.sharding, lambda _: True, x.dtype
          ),
          get_abstract_dummy_result(input_arrays),
      )

    return cp_wrapper.specialize(out_specs_fn=get_abstract_dummy_result)

  def dispatch_arrays(
      self,
      fn: Callable[..., Any],
      arrays: Sequence[jax.Array],
      *args,
      **kwargs,
  ) -> future_lib.JaxBlockUntilReadyFuture:
    """Dispatches fn with given arrays to colocated CPU devices.

    The arrays are first copied to colocated CPU devices before being passed to
    fn.

    Args:
      fn: Function to dispatch. It will be called with `fn(arrays, *args,
        **kwargs)`.
      arrays: A sequence of jax.Arrays to be passed to fn.
      *args: Additional positional arguments for fn.
      **kwargs: Additional keyword arguments for fn.

    Returns:
      A future that blocks until the computation is complete.
    """

    cp_wrapper = self._dispatch_arrays_wrapper(fn, *args, **kwargs)
    result = cp_wrapper(to_colocated_python(arrays))

    return future_lib.JaxBlockUntilReadyFuture(result)


class DirectDispatcher(Dispatcher):
  """Dispatches functions directly by calling them."""

  def dispatch_devices(
      self,
      fn: Callable[..., Any],
      devices: Sequence[jax.Device] | None = None,
      *args,
      **kwargs,
  ) -> future_lib.NoopFuture:
    """Dispatches fn on the current process.

    Device argument is ignored and the function is executed on the current
    process.

    Args:
      fn: Function to dispatch.
      devices: Not used currently for DirectDispatcher.
      *args: Positional arguments for fn.
      **kwargs: Keyword arguments for fn.

    Returns:
      A no-op future.
    """
    del devices
    logging.vlog(
        1,
        'Executing function %r via DirectDispatcher on process=%s/%s',
        fn,
        multihost.process_index(),
        multihost.process_count(),
    )
    fn(*args, **kwargs)
    return future_lib.NoopFuture()

  def dispatch_arrays(
      self,
      fn: Callable[..., Any],
      arrays: Sequence[jax.Array],
      *args,
      **kwargs,
  ) -> future_lib.NoopFuture:
    """Dispatches fn with given arrays directly on the current process.

    Args:
      fn: Function to dispatch. It will be called with `fn(arrays, *args,
        **kwargs)`.
      arrays: A sequence of jax.Arrays to be passed to fn.
      *args: Additional positional arguments for fn.
      **kwargs: Keyword arguments for fn.

    Returns:
      A no-op future.
    """
    logging.vlog(
        1,
        'Executing function %r via DirectDispatcher on process=%s/%s',
        fn,
        multihost.process_index(),
        multihost.process_count(),
    )
    fn(arrays, *args, **kwargs)
    return future_lib.NoopFuture()


