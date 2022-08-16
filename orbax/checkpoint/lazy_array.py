# Copyright 2022 The Orbax Authors.
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

"""Orbax LazyArray."""

import abc
import asyncio
import logging
from typing import Callable, Optional, Sequence, Tuple, Union

import jax
from jax import numpy as jnp
from jax.experimental.global_device_array import GlobalDeviceArray
import numpy as np
import tensorstore as ts

Array = Union[np.ndarray, jnp.ndarray, GlobalDeviceArray]
ScalarOrArray = Union[int, float, Array]


class LazyArray(abc.ABC):
  """Lazily and asynchronously loads an array.

  LazyArray behaves in the same way as a `numpy` or `jax.numpy` array
  while instantiating lazily. All properties, including shape, dtype, and nbytes
  are created when the LazyArray is created, but no data is materialized until
  `get` or `get_async` are called. Data is materialized using a specified
  `get_fn`.

  This class can be used to implement lazy restoration in checkpointing APIs,
  where the data is only read from disk when explicitly needed by the user.
  """

  def __init__(self, shape: Sequence[int], dtype: jnp.dtype,
               get_fn: Callable[[], np.ndarray]):
    self._shape = tuple(shape) if shape is not None else shape
    self._dtype = jnp.dtype(dtype) if dtype is not None else dtype
    self._get_fn = get_fn

  @property
  def shape(self) -> Tuple[int, ...]:
    return self._shape

  @property
  def dtype(self) -> jnp.dtype:
    return self._dtype

  @property
  def nbytes(self) -> int:
    return np.prod(self._shape) * self._dtype.itemsize

  def astype(self, dtype: np.dtype) -> 'LazyArray':
    return type(self)(self._shape, dtype, self._get_fn)  # pytype: disable=not-instantiable

  @abc.abstractmethod
  async def get_async(self) -> ScalarOrArray:
    raise NotImplementedError

  @abc.abstractmethod
  def get(self) -> ScalarOrArray:
    raise NotImplementedError

  def __repr__(self):
    return f'{type(self).__name__}(shape={self.shape}, dtype={self.dtype})'


class LazyAwaitableArray(LazyArray):
  """Lazily and asynchronously loads an array when the `get_fn` is async."""

  async def get_async(self) -> ScalarOrArray:
    return await self._get_fn()  # pytype: disable=bad-return-type

  def get(self) -> ScalarOrArray:
    return asyncio.run(self.get_async())

  def astype(self, dtype: np.dtype) -> 'LazyArray':
    logging.warning('Orbax LazyAwaitableArray cannot be cast.')
    return self

  @classmethod
  def from_tensor_store_spec(
      cls,
      ts_spec: ts.Spec,
      get_fn: Callable[[], np.ndarray],
      dtype: Optional[jnp.dtype] = None) -> 'LazyAwaitableArray':
    """Create a LazyAwaitableArray based on a tensorstore.Spec."""
    ts_spec = ts_spec.to_json()
    shape = None
    if 'metadata' in ts_spec:
      if 'shape' in ts_spec['metadata']:
        shape = ts_spec['metadata']['shape']
    dtype = None
    if dtype is None:
      if 'dtype' in ts_spec:
        dtype = jnp.dtype(ts_spec['dtype'])
    else:
      dtype = jnp.dtype(dtype)
    return cls(shape, dtype, get_fn)

  @classmethod
  def from_array(cls,
                 array: ScalarOrArray,
                 dtype: Optional[jnp.dtype] = None) -> 'LazyAwaitableArray':
    """Create a LazyAwaitableArray based on an array or python number."""
    if isinstance(array, (np.ndarray, jnp.ndarray, GlobalDeviceArray)):
      shape = array.shape  # pytype: disable=attribute-error
      if dtype is None:
        dtype = array.dtype  # pytype: disable=attribute-error
    else:
      shape = ()
      dtype = jnp.dtype(np.asarray(array).dtype)
    if dtype is not None:
      dtype = jnp.dtype(dtype)

    async def get_fn():
      return array

    return cls(shape, dtype, get_fn)


async def maybe_get_async(arr):

  async def identity(x):
    return x

  if isinstance(arr, LazyArray):
    return await arr.get_async()
  else:
    return await identity(arr)


def maybe_get(arr):
  return asyncio.run(maybe_get_async(arr))


async def maybe_get_tree_async(pytree):
  flat, structure = jax.tree_util.tree_flatten(
      jax.tree_util.tree_map(maybe_get_async, pytree))
  flat = await asyncio.gather(*flat)
  return jax.tree_util.tree_unflatten(structure, flat)


def maybe_get_tree(pytree):
  return asyncio.run(maybe_get_tree_async(pytree))
