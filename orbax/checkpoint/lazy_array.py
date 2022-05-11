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
from typing import Callable, Optional, Sequence, Tuple, Union

from jax import numpy as jnp
from jax.experimental.global_device_array import GlobalDeviceArray
import numpy as np
import tensorstore as ts

Array = Union[np.ndarray, jnp.ndarray, GlobalDeviceArray]


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
    self._shape = tuple(shape)
    self._dtype = jnp.dtype(dtype)
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
  async def get_async(self) -> Array:
    raise NotImplementedError

  @abc.abstractmethod
  def get(self) -> Array:
    raise NotImplementedError

  def __repr__(self):
    return f'{type(self).__name__}(shape={self.shape}, dtype={self.dtype})'


class LazyAwaitableArray(LazyArray):
  """Lazily and asynchronously loads an array when the `get_fn` is async."""

  async def get_async(self) -> Array:
    arr = await self._get_fn()  # pytype: disable=bad-return-type
    assert arr.dtype == self.dtype
    return arr

  def get(self) -> Array:
    return asyncio.run(self.get_async())

  @classmethod
  def from_tensor_store_spec(
      cls,
      ts_spec: ts.Spec,
      get_fn: Callable[[], np.ndarray],
      dtype: Optional[jnp.dtype] = None) -> 'LazyAwaitableArray':
    """Create a LazyAwaitableArray based on a tensorstore.Spec."""
    ts_spec = ts_spec.to_json()
    shape = ts_spec['metadata']['shape']
    if dtype is None:
      dtype = jnp.dtype(ts_spec['dtype'])
    else:
      dtype = jnp.dtype(dtype)
    return cls(shape, dtype, get_fn)

  @classmethod
  def from_array(cls,
                 array: np.ndarray,
                 dtype: Optional[jnp.dtype] = None) -> 'LazyAwaitableArray':
    """Create a LazyAwaitableArray based on an array or python number."""
    if dtype is None:
      dtype = array.dtype
    else:
      dtype = jnp.dtype(dtype)

    async def get_fn():
      return array

    return cls(array.shape, dtype, get_fn)
