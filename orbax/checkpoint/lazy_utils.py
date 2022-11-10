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

import asyncio
from typing import Any, Callable

import jax


class LazyValue:
  """An interface allowing the real object to be fetched asynchronously."""

  def __init__(self, get_fn: Callable[[], Any]):
    self._get_fn = get_fn

  async def get_async(self) -> Any:
    return await self._get_fn()

  def get(self) -> Any:
    return asyncio.run(self.get_async())


def identity(value):
  """Constructs an async function that returns the given value."""

  async def get_fn():
    return value

  return get_fn


async def maybe_get_async(value):
  """Gets the value asynchronously if it is a LazyValue."""
  if isinstance(value, LazyValue):
    return await value.get_async()
  else:
    return await identity(value)()


def maybe_get(value):
  """Gets the value if it is a LazyValue."""
  return asyncio.run(maybe_get_async(value))


async def maybe_get_tree_async(pytree):
  """Gets tree values asynchronously if they are LazyValue."""
  flat, structure = jax.tree_util.tree_flatten(
      jax.tree_util.tree_map(maybe_get_async, pytree))
  flat = await asyncio.gather(*flat)
  return jax.tree_util.tree_unflatten(structure, flat)


def maybe_get_tree(pytree):
  """Gets tree values if they are LazyValue."""
  return asyncio.run(maybe_get_tree_async(pytree))
