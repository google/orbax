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

"""Defines types for `CheckpointableHandler`."""

from typing import Any, Awaitable, Protocol, Type, TypeVar, runtime_checkable
from orbax.checkpoint.experimental.v1._src.path import types as path_types


T = TypeVar('T')
AbstractT = TypeVar('AbstractT')


@runtime_checkable
class StatefulCheckpointable(Protocol[T]):
  """An interface that defines save/load logic for a `checkpointable` object."""

  async def save(
      self, directory: path_types.PathAwaitingCreation
  ) -> Awaitable[None]:
    """Saves the given `checkpointable` to the given `directory`."""
    ...

  async def load(
      self,
      directory: path_types.Path,
      abstract_checkpointable: T | None = None,
  ) -> Awaitable[None]:
    """Loads the checkpointable from the given `directory`."""
    ...


class CheckpointableHandler(Protocol[T, AbstractT]):
  """An interface that defines save/load logic for a `checkpointable` object.

  NOTE: Prefer to use `StatefulCheckpointable` interface when possible.

  A "checkpointable" is a fundamental concept in Orbax. A “checkpointable”
  refers to a logical piece of the checkpoint that is distinct in some way from
  other pieces. Checkpointables are separable; they may or may not be loaded
  concurrently and some may be omitted from the checkpoint entirely.
  Checkpointables are often represented by different types, and have different
  representations on disk. The quintessential example is model params vs.
  dataset.

  A PyTree of arrays, representing model parameters, is the most basic
  "checkpointable". A singular array is also a checkpointable.

  In most contexts, when dealing with just a PyTree, the API of choice is::

    ocp.save_pytree(directory, pytree)

  The concept of "checkpointable" is not so obvious in this case. When dealing
  with multiple objects, we can use::

    ocp.save_checkpointables(
        directory,
        dict(
            pytree=model_params,
            dataset=dataset_iterator,
            # other checkpointables, e.g. extra metadata, etc.
        ),
    )

  Now, it is easy to simply skip loading the dataset, as is commonly desired
  when running evals or inference::

    ocp.load_checkpointables(
        directory,
        dict(
            pytree=abstract_model_params,
        ),
    )
    # Equivalently,
    ocp.load_pytree(directory, abstract_model_params)

  With the methods defined in this Protocol (`save`, `load`),
  logic within the method itself is executed in the main thread,
  in a blocking fashion. Additional logic can be executed in the background by
  returning an `Awaitable` function (which itself may return a result).

  Let's look at some suggestions on how to implement a `CheckpointableHandler`.
  TODO(b/398249409): Include more details on implementing this Protocol.

  First, take a look at
  orbax/checkpoint/experimental/v1/_src/testing/handler_utils.py
  for some toy implementations used for unit testing.

  Here are some details on how to implement `is_handleable` and
  `is_abstract_handleable`.

  For example, if a handler may be defined as follows::

    class FooHandler(CheckpointableHandler[Foo, AbstractFoo]):

      def is_handleable(self, checkpointable: Foo) -> bool:
        return isinstance(foo, Foo)

      def is_abstract_handleable(
          self, abstract_checkpointable: AbstractFoo) -> bool:
        return isinstance(abstract_foo, AbstractFoo)

  This is simple because the handler only works with `Foo` and `AbstractFoo`.
  But the handler may work on more generic types. In a toy
  example, let's say we've developed an improved way of storing very large
  arrays, which is still suboptimal for more normal-sized arrays. We can
  implement the handler as::

    class FooHandler(CheckpointableHandler[jax.Array, jax.ShapeDtypeStruct]):

      def is_handleable(self, checkpointable: jax.Array) -> bool:
        return (
            isinstance(checkpointable, jax.Array)
            and checkpointable.size > LARGE_ARRAY_THRESHOLD
        )

      def is_abstract_handleable(
          self, abstract_checkpointable: jax.ShapeDtypeStruct) -> bool:
        return (
            isinstance(abstract_checkpointable, jax.ShapeDtypeStruct)
            and abstract_checkpointable.size > LARGE_ARRAY_THRESHOLD
        )

  In many cases, no information is needed for loading. In this case,
  `AbstractT` may be defined as `None`. For example::

    class FooHandler(CheckpointableHandler[Foo, None]):

      def is_handleable(self, checkpointable: Foo) -> bool:
        return isinstance(checkpointable, Foo)

      def is_abstract_handleable(self, abstract_checkpointable: None) -> bool:
        return abstract_checkpointable is None
  """

  async def save(
      self, directory: path_types.PathAwaitingCreation, checkpointable: T
  ) -> Awaitable[None]:
    """Saves the given `checkpointable` to the given `directory`.

    Save should perform any operations that need to block the main thread, such
    as device-to-host copying of on-device arrays. It then creates a background
    operation to continue writing the object to the storage location.

    IMPORTANT: Do not assume that `directory` already exists at the start of
    this method. All directories are created by upper layers of the Orbax
    library, for performance reasons in a multihost setting and because upper
    layers also need to modify the directories. Before engaging in any
    filesystem operations, wait for the directory to exist. For example::

      async def _background_save(
          self,
          directory: path_types.PathAwaitingCreation,
          checkpointable: T,
      ) -> None:
        directory = await directory.await_creation()
        # Write to `directory` here.
        ...

      async def save(
          self,
          directory: path_types.PathAwaitingCreation,
          checkpointable: T,
      ) -> Awaitable[None]:
        # OK to access path properties, as long as we don't touch the actual
        # directory in the filesystem.
        logging.info(directory.name)
        return self._background_save(directory, checkpointable)

    Args:
      directory: The directory to save the checkpoint to. Note that the
        directory should not be expected to exist yet - it is in the process of
        being created. To wait for it to be created, use `await_creation`,
        preferably in a background awaitable to avoid blocking the main thread.
      checkpointable: The checkpointable object to save.

    Returns:
      An `Awaitable`. This object represents the result of the save
      operation running in the background.
    """
    ...

  async def load(
      self,
      directory: path_types.Path,
      abstract_checkpointable: AbstractT | None = None,
  ) -> Awaitable[T]:
    """Loads the checkpointable from the given `directory`.

    Args:
      directory: The directory to load the checkpoint from.
      abstract_checkpointable: An optional abstract representation of the
        checkpointable to load. If provided, this is used to provide properties
        to guide the restoration logic of the checkpoint. In the case of arrays,
        for example, this conveys properties like shape and dtype, for casting
        and reshaping. In some cases, no information is needed, and `AbstractT`
        may always be None. In other cases, the abstract representation may be a
        hard requirement for loading.

    Returns:
      An `Awaitable` that continues to load the checkpointable in the background
      and returns the loaded checkpointable when complete.
    """
    ...

  async def metadata(self, directory: path_types.Path) -> AbstractT:
    """Returns the metadata for the given `directory`.

    The logic in this method must be executed fully in the main thread; metadata
    access is expected to be cheap and fast.

    In many cases it is desirable to return additional metadata properties
    beyond the limited set in `AbstractT`. In this case, `AbstractT` should
    be subclasses, and this subclass can be returned from `metadata`.

    Args:
      directory: The directory where the checkpoint is located.

    Returns:
      AbstractT: The metadata is an `AbstractT`, which is the abstract
      representation of the checkpointable.
    """
    ...

  def is_handleable(self, checkpointable: Any) -> bool:
    """Returns whether the handler can handle the given checkpointable.

    The method should return `True` if it is possible to save such an object.

    See class docstring for more details.

    Args:
      checkpointable: Either a concrete checkpointable, for saving.

    Returns:
      True if the handler can handle the given checkpointable.
    """
    ...

  def is_abstract_handleable(self, abstract_checkpointable: Any) -> bool | None:
    """Returns whether the handler can handle the abstract checkpointable.

    The method should return `True` if it is possible to use the given
    `abstract_checkpointable` for loading a concrete `T`. Note that `None` is
    always considered handleable for loading, so this method does not need to
    check for it. If an implementation defines
    `AbstractT` as `None`, then this method should only return True for values
    of `None`.

    See class docstring for more details.

    Args:
      abstract_checkpointable: An abstract checkpointable, for loading.

    Returns:
      True if the handler can handle the given checkpointable.
      None if the handler cannot decide whether it can handle the abstract
      checkpointable and defers to the typestr.
    """
    ...


def typestr(handler_cls: Type[CheckpointableHandler]) -> str:
  """A name for the handler class that uniquely identifies it."""
  return f'{handler_cls.__module__}.{handler_cls.__qualname__}'
