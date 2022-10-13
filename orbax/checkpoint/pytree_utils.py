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

"""Provides utils for PytreeCheckpointHandler."""

import abc
import dataclasses
from typing import Any, Dict, List, Optional

from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.future import Future

PyTreeDef = jax.tree_util.PyTreeDef


@dataclasses.dataclass
class ParamInfo:
  """Information describing a parameter in a PyTree.

  Note that ParamInfo is distinct from SaveArgs and RestoreArgs in that in
  represents information not provided by a user, and should be computed
  internally.

  name: name of the parameter.
  aggregate: whether the parameter should be / was aggregated.
  tspec: Tensorstore spec in JSON format.
  """
  name: Optional[str] = None
  aggregate: Optional[bool] = None
  tspec: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class SaveArgs:
  """Extra arguments that can be provided for saving.

  aggregate: if true, saves the given parameter in an aggregated tree format
    rather than individually. See AggregateHandler.
  dtype: if provided, casts the parameter to the given dtype before saving.
    Note that the parameter must be compatible with the given type (e.g.
    jnp.bfloat16 is not compatible with np.ndarray).
  """
  aggregate: bool = False
  dtype: Optional[jnp.dtype] = None


@dataclasses.dataclass
class RestoreArgs:
  """Extra arguments that can be provided for restoration.

  lazy: if True, restores using LazyArray. The actual read operation will not be
    performed until `get` is called for the restored LazyArray.
  restore_type: Specifies the type of the restored parameter. The type must have
    a corresponding TypeHandler for restoration. Ignored if the parameter is
    restored from an aggregated checkpoint file.
  dtype: if provided, casts the parameter to the given dtype after restoring.
    Note that the parameter must be compatible with the given type (e.g.
    jnp.bfloat16 is not compatible with np.ndarray).
  """
  lazy: bool = False
  # TODO(b/253238305) Deprecate this in favor of saving type information in
  # checkpoint.
  restore_type: Any = np.ndarray
  dtype: Optional[jnp.dtype] = None


class TypeHandler(abc.ABC):
  """Interface for reading and writing a PyTree leaf."""

  # TODO(b/253238305) Consider providing SaveArgs / RestoreArgs.
  @abc.abstractmethod
  def param_info(self, directory: epath.Path, name: str,
                 value: Any) -> ParamInfo:
    """Determines information necessary to save and restore the parameter.

    Note that the ParamInfo represents internal information not provided by a
    user.

    Args:
      directory: filepath where the parameter should be saved.
      name: name of the parameter.
      value: the parameter itself.

    Returns:
      ParamInfo
    """
    pass

  @abc.abstractmethod
  async def serialize(
      self,
      value: Any,
      info: ParamInfo,
      args: Optional[SaveArgs] = None) -> List[Future]:
    """Writes the parameter to a storage location.

    This method is responsible for copying the parameter from a remote device in
    a synchronous fashion (if applicable). It should then return a list of
    futures which can be later awaited to complete the final commit operation
    to a storage location.

    The function can be used in a multihost setting, but should not implement
    extra logic to ensure atomicity.

    Args:
      value: the parameter to save.
      info: contains relevant information for serialization.
      args: additional arguments for serialization, provided by the user.

    Returns:
      List of commit futures which can be awaited to complete the save
      operation.
    """
    pass

  @abc.abstractmethod
  async def deserialize(self,
                        info: ParamInfo,
                        args: Optional[RestoreArgs] = None) -> Any:
    """Reads the parameter from a storage location.

    Args:
      info: parameter information.
      args: user-provided restoration information.

    Returns:
      The deserialized parameter.
    """
    pass


_TYPE_HANDLER_REGISTRY = {}


def register_type_handler(ty: Any,
                          handler: TypeHandler,
                          override: bool = False):
  """Registers a type for serialization/deserialization with a given handler.

  Args:
    ty: an object type to register.
    handler: a TypeHandler capable of reading and writing parameters of type
      `ty`.
    override: if True, will override an existing mapping of type to handler.

  Raises:
    ValueError if a type is already registered and override is False.
  """
  if ty in _TYPE_HANDLER_REGISTRY and not override:
    raise ValueError(
        f'A TypeHandler for "{ty.__name__}" is already registered.')
  _TYPE_HANDLER_REGISTRY[ty] = handler


def get_type_handler(ty: Any) -> TypeHandler:
  """Returns the handler registered for a given type, if available.

  Args:
    ty: an object type.

  Returns:
    The TypeHandler that is registered for the given type.

  Raises:
    ValueError if the given type has no registered handler.
  """
  if ty not in _TYPE_HANDLER_REGISTRY:
    raise ValueError(
        f'Unkown type: "{ty.__name__}". Must register a TypeHandler.')
  return _TYPE_HANDLER_REGISTRY[ty]


class AggregateHandler(abc.ABC):
  """Interface for reading and writing a PyTree using a specific format."""

  @abc.abstractmethod
  async def serialize(self, directory: epath.Path, item: PyTreeDef):
    """Serializes and writes `item` to a given `directory`.

    The function is compatible with a multihost setting, but does not include
    extra logic to ensure atomicity.

    Args:
      directory: the folder to which the item should be written.
      item: a PyTree.
    """
    pass

  @abc.abstractmethod
  def deserialize(self, directory: epath.Path) -> PyTreeDef:
    """Reads and deserializes a PyTree from the given directory."""
    pass
