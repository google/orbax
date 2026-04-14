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

"""RandomKeyCheckpointHandlers for saving and restoring individual Jax and Numpy random keys."""

from __future__ import annotations

import dataclasses
from typing import Any, List, Optional, Union

from etils import epath
import jax
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.handlers import array_checkpoint_handler
from orbax.checkpoint._src.handlers import async_checkpoint_handler
from orbax.checkpoint._src.handlers import composite_checkpoint_handler
from orbax.checkpoint._src.handlers import json_checkpoint_handler
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.serialization import type_handlers


NumpyRandomKeyType = Union[tuple, dict]

ArrayRestoreArgs = array_checkpoint_handler.ArrayRestoreArgs
ArraySaveArgs = array_checkpoint_handler.ArraySaveArgs
CheckpointArgs = checkpoint_args.CheckpointArgs
CompositeArgs = composite_checkpoint_handler.CompositeArgs
CompositeCheckpointHandler = (
    composite_checkpoint_handler.CompositeCheckpointHandler
)
JsonRestoreArgs = json_checkpoint_handler.JsonRestoreArgs
JsonSaveArgs = json_checkpoint_handler.JsonSaveArgs
PyTreeRestoreArgs = pytree_checkpoint_handler.PyTreeRestoreArgs
PyTreeSaveArgs = pytree_checkpoint_handler.PyTreeSaveArgs
register_with_handler = checkpoint_args.register_with_handler


class BaseRandomKeyCheckpointHandler(
    async_checkpoint_handler.AsyncCheckpointHandler
):
  """Base handle saving and restoring individual Jax random key in both typed and untyped format."""

  def __init__(self, key_name: str):
    """Initializes the handler.

    Args:
      key_name: Provides a name for the directory under which Tensorstore files
        will be saved. Defaults to 'random_key'.

    Raises:
      DeprecationWarning: If the handler is used.
    """
    del key_name
    raise DeprecationWarning(
        'BaseRandomKeyCheckpointHandler is deprecated. Use '
        'PyTreeCheckpointHandler instead.'
    )

  async def async_save(
      self,
      directory: epath.Path,
      args: CheckpointArgs,
  ) -> Optional[List[future.Future]]:
    pass

  def save(self, directory: epath.Path, *args, **kwargs):
    pass

  def restore(
      self,
      directory: epath.Path,
      args: CheckpointArgs,
  ) -> Any:
    pass


class JaxRandomKeyCheckpointHandler(BaseRandomKeyCheckpointHandler):
  """Handles saving and restoring individual Jax random key in both typed and untyped format."""

  def __init__(self, key_name: Optional[str] = None):
    """Initializes the handler.

    Args:
      key_name: Provides a name for the directory under which Tensorstore files
        will be saved. Defaults to 'jax_random_key'.

    Raises:
      DeprecationWarning: If the handler is used.
    """
    raise DeprecationWarning(
        'JaxRandomKeyCheckpointHandler is deprecated. Use '
        'PyTreeCheckpointHandler instead.'
    )
    super().__init__(key_name or 'jax_random_key')  # pylint: disable=unreachable


@register_with_handler(JaxRandomKeyCheckpointHandler, for_save=True)
@dataclasses.dataclass
class JaxRandomKeySaveArgs(CheckpointArgs):
  """Parameters for saving a JAX random key.

  Attributes:
    item (required): a JAX random key.
    save_args: a `ocp.SaveArgs` object specifying save options.
  """

  item: jax.Array
  save_args: Optional[type_handlers.SaveArgs] = None

  def __post_init__(self):
    raise DeprecationWarning(
        'JaxRandomKeySaveArgs is deprecated. Use PyTreeCheckpointHandler'
        ' instead.'
    )


@register_with_handler(JaxRandomKeyCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class JaxRandomKeyRestoreArgs(CheckpointArgs):
  """Jax random key restore args.

  Attributes:
    restore_args: a `ocp.RestoreArgs` object specifying restore options for
      JaxArray.
  """

  restore_args: Optional[type_handlers.RestoreArgs] = None

  def __post_init__(self):
    raise DeprecationWarning(
        'JaxRandomKeyRestoreArgs is deprecated. Use PyTreeCheckpointHandler'
        ' instead.'
    )


class NumpyRandomKeyCheckpointHandler(BaseRandomKeyCheckpointHandler):
  """Saves Nnumpy random key in legacy or non-lagacy format."""

  def __init__(self, key_name: Optional[str] = None):
    """Initializes NumpyRandomKeyCheckpointHandler.

    Args:
      key_name: Provides a name for the directory under which Tensorstore files
        will be saved. Defaults to 'np_random_key'.

    Raises:
      DeprecationWarning: Raise deprecation error.
    """
    raise DeprecationWarning(
        'NumpyRandomKeyCheckpointHandler is deprecated. Use'
        ' PyTreeCheckpointHandler to save or restore numpy random keys.'
    )
    super().__init__(key_name or 'np_random_key')  # pylint: disable=unreachable


@dataclasses.dataclass
class NumpyRandomKeySaveArgs(CheckpointArgs):
  """Parameters for saving a Numpy random key.

  Attributes:
    item (required): a Numpy random key in legacy or nonlegacy format
  """

  item: NumpyRandomKeyType

  def __post_init__(self):
    raise DeprecationWarning(
        'NumpyRandomKeySaveArgs is deprecated. Use PyTreeCheckpointHandler'
        ' instead.'
    )


@register_with_handler(NumpyRandomKeyCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class NumpyRandomKeyRestoreArgs(CheckpointArgs):
  """Numpy random key restore args."""

  pass

  def __post_init__(self):
    raise DeprecationWarning(
        'NumpyRandomKeyRestoreArgs is deprecated. Use PyTreeCheckpointHandler'
        ' instead.'
    )
