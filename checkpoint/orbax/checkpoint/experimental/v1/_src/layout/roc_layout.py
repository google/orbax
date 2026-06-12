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

"""Wrapper layout for loading Roc-format checkpoints."""

import asyncio
from typing import Any, Awaitable

import jax
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types

from .learning.deepmind.jax.roc import core as roc
from .learning.deepmind.jax.roc.file_systems import colossus as _colossus
from .learning.deepmind.jax.roc.file_systems import tfhub as _tfhub
from .learning.deepmind.jax.roc.formats.einshape_numpy_pickle import einshape_numpy_pickle as _einshape_numpy_pickle
from .learning.deepmind.jax.roc.formats.einshape_numpy_proto import einshape_numpy_proto as _einshape_numpy_proto
from .learning.deepmind.jax.roc.formats.einshape_numpy_proto import einshape_numpy_proto_local as _einshape_numpy_proto_local

_colossus.register()
_tfhub.register()
_einshape_numpy_pickle.register()
_einshape_numpy_proto.register()
_einshape_numpy_proto_local.register()


CheckpointLayout = checkpoint_layout.CheckpointLayout
InvalidLayoutError = checkpoint_layout.InvalidLayoutError
Path = types.Path
Checkpointable = checkpoint_layout.Checkpointable
AbstractCheckpointable = checkpoint_layout.AbstractCheckpointable


_STATE_CHECKPOINTABLE_NAME = 'state'


def _get_state_path(path: Path, checkpointable_name: str | None) -> Path:
  if checkpointable_name is None:
    state_path = path
  elif checkpointable_name == _STATE_CHECKPOINTABLE_NAME:
    state_path = path / checkpointable_name
  else:
    raise InvalidLayoutError(
        f'Only `None` and "{_STATE_CHECKPOINTABLE_NAME}" are supported values'
        ' for `checkpointable_name`.'
    )
  return state_path


class RocLayout(CheckpointLayout):
  """Handles Roc format checkpoints.

  Supports only flat (checkpointable_name=None) and nested
  (checkpointable_name='state') checkpoint structures. All other
  checkpointable names are unsupported.
  """

  async def validate_checkpointables(self, path: Path) -> None:
    try:
      await self.validate(path, None)
      return
    except InvalidLayoutError:
      pass

    try:
      await self.validate(path, _STATE_CHECKPOINTABLE_NAME)
      return
    except InvalidLayoutError:
      pass

    raise InvalidLayoutError(
        f'Failed to identify path {path} or nested'
        f' "{_STATE_CHECKPOINTABLE_NAME}" directory as a valid Roc'
        ' checkpoint.'
    )

  async def get_checkpointable_names(self, path: Path) -> list[str | None]:
    if await async_path.exists(path / _STATE_CHECKPOINTABLE_NAME):
      return [_STATE_CHECKPOINTABLE_NAME]
    return [None]

  async def validate(self, path: Path, checkpointable_name: str | None) -> None:
    state_path = _get_state_path(path, checkpointable_name)
    if not await async_path.exists(state_path):
      raise InvalidLayoutError(
          f'Failed to identify path {state_path} as a valid Roc checkpoint: '
          'directory does not exist.'
      )
    try:
      roc_path = roc.checkpoint.Path(state_path.as_posix())
      await asyncio.to_thread(
          roc.guess.guess_checkpoint_format_or_die, roc_path
      )
    except roc.guess.FormatNotFoundError as e:
      raise InvalidLayoutError(
          f'Failed to identify path {state_path} as a valid Roc checkpoint.'
      ) from e

  async def checkpointables_metadata(
      self, path: Path
  ) -> metadata_types.CheckpointMetadata[dict[str, tree_types.PyTreeOf[Any]]]:
    raise NotImplementedError()

  async def metadata(
      self, path: Path, checkpointable_name: str | None
  ) -> metadata_types.CheckpointMetadata[AbstractCheckpointable]:
    """Returns the metadata describing a single checkpointable in the Roc checkpoint."""
    state_path = _get_state_path(path, checkpointable_name)

    roc_path = roc.checkpoint.Path(state_path.as_posix())
    checkpoint_format = await asyncio.to_thread(
        roc.guess.guess_checkpoint_format_or_die, roc_path
    )
    checkpoint = roc.checkpoint.at(roc_path, checkpoint_format)
    index = await asyncio.to_thread(checkpoint.load_index)
    return metadata_types.CheckpointMetadata(path=path, metadata=index)

  async def load(
      self,
      path: Path,
      checkpointable_name: str | None = None,
      abstract_state: Any | None = None,
  ) -> Awaitable[tree_types.PyTreeOf[Any]]:
    state_path = _get_state_path(path, checkpointable_name)
    roc_path = roc.checkpoint.Path(state_path.as_posix())
    checkpoint_format = await asyncio.to_thread(
        roc.guess.guess_checkpoint_format_or_die, roc_path
    )
    checkpoint = roc.checkpoint.at(roc_path, checkpoint_format)

    if abstract_state is None:
      raise ValueError('`abstract_state` is required and cannot be None.')

    for x in jax.tree.leaves(abstract_state):
      if not isinstance(x, jax.ShapeDtypeStruct):
        raise ValueError(
            f'Unsupported type in abstract_state: {type(x)}. Only'
            ' ShapeDtypeStruct is supported.'
        )
      if x.sharding is None:
        raise ValueError(
            'All leaves in abstract_state must have a valid sharding.'
        )

    return asyncio.to_thread(roc.load_jax_arrays, checkpoint, abstract_state)

  async def load_checkpointables(
      self,
      path: Path,
      abstract_checkpointables: dict[str, AbstractCheckpointable] | None = None,
  ) -> Awaitable[dict[str, Checkpointable]]:
    raise NotImplementedError(
        '`load_checkpointables` not supported for Roc checkpoints. Use'
        ' `ocp.v1.load`.'
    )

  async def save_checkpointables(
      self,
      path: types.PathAwaitingCreation,
      *,
      checkpointables: dict[str, Any],
  ) -> Awaitable[None]:
    raise NotImplementedError('Saving Roc checkpoints is not supported.')
