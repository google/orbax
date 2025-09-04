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

"""A handler for JAX arrays that uses colocated python."""

from __future__ import annotations

import asyncio
from typing import Sequence

from absl import logging
import jax
from jax.experimental import colocated_python
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.serialization.type_handlers import ParamInfo
from orbax.checkpoint._src.serialization.type_handlers import RestoreArgs
from orbax.checkpoint._src.serialization.type_handlers import SaveArgs


@colocated_python.colocated_python
async def _serialize(
    info: ParamInfo,
    value: jax.Array,
    args: SaveArgs | None,
) -> None:
  """Function to be run on a remote host to serialize a single array."""
  # TODO(b/283161063): remove this logging.
  logging.info('Beginning serialization for %s.', info.name)
  logging.info(
      '[Colocated] _serialize started for param: %s on device: %s',
      info.name,
      value.device(),
  )
  # Must create a new handler on the remote host.
  logging.info('[Colocated] Creating remote ArrayHandler for: %s.', info.name)
  handler = type_handlers.ArrayHandler()
  logging.info(
      '[Colocated] Calling handler.serialize for: %s with args: %s.',
      info.name,
      args,
  )
  commit_futures = await handler.serialize([value], infos=[info], args=[args])
  logging.info(
      '[Colocated] Received %d commit futures for: %s.',
      len(commit_futures),
      info.name,
  )
  # All futures should be awaited.
  for i, f in enumerate(commit_futures):
    logging.info('[Colocated] Awaiting commit future %d for: %s.', i, info.name)
    f.result()
    logging.info('[Colocated] Commit future %d for %s completed.', i, info.name)
  logging.info('[Colocated] _serialize finished for param: %s.', info.name)


@colocated_python.colocated_python
async def _deserialize(
    info: ParamInfo,
    args: RestoreArgs | None,
) -> jax.Array:
  """Function to be run on a remote host to deserialize a single array."""
  logging.info(
      '[Colocated] _deserialize started for param: %s with args: %s.',
      info.name,
      args,
  )
  # Must create a new handler on the remote host.
  logging.info('[Colocated] Creating remote ArrayHandler for: %s.', info.name)
  handler = type_handlers.ArrayHandler()
  logging.info('[Colocated] Calling handler.deserialize for: %s.', info.name)
  restored = await handler.deserialize([info], args=[args])
  logging.info(
      '[Colocated] Deserialization complete for: %s. Result count: %d.',
      info.name,
      len(restored),
  )
  if not restored:
    raise ValueError(f'Failed to deserialize {info.name}.')
  logging.info('[Colocated] _deserialize finished for param: %s.', info.name)
  return restored[0]


class ColocatedPythonArrayHandler(type_handlers.ArrayHandler):
  """An implementation of TypeHandler for jax.Array on Pathways."""

  async def serialize(
      self,
      values: Sequence[jax.Array],
      infos: Sequence[ParamInfo],
      args: Sequence[SaveArgs] | None = None,
  ) -> Sequence[future.Future]:
    """Serializes a jax.Array using colocated python."""
    logging.info(
        'ColocatedPythonArrayHandler.serialize called for %d values.',
        len(values),
    )
    args = args or ([SaveArgs()] * len(values))
    type_handlers.check_input_arguments(values, infos, args)
    logging.info('Input arguments checked successfully.')

    async def _serialize_all():
      logging.info(
          'Dispatching %d colocated serialization tasks.', len(values)
      )
      tasks = [
          _serialize(info, v, arg) for info, v, arg in zip(infos, values, args)
      ]
      await asyncio.gather(*tasks)
      logging.info(
          'All %d colocated serialization tasks completed.', len(values)
      )

    await _serialize_all()
    logging.info('ColocatedPythonArrayHandler.serialize finished.')
    # Returning a single future that is already complete.
    # The actual work is awaited above.
    return [future.Future()]

  async def deserialize(
      self,
      infos: Sequence[ParamInfo],
      args: Sequence[RestoreArgs] | None = None,
  ) -> Sequence[jax.Array]:
    """See superclass documentation."""
    logging.info(
        'ColocatedPythonArrayHandler.deserialize called for %d infos.',
        len(infos),
    )
    args = args or ([RestoreArgs()] * len(infos))
    logging.info(
        'Dispatching %d colocated deserialization tasks.', len(infos)
    )
    tasks = [_deserialize(info, arg) for info, arg in zip(infos, args)]
    results = await asyncio.gather(*tasks)
    logging.info(
        'All %d colocated deserialization tasks completed. Returning results.',
        len(results),
    )
    return results
