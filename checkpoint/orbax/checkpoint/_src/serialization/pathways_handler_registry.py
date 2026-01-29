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

"""Registers the Pathways handlers with the given options."""

from __future__ import annotations

import types

from absl import logging
import jax
import numpy as np
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.serialization import jax_array_handlers
from orbax.checkpoint._src.serialization import pathways_types
from orbax.checkpoint._src.serialization import type_handler_registry
from orbax.checkpoint._src.serialization import type_handlers


CheckpointingImpl = pathways_types.CheckpointingImpl




def _get_array_hander_with_dispatcher(
    dispatcher: dispatchers.Dispatcher | None,
    use_single_replica_array_handler: bool,
    **kwargs,
) -> type_handlers.ArrayHandler:
  """Returns the Pathways ArrayHandler."""
  # Inject default array_metadata_store if not provided
  if 'array_metadata_store' not in kwargs:
    logging.warn('Array Metadata Store not specified, setting to default Store')
    kwargs['array_metadata_store'] = array_metadata_store_lib.Store()
  if use_single_replica_array_handler:
    logging.info('Using SingleReplicaArrayHandler')
    return jax_array_handlers.SingleReplicaArrayHandler(
        dispatcher=dispatcher, **kwargs
    )
  else:
    return jax_array_handlers.ArrayHandler(dispatcher=dispatcher, **kwargs)


def get_pathways_numpy_handler() -> type_handlers.NumpyHandler:
  """Returns the Pathways NumpyHandler."""
  return type_handlers.NumpyHandler(ocdbt_process_id='pwcontroller')


def get_pathways_scalar_handler() -> type_handlers.ScalarHandler:
  """Returns the Pathways ScalarHandler."""
  return type_handlers.ScalarHandler(ocdbt_process_id='pwcontroller')


def _register_numpy_and_scalar_handlers():
  """Registers the Numpy and Scalar handlers."""
  numpy_handler = get_pathways_numpy_handler()
  scalar_handler = get_pathways_scalar_handler()
  type_handler_registry.register_type_handler(
      int, scalar_handler, override=True
  )
  type_handler_registry.register_type_handler(
      float, scalar_handler, override=True
  )
  type_handler_registry.register_type_handler(
      bytes, scalar_handler, override=True
  )
  type_handler_registry.register_type_handler(
      np.number, scalar_handler, override=True
  )
  type_handler_registry.register_type_handler(
      np.ndarray, numpy_handler, override=True
  )


def get_pathways_array_handler(
    use_single_replica_array_handler: bool = False,
    checkpointing_impl: CheckpointingImpl | None = None,
    **kwargs,
) -> type_handlers.ArrayHandler:
  """Returns the Pathways ArrayHandler with the given options."""
  # If not set, use whichever dispatcher implementation is available.
  checkpointing_impl = checkpointing_impl or CheckpointingImpl.from_options(
      use_colocated_python=True,
  )
  match checkpointing_impl:
    case CheckpointingImpl.COLOCATED_PYTHON:
      logging.info('Using ColocatedPythonDispatcher')
      dispatcher = dispatchers.ColocatedPythonDispatcher()
    case CheckpointingImpl.NO_DISPATCHER:
      logging.info('Not using dispatcher')
      dispatcher = None
    case _:
      raise ValueError(f'Unsupported CheckpointingImpl: {checkpointing_impl}')

  return _get_array_hander_with_dispatcher(
      dispatcher,
      use_single_replica_array_handler,
      **kwargs,
  )


def register_pathways_handlers(
    use_single_replica_array_handler: bool = False,
    checkpointing_impl: CheckpointingImpl | None = None,
    **kwargs,
):
  """Registers the Pathways handlers with the given options.

  Args:
    use_single_replica_array_handler: Whether to use the
      SingleReplicaArrayHandler.
    checkpointing_impl: The implementation to use for Pathways checkpointing.
    **kwargs: Keyword arguments to pass to the ArrayHandler.
  """
  _register_numpy_and_scalar_handlers()

  type_handler_registry.register_type_handler(
      jax.Array,
      get_pathways_array_handler(
          use_single_replica_array_handler,
          checkpointing_impl=checkpointing_impl,
          **kwargs,
      ),
      override=True,
  )
