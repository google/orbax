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

import types

from absl import logging
import jax
import numpy as np
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.serialization import jax_array_handlers
from orbax.checkpoint._src.serialization import type_handler_registry
from orbax.checkpoint._src.serialization import type_handlers




def _get_array_hander_with_dispatcher(
    dispatcher: dispatchers.Dispatcher | None,
    use_single_replica_array_handler: bool,
    **kwargs
):
  """Returns the Pathways ArrayHandler."""
  if use_single_replica_array_handler:
    logging.info('Using SingleReplicaArrayHandler')
    return jax_array_handlers.SingleReplicaArrayHandler(
        dispatcher=dispatcher, **kwargs
    )
  else:
    return jax_array_handlers.ArrayHandler(dispatcher=dispatcher, **kwargs)


def _register_numpy_and_scalar_handlers(**kwargs):
  """Registers the Numpy and Scalar handlers."""
  metadata_key = kwargs.get('metadata_key', None)
  numpy_handler = type_handlers.NumpyHandler(
      ocdbt_process_id='pwcontroller', metadata_key=metadata_key
  )
  scalar_handler = type_handlers.ScalarHandler(
      ocdbt_process_id='pwcontroller', metadata_key=metadata_key
  )
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


def register_pathways_handlers(
    use_single_replica_array_handler: bool = False,
    use_colocated_python: bool = True,
    **kwargs
):
  """Registers the Pathways handlers with the given options.

  Args:
    use_single_replica_array_handler: Whether to use the
      SingleReplicaArrayHandler.
    use_colocated_python: Use ColocatedPythonDispatcher with jax array handler.
    **kwargs: Keyword arguments to pass to the ArrayHandler.
  """
  _register_numpy_and_scalar_handlers(**kwargs)


  if use_colocated_python:
    logging.info('Using ColocatedPythonDispatcher')
    dispatcher = dispatchers.ColocatedPythonDispatcher()
  else:
    logging.info('Not using dispatcher')
    dispatcher = None

  array_handler = _get_array_hander_with_dispatcher(
      dispatcher,
      use_single_replica_array_handler,
      **kwargs,
  )
  type_handler_registry.register_type_handler(
      jax.Array, array_handler, override=True
  )
