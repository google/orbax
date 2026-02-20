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

"""Defines helpers for creating array `v0.TypeHandler`.

Functions should return a `TypeHandler` that can be wrapped into a
:py:class:`LeafHandler`. It should return the appropriate handler based on
global settings and runtime (Pathways vs. mcJAX).

This structure also helps prevent users from including Pathways dependencies in
their
binaries when they are not running on Pathways. The Pathways imports are
deferred until `is_pathways_backend()` returns True.

Pathways dependencies should not be added to this file.
"""

from orbax.checkpoint._src.serialization import jax_array_handlers
from orbax.checkpoint._src.serialization import pathways_handler_registry
from orbax.checkpoint._src.serialization import pathways_types
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.synchronization import multihost

_PATHWAYS_IMPORT_ERROR_MSG = """
Failed to import Pathways dependencies. Please ensure you have
linked the Pathways dependencies required by Orbax to your binary.
These are found at
//orbax/checkpoint/experimental/v1:pathways_support.
Please note that such dependencies are not linked automatically
because Pathways has a lot of dependencies, which non-Pathways
users wish to avoid linking.
"""


def resolve_pathways_checkpointing_impl(
    context: context_lib.Context,
) -> pathways_types.CheckpointingImpl:
  """Returns the Pathways checkpointing implementation."""
  try:
    # pylint: disable=g-import-not-at-top
    # pytype: disable=import-error
    from .learning.deepmind.jax.ocean.remote_python import rp
    # pytype: enable=import-error
    # pylint: enable=g-import-not-at-top
  except ImportError as e:
    raise ImportError(_PATHWAYS_IMPORT_ERROR_MSG) from e
  checkpointing_impl = context.pathways_options.checkpointing_impl
  return checkpointing_impl or pathways_types.CheckpointingImpl.from_options(
      use_colocated_python=False,  # Not enabled unless explicitly requested.
      use_remote_python=rp.available(),
      use_persistence_array_handler=True,  # Only used as a fallback.
  )


def get_array_handler(
    context: context_lib.Context,
) -> type_handlers.ArrayHandler:
  """Returns the TypeHandler for JAX arrays (pytree leaves)."""
  saving_options = context.array_options.saving
  loading_options = context.array_options.loading
  primary_host = context.multiprocessing_options.primary_host
  common_kwargs = dict(
      primary_host=primary_host,
      replica_id=None if primary_host is None else 0,
      use_replica_parallel=saving_options.use_replica_parallel,
      min_slice_bytes_for_replica_parallel=saving_options.min_slice_bytes_for_replica_parallel,
      max_replicas_for_replica_parallel=saving_options.max_replicas_for_replica_parallel,
      enable_replica_parallel_separate_folder=saving_options.enable_replica_parallel_separate_folder,
      enable_write_sharding_file=saving_options.enable_write_sharding_file,
      array_metadata_store=saving_options.array_metadata_store,
  )
  if loading_options.use_load_and_broadcast:
    load_and_broadcast_kwargs = dict(
        replica_axis_index=loading_options.load_and_broadcast_options.replica_axis_index,
        primary_replica_id=loading_options.load_and_broadcast_options.primary_replica_id,
        broadcast_memory_limit_bytes=loading_options.load_and_broadcast_options.broadcast_memory_limit_bytes,
        broadcast_memory_scaling_factor=loading_options.load_and_broadcast_options.broadcast_memory_scaling_factor,
    )
  else:
    load_and_broadcast_kwargs = dict()

  if multihost.is_pathways_backend():
    checkpointing_impl = resolve_pathways_checkpointing_impl(context)
    return pathways_handler_registry.get_pathways_array_handler(
        use_single_replica_array_handler=loading_options.use_load_and_broadcast,
        checkpointing_impl=checkpointing_impl,
        **common_kwargs,
        **load_and_broadcast_kwargs,
    )
  else:
    if loading_options.use_load_and_broadcast:
      return jax_array_handlers.SingleReplicaArrayHandler(
          dispatcher=None,
          **common_kwargs,
          **load_and_broadcast_kwargs,
      )
    return jax_array_handlers.ArrayHandler(dispatcher=None, **common_kwargs)


def get_numpy_handler() -> type_handlers.NumpyHandler:
  """Returns the TypeHandler for Numpy arrays."""
  if multihost.is_pathways_backend():
    return pathways_handler_registry.get_pathways_numpy_handler()
  return type_handlers.NumpyHandler()


def get_scalar_handler() -> type_handlers.ScalarHandler:
  """Returns the TypeHandler for scalars."""
  if multihost.is_pathways_backend():
    return pathways_handler_registry.get_pathways_scalar_handler()
  return type_handlers.ScalarHandler()
