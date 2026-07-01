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

"""CloudPathwaysArrayHandler for Pathways on Cloud with Persistence API."""

from __future__ import annotations

import collections
from collections.abc import Coroutine, Sequence
import concurrent.futures
import datetime
import functools
from typing import Any, cast

from absl import logging
import jax
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.metadata import array_metadata as array_metadata_lib
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.serialization import cloud_pathways_helper
from orbax.checkpoint._src.serialization import jax_array_handlers
from orbax.checkpoint._src.serialization import jax_array_restore_args
from orbax.checkpoint._src.serialization import types


ArrayMetadata = array_metadata_lib.ArrayMetadata


def extract_parent_dir_and_name(
    infos: Sequence[types.ParamInfo],
) -> tuple[Sequence[str], Sequence[str]]:
  """Extracts names and locations from ParamInfos."""
  parent_dirs = [str(info.parent_dir) for info in infos]
  names = [str(info.name) for info in infos]
  return parent_dirs, names


class CloudPathwaysArrayHandler(jax_array_handlers.ArrayHandler):
  """A TypeHandler for array types when using Pathways."""

  def __init__(
      self,
      timeout: datetime.timedelta | None = None,
      use_ocdbt: bool = False,
      array_metadata_store: array_metadata_store_lib.Store | None = None,
      **kwargs,
  ):
    """Orbax array handler for Pathways on Cloud with Persistence API.

    Args:
      timeout: Duration indicating the timeout for reading and writing arrays.
        Default is 1 hour.
      use_ocdbt: allows using Tensorstore OCDBT driver.
      array_metadata_store: An optional store for writing and reading array
        metadata. Only required for saving new-style jax random keys.
      **kwargs: Other keyword arguments.
    """
    del kwargs
    if timeout is None:
      timeout = datetime.timedelta(hours=1)
    self.timeout = timeout

    if use_ocdbt:
      raise ValueError("OCDBT not supported for Pathways.")
    super().__init__(array_metadata_store=array_metadata_store)

  async def _background_serialize(
      self,
      futures_results: Sequence[concurrent.futures.Future[None]],
      metadata_coroutine: Coroutine[Any, Any, None] | None = None,
  ) -> None:
    if metadata_coroutine:
      await metadata_coroutine

    for future_result in futures_results:
      future_result.result()

  def _wait_for_directory_creation_signals(self):
    async def _no_op():
      pass

    # Wait for directory creation signals to be set.
    future.CommitFutureAwaitingContractedSignals(_no_op()).result()

  async def serialize(
      self,
      values: Sequence[jax.Array],
      infos: Sequence[types.ParamInfo],
      args: Sequence[types.SaveArgs] | None = None,
  ) -> Sequence[future.Future]:
    """Uses Pathways Persistence API to serialize a jax array."""
    types.check_input_arguments(values, infos, args)

    if any([arg.dtype is not None for arg in args]):  # pyrefly: ignore[not-iterable]
      raise ValueError("Casting during save not supported for Pathways.")

    array_metadatas = []
    any_random_key = False
    arrays = []
    for v, info, arg in zip(values, infos, args):  # pyrefly: ignore[bad-argument-type]
      ext_metadata = None
      if jax.dtypes.issubdtype(v.dtype, jax.dtypes.prng_key):
        any_random_key = True
        impl = str(jax.random.key_impl(v))
        v = jax.random.key_data(v)
        ext_metadata = {array_metadata_lib.RANDOM_KEY_IMPL: impl}

      array_metadatas.append(
          ArrayMetadata(
              param_name=info.name,
              shape=v.shape,
              dtype=(arg.dtype if arg is not None else v.dtype),  # pyrefly: ignore[bad-argument-type]
              write_shape=getattr(v, "local_shape", v.shape),
              chunk_shape=getattr(v, "local_shape", v.shape),
              use_ocdbt=False,
              use_zarr3=False,
              ext_metadata=ext_metadata,
          )
      )
      arrays.append(v)

    if any_random_key and self._array_metadata_store is None:
      raise ValueError(
          "Array metadata store is not set with a checkpoint that requires"
          f" it. Array metadata: {array_metadatas}"
      )

    metadata_coroutine = None
    if self._array_metadata_store is not None:
      metadata_coroutine = self._array_metadata_store.write(
          checkpoint_dir=infos[0].parent_dir,
          array_metadatas=array_metadatas,
          process_index=0,
      )

    self._wait_for_directory_creation_signals()
    locations, names = extract_parent_dir_and_name(infos)
    f = functools.partial(
        cloud_pathways_helper.write_one_array, timeout=self.timeout
    )
    futures_results = list(map(f, locations, names, arrays))

    return [
        future.CommitFutureAwaitingContractedSignals(
            self._background_serialize(futures_results, metadata_coroutine),
            name="cloud_pathways_array_handler",
        )
    ]

  async def deserialize(
      self,
      infos: Sequence[types.ParamInfo],
      args: Sequence[types.RestoreArgs] | None = None,
  ) -> Sequence[jax.Array]:
    """Uses Pathways Persistence API to deserialize a jax array."""
    if args is None:
      raise ValueError("Must provide ArrayRestoreArgs to restore as jax.Array.")
    types.check_input_arguments(infos, args)

    global_meshes = []
    mesh_axes = []
    global_shapes = []
    dtypes = []
    shardings = []

    should_open_metadata = False
    for arg in args:
      if not isinstance(arg, jax_array_restore_args.ArrayRestoreArgs):
        raise ValueError(
            "To restore jax.Array, provide ArrayRestoreArgs; found"
            f" {type(arg).__name__}"
        )
      arg = cast(jax_array_restore_args.ArrayRestoreArgs, arg)
      if arg.sharding is None and (arg.mesh is None or arg.mesh_axes is None):
        raise ValueError(
            "Sharding of jax.Array cannot be None. Provide `mesh`"
            " and `mesh_axes` OR `sharding`."
        )
      if arg.sharding is None:
        global_meshes.append(arg.mesh)
        mesh_axes.append(arg.mesh_axes)
        shardings.append(
            jax.sharding.NamedSharding(mesh=arg.mesh, spec=arg.mesh_axes)  # pyrefly: ignore[bad-argument-type]
        )
      else:
        if not isinstance(arg.sharding, jax.sharding.NamedSharding):
          raise ValueError("Pathways only supports jax.sharding.NamedSharding.")
        sharding = cast(jax.sharding.NamedSharding, arg.sharding)
        global_meshes.append(sharding.mesh)
        mesh_axes.append(sharding.spec)
        shardings.append(sharding)
      if arg.global_shape is None or arg.dtype is None:
        logging.warning(
            "Shape or dtype not provided for restoration. Provide these"
            " properties for improved performance."
        )
        should_open_metadata = True
      global_shapes.append(arg.global_shape)
      dtypes.append(arg.dtype)

    if should_open_metadata:
      metadatas = await self.metadata(infos)
      global_shapes = [
          m.shape if s is None else s for m, s in zip(metadatas, global_shapes)
      ]
      dtypes = [m.dtype if d is None else d for m, d in zip(metadatas, dtypes)]

    array_metadatas_cache = {}
    if self._array_metadata_store is not None:
      if array_metadatas := await self._array_metadata_store.read(
          checkpoint_dir=infos[0].parent_dir,
          process_index=0,
      ):
        if not isinstance(array_metadatas, list):
          raise ValueError(
              "Array metadata store returned unexpected result:"
              f" {array_metadatas}"
          )

        array_metadatas_cache = {
            array_metadata.param_name: array_metadata
            for array_metadata in array_metadatas
        }

    # Group inputs by global_mesh so that we can perform batched Array
    # construction for each global_mesh.
    inputs_by_global_mesh = collections.defaultdict(list)
    for i, global_mesh in enumerate(global_meshes):
      inputs_by_global_mesh[global_mesh].append(i)

    results = cast(list[jax.Array], [None] * len(infos))

    for global_mesh, idxs in inputs_by_global_mesh.items():
      grouped_infos = [infos[idx] for idx in idxs]
      grouped_global_shapes = [global_shapes[idx] for idx in idxs]
      grouped_dtypes = [dtypes[idx] for idx in idxs]
      grouped_shardings = [shardings[idx] for idx in idxs]
      locations, names = extract_parent_dir_and_name(grouped_infos)
      grouped_arrays, read_future = cloud_pathways_helper.read_arrays(
          locations[0],
          names,
          grouped_dtypes,  # pyrefly: ignore[bad-argument-type]
          grouped_global_shapes,
          grouped_shardings,
          global_mesh.devices,
          timeout=self.timeout,
      )
      # each persistence call is awaited serially.
      read_future.result()
      for idx, info, arr in zip(idxs, grouped_infos, grouped_arrays):
        if meta := array_metadatas_cache.get(info.name):
          assert isinstance(
              meta, array_metadata_lib.SerializedArrayMetadata
          ), f"Expecting SerializedArrayMetadata but got {type(meta)}."
          if meta.ext_metadata:
            assert isinstance(meta.ext_metadata, dict), (
                "Expecting ext_metadata to be a dict but got"
                f" {type(meta.ext_metadata)}."
            )

            if impl := meta.ext_metadata.get(
                array_metadata_lib.RANDOM_KEY_IMPL
            ):
              arr = jax.random.wrap_key_data(arr, impl=impl)
        results[idx] = arr

    return results
