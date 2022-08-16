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

"""PyTreeCheckpointHandler class.

Implementation of CheckpointHandler interface.
"""

import asyncio
import dataclasses
import functools
import os
import re
from typing import Any, List, MutableMapping, Optional, Tuple, Union

from etils import epath
import flax
from flax import traverse_util
import jax
from jax.experimental import multihost_utils
from jax.experimental import pjit
from jax.experimental.gda_serialization import serialization

from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental.maps import Mesh
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import lazy_array
from orbax.checkpoint import transform_utils
from orbax.checkpoint import utils
from orbax.checkpoint.async_checkpoint_handler import AsyncCheckpointHandler
import tensorstore as ts

PyTree = type(jax.tree_util.tree_structure(None))
ArrayOrScalar = Union[int, float, np.number, np.ndarray, jnp.ndarray,
                      GlobalDeviceArray]
Array = Union[np.ndarray, jnp.ndarray, GlobalDeviceArray]
StructureLeaf = Union[str, dict, ts.Spec, ArrayOrScalar]

_FLAX_CHECKPOINT_FILE = 'checkpoint'

utils.register_ts_spec_for_serialization()


@dataclasses.dataclass
class ParamInfo:
  """Information describing a parameter in a PyTree.

  name: name of the parameter.
  tspec: Tensorstore spec in JSON format.
  """
  name: Optional[str]
  tspec: Optional[MutableMapping[str, Any]]


@dataclasses.dataclass
class SaveArgs:
  """Extra arguments that can be provided for saving.

  use_flax: if true, saves the given parameter using flax.serialization to a
    unified checkpoint file. Must be false if the given array value is a GDA.
  dtype: if provided, casts the parameter to the given dtype before saving.
    Note that the parameter must be compatible with the given type (e.g.
    jnp.bfloat16 is not compatible with np.ndarray).
  """
  use_flax: bool = False
  dtype: Optional[jnp.dtype] = None


# TODO(b/233807751): Raise an error if mesh/mesh_axes are None.
@dataclasses.dataclass
class RestoreArgs:
  """Extra arguments that can be provided for restoration.

  as_gda: if true, restores the given paramater as a GlobalDeviceArray
    regardless of how it was saved. If the array was not saved as a GDA, mesh
    and mesh_axes are required.
  mesh: the device mesh that the array should be restored as. If restoring as
    GDA, cannot be None.
  mesh_axes: the mesh_axes that the array should be restored as. If restoring as
    GDA, cannot be None.
  global_shapes: the global shape that the array should be restored into. If not
    provided, the shape will be restored as written.
  lazy: if True, restores using LazyArray. The actual read operation will not be
    performed until `get` is called for the restored LazyArray
  dtype: if provided, casts the parameter to the given dtype after restoring.
    Note that the parameter must be compatible with the given type (e.g.
    jnp.bfloat16 is not compatible with np.ndarray).
  """
  as_gda: bool = True
  mesh: Optional[Mesh] = None
  mesh_axes: Optional[pjit.PartitionSpec] = None
  global_shape: Optional[Tuple[int]] = None
  lazy: bool = False
  dtype: Optional[jnp.dtype] = None


def _validate_save_args(args: SaveArgs, enable_flax: bool):
  if args.use_flax and not enable_flax:
    raise ValueError(
        'Saving with flax disabled for this handler, but saving with flax was requested for at least one parameter.'
    )


def _validate_restore_args(args: RestoreArgs):
  if args.mesh is None or args.mesh_axes is None:
    raise ValueError('Restoring as GDA: `mesh` and `mesh_axes` cannot be None.')


async def _create_param_save_dir(param_info: ParamInfo):
  tspec = param_info.tspec
  if tspec is None:
    return
  path = tspec['kvstore']['path']
  if jax.process_index() == 0:
    await utils.async_makedirs(epath.Path(path))


async def _maybe_deserialize(args, value, info):
  """Deserialize from tensorstore or return value if already deserialized."""
  if value is None or (isinstance(value, dict) and not value):
    return {}
  if isinstance(value, (ts.Spec, utils.Leaf)):
    result = lazy_array.LazyAwaitableArray.from_tensor_store_spec(
        ts.Spec(info.tspec),
        get_fn=lambda: _deserialize_array(args, info),
        dtype=args.dtype)
  else:  # Already initialized as np.ndarray or GDA.
    if args.dtype is not None:
      if utils.is_scalar(value):
        value = np.asarray(value).astype(args.dtype).item()
      else:
        value = value.astype(args.dtype)
    if args.as_gda and not isinstance(value, GlobalDeviceArray):
      if utils.is_scalar(value):
        value = np.asarray(value)
      _validate_restore_args(args)
      shape = args.global_shape or value.shape
      result = GlobalDeviceArray.from_callback(
          shape, args.mesh, args.mesh_axes,
          lambda idx: value[idx].astype(value.dtype))
    else:
      result = value
    result = lazy_array.LazyAwaitableArray.from_array(result, dtype=args.dtype)
  if args.lazy:
    return result
  else:
    return await result.get_async()


def _get_array_tensorstore_spec(path: epath.Path, arr: ArrayOrScalar):
  """Gets ts.Spec in json format for the given path and array."""
  if arr is None:
    return None
  path = os.fspath(path)
  tspec = serialization.get_tensorstore_spec(path)
  if isinstance(arr, GlobalDeviceArray):
    tspec['metadata'] = serialization._get_metadata(arr)  # pylint: disable=protected-access
    del tspec['metadata']['dtype']  # pytype: disable=unsupported-operands
  # Note: should match ArrayOrScalar defined at the top of the file.
  elif isinstance(arr, (int, float, np.number, np.ndarray, jnp.ndarray)):
    if utils.is_scalar(arr):
      arr = np.asarray(arr)
    tspec['metadata'] = {
        'compressor': {
            'id': 'gzip'
        },
        'shape': arr.shape,  # pytype: disable=attribute-error
        'chunks': arr.shape,  # pytype: disable=attribute-error
    }
  else:
    raise ValueError(f'Unsupported array type: {type(arr)}')
  return tspec


def _get_param_infos_from_structure(directory: epath.Path,
                                    structure: PyTree) -> PyTree:
  """Construct ParamInfos based on structure()."""

  def _get_param_info(leaf):
    if leaf is None:
      tspec = None
    elif isinstance(leaf, dict):
      tspec = None
    elif isinstance(leaf, epath.Path):
      # Leaf is a param name.
      path = os.fspath(directory / leaf)
      tspec = serialization.get_tensorstore_spec(path)
    elif isinstance(leaf, ts.Spec):
      tspec = leaf.to_json()  # pytype: disable=attribute-error
      # Skip '.', since we need special regex handling for this char.
      pattern = r'\.' + utils.TMP_DIR_SUFFIX[1:] + r'\d+'
      tspec['kvstore']['path'] = re.sub(pattern, '', tspec['kvstore']['path'])
    elif isinstance(leaf, (int, float, np.number, np.ndarray, jnp.ndarray)):
      # Array already restored, do not need ts.Spec.
      tspec = None
    else:
      raise ValueError(f'Unsupported type: {type(leaf)}')
    return ParamInfo(name=None, tspec=tspec)

  return jax.tree_util.tree_map(_get_param_info, structure)


def _get_flax_tree_value(param_info, arg, arr):
  """Get leaf value for Flax checkpoint."""
  if param_info.tspec is None:
    return None
  if arg.use_flax:
    if arg.dtype is not None:
      if utils.is_scalar(arr):
        arr = np.asarray(arr).astype(arg.dtype).item()
      else:
        arr = arr.astype(arg.dtype)
    return arr
  else:
    return ts.Spec(param_info.tspec)


async def _serialize_array(
    param_info: ParamInfo, save_args: SaveArgs,
    arr: Union[ArrayOrScalar,
               GlobalDeviceArray]) -> Optional[List[asyncio.Future]]:
  """Writes an array (np.ndarray) or GlobalDeviceArray."""
  tspec = param_info.tspec
  if tspec is None:
    return
  if save_args.use_flax:
    if isinstance(arr, GlobalDeviceArray):
      raise ValueError('Cannot serialize GlobalDeviceArray with flax')
    return

  def set_tspec_dtype(arr: Array, spec):
    if save_args.dtype is None:
      spec['base']['dtype'] = jnp.dtype(arr.dtype).name
    else:
      spec['base']['dtype'] = jnp.dtype(save_args.dtype).name
    return spec

  tspec = {
      'base': tspec,
      'driver': 'cast',
  }

  if isinstance(arr, GlobalDeviceArray):
    # Origin dtype.
    tspec['dtype'] = jnp.dtype(arr.dtype).name
    # Destination dtype.
    tspec = set_tspec_dtype(arr, tspec)
    commit_futures = []
    await serialization.async_serialize(
        arr, tspec, commit_future=commit_futures)
    return commit_futures
  # Note: should match ArrayOrScalar defined at the top of the file.
  elif isinstance(arr, (int, float, np.number, np.ndarray, jnp.ndarray)):
    if isinstance(arr, (int, float, np.number)):
      arr = np.asarray(arr)
    # Origin dtype.
    tspec['dtype'] = jnp.dtype(arr.dtype).name
    # Destination dtype.
    tspec = set_tspec_dtype(arr, tspec)
    t = await ts.open(
        ts.Spec(tspec),
        create=True,
        open=True,
        context=ts.Context({'file_io_concurrency': {
            'limit': 128
        }}))
    write_future = t.write(arr)
    await write_future.copy
    return [write_future.commit]
  else:
    raise ValueError(f'Unsupported array type: {type(arr)}')


async def _deserialize_array(
    restore_args: RestoreArgs,
    param_info: ParamInfo) -> Union[ArrayOrScalar, GlobalDeviceArray]:
  """Writes an array (np.ndarray) or GlobalDeviceArray."""
  tspec = param_info.tspec

  if restore_args.dtype is not None:
    tspec = {
        'base': tspec,
        'driver': 'cast',
        'dtype': jnp.dtype(restore_args.dtype).name,
    }

  if restore_args.as_gda:
    _validate_restore_args(restore_args)
    return await serialization.async_deserialize(
        restore_args.mesh,
        restore_args.mesh_axes,
        tspec,
        global_shape=restore_args.global_shape)
  else:
    t = await ts.open(ts.Spec(tspec), open=True)
    result = await t.read()
    if result.ndim == 0:
      result = result.item()
    return result


class PyTreeCheckpointHandler(AsyncCheckpointHandler):
  """A CheckpointHandler implementation for any PyTree structure.

  The PyTree is assumed to be a nested dictionary with array values represented
  as `GlobalDeviceArray` objects or `np.ndarray` or `jnp.ndarray`. If not
  `GlobalDeviceArray`, arrays are expected to be non-partitioned.
  """

  def __init__(self, enable_flax=True):
    self._structure = None
    self._enable_flax = enable_flax

  def _get_param_names(self, item: PyTree) -> PyTree:
    """Gets parameter names for PyTree elements."""
    state_dict = utils.to_state_dict(item)
    names = traverse_util.unflatten_dict({
        k: '.'.join(k)
        for k in traverse_util.flatten_dict(state_dict, keep_empty_nodes=True)
    })
    r = flax.serialization.from_state_dict(item, names)
    return r

  def _get_param_infos(self, item: PyTree, directory: epath.Path) -> PyTree:
    """Returns parameter information for elements in `item`.

    At minimum, this method should extract the names of each parameter for
    saving/restoring.

    Args:
      item: a PyTree to extract information from.
      directory: a directory where checkpoint files are located.

    Returns:
      A PyTree matching `item` of ParamInfo.
    """
    if not item:
      raise ValueError('Found empty item')
    names = self._get_param_names(item)

    def _param_info(name, arr_or_spec):
      path = directory / name
      return ParamInfo(name, _get_array_tensorstore_spec(path, arr_or_spec))

    return jax.tree_util.tree_map(_param_info, names, item)

  async def async_save(
      self,
      directory: epath.Path,
      item: PyTree,
      save_args: Optional[PyTree] = None) -> Optional[List[asyncio.Future]]:
    """Saves a PyTree from a given training step.

    This operation is compatible with a multi-host, multi-device setting. Array
    values must be represented as `GlobalDeviceArray` or `np.ndarray`.

    After saving, all files will be located in directory/

    Saves an additional file to directory/checkpoint on host 0 which
    contains the serialized structure of `item`, along with any parameters that
    request Flax serialization.

    Args:
      directory: save location directory.
      item: a PyTree to be saved.
      save_args: a PyTree matching `item` which consists of SaveArgs objects as
        values.

    Returns:
      A Future that will commit the data to `directory` when awaited. Copying
      the data from its source will be awaited in this function.
    """
    item = await lazy_array.maybe_get_tree_async(item)

    if save_args is None:
      save_args = jax.tree_util.tree_map(lambda x: SaveArgs(), item)
    else:
      save_args = utils.to_state_dict(save_args)
    jax.tree_util.tree_map(
        functools.partial(_validate_save_args, enable_flax=self._enable_flax),
        save_args)

    param_infos = self._get_param_infos(item, directory)
    # Create directories in parallel.
    await asyncio.gather(*jax.tree_util.tree_flatten(
        jax.tree_util.tree_map(_create_param_save_dir, param_infos))[0])
    multihost_utils.sync_global_devices(
        'PyTreeCheckpointHandler:create_param_save_dirs')

    future_tree = jax.tree_util.tree_map(_serialize_array, param_infos,
                                         save_args, item)
    futures, _ = jax.tree_util.tree_flatten(future_tree)
    assert isinstance(futures, list)
    # Await copy futures.
    commit_futures = await asyncio.gather(*futures)
    commit_futures, _ = jax.tree_util.tree_flatten(commit_futures)

    if self._enable_flax and jax.process_index() == 0:
      flax_item = jax.tree_util.tree_map(_get_flax_tree_value, param_infos,
                                         save_args, item)
      msgpack = flax.serialization.to_bytes(flax_item)
      (directory / _FLAX_CHECKPOINT_FILE).write_bytes(msgpack)

    return commit_futures

  def save(self, directory: epath.Path, item: Any, *args, **kwargs):
    """Saves the provided item.

    See async_save.

    Args:
      directory: the directory to save to.
      item: the item to be saved.
      *args: additional arguments for save.
      **kwargs: additional arguments for save.
    """

    async def async_save(*args, **kwargs):
      futures = await self.async_save(*args, **kwargs)
      await asyncio.gather(*futures)

    asyncio.run(async_save(directory, item, *args, **kwargs))
    multihost_utils.sync_global_devices('PyTreeCheckpointHandler:save')

  def restore(self,
              directory: epath.Path,
              item: Optional[PyTree] = None,
              restore_args: Optional[PyTree] = None,
              transforms: Optional[PyTree] = None,
              transforms_default_to_original: bool = True) -> PyTree:
    """Restores a PyTree from the checkpoint directory at the given step.

    Optional arguments meshes and mesh_axes define how each array in the
    restored tree should be partitioned. For more information, see below and see
    pjit documentation.

    Args:
      directory: save location directory.
      item: provides the structure for the restored item. If not provided, will
        infer the structure from the saved checkpoint. Transformations will not
        be run.
      restore_args: optional object containing additional arguments for
        restoration. It should be a PyTree matching the structure of `item`, and
        should contain a RestoreArgs object for every value. If `item` is not
        provided, should match the structure of the checkpoint.
      transforms: a PyTree of transformations that should be applied to the
        saved item in order to obtain a final structure. See `transform_utils`
        for further information. Note that if `transforms` is provided, all
        other arguments are structured relative to the saved object, not to the
        restored object post-transformation.
      transforms_default_to_original: See transform_utils.apply_transformations.

    Returns:
      A PyTree matching the structure of `item` with restored array values as
      `GlobalDeviceArray` if `as_gda` or `np.ndarray` otherwise. If `lazy`
      restoration is enabled, `LazyArray` will be returned.

    Raises:
      FileNotFoundError: `directory` does not exist or is missing required files
      ValueError: `transforms` is provided without `item`.
      ValueError: `transforms` contains elements with `value_fn` or
        `multi_value_fn`.
    """
    if not directory.exists():
      raise FileNotFoundError(
          f'Requested directory for restore does not exist at {directory}')

    structure = self.structure(directory)
    param_infos = _get_param_infos_from_structure(directory, structure)
    if item is None:
      if transforms is not None:
        raise ValueError(
            'If providing `transforms`, must provide `item` matching structure of expected result.'
        )
      item = structure
    else:
      if transforms is None:
        item = flax.serialization.from_state_dict(item, structure)
        param_infos = flax.serialization.from_state_dict(item, param_infos)
      else:
        if transform_utils.has_value_functions(transforms):
          raise ValueError(
              'Found disallowed `value_fn` or `multi_value_fn` in `transforms`.'
          )
        item = transform_utils.apply_transformations(
            structure, transforms, item, transforms_default_to_original)
        # param_infos must be transformed because the ts.Spec of saved params
        # may correspond to the original structure rather than the new.
        param_infos = transform_utils.apply_transformations(
            param_infos, transforms, item, transforms_default_to_original)

    if restore_args is None:
      restore_args = jax.tree_util.tree_map(lambda x: RestoreArgs(as_gda=False),
                                            item)

    future_arrays = jax.tree_util.tree_map(
        _maybe_deserialize,
        restore_args,
        item,
        param_infos,
        is_leaf=lambda args: isinstance(args, RestoreArgs) or not args)

    future_arrays, item_def = jax.tree_util.tree_flatten(future_arrays)

    async def _async_restore(future_arrays):
      return await asyncio.gather(*future_arrays)

    result = asyncio.run(_async_restore(future_arrays))

    restored_item = jax.tree_util.tree_unflatten(item_def, result)

    multihost_utils.sync_global_devices('PyTreeCheckpointHandler:restore')
    return restored_item

  def structure(self, directory: epath.Path) -> PyTree:
    """Restores the PyTree structure from a Flax checkpoint.

    Args:
      directory: the directory to restore from.

    Returns:
      The structure of the checkpointed PyTree. Leaves may be of type
      StructureLeaf.

    Raises:
      FileNotFoundError: if the flax checkpoint is not found.
    """
    if self._structure is not None:
      return self._structure
    flax_file = directory / _FLAX_CHECKPOINT_FILE
    if flax_file.exists():
      msgpack = flax_file.read_bytes()
      self._structure = utils.msgpack_restore(msgpack)
    else:
      if self._enable_flax:
        raise FileNotFoundError(f'Checkpoint does not exist at {directory}.')
      else:
        self._structure = utils.pytree_structure(directory)
    return self._structure
