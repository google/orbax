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

"""PyTreeCheckpointer class. Implementation of Checkpointer interface."""

import asyncio
import dataclasses
from typing import Any, MutableMapping, Optional, Sequence, Union

import flax
from flax import traverse_util
import jax
from jax.experimental import pjit
from jax.experimental.gda_serialization import serialization

from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental.maps import Mesh
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import lazy_array
from orbax.checkpoint import transform_utils
from orbax.checkpoint import utils
from orbax.checkpoint.checkpointer import Checkpointer
import tensorflow as tf
import tensorstore as ts

PyTree = type(jax.tree_structure(None))
ArrayOrScalar = Union[int, float, np.number, np.ndarray, jnp.ndarray]
Array = Union[np.ndarray, jnp.ndarray]

_FLAX_CHECKPOINT_FILE = 'checkpoint'


# Register functions with flax.serialization to handle `ts.Spec`.
flax.serialization.register_serialization_state(
    ts.Spec,
    ty_to_state_dict=lambda t: t.to_json(),
    ty_from_state_dict=lambda t, s: ts.Spec(s))


@dataclasses.dataclass
class ParamInfo:
  """Information describing a parameter in a PyTree.

  name: name of the parameter.
  tspec: Tensorstore spec in JSON format.
  """
  name: str
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


@dataclasses.dataclass
class RestoreArgs:
  """Extra arguments that can be provided for restoration.

  as_gda: if true, restores the given paramater as a GlobalDeviceArray
    regardless of how it was saved. If the array was not saved as a GDA, mesh
    and mesh_axes are required.
  mesh: the device mesh that the array should be restored as. If None, uses a
    linear mesh of jax.devices.
  mesh_axes: the mesh_axes that the array should be restored as. If None, fully
    replicates the array to every device.
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
  global_shape: Optional[Sequence[int]] = None
  lazy: bool = False
  dtype: Optional[jnp.dtype] = None


async def _maybe_deserialize(args, value, info):
  """Deserialize from tensorstore or return value if already deserialized."""
  if value is None:
    return {}
  if isinstance(value, ts.Spec):
    result = lazy_array.LazyAwaitableArray.from_tensor_store_spec(
        ts.Spec(info.tspec),
        get_fn=lambda: _deserialize_array(args, info),
        dtype=args.dtype)
  else:  # already initialized as np.ndarray or GDA.
    if utils.is_scalar(value):
      value = np.asarray(value)
    if args.dtype is not None:
      value = value.astype(args.dtype)
    if args.as_gda:
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


def _msgpack_restore(msgpack):
  """Restores tree serialized using Flax. Converts ts_spec dict to ts.Spec."""
  state_dict = flax.serialization.msgpack_restore(msgpack)

  def is_leaf(x):
    if isinstance(x, dict):
      return set(x.keys()) >= {'driver', 'kvstore'}
    return False

  state_dict = jax.tree_map(
      lambda x: ts.Spec(x) if isinstance(x, dict) else x,
      state_dict,
      is_leaf=is_leaf)
  return state_dict


def _get_tensorstore_spec(path: str,
                          arr_or_spec: Union[ArrayOrScalar, GlobalDeviceArray],
                          cast_dtype: Optional[jnp.dtype] = None):
  """Gets ts.Spec in json format for the given path and array."""

  def set_tspec_dtype(arr: Array, spec):
    if cast_dtype is None:
      spec['dtype'] = jnp.dtype(arr.dtype).name
    else:
      spec['dtype'] = jnp.dtype(cast_dtype).name
    return spec

  if arr_or_spec is None:
    return None
  tspec = serialization.get_tensorstore_spec(path)
  if isinstance(arr_or_spec, GlobalDeviceArray):
    arr = arr_or_spec
    tspec = set_tspec_dtype(arr, tspec)
    tspec['metadata'] = serialization._get_metadata(arr)  # pylint: disable=protected-access
    del tspec['metadata']['dtype']  # pytype: disable=unsupported-operands
  # Note: should match ArrayOrScalar defined at the top of the file.
  elif isinstance(arr_or_spec,
                  (int, float, np.number, np.ndarray, jnp.ndarray)):
    arr = arr_or_spec
    if utils.is_scalar(arr):
      arr = np.asarray(arr)
    tspec = set_tspec_dtype(arr, tspec)
    tspec['metadata'] = {
        'compressor': {
            'id': 'gzip'
        },
        'shape': arr.shape,  # pytype: disable=attribute-error
        'chunks': arr.shape,  # pytype: disable=attribute-error
    }
  elif isinstance(arr_or_spec, ts.Spec):
    tspec = arr_or_spec.to_json()
    tspec['kvstore']['path'] = path
  elif isinstance(arr_or_spec, dict):
    assert not arr_or_spec  # expected empty
    tspec = None
  else:
    raise ValueError(f'Unsupported array type: {type(arr_or_spec)}')
  return tspec


async def _serialize_array(param_info: ParamInfo, save_args: SaveArgs,
                           arr: Union[ArrayOrScalar, GlobalDeviceArray]):
  """Writes an array (np.ndarray) or GlobalDeviceArray."""
  tspec = param_info.tspec
  if tspec is None:
    return
  if save_args.use_flax:
    if isinstance(arr, GlobalDeviceArray):
      raise ValueError('Cannot serialize GlobalDeviceArray with flax')
    return

  tspec = {
      'base': tspec,
      'driver': 'cast',
  }

  if isinstance(arr, GlobalDeviceArray):
    # origin dtype
    tspec['dtype'] = jnp.dtype(arr.dtype).name
    return await serialization.async_serialize(arr, tspec)
  # Note: should match ArrayOrScalar defined at the top of the file.
  elif isinstance(arr, (int, float, np.number, np.ndarray, jnp.ndarray)):
    if isinstance(arr, (int, float, np.number)):
      arr = np.asarray(arr)
    # origin dtype
    tspec['dtype'] = jnp.dtype(arr.dtype).name
    t = await ts.open(
        ts.Spec(tspec),
        create=True,
        open=True,
        context=ts.Context({'file_io_concurrency': {
            'limit': 128
        }}))
    return await t.write(arr)
  else:
    raise ValueError(f'Unsupported array type: {type(arr)}')


async def _deserialize_array(
    restore_args: RestoreArgs,
    param_info: ParamInfo) -> Union[ArrayOrScalar, GlobalDeviceArray]:
  """Writes an array (np.ndarray) or GlobalDeviceArray."""
  tspec = param_info.tspec
  if not tf.io.gfile.listdir(tspec['kvstore']['path']):
    # empty dictionaries are written as directories containing no files.
    return {}

  if restore_args.dtype is not None:
    tspec = {
        'base': tspec,
        'driver': 'cast',
        'dtype': jnp.dtype(restore_args.dtype).name,
    }

  if restore_args.as_gda:
    mesh = restore_args.mesh
    mesh_axes = restore_args.mesh_axes
    global_shape = restore_args.global_shape
    if mesh is None:
      mesh = Mesh(np.asarray(jax.devices()), ('devices',))
    if mesh_axes is None:
      mesh_axes = pjit.PartitionSpec(None,)
    return await serialization.async_deserialize(
        mesh, mesh_axes, tspec, global_shape=global_shape)
  else:
    t = await ts.open(ts.Spec(tspec), open=True)
    return await t.read()


class PyTreeCheckpointer(Checkpointer):
  """A Checkpointer implementation for any PyTree structure.

  The PyTree is assumed to be a nested dictionary with array values represented
  as `GlobalDeviceArray` objects or `np.ndarray` or `jnp.ndarray`. If not
  `GlobalDeviceArray`, arrays are expected to be non-partitioned.
  """

  def _get_param_infos(self, state_dict: PyTree, directory: str,
                       cast_dtypes: PyTree) -> PyTree:
    """Returns parameter information for elements in `state_dict`.

    At minimum, this method should extract the names of each parameter for
    saving/restoring.

    Args:
      state_dict: a nested dict to extract information from.
      directory: a directory where checkpoint files are located.
      cast_dtypes: a nested dict matching state_dict of dtypes to cast the
        parameters to. Leaves may be None.

    Returns:
      A PyTree matching `state_dict` of ParamInfo.
    """
    if not state_dict:
      raise ValueError('Found empty state_dict')
    names = traverse_util.unflatten_dict({
        k: '.'.join(k)
        for k in traverse_util.flatten_dict(state_dict, keep_empty_nodes=True)
    })

    def _param_info(name, arr_or_spec, cast_dtype):
      path = tf.io.gfile.join(directory, name)
      return ParamInfo(name, _get_tensorstore_spec(path, arr_or_spec,
                                                   cast_dtype))

    return jax.tree_map(_param_info, names, state_dict, cast_dtypes)

  async def async_save(self,
                       directory: str,
                       item: PyTree,
                       save_args: Optional[PyTree] = None):
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
    """
    # convert arbitrary pytree into dictionary
    item = flax.serialization.to_state_dict(item)

    if save_args is None:
      save_args = jax.tree_map(lambda x: SaveArgs(), item)
    else:
      save_args = flax.serialization.to_state_dict(save_args)
    cast_dtypes = jax.tree_map(lambda x: x.dtype, save_args)

    param_infos = self._get_param_infos(item, directory, cast_dtypes)

    future_tree = jax.tree_map(_serialize_array, param_infos, save_args, item)
    futures, _ = jax.tree_flatten(future_tree)
    await asyncio.gather(*futures)

    def flax_tree_value(param_info, arg, arr):
      if param_info.tspec is None:
        return None
      if arg.use_flax:
        if arg.dtype is not None:
          if utils.is_scalar(arr):
            arr = np.asarray(arr).astype(arg.dtype)
          else:
            arr = arr.astype(arg.dtype)
          assert arg.dtype == arr.dtype
        return arr
      else:
        return ts.Spec(param_info.tspec)

    if jax.process_index() == 0:
      flax_item = jax.tree_map(flax_tree_value, param_infos, save_args, item)
      msgpack = flax.serialization.to_bytes(flax_item)
      with tf.io.gfile.GFile(
          tf.io.gfile.join(directory, _FLAX_CHECKPOINT_FILE), mode='wb') as f:
        f.write(msgpack)

  async def async_restore(self,
                          directory: str,
                          item: Optional[PyTree] = None,
                          restore_args: Optional[PyTree] = None,
                          transforms: Optional[PyTree] = None) -> PyTree:
    """Restores a PyTree from the checkpoint directory at the given step.

    Optional arguments meshes and mesh_axes define how each array in the
    restored tree should be partitioned. For more information, see below and see
    pjit documentation.

    Args:
      directory: save location directory.
      item: provides the structure for the restored item. If not provided, will
        infer the structure from the saved checkpoint.
      restore_args: optional object containing additional arguments for
        restoration. It should be a PyTree matching the structure of the saved
        checkpoint, and should contain a RestoreArgs object for every value.
      transforms: a PyTree of transformations that should be applied to the
        saved item in order to obtain a final structure. See `transform_utils`
        for further information. Note that if `transforms` is provided, all
        other arguments are structured relative to the saved object, not to the
        restored object post-transformation.

    Returns:
      A PyTree matching the structure of `item` with restored array values as
      `GlobalDeviceArray` if `as_gda` or `np.ndarray` otherwise. If `lazy`
      restoration is enabled, `LazyArray` will be returned.

    Raises:
      FileNotFoundError: `directory` does not exist or is missing required files
    """
    if not tf.io.gfile.exists(directory):
      raise FileNotFoundError(
          f'Requested directory for restore does not exist at {directory}')
    flax_path = tf.io.gfile.join(directory, _FLAX_CHECKPOINT_FILE)
    if not tf.io.gfile.exists(flax_path):
      raise FileNotFoundError(f'Checkpoint does not exist at {flax_path}.')
    with tf.io.gfile.GFile(flax_path, mode='rb') as f:
      msgpack = f.read()

    # Note: because of None values in `state_dict`, do not jax.tree_map with
    # item in first position. Use param_infos.
    state_dict = _msgpack_restore(msgpack)

    if restore_args is None:
      restore_args = jax.tree_map(lambda x: RestoreArgs(), state_dict)
    else:
      restore_args = flax.serialization.to_state_dict(restore_args)
    cast_dtypes = jax.tree_map(lambda x: x.dtype, restore_args)

    param_infos = self._get_param_infos(state_dict, directory, cast_dtypes)

    future_arrays = jax.tree_map(
        _maybe_deserialize,
        restore_args,
        state_dict,
        param_infos,
        is_leaf=lambda args: isinstance(args, RestoreArgs) or not args)

    future_arrays, state_dict_def = jax.tree_flatten(future_arrays)
    result = await asyncio.gather(*future_arrays)

    restored_state_dict = jax.tree_unflatten(state_dict_def, result)
    # convert back into original object
    restored = flax.serialization.from_state_dict(item, restored_state_dict)
    if transforms is None:
      return restored
    return transform_utils.apply_transformations(restored, transforms)
