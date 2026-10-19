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

"""Functions for checkpoint generation and loading in Orbax benchmark tests."""

from collections.abc import Sequence
import json
from typing import Any

from absl import logging
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import checkpoint_utils
from orbax.checkpoint import pathways
from orbax.checkpoint import v1 as ocp
from orbax.checkpoint._src.arrays import abstract_arrays
from orbax.checkpoint._src.arrays import sharding as sharding_utils
from orbax.checkpoint._src.checkpointers import checkpointer
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.testing.benchmarks.core import configs
from orbax.checkpoint._src.tree import utils as tree_utils



def _create_array(
    spec: dict[str, Any], mesh: jax.sharding.Mesh | None
) -> jax.typing.ArrayLike:
  """Creates a jax.Array based on the spec.

  Args:
    spec: A dictionary defining the array properties. Example: {'dtype':
      'float32', 'shape': [32], 'sharding': [None]}
    mesh: The mesh to use for sharding. Required if 'sharding' is in spec.

  Returns:
    A jax array.
  """
  dtype = getattr(jnp, spec['dtype'])
  shape = tuple(spec['shape'])
  sharding_spec = spec.get('sharding')

  if sharding_spec is not None and mesh is None:
    raise ValueError('Mesh is required when sharding spec is provided.')

  if mesh is None:
    logging.info(
        'No mesh and sharding spec provided, create an array with no sharding.'
    )
    return jnp.asarray(
        np.random.normal(size=shape, scale=np.prod(shape)), dtype=dtype
    )
  else:
    if sharding_spec is None:
      logging.info(
          'No sharding spec provided, creating a fully replicated array.'
      )
      pspec = jax.sharding.PartitionSpec()
    else:
      pspec = jax.sharding.PartitionSpec(*sharding_spec)
    sharding = jax.sharding.NamedSharding(mesh, pspec)
    logging.info(
        'Creating sharded array with shape=%s, dtype=%s, sharding=%s',
        shape,
        dtype,
        sharding,
    )
    np_array = np.random.normal(size=shape, scale=np.prod(shape)).astype(dtype)
    return jax.make_array_from_callback(
        shape, sharding, lambda index: np_array[index]
    )


def generate_checkpoint(
    config: configs.CheckpointConfig, mesh: jax.sharding.Mesh | None = None
) -> Any:
  """Generates a PyTree of test checkpoint data based on a provided specification.

  Args:
      config: A CheckpointConfig object containing the data specification.
      mesh: The mesh to use for sharding the generated data. If None, the data
        will not be sharded.

  Returns:
      A dictionary (PyTree) containing the generated data.

  Raises:
      ValueError: If the spec string is not supported.
  """
  if config.spec is None:
    raise ValueError(
        'CheckpointConfig must have a `spec` if `path` is not provided.'
    )
  pytree = {}
  if config.random_seed is not None:
    np.random.seed(config.random_seed)

  for name, spec in config.spec.items():
    if isinstance(spec, str):
      if spec == 'int':
        pytree[name] = 0
      elif spec == 'str':
        pytree[name] = 'default_string'
      else:
        raise ValueError(f'Unsupported spec string: {spec}')
    elif isinstance(spec, dict):
      pytree[name] = _create_array(spec, mesh)
    else:
      raise ValueError(f'Unsupported spec type: {type(spec)}')
  logging.info('Generated data with keys: %s', list(pytree.keys()))
  return pytree


def _partition_axis_name(offset: int) -> str:
  return str(chr(ord('a') + offset))




def get_abstract_state_with_generated_shardings(pytree_metadata: Any) -> Any:
  abstract_state = jax.tree.map(
      abstract_arrays.to_shape_dtype_struct, pytree_metadata
  )
  shardings = sharding_utils.construct_maximal_shardings(abstract_state)
  return jax.tree.map(
      lambda sds, sharding: jax.ShapeDtypeStruct(
          sds.shape, sds.dtype, sharding=sharding
      ),
      abstract_state,
      shardings,
  )


def partition_spec(
    shape: Sequence[int],
    axis_sizes: dict[str, int],
    replicated_axes: Sequence[str],
) -> jax.sharding.PartitionSpec:
  """Shards each mesh axis not in `replicated_axes` over the largest dim it divides.

  Args:
    shape: The array's shape.
    axis_sizes: Mesh axis name to size, in mesh order.
    replicated_axes: Mesh axes the array is replicated over.

  Returns:
    The spec; dims no axis divides are not sharded.
  """
  remaining = list(shape)
  axes_per_dim: list[list[str]] = [[] for _ in shape]
  for axis, size in axis_sizes.items():
    if axis in replicated_axes:
      continue
    for dim in sorted(range(len(shape)), key=lambda d: -remaining[d]):
      if remaining[dim] % size == 0:
        axes_per_dim[dim].append(axis)
        remaining[dim] //= size
        break
  return jax.sharding.PartitionSpec(*(
      tuple(axes) if len(axes) > 1 else (axes or [None])[0]
      for axes in axes_per_dim
  ))


def get_abstract_state_on_mesh(
    pytree_metadata: Any,
    mesh: jax.sharding.Mesh,
    replicated_axes: Sequence[str],
) -> Any:
  """Abstract state on `mesh`, laid out by `partition_spec`."""
  axis_sizes = dict(mesh.shape)

  def on_mesh(sds: Any) -> Any:
    if not isinstance(sds, jax.ShapeDtypeStruct):
      return sds
    spec = partition_spec(sds.shape, axis_sizes, replicated_axes)
    return jax.ShapeDtypeStruct(
        sds.shape, sds.dtype, sharding=jax.sharding.NamedSharding(mesh, spec)
    )

  return jax.tree.map(
      on_mesh,
      jax.tree.map(abstract_arrays.to_shape_dtype_struct, pytree_metadata),
  )


def get_abstract_state_from_sharding_config(
    sharding_config_path: epath.Path,
    metadata: Any,
    *,
    devices: list[jax.Device],
    mesh: jax.sharding.Mesh | None = None,
) -> Any:
  """Loads abstract state from a JSON file.

  Args:
    sharding_config_path: The sharding file.
    metadata: The checkpoint's metadata tree.
    devices: The devices a file's mesh is laid over, in order.
    mesh: The run's mesh. An array whose mesh in the file has the same axes and
      shape uses it, so it follows the run's (e.g. slice-aware) device order.

  Returns:
    The abstract state.
  """
  path = epath.Path(sharding_config_path)
  parsed_config = json.loads(path.read_text())
  flat_abstract_state = {}
  for k, v in parsed_config.items():
    file_mesh = v['sharding']['mesh']
    if (
        mesh is not None
        and tuple(file_mesh['axes']) == tuple(mesh.axis_names)
        and tuple(file_mesh['shape']) == mesh.devices.shape
    ):
      array_mesh = mesh
    else:
      array_mesh = jax.sharding.Mesh(
          np.array(devices).reshape(file_mesh['shape']), file_mesh['axes']
      )
    flat_abstract_state[k] = jax.ShapeDtypeStruct(
        shape=tuple(v['shape']),
        dtype=jnp.dtype(v['dtype']),
        sharding=jax.sharding.NamedSharding(
            mesh=array_mesh,
            spec=jax.sharding.PartitionSpec(*v['sharding']['spec']),
        ),
    )
  return tree_utils.from_flat_dict(flat_abstract_state, metadata, sep='.')


def _get_abstract_state(
    config: configs.CheckpointConfig,
    *,
    use_ocdbt: bool,
    devices: list[jax.Device] | None = None,
    mesh: jax.sharding.Mesh | None = None,
) -> Any:
  """Creates abstract state for a provided CheckpointConfig."""
  path = epath.Path(config.path)  # pyrefly: ignore[bad-argument-type]
  devices = devices or jax.devices()
  with checkpointer.Checkpointer(
      pytree_checkpoint_handler.PyTreeCheckpointHandler(use_ocdbt=use_ocdbt)
  ) as ckptr:
    metadata = ckptr.metadata(path).item_metadata

  if config.sharding_config_path is not None:
    return get_abstract_state_from_sharding_config(
        epath.Path(config.sharding_config_path),
        metadata,
        devices=devices,
        mesh=mesh,
    )
  if config.mesh_replicated_axes is not None and mesh is not None:
    return get_abstract_state_on_mesh(
        metadata.tree, mesh, config.mesh_replicated_axes
    )
  return get_abstract_state_with_generated_shardings(metadata.tree)


def generate_and_save_checkpoint(
    config: configs.CheckpointConfig,
    output_dir: epath.PathLike,
    mesh: jax.sharding.Mesh | None = None,
) -> None:
  """Generates a synthetic checkpoint from a spec and writes it with `ocp.save`.

  Materialises a synthetic Orbax checkpoint on disk so a load benchmark has
  something to read without depending on a real model download. The fixture is
  deterministic for a fixed `config.random_seed`, which the digest-mode
  correctness check relies on.

  Args:
    config: The checkpoint config; must carry a `spec`.
    output_dir: Directory to write the Orbax checkpoint to.
    mesh: Mesh used to shard the generated data. If None, the data is not
      sharded.
  """
  out = epath.Path(output_dir)
  if out.exists() and any(out.iterdir()):
    logging.warning('A fixture is already present at %s; overwriting it.', out)
  pytree = generate_checkpoint(config, mesh=mesh)
  ocp.save(out, pytree, overwrite=True)


def load_checkpoint(
    config: configs.CheckpointConfig, mesh: jax.sharding.Mesh | None = None
) -> Any:
  """Loads a PyTree of test checkpoint from a provided path.

  Constructs a checkpoint from a reference checkpoint path specified in the
  config, which is expected to be provided. The checkpoint will be sharded
  according to an opaque strategy intended to minimize memory footprint, or will
  use the sharding config specified in `config` if provided.

  Args:
      config: A CheckpointConfig object allowing the checkpoint to be loaded
        from a reference or generated from a spec.
      mesh: The run's mesh. Used by a sharding file whose mesh matches it, and
        by `config.mesh_replicated_axes`.

  Returns:
      A PyTree containing the loaded checkpoint.
  """
  if config.path is None:
    raise ValueError(
        'CheckpointConfig must have a `path` if `spec` is not provided.'
    )
  logging.info('Loading checkpoint from path: %s', config.path)
  path = epath.Path(config.path)


  use_ocdbt = type_handlers.is_ocdbt_checkpoint(path)
  abstract_state = _get_abstract_state(config, use_ocdbt=use_ocdbt, mesh=mesh)
  restore_args = checkpoint_utils.construct_restore_args(abstract_state)

  if multihost.is_pathways_backend():
    checkpointing_impl = pathways.CheckpointingImpl.from_options(
        use_colocated_python=config.load_with_colocated_python,
    )
    pathways.register_type_handlers(
        checkpointing_impl=checkpointing_impl,
        use_replica_parallel=False,
        enable_replica_parallel_separate_folder=False,
    )

  with checkpointer.Checkpointer(
      pytree_checkpoint_handler.PyTreeCheckpointHandler(use_ocdbt=use_ocdbt)
  ) as ckptr:
    pytree = ckptr.restore(
        path,
        args=pytree_checkpoint_handler.PyTreeRestoreArgs(
            restore_args=restore_args
        ),
    )
  return tree_utils.serialize_tree(pytree, keep_empty_nodes=True)
