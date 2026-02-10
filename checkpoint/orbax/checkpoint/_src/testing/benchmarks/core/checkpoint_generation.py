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

import json
from typing import Any

from absl import logging
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import checkpoint_utils
from orbax.checkpoint._src.arrays import abstract_arrays
from orbax.checkpoint._src.arrays import sharding as sharding_utils
from orbax.checkpoint._src.checkpointers import checkpointer
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
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




def _get_abstract_state(
    config: configs.CheckpointConfig,
    *,
    use_ocdbt: bool,
    devices: list[jax.Device] | None = None,
) -> Any:
  """Loads sharding configuration from a JSON file."""
  path = epath.Path(config.path)
  devices = devices or jax.devices()
  with checkpointer.Checkpointer(
      pytree_checkpoint_handler.PyTreeCheckpointHandler(use_ocdbt=use_ocdbt)
  ) as ckptr:
    metadata = ckptr.metadata(path).item_metadata

  if config.sharding_config_path is None:
    abstract_state = jax.tree.map(
        abstract_arrays.to_shape_dtype_struct, metadata.tree
    )
    shardings = sharding_utils.construct_maximal_shardings(abstract_state)
    return jax.tree.map(
        lambda sds, sharding: jax.ShapeDtypeStruct(
            sds.shape, sds.dtype, sharding=sharding
        ),
        abstract_state,
        shardings,
    )

  path = epath.Path(config.sharding_config_path)
  parsed_config = json.loads(path.read_text())
  flat_abstract_state = {}
  for k, v in parsed_config.items():
    flat_abstract_state[k] = jax.ShapeDtypeStruct(
        shape=tuple(v['shape']),
        dtype=jnp.dtype(v['dtype']),
        sharding=jax.sharding.NamedSharding(
            mesh=jax.sharding.Mesh(
                np.array(devices).reshape(v['sharding']['mesh']['shape']),
                v['sharding']['mesh']['axes'],
            ),
            spec=jax.sharding.PartitionSpec(*v['sharding']['spec']),
        ),
    )
    return tree_utils.from_flat_dict(
        flat_abstract_state, metadata.tree, sep='.'
    )


def load_checkpoint(config: configs.CheckpointConfig) -> Any:
  """Loads a PyTree of test checkpoint from a provided path.

  Constructs a checkpoint from a reference checkpoint path specified in the
  config, which is expected to be provided. The checkpoint will be sharded
  according to an opaque strategy intended to minimize memory footprint, or will
  use the sharding config specified in `config` if provided.

  Args:
      config: A CheckpointConfig object allowing the checkpoint to be loaded
        from a reference or generated from a spec.

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
  abstract_state = _get_abstract_state(config, use_ocdbt=use_ocdbt)
  restore_args = checkpoint_utils.construct_restore_args(abstract_state)

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
