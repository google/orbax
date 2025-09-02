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

"""Functions for checkpoint generation and loading in Orbax benchmark tests."""

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
from orbax.checkpoint._src.tree import utils


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
    return jnp.arange(np.prod(shape), dtype=dtype).reshape(shape)
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
    np_array = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
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
  pytree = {}
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


def load_checkpoint(path: str) -> Any:
  """Loads a PyTree of test checkpoint from a provided path."""
  logging.info('Loading checkpoint from path: %s', path)
  path = epath.Path(path)
  use_ocdbt = type_handlers.is_ocdbt_checkpoint(path)
  with checkpointer.Checkpointer(
      pytree_checkpoint_handler.PyTreeCheckpointHandler(use_ocdbt=use_ocdbt)
  ) as ckptr:
    metadata = ckptr.metadata(path).item_metadata
    abstract_state = jax.tree.map(
        abstract_arrays.to_shape_dtype_struct, metadata.tree
    )
    shardings = sharding_utils.construct_maximal_shardings(abstract_state)
    abstract_state = jax.tree.map(
        lambda sds, sharding: jax.ShapeDtypeStruct(
            sds.shape, sds.dtype, sharding=sharding
        ),
        abstract_state,
        shardings,
    )
    restore_args = checkpoint_utils.construct_restore_args(abstract_state)
    pytree = ckptr.restore(
        path,
        args=pytree_checkpoint_handler.PyTreeRestoreArgs(
            restore_args=restore_args
        ),
    )
  pytree = utils.serialize_tree(pytree, keep_empty_nodes=True)
  return pytree
