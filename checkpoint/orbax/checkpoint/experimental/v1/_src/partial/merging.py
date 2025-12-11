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

"""Partial merging utils."""

from typing import Any, TypeVar

from etils import epath
import jax
from orbax.checkpoint._src.arrays import sharding as array_sharding
from orbax.checkpoint._src.tree import structure_utils
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout

PYTREE_CHECKPOINTABLE_KEY = checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY

PyTree = Any
T = TypeVar('T')
PyTreeOf = PyTree | T


# TODO(b/447415436): Use OrbaxLayout for this once the implementation is ready.
def resolve_pytree_path(path: epath.Path) -> epath.Path:
  """Resolves the path to the pytree checkpoint."""
  if not (path / PYTREE_CHECKPOINTABLE_KEY).exists():
    raise ValueError(f'Path {path} does not contain a pytree checkpoint.')

  return path / PYTREE_CHECKPOINTABLE_KEY


def merge_transform_fn(*args: PyTree) -> PyTree:
  """Merges trees, overwriting existing keys."""
  return structure_utils.merge_trees(*args, overwrite=True)


def resolve_target_structure(
    abstract_sources: list[PyTree], host_cpus: list[jax.Device]
) -> PyTree:
  """Resolves output structure and output sharding for merged sources."""
  abstract_target = jax.eval_shape(merge_transform_fn, *abstract_sources)

  shardings = array_sharding.construct_maximal_shardings(
      abstract_target, devices=host_cpus
  )
  sharded_abstract_target = jax.tree.map(
      lambda x, s: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s),
      abstract_target,
      shardings,
  )

  return sharded_abstract_target
