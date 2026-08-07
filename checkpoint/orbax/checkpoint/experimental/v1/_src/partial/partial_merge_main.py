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

"""Partial merge binary."""

import collections
import dataclasses
import random
from typing import Any, Iterator, List

from absl import app
from absl import flags
from etils import epath
import jax
import numpy as np
from orbax.checkpoint._src.arrays import fragments as array_fragments
from orbax.checkpoint._src.arrays import sharding as array_sharding
from orbax.checkpoint._src.tree import parts_of
from orbax.checkpoint._src.tree import structure_utils
from orbax.checkpoint._src.tree import utils as tree_utils
from orbax.checkpoint.experimental.model_surgery import source_checkpoint
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.partial import saving as partial_saving

# Note: Ensure you have access to numpy_utils if copying _from_fragments,
# otherwise rely on public APIs if available.
from .learning.deepmind.jax.roc import numpy_utils
from .learning.deepmind.jax.roc.experimental import eval_fragments

FLAGS = flags.FLAGS

_IN_PATHS = flags.DEFINE_multi_string(
    'in_paths',
    None,
    'Paths of checkpoints to merge.',
    required=True,
)
_OUT_PATH = flags.DEFINE_string(
    'out_path',
    None,
    'Output checkpoint path.',
    required=True,
)
_PER_HOST_MEMORY_LIMIT_GB = flags.DEFINE_integer(
    'per_host_memory_limit_gb',
    16,
    'Memory limit in GB per CPU host for partial loading and saving.'
    ' Non-uniform memory limits are not supported.',
)

PyTree = Any
Keypath = tuple[Any, ...]
PartsOf = parts_of.PartsOf


def fragments_to_arrays(
    fragments_or_arrays: PyTree,
    target: PyTree,
) -> PyTree:
  """Creates jax.Array from a tree of Fragments."""

  def _to_jax_array(frags_or_arr, abstract_target):
    if not isinstance(frags_or_arr, eval_fragments.ConcreteFragments):
      return frags_or_arr

    def extract_shard(idx) -> jax.Array:
      idx = numpy_utils.resolve_slice(idx, abstract_target.shape)
      shard_data = eval_fragments._extract_fragment(  # pylint: disable=protected-access
          frags_or_arr.fragments,
          eval_fragments.AbstractFragment(index=idx),
      ).value
      assert shard_data is not None
      return jax.numpy.asarray(shard_data)

    sharding = abstract_target.sharding
    return jax.make_array_from_callback(
        abstract_target.shape, sharding, extract_shard
    )

  return jax.tree.map(_to_jax_array, fragments_or_arrays, target)


@dataclasses.dataclass(frozen=True)
class FragmentInfo:
  """Information about a fragment to be used for batching."""

  ckpt_idx: int
  keypath: Keypath
  fragment: array_fragments.AbstractFragment
  dtype: np.dtype

  @property
  def size_bytes(self) -> int:
    return self.fragment.nbytes_astype(self.dtype)


def merge_transform_fn(*args: PyTree) -> PyTree:
  """Merges trees, overwriting existing keys."""
  return structure_utils.merge_trees(*args, overwrite=True)


def batch_fragments(
    fragment_infos: list[FragmentInfo], memory_limit_gb: int
) -> Iterator[list[FragmentInfo]]:
  """Groups leaves into batches based on memory availability."""
  memory_limit_bytes = memory_limit_gb * 1024**3
  current_batch_leaves = []
  current_batch_size = 0

  for finfo in fragment_infos:
    if finfo.size_bytes > memory_limit_bytes:
      raise ValueError(
          f'Fragment size {finfo.size_bytes} is larger than memory limit.'
      )

    if current_batch_size + finfo.size_bytes > memory_limit_bytes:
      # Yield the current batch and start a new one.
      yield current_batch_leaves
      current_batch_leaves = [finfo]
      current_batch_size = finfo.size_bytes
    else:
      # Add the leaf to the current batch.
      current_batch_leaves.append(finfo)
      current_batch_size += finfo.size_bytes

  if current_batch_leaves:
    # Yield the final batch.
    yield current_batch_leaves


def resolve_pytree_path(path: epath.Path) -> epath.Path:
  """Resolves the path to the pytree checkpoint."""
  if not (path / checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY).exists():
    raise ValueError(f'Path {path} does not contain a pytree checkpoint.')

  return path / checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY


def resolve_target_structure(
    abstract_sources: list[PyTree], host_cpus: list[jax.Device]
) -> PyTree:
  """Resolves output structure and output sharding."""
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


def resolve_merge_topology(
    sharded_abstract_target: PyTree, abstract_sources: list[PyTree]
) -> tuple[PyTree, Any]:
  """Uses Model Surgery to resolve topology."""

  # Determine Fragments
  abstract_fragments_to_load = jax.tree.map(
      array_fragments.abstract_fragments, sharded_abstract_target
  )

  # The "Surgery": Map inputs to outputs
  return eval_fragments.eval_fragments(
      merge_transform_fn,
      abstract_sources,
      abstract_fragments_to_load,
  )


def create_fragment_infos(required_input_fragments: Any) -> list[FragmentInfo]:
  """Flattens fragments into FragmentInfos for batching."""
  fragment_infos = []
  for ckpt_idx, fragments_tree in enumerate(required_input_fragments):
    flat_fragments = tree_utils.to_flat_dict(fragments_tree)
    ckpt_fragment_infos = []

    for keypath, fragments in flat_fragments.items():
      for fragment in fragments.fragments:
        ckpt_fragment_infos.append(
            FragmentInfo(
                ckpt_idx=ckpt_idx,
                keypath=keypath,
                fragment=fragment,
                dtype=fragments.dtype,
            )
        )

    # Randomize the order of leaves *within* this checkpoint. This helps mix
    # large and small arrays in batches to avoid wasting batch space.
    random.shuffle(ckpt_fragment_infos)
    fragment_infos.extend(ckpt_fragment_infos)
  return fragment_infos


def load_batch_fragments(
    abstract_sources: list[PyTree],
    batch_fragments_map: dict[
        int, dict[tuple[Any, ...], list[array_fragments.AbstractFragment]]
    ],
    source_checkpoints: list[source_checkpoint.SourceCheckpoint],
    memory_limit_gb: int,
) -> list[PyTree]:
  """Loads fragments for a batch."""
  loaded_fragments = []
  # Reconstruct trees for loading
  for i, abstract_source in enumerate(abstract_sources):
    # We need to construct a request tree that matches the source structure
    # but only contains the fragments for this batch.

    def _get_fragments_for_leaf(
        path, meta, keypath_fragments=batch_fragments_map[i]
    ):
      # Convert JAX KeyPath to tuple for dict lookup
      path_tuple = tree_utils.tuple_path_from_keypath(path)

      frags = keypath_fragments.get(path_tuple)

      if frags:
        return array_fragments.AbstractFragments(
            shape=meta.shape,
            dtype=meta.dtype,  # Use source dtype
            fragments=frags,
        )
      return array_fragments.AbstractFragments(
          shape=meta.shape, dtype=meta.dtype, fragments=[]
      )

    source_request_tree = jax.tree_util.tree_map_with_path(
        _get_fragments_for_leaf, abstract_source
    )

    loaded_fragments.append(
        source_checkpoints[i].load_fragments(
            source_request_tree, concurrent_gb=memory_limit_gb
        )
    )
  return loaded_fragments


def main(argv: List[str] | None = None) -> None:
  if argv is not None and len(argv) > 1:
    raise app.UsageError(f'Too many command-line arguments: {argv[1:]}')

  all_cpus = jax.devices('cpu')
  host_cpus = all_cpus[: jax.process_count()]

  random.seed(0)

  ckpts_to_merge = [epath.Path(path) for path in _IN_PATHS.value]
  merged_ckpt_path = epath.Path(_OUT_PATH.value)

  # Load metadata for all input checkpoints to understand their structure and
  # contents.
  source_checkpoints = [
      source_checkpoint.checkpoint_at(resolve_pytree_path(path))
      for path in ckpts_to_merge
  ]
  abstract_sources = [sc.metadata for sc in source_checkpoints]

  # Determine the structure and sharding of the final merged checkpoint. This
  # acts as the blueprint for the output, derived by merging the metadata of the
  # input checkpoints.
  sharded_abstract_target = resolve_target_structure(
      abstract_sources, host_cpus
  )

  # Plan the merge operation by identifying exactly which data fragments need to
  # be read from the inputs to construct the output. This also prepares a
  # transformation function to assemble the loaded data.
  required_input_fragments, fragment_transform_fn = resolve_merge_topology(
      sharded_abstract_target, abstract_sources
  )

  # Prepare for execution by flattening the required data fragments into a list
  # of tasks. This allows us to process the merge in memory-constrained batches.
  fragment_infos = create_fragment_infos(required_input_fragments)

  for batch in batch_fragments(fragment_infos, _PER_HOST_MEMORY_LIMIT_GB.value):
    # Group the fragments in the current batch by their source checkpoint and
    # original keypath.
    batch_fragments_map = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for finfo in batch:
      batch_fragments_map[finfo.ckpt_idx][finfo.keypath].append(finfo.fragment)

    # Execute the load for the current batch: fetch the specific data fragments
    # from the source checkpoints into memory.
    loaded_fragments = load_batch_fragments(
        abstract_sources,
        batch_fragments_map,
        source_checkpoints,
        _PER_HOST_MEMORY_LIMIT_GB.value,
    )

    # Apply the transformation function to assemble the loaded fragments into
    # the desired target structure.
    target_fragments = fragment_transform_fn(loaded_fragments)

    # Convert the assembled fragments into concrete, sharded JAX arrays.
    target_tree = fragments_to_arrays(target_fragments, sharded_abstract_target)

    # Save the current batch of merged arrays to the output checkpoint
    # directory.
    partial_saving.save_pytree(merged_ckpt_path, target_tree)

  # Finalize the checkpoint, completing the merge process.
  partial_saving.finalize(merged_ckpt_path)


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)
