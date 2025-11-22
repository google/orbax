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

import collections
import dataclasses
import random
from typing import Any, NamedTuple, TypeVar

from etils import epath
import jax
import numpy as np
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.tree import structure_utils
from orbax.checkpoint._src.tree import utils as tree_utils
from orbax.checkpoint.experimental.v1._src.metadata import loading as metadata_loading

PyTree = Any
T = TypeVar('T')
PyTreeOf = PyTree | T
Keypath = tuple[Any, ...]
ArrayMetadata = value_metadata.ArrayMetadata


class SourceIndexedMetadata(NamedTuple):
  ckpt_idx: int
  metadata: ArrayMetadata


@dataclasses.dataclass(frozen=True)
class LeafInfo:
  """Information about a leaf to be used for batching."""

  keypath: Keypath
  ckpt_idx: int
  metadata: ArrayMetadata
  sharding: jax.sharding.Sharding
  size_bytes: int | None = dataclasses.field(default=None, init=False)

  def __post_init__(self):
    if self.metadata.dtype is None:
      raise ValueError(f'Metadata dtype is None for keypath {self.keypath}')

    # Calculate size lazily, keep logic out of the main loop
    object.__setattr__(
        self,
        'size_bytes',
        np.prod(self.sharding.shard_shape(self.metadata.shape))
        * self.metadata.dtype.itemsize,
    )


def merge_ckpt_metadata(
    ckpts_to_merge: list[epath.Path],
) -> PyTreeOf[SourceIndexedMetadata]:
  """Merges metadata from multiple checkpoints, labeling each leaf with its source index."""
  labeled_ckpt_metadata: list[PyTreeOf[SourceIndexedMetadata]] = [
      jax.tree.map(
          lambda metadata, index=i: SourceIndexedMetadata(index, metadata),
          metadata_loading.pytree_metadata(ckpt_path).metadata,
      )
      for i, ckpt_path in enumerate(ckpts_to_merge)
  ]
  return structure_utils.merge_trees(
      *labeled_ckpt_metadata,
      overwrite=True,
      is_leaf=lambda x: isinstance(x, SourceIndexedMetadata)
  )


def group_leaves_by_ckpt(
    merged_metadata: PyTreeOf[SourceIndexedMetadata],
) -> dict[int, dict[Keypath, ArrayMetadata]]:
  """Groups leaves by the checkpoint index they belong to."""
  leaves_by_ckpt = collections.defaultdict(dict)
  for keypath, (ckpt_idx, metadata) in sorted(
      tree_utils.to_flat_dict(
          merged_metadata,
          is_leaf=lambda x: isinstance(x, SourceIndexedMetadata),
      ).items(),
      key=lambda x: x[0],
  ):
    leaves_by_ckpt[ckpt_idx][keypath] = metadata
  return leaves_by_ckpt


def construct_leaf_infos(
    leaf_metadata_by_ckpt: dict[int, dict[Keypath, ArrayMetadata]],
    leaf_shardings_by_ckpt: dict[int, dict[Keypath, jax.sharding.Sharding]],
) -> list[LeafInfo]:
  """Groups the necessary per-leaf information to batch leaves."""
  leaf_infos: list[LeafInfo] = []

  # Sort the checkpoints by index to minimize disk reads. (vs worst case where
  # all checkpoints are read for each batch.)
  for ckpt_idx in sorted(leaf_metadata_by_ckpt.keys()):
    ckpt_leaf_infos: list[LeafInfo] = []
    shardings_map = leaf_shardings_by_ckpt[ckpt_idx]

    for keypath, metadata in leaf_metadata_by_ckpt[ckpt_idx].items():
      ckpt_leaf_infos.append(
          LeafInfo(
              keypath=keypath,
              ckpt_idx=ckpt_idx,
              metadata=metadata,
              sharding=shardings_map[keypath],
          )
      )

    # Randomize the order of leaves *within* this checkpoint. This helps mix
    # large and small arrays in batches to avoid wasting batch space.
    random.Random(len(ckpt_leaf_infos)).shuffle(ckpt_leaf_infos)
    leaf_infos.extend(ckpt_leaf_infos)

  return leaf_infos
