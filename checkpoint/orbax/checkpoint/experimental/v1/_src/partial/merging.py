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

from typing import Any, NamedTuple, TypeVar

from etils import epath
import jax
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.tree import structure_utils
from orbax.checkpoint.experimental.v1._src.metadata import loading as metadata_loading

PyTree = Any
T = TypeVar('T')
PyTreeOf = PyTree | T
ArrayMetadata = value_metadata.ArrayMetadata


class SourceIndexedMetadata(NamedTuple):
  ckpt_idx: int
  metadata: ArrayMetadata


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
