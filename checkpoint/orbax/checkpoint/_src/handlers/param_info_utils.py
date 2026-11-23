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

"""Utilities for constructing `ParamInfo`s for PyTree checkpointing."""

from __future__ import annotations

from typing import Any

from etils import epath
import jax
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.path import types as path_types
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.serialization import type_handler_registry as type_handler_registry_lib
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.serialization import types
from orbax.checkpoint._src.tree import utils as tree_utils

PyTree = Any


def _skip_deserialize(value: Any) -> bool:
  if isinstance(value, tree_metadata.ValueMetadataEntry):
    return value.skip_deserialize
  return isinstance(value, type(type_handlers.PLACEHOLDER))


def build_param_infos(
    item: PyTree,
    directory: epath.Path | path_types.PathAwaitingCreation,
    *,
    names: PyTree | None = None,
    type_handler_registry: types.TypeHandlerRegistry = (
        type_handler_registry_lib.GLOBAL_TYPE_HANDLER_REGISTRY
    ),
    pytree_metadata_options: tree_metadata.PyTreeMetadataOptions = (
        tree_metadata.PYTREE_METADATA_OPTIONS
    ),
    use_ocdbt: bool = True,
    **param_info_kwargs: Any,
) -> PyTree:
  """Returns a PyTree of `ParamInfo` matching `item`.

  Populates the fields of `ParamInfo` that are derived from the PyTree itself
  (`name`, `keypath`, `parent_dir`, `skip_deserialize`, `is_ocdbt_checkpoint`,
  `ts_context` and `value_typestr`). This is sufficient to build
  `tree_metadata.InternalTreeMetadata` for the `_METADATA` file. Any other
  `ParamInfo` fields (e.g. byte limiters, compression settings) can be provided
  via `param_info_kwargs`.

  Args:
    item: The PyTree to build `ParamInfo`s for. Empty nodes are treated as
      leaves.
    directory: The checkpoint directory, used as `ParamInfo.parent_dir`.
    names: Optional PyTree of parameter names matching `item`. Defaults to
      names derived from each leaf's keypath.
    type_handler_registry: Registry used to resolve each leaf's typestr.
    pytree_metadata_options: Options used to resolve each leaf's typestr.
    use_ocdbt: Whether the checkpoint uses OCDBT.
    **param_info_kwargs: Additional keyword arguments passed to every
      `ParamInfo`.

  Returns:
    A PyTree matching `item` of `ParamInfo`.
  """
  if names is None:
    names = tree_utils.get_param_names(item)
  ts_context = ts_utils.get_ts_context(use_ocdbt=use_ocdbt)

  def _param_info(keypath, name, value):
    return types.ParamInfo(
        name=name,
        keypath=keypath,
        parent_dir=directory,
        skip_deserialize=_skip_deserialize(value),
        is_ocdbt_checkpoint=use_ocdbt,
        ts_context=ts_context,
        value_typestr=type_handler_registry_lib.get_param_typestr(
            value, type_handler_registry, pytree_metadata_options
        ),
        **param_info_kwargs,
    )

  return jax.tree.map_with_path(
      _param_info, names, item, is_leaf=tree_utils.is_empty_or_leaf
  )
