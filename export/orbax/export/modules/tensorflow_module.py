# Copyright 2024 The Orbax Authors.
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

"""Wraps JAX functions and parameters into a tf.Module."""

from collections.abc import Callable, Mapping
import dataclasses
from typing import Any, Optional, Union

from orbax.export import typing as orbax_export_typing
from orbax.export.modules import orbax_module_base
import tensorflow as tf

PyTree = orbax_export_typing.PyTree
ApplyFn = orbax_export_typing.ApplyFn


@dataclasses.dataclass(frozen=True)
class _NonTrackableMetadata:
  """A container that holds the metadata required for variable update.

  Most fields of this dataclass are python containers (dict, list, tuple). If
  they are attached a tf.Module directly, TF will turn them into TF trackable
  wrappers (DictWrapper, ListWrapper, etc.), thus mutate their orginal PyTree
  def. Therefore, we create this dataclass to hold the metadata to avoid such
  implicit conversion. See also
  https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#errors-due-to-tfmodule-magic-conversion-during-attribute-assignment
  """

  apply_fn_map: Mapping[str, ApplyFn]
  tf_var_treedef: Any
  var_trainable: Mapping[str, bool]
  var_pspecs: Optional[Mapping[str, PyTree]]
  model_params: PyTree
  jax2tf_kwargs_map: Mapping[str, Any]
  input_polymorphic_shape_map: Mapping[str, Any]
  allow_multi_axis_sharding_consolidation: Optional[bool]


class TensorFlowModule(orbax_module_base.OrbaxModuleBase, tf.Module):
  """An exportable module for JAX functions and parameters.

  Holds tf.Variables converted from JAX parameters, as well as TF functions
  converted from JAX functions and bound with the tf.Variables.
  """

  def __init__(
      self,
      params: PyTree,
      apply_fn_map: Union[Mapping[str, ApplyFn], dict[str, ApplyFn]],
      trainable: Optional[Union[bool, PyTree]] = None,
      input_polymorphic_shape: Union[PyTree, Mapping[str, PyTree], None] = None,
      jit_compile: Union[bool, Mapping[str, bool]] = True,
      pspecs: Optional[PyTree] = None,
      allow_multi_axis_sharding_consolidation: Optional[bool] = None,
      preprocess_only: bool = False,
      with_gradient: bool = False,
      **kwargs: Any,
  ):
    pass

  @property
  def apply_fn_map(self) -> Mapping[str, ApplyFn]:
    """Returns the apply_fn_map."""
    return {}

  @property
  def model_params(self) -> PyTree:
    """Returns the model parameters."""
    return {}

  @property
  def jax2tf_kwargs_map(self) -> Mapping[str, Any]:
    """Returns the jax2tf_kwargs_map."""
    return {}

  @property
  def input_polymorphic_shape_map(self) -> Mapping[str, PyTree]:
    """Returns the polymorphic shapes."""
    return {}

  @property
  def methods(self) -> Mapping[str, Callable[..., Any]]:
    """Named methods in TF context."""
    return {}

  @property
  def jax_methods(self) -> Mapping[str, Callable[..., Any]]:
    """Named methods in JAX context for validation."""
    return {}
