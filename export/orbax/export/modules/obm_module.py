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

from collections.abc import Callable, Mapping, Sequence
import copy
import logging
from typing import Any, Optional, Union

import jax
from orbax.export import constants
from orbax.export import typing as orbax_export_typing
from orbax.export.modules import orbax_module_base
from orbax.export.typing import PyTree
from orbax.export import utils
import tensorflow as tf

ApplyFn = orbax_export_typing.ApplyFn


class ObmModule(orbax_module_base.OrbaxModuleBase):
  """A data module for encapsulating the data for a Jax model to be serialized through the Orbax Model export flow."""

  def __init__(
      self,
      params: PyTree,
      apply_fn: Union[ApplyFn, Mapping[str, ApplyFn]],
      jax2obm_kwargs: Union[Mapping[str, Any], None] = None,
  ):
    """Data container for Orbax Model export.

    Args:
      params: The model parameter specs (e.g. `jax.ShapeDtypeStruct`s).
      apply_fn: The apply_fn for the model.
      jax2obm_kwargs: A dictionary of kwargs to pass to the jax2obm conversion
        library. Accepted arguments to jax2obm_kwargs are
        'native_serialization_platforms', 'flatten_signature', 'weights_name'and
        'checkpoint_path'.
    """

    # It is possible for jax2obm_kwargs to be None if the key is present.
    if not jax2obm_kwargs:
      jax2obm_kwargs = {}

    self._apply_fn_map = self._normalize_apply_fn_map(
        self._normalize_apply_fn_map(apply_fn)
    )

    if len(self._apply_fn_map) != 1:
      raise NotImplementedError(
          'ObmModule: Currently the ObmExport only supports a single method'
          f' for export. Received: {self._apply_fn_map}'
      )

    self._native_serialization_platforms = utils.get_lowering_platforms(
        jax2obm_kwargs
    )

    self._flatten_signature = (
        jax2obm_kwargs[constants.FLATTEN_SIGNATURE]
        if constants.FLATTEN_SIGNATURE in jax2obm_kwargs
        else False
    )
    self._support_tf_resources = jax2obm_kwargs.get(
        constants.OBM_SUPPORT_TF_RESOURCES, None
    )
    if self._support_tf_resources is None:
      self._support_tf_resources = False

    self._params_args_spec = params

    self._checkpoint_path: str = None
    # Set the Orbax checkpoint path if provided in the jax2obm_kwargs.
    self._maybe_set_orbax_checkpoint_path(jax2obm_kwargs)

  def _normalize_apply_fn_map(
      self, apply_fn: Union[ApplyFn, Mapping[str, ApplyFn]]
  ) -> Mapping[str, ApplyFn]:
    if callable(apply_fn):
      apply_fn_map = {constants.DEFAULT_METHOD_KEY: apply_fn}
    elif len(apply_fn) > 1:
      raise NotImplementedError(
          'ObmModule: Currently the ObmExport only supports a single method'
          f' per module. Received: {apply_fn}'
      )
    else:
      apply_fn_map = apply_fn
    return apply_fn_map

  def _maybe_set_orbax_checkpoint_path(self, jax2obm_kwargs):
    if constants.CHECKPOINT_PATH not in jax2obm_kwargs:
      return

    # TODO: b/374195447 - Add a version check for the Orbax checkpointer.
    self._checkpoint_path = jax2obm_kwargs[constants.CHECKPOINT_PATH]
    self._weights_name = (
        jax2obm_kwargs[constants.WEIGHTS_NAME]
        if constants.WEIGHTS_NAME in jax2obm_kwargs
        else constants.DEFAULT_WEIGHTS_NAME
    )

  def export_module(
      self,
  ) -> Union[tf.Module, orbax_module_base.OrbaxModuleBase]:
    return self

  @property
  def apply_fn_map(self) -> Mapping[str, ApplyFn]:
    """Returns the apply_fn_map from function name to jit'd apply function."""
    return self._apply_fn_map

  @property
  def native_serialization_platforms(
      self,
  ) -> Optional[Sequence[constants.OrbaxNativeSerializationType]]:
    """Returns the native serialization platform."""
    return self._native_serialization_platforms

  @property
  def flatten_signature(self) -> bool:
    """Returns the flatten signature."""
    return self._flatten_signature

  @property
  def export_version(self) -> constants.ExportModelType:
    """Returns the export version."""
    return constants.ExportModelType.ORBAX_MODEL

  def support_tf_resources(self) -> bool:
    """Returns True if the model supports TF resources."""
    return self._support_tf_resources

  @property
  def model_params(self) -> PyTree:
    """Returns the model parameter specs."""
    return self._params_args_spec

  def obm_module_to_jax_exported_map(
      self,
      model_inputs: PyTree,
  ) -> Mapping[str, jax.export.Exported]:
    """Converts the OrbaxModel to jax_export.Exported."""
    raise NotImplementedError(
        'ObmModule.methods not implemented yet. See b/363061755.'
    )

  @property
  def with_gradient(self) -> bool:
    """Returns True if a gradient function is defined."""
    raise NotImplementedError(
        'ObmModule.methods not implemented yet. See b/363061755.'
    )

  @property
  def methods(self) -> Mapping[str, Callable[..., Any]]:
    """Named methods in the context of the chosen export pathway."""
    raise NotImplementedError('apply_fn_map is not implemented for ObmModule.')

  @property
  def jax_methods(self) -> Mapping[str, Callable[..., Any]]:
    """Named methods in JAX context for validation."""
    raise NotImplementedError('apply_fn_map is not implemented for ObmModule.')
