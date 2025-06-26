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

"""Wraps JAX functions and parameters into a ObmModule(Orbax Model Module)."""

from collections.abc import Callable, Mapping, Sequence
import copy
import logging
from typing import Any, Optional, Tuple, Union

import jax
from orbax.export import constants
from orbax.export import typing as orbax_export_typing
from orbax.export import utils
from orbax.export.modules import orbax_module_base
from orbax.export.typing import PyTree
import tensorflow as tf

ApplyFn = orbax_export_typing.ApplyFn


class ObmModule(orbax_module_base.OrbaxModuleBase):
  """A data module for encapsulating the data for a Jax model to be serialized through the Orbax Model export flow."""

  def __init__(
      self,
      params: PyTree,
      apply_fn: Union[ApplyFn, Mapping[str, ApplyFn]],
      *,
      input_polymorphic_shape: Any = None,
      jax2obm_kwargs: Union[Mapping[str, Any], None] = None,
  ):
    """Data container for Orbax Model export.

    Args:
      params: The model parameter specs (e.g. `jax.ShapeDtypeStruct`s).
      apply_fn: The apply_fn for the model.
      input_polymorphic_shape: polymorhpic shape for the inputs of `apply_fn`.
      jax2obm_kwargs: A dictionary of kwargs to pass to the jax2obm conversion
        library. Accepted arguments to jax2obm_kwargs are
        'native_serialization_platforms', 'weights_name', 'checkpoint_path' and
        'polymorphic_constraints'.
    """

    # It is possible for jax2obm_kwargs to be None if the key is present.
    if not jax2obm_kwargs:
      jax2obm_kwargs = {}

    self._apply_fn_map, self.input_polymorphic_shape_map = (
        self._normalize_apply_fn_map(apply_fn, input_polymorphic_shape)
    )

    self._xla_compile_options = jax2obm_kwargs.get(
        constants.XLA_COMPILE_OPTIONS, None
    )

    self.polymorphic_constraints = self._maybe_set_polymorphic_constraints(
        jax2obm_kwargs
    )
    if self.polymorphic_constraints is None:
      self.polymorphic_constraints = ()

    self._native_serialization_platforms = utils.get_lowering_platforms(
        jax2obm_kwargs
    )
    self._params_args_spec = params

    self._checkpoint_path: str = None
    # Set the Orbax checkpoint path if provided in the jax2obm_kwargs.
    self._maybe_set_orbax_checkpoint_path(jax2obm_kwargs)

  def _normalize_apply_fn_map(
      self,
      apply_fn: Union[ApplyFn, Mapping[str, ApplyFn]],
      input_polymorphic_shape: Union[PyTree, Mapping[str, PyTree], None],
  ) -> Tuple[Mapping[str, ApplyFn], Mapping[str, Union[PyTree, None]]]:
    if callable(apply_fn):
      apply_fn_map = {constants.DEFAULT_METHOD_KEY: apply_fn}
      input_polymorphic_shape_map = {
          constants.DEFAULT_METHOD_KEY: input_polymorphic_shape
      }
    elif len(apply_fn) > 1:
      apply_fn_map = apply_fn
      input_polymorphic_shape_map = input_polymorphic_shape
      if not isinstance(input_polymorphic_shape, Mapping):
        raise TypeError(
            'When apply_fn is a mapping, input_polymorphic_shape must also be a'
            ' mapping.'
        )
      if len(apply_fn_map) != len(input_polymorphic_shape_map):
        raise ValueError(
            f'The size of apply_fn_map:{len(apply_fn_map)} should be equal to'
            ' the size of'
            f' input_polymorphic_shape_map:{len(input_polymorphic_shape_map)}.'
        )
    else:
      apply_fn_map = apply_fn
      if not apply_fn_map:
        raise ValueError(
            'apply_fn_map is empty. Please provide a valid apply_fn_map.'
        )
      elif input_polymorphic_shape is None:
        input_polymorphic_shape_map = {next(iter(apply_fn_map)): None}
      elif not isinstance(input_polymorphic_shape, Mapping):
        raise TypeError(
            'When apply_fn is a mapping, input_polymorphic_shape must also be a'
            ' mapping.'
        )
      elif next(iter(apply_fn_map)) not in input_polymorphic_shape:
        raise ValueError(
            f'The key {next(iter(apply_fn_map))} is not found in'
            ' input_polymorphic_shape.'
        )
      else:
        input_polymorphic_shape_map = input_polymorphic_shape
    return apply_fn_map, input_polymorphic_shape_map

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

  def _maybe_set_polymorphic_constraints(
      self, jax2obm_kwargs
  ) -> Mapping[str, Sequence[Any]]:
    """Sets the polymorphic constraints for the model.

    Args:
      jax2obm_kwargs: A dictionary of kwargs to pass to the jax2obm conversion
        library.

    Returns:
      A mapping of function name to polymorphic constraints.

    Raises:
      TypeError: If the polymorphic_constraints is not a Sequence or a Mapping.
      ValueError: If the size of the polymorphic_constraints is not equal to the
        size of the apply_fn_map or if a key in apply_fn_map is not found in
        polymorphic_constraints.
    """
    polymorphic_constraints = jax2obm_kwargs.get(
        constants.POLYMORPHIC_CONSTRAINTS, None
    )
    if not isinstance(polymorphic_constraints, Mapping):
      polymorphic_constraints_mapping = {}
      if polymorphic_constraints is None:
        polymorphic_constraints = ()
      if not isinstance(polymorphic_constraints, Sequence):
        raise TypeError(
            'If not a Mapping, polymorphic_constraints needs to be a Sequence.'
            f' Got type: {type(polymorphic_constraints)} .'
        )
      for key in self._apply_fn_map:
        # If the polymorphic_constraints is a non-Mapping (in which case it
        # needs to be a Sequence), which means it is the same for all
        # functions, we need to map it to a mapping of function name to
        # constraint.
        polymorphic_constraints_mapping[key] = polymorphic_constraints
      polymorphic_constraints = polymorphic_constraints_mapping
    else:
      if len(polymorphic_constraints) != len(self._apply_fn_map):
        raise ValueError(
            'The size of'
            f' polymorphic_constraints:{len(polymorphic_constraints)} should be'
            f' equal to the size of the apply_fn_map:{len(self._apply_fn_map)}.'
        )
      for key in self._apply_fn_map:
        if key not in polymorphic_constraints:
          raise ValueError(
              f'The key {key} is not found in polymorphic_constraints:'
              f' {polymorphic_constraints}'
          )
    return polymorphic_constraints  # pytype: disable=bad-return-type

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
  def export_version(self) -> constants.ExportModelType:
    """Returns the export version."""
    return constants.ExportModelType.ORBAX_MODEL

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
