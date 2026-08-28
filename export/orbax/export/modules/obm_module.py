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

"""Wraps JAX functions and parameters into a ObmModule(Orbax Model Module)."""

from collections.abc import Callable, Mapping, Sequence
import copy
import dataclasses
import logging
from typing import Any, Optional, Union
import warnings

import jax
from orbax.export import constants
from orbax.export import obm_configs
from orbax.export import typing as orbax_export_typing
from orbax.export import utils
from orbax.export.modules import orbax_module_base
from orbax.export.typing import PyTree
import tensorflow as tf

ApplyFn = orbax_export_typing.ApplyFn

_KNOWN_JAX2OBM_KEYS: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(obm_configs.Jax2ObmOptions)
) | frozenset([getattr(constants, 'PSPECS', 'pspecs')])


def _get_shared_value(
    m: Mapping[str, obm_configs.Jax2ObmOptions] | obm_configs.Jax2ObmOptions,
    keys: Sequence[str] | None,
    field_name: str,
) -> Any:
  """Returns attribute `field_name` from `m` or checks if it is shared in `m`.

  If `m` is a Jax2ObmOptions object, returns `getattr(m, field_name)`. If `m`
  is a mapping, it checks if all keys in `keys` are present in `m`, and if
  `field_name` has the same value across all values in `m`.

  Args:
    m: A Jax2ObmOptions object or a mapping where values are Jax2ObmOptions
      objects.
    keys: A sequence of keys that must be present in `m` if `m` is a mapping.
    field_name: The attribute name to get or check for shared value.

  Returns:
    The attribute value if `m` is a Jax2ObmOptions object, or the shared
    attribute value if `m` is a mapping and all values for `field_name` are
    the same.

  Raises:
    ValueError: If `m` is empty, or if `m` is a mapping and any key in `keys`
      is not found in `m`, or if values for `field_name` are not all same.
    AttributeError: If `field_name` is not an attribute of `Jax2ObmOptions`.
  """
  if not m:
    raise ValueError('Input mapping is empty.')
  if isinstance(m, obm_configs.Jax2ObmOptions):
    return getattr(m, field_name)
  if keys:
    for key in keys:
      if key not in m:
        raise ValueError(f'Key {key} is not found in mapping {m}.')
  value = getattr(next(iter(m.values())), field_name)
  if not all(getattr(v, field_name) == value for v in m.values()):
    raise ValueError(
        f'Not all values for `{field_name}` in the mapping are the same.'
    )
  return value


def _normalize_polymorphic_constraints(
    constraints: Mapping[str, Sequence[str]] | Sequence[str] | None,
    apply_fn_keys: Sequence[str],
) -> Mapping[str, Sequence[str]]:
  """Normalizes polymorphic constraints into a function-to-constraints mapping.

  Args:
    constraints: A mapping from function name to constraints, a single sequence
      of constraints to broadcast across all functions, or None.
    apply_fn_keys: Sequence of function names in the model.

  Returns:
    A mapping from each function name to its polymorphic constraints sequence.

  Raises:
    TypeError: If constraints is not a Mapping, Sequence, or None (e.g. str).
    ValueError: If constraints mapping size does not match apply_fn_keys or if
      a function key is missing from the constraints mapping.
  """
  if isinstance(constraints, Mapping):
    if len(constraints) != len(apply_fn_keys):
      raise ValueError(
          f'The size of polymorphic_constraints:{len(constraints)} should'
          f' be equal to the size of the apply_fn_map:{len(apply_fn_keys)}.'
      )
    for key in apply_fn_keys:
      if key not in constraints:
        raise ValueError(
            f'The key {key} is not found in polymorphic_constraints:'
            f' {constraints}'
        )
    return constraints

  if constraints is None:
    return {key: () for key in apply_fn_keys}

  if isinstance(constraints, Sequence) and not isinstance(
      constraints, (str, bytes)
  ):
    # If the polymorphic_constraints is a non-Mapping (in which case it
    # needs to be a Sequence), which means it is the same for all
    # functions, we need to map it to a mapping of function name to
    # constraint.
    return {key: constraints for key in apply_fn_keys}

  raise TypeError(
      'If not a Mapping, polymorphic_constraints needs to be a Sequence.'
      f' Got type: {type(constraints)} .'
  )


class ObmModule(orbax_module_base.OrbaxModuleBase):
  """Container for serializing a Jax model via the Orbax Model export flow."""

  def __init__(
      self,
      params: PyTree,
      apply_fn: (
          orbax_export_typing.ApplyFn
          | orbax_export_typing.ApplyFnInfo
          | Mapping[
              str, orbax_export_typing.ApplyFn | orbax_export_typing.ApplyFnInfo
          ]
      ),
      *,
      input_polymorphic_shape: Any = None,
      input_polymorphic_shape_symbol_values: Union[
          Mapping[str, PyTree], Mapping[str, Mapping[str, PyTree]], None
      ] = None,
      jax2obm_options: (
          obm_configs.Jax2ObmOptions
          | Mapping[str, obm_configs.Jax2ObmOptions]
          | None
      ) = None,
      jax2obm_kwargs: Union[Mapping[str, Any], None] = None,
  ):
    """Data container for Orbax Model export.

    Args:
      params: The model parameter specs (e.g. `jax.ShapeDtypeStruct`s).
      apply_fn: A single `ApplyFn` (taking `model_params` and `model_inputs`), a
        single `ApplyFnInfo` object (containing `ApplyFn` and input/output
        keys), or a mapping of method keys to `ApplyFn`s or `ApplyFnInfo`
        objects. If it is a single ``ApplyFn`` or ``ApplyFnInfo``, it will be
        assigned a key ``constants.DEFAULT_METHOD_KEY`` automatically.
      input_polymorphic_shape: polymorphic shape for the inputs of `apply_fn`.
      input_polymorphic_shape_symbol_values: optional mapping of symbol names
        presented in `input_polymorphic_shape` to discrete values (e.g. {'b':
        (1, 2), 'l': (128, 512)}). When there are multiple ``apply_fn``s in the
        form of a flat mapping, this argument must be a flat mapping with the
        same keys (e.g. { 'serving_default': { 'b': (1, 2), 'l': (128, 512)}).
        When this argument is set, the polymorphic shape will be concretized to
        a set of all possible concretized input shape combinations.
      jax2obm_options: Options for jax2obm conversion. If `apply_fn` is a
        mapping, this can also be a mapping from method keys to
        `Jax2ObmOptions`. When it is a mapping, all options must be shared
        across different apply functions, except for `enable_auto_layout` and
        `native_serialization_disabled_checks`.
      jax2obm_kwargs: DEPRECATED. Use `jax2obm_options` instead. A dictionary of
        kwargs to pass to the jax2obm conversion library. Accepted arguments to
        jax2obm_kwargs are 'native_serialization_platforms', 'weights_name',
        'checkpoint_path' and 'polymorphic_constraints'.
    """
    if (
        input_polymorphic_shape is None
        and input_polymorphic_shape_symbol_values is not None
    ):
      raise ValueError(
          '`input_polymorphic_shape` is required when'
          ' `input_polymorphic_shape_symbol_values` is provided.'
      )

    if jax2obm_kwargs:
      if jax2obm_options is not None:
        raise ValueError(
            'Both `jax2obm_kwargs` and `jax2obm_options` are set. Please only'
            ' use `jax2obm_options`.'
        )
      warnings.warn(
          '`jax2obm_kwargs` is deprecated, use `jax2obm_options` instead.',
          DeprecationWarning,
      )
      jax2obm_options = self._jax2obm_kwargs_to_options(jax2obm_kwargs)
    elif jax2obm_options is None:
      jax2obm_options = obm_configs.Jax2ObmOptions()
    self._jax2obm_options = jax2obm_options

    if isinstance(self._jax2obm_options, Mapping):
      if not isinstance(apply_fn, Mapping):
        raise ValueError(
            'If `jax2obm_options` is a mapping, `apply_fn` must be a mapping.'
        )
      self._apply_fn_keys = list(apply_fn.keys())
    else:
      self._apply_fn_keys = None

    self._enable_bf16_optimization = _get_shared_value(
        self._jax2obm_options,
        self._apply_fn_keys,
        constants.ENABLE_BF16_OPTIMIZATION,
    )
    if self._enable_bf16_optimization:
      mapped_apply_fn = utils.to_bfloat16(apply_fn)
      self._params_args_spec = utils.to_bfloat16(params)
    else:
      mapped_apply_fn = apply_fn
      self._params_args_spec = params
    (
        self._apply_fn_map,
        self.input_polymorphic_shape_map,
        self.input_polymorphic_shape_symbol_values_map,
    ) = self._normalize_apply_fn_map(
        mapped_apply_fn,
        input_polymorphic_shape,
        input_polymorphic_shape_symbol_values,
    )
    self._jax_mesh = _get_shared_value(
        self._jax2obm_options,
        self._apply_fn_keys,
        constants.JAX_MESH,
    )
    self.polymorphic_constraints = self._maybe_set_polymorphic_constraints()
    self._native_serialization_platforms = utils.get_lowering_platforms(
        _get_shared_value(
            self._jax2obm_options,
            self._apply_fn_keys,
            constants.NATIVE_SERIALIZATION_PLATFORMS,
        )
    )
    xla_flags_per_platform = _get_shared_value(
        self._jax2obm_options,
        self._apply_fn_keys,
        constants.XLA_FLAGS_PER_PLATFORM,
    )
    persist_xla_flags = _get_shared_value(
        self._jax2obm_options,
        self._apply_fn_keys,
        constants.PERSIST_XLA_FLAGS,
    )
    self._save_shlo_to_file = _get_shared_value(
        self._jax2obm_options,
        self._apply_fn_keys,
        constants.SAVE_SHLO_TO_FILE,
    )

    self._checkpoint_path: str | None = None
    # Set the Orbax checkpoint path if provided in the jax2obm_kwargs.
    self._maybe_set_orbax_checkpoint_path()

  def _jax2obm_kwargs_to_options(
      self, jax2obm_kwargs: Mapping[str, Any]
  ) -> obm_configs.Jax2ObmOptions:
    """Converts jax2obm_kwargs to Jax2ObmOptions."""
    unrecognized = set(jax2obm_kwargs.keys()) - _KNOWN_JAX2OBM_KEYS
    if unrecognized:
      logging.warning(
          'Unrecognized keys in jax2obm_kwargs: %s. '
          'Please migrate to `Jax2ObmOptions` or remove unmapped options.',
          unrecognized,
      )
    return obm_configs.Jax2ObmOptions(
        native_serialization_platforms=jax2obm_kwargs.get(
            constants.NATIVE_SERIALIZATION_PLATFORMS
        ),
        checkpoint_path=jax2obm_kwargs.get(constants.CHECKPOINT_PATH),
        weights_name=jax2obm_kwargs.get(constants.WEIGHTS_NAME),
        polymorphic_constraints=jax2obm_kwargs.get(
            constants.POLYMORPHIC_CONSTRAINTS
        ),
        xla_flags_per_platform=jax2obm_kwargs.get(
            constants.XLA_FLAGS_PER_PLATFORM
        ),
        jax_mesh=jax2obm_kwargs.get(constants.JAX_MESH),
        persist_xla_flags=jax2obm_kwargs.get(constants.PERSIST_XLA_FLAGS, True),
        enable_bf16_optimization=jax2obm_kwargs.get(
            constants.ENABLE_BF16_OPTIMIZATION, False
        ),
        save_shlo_to_file=jax2obm_kwargs.get(
            constants.SAVE_SHLO_TO_FILE, False
        ),
        loader_type=jax2obm_kwargs.get(constants.LOADER_TYPE),
    )

  def _normalize_apply_fn_map(
      self,
      apply_fn: (
          orbax_export_typing.ApplyFn
          | orbax_export_typing.ApplyFnInfo
          | Mapping[
              str, orbax_export_typing.ApplyFn | orbax_export_typing.ApplyFnInfo
          ]
      ),
      input_polymorphic_shape: Union[PyTree, Mapping[str, PyTree], None],
      input_polymorphic_shape_symbol_values: Union[
          PyTree, Mapping[str, PyTree], None
      ],
  ) -> tuple[
      Mapping[
          str, orbax_export_typing.ApplyFn | orbax_export_typing.ApplyFnInfo
      ],
      Mapping[str, Union[PyTree, None]],
      Mapping[str, Union[PyTree, None]],
  ]:
    """Converts all the inputs to maps that share the same keys."""

    # Single apply_fn case. Will use the default method key.
    if not isinstance(apply_fn, Mapping):
      apply_fn: orbax_export_typing.ApplyFnInfo | orbax_export_typing.ApplyFn
      apply_fn_map = {constants.DEFAULT_METHOD_KEY: apply_fn}
      input_polymorphic_shape_map = {
          constants.DEFAULT_METHOD_KEY: input_polymorphic_shape
      }
      input_polymorphic_shape_symbol_values_map = {
          constants.DEFAULT_METHOD_KEY: input_polymorphic_shape_symbol_values
      }
      return (
          apply_fn_map,
          input_polymorphic_shape_map,
          input_polymorphic_shape_symbol_values_map,
      )

    # Mapping from method key to apply_fn.
    if isinstance(apply_fn, Mapping):
      if not apply_fn:
        raise ValueError('`apply_fn` should be a non-empty mapping')
      apply_fn_map = apply_fn

      # Handle `input_polymorphic_shape`
      if input_polymorphic_shape is None:
        input_polymorphic_shape_map = {key: None for key in apply_fn_map}
      else:
        if not isinstance(input_polymorphic_shape, Mapping):
          raise TypeError(
              '`input_polymorphic_shape` must be a mapping, but got'
              f' {type(input_polymorphic_shape)}.'
          )
        input_polymorphic_shape_map = input_polymorphic_shape
      if apply_fn_map.keys() != input_polymorphic_shape_map.keys():
        raise ValueError(
            'The keys of `apply_fn` and `input_polymorphic_shape` should be'
            f' the same, but got ({apply_fn_map.keys()}) vs'
            f' ({input_polymorphic_shape_map.keys()})'
        )

      # Handle `input_polymorphic_shape_symbol_values`
      if input_polymorphic_shape_symbol_values is None:
        input_polymorphic_shape_symbol_values_map = {
            key: None for key in apply_fn_map
        }
      else:
        if not isinstance(input_polymorphic_shape_symbol_values, Mapping):
          raise TypeError(
              '`input_polymorphic_shape_symbol_values` must be a mapping, but'
              f' got {type(input_polymorphic_shape_symbol_values)}.'
          )
        input_polymorphic_shape_symbol_values_map = (
            input_polymorphic_shape_symbol_values
        )
      if (
          apply_fn_map.keys()
          != input_polymorphic_shape_symbol_values_map.keys()
      ):
        raise ValueError(
            'The keys of `apply_fn` and'
            ' `input_polymorphic_shape_symbol_values` should be the same, but'
            f' got ({apply_fn_map.keys()}) vs'
            f' ({input_polymorphic_shape_symbol_values_map.keys()})'
        )

      return (
          apply_fn_map,
          input_polymorphic_shape_map,
          input_polymorphic_shape_symbol_values_map,
      )

    raise TypeError(
        f'`apply_fn` must be a callable or a mapping, but got {type(apply_fn)}.'
    )

  def _maybe_set_orbax_checkpoint_path(self):
    if (
        _get_shared_value(
            self._jax2obm_options,
            self._apply_fn_keys,
            constants.CHECKPOINT_PATH,
        )
        is None
    ):
      self._weights_name = None
      return

    # TODO: b/374195447 - Add a version check for the Orbax checkpointer.
    self._checkpoint_path = _get_shared_value(
        self._jax2obm_options, self._apply_fn_keys, constants.CHECKPOINT_PATH
    )
    weights_name = _get_shared_value(
        self._jax2obm_options, self._apply_fn_keys, constants.WEIGHTS_NAME
    )
    self._weights_name = weights_name or constants.DEFAULT_WEIGHTS_NAME

  def _maybe_set_polymorphic_constraints(self) -> Mapping[str, Sequence[str]]:
    """Sets the polymorphic constraints for the model."""
    constraints = _get_shared_value(
        self._jax2obm_options,
        self._apply_fn_keys,
        constants.POLYMORPHIC_CONSTRAINTS,
    )
    return _normalize_polymorphic_constraints(
        constraints, list(self._apply_fn_map.keys())
    )

  def export_module(  # pyrefly: ignore[bad-override]
      self,
  ) -> Union[tf.Module, orbax_module_base.OrbaxModuleBase]:
    return self

  @property
  def apply_fn_map(  # pyrefly: ignore[bad-override]
      self,
  ) -> Mapping[
      str, orbax_export_typing.ApplyFn | orbax_export_typing.ApplyFnInfo
  ]:
    """Returns the apply_fn_map from function name to jit'd apply function."""
    return self._apply_fn_map

  @property
  def export_version(  # pyrefly: ignore[bad-override]
      self,
  ) -> constants.ExportModelType:
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

  @property
  def jax2obm_options(
      self,
  ) -> obm_configs.Jax2ObmOptions | Mapping[str, obm_configs.Jax2ObmOptions]:
    """Returns the jax2obm_options."""
    return self._jax2obm_options
