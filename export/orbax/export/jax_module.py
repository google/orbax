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

"""Wraps JAX functions and parameters into a tf.Module."""

from collections.abc import Callable, Mapping
from typing import Any, Optional, Sequence, Union, cast

import jax
from jax import export as jax_export
from orbax.export import constants
from orbax.export import typing as orbax_export_typing
from orbax.export.modules import obm_module
from orbax.export.modules import orbax_module_base
from orbax.export.modules import tensorflow_module

PyTree = orbax_export_typing.PyTree
ApplyFn = orbax_export_typing.ApplyFn


class JaxModule(orbax_module_base.OrbaxModuleBase):
  """An exportable module for JAX functions and parameters.

  Holds tf.Variables converted from JAX parameters, as well as TF functions
  converted from JAX functions and bound with the tf.Variables.
  """

  def __init__(
      self,
      params: PyTree,
      apply_fn: Union[ApplyFn, Mapping[str, ApplyFn]],
      trainable: Optional[Union[bool, PyTree]] = None,
      input_polymorphic_shape: Union[PyTree, Mapping[str, PyTree], None] = None,
      input_polymorphic_shape_symbol_values: Union[
          PyTree, Mapping[str, PyTree], None
      ] = None,
      jax2tf_kwargs: Optional[Mapping[str, Any]] = None,
      jit_compile: Union[bool, Mapping[str, bool]] = True,
      pspecs: Optional[PyTree] = None,
      allow_multi_axis_sharding_consolidation: Optional[bool] = None,
      export_version: constants.ExportModelType = constants.ExportModelType.TF_SAVEDMODEL,
      jax2obm_kwargs: Optional[Mapping[str, Any]] = None,
  ):
    """JaxModule constructor.

    Args:
      params: a pytree of JAX parameters or parameter specs (e.g.
        `jax.ShapeDtypeStruct`s).
      apply_fn: A JAX ``ApplyFn`` (i.e. of signature ``apply_fn(params, x)``),
        or a mapping of method key to ``ApplyFn``. If it is an ``ApplyFn``, it
        will be assigned a key ``constants.DEFAULT_METHOD_KEY`` automatically,
        which can be used to look up the TF function converted from it.
      trainable: a pytree in the same structure as ``params`` and boolean leaves
        to tell if a parameter is trainable. Alternatively, it can be a single
        boolean value to tell if all the parameters are trainable or not. By
        default all parameters are non-trainable. The default value is subject
        to change in the future, thus it is recommended to specify the value
        explicitly. Currently trainable is only relevant for TF SavedModel
        export.
      input_polymorphic_shape: the polymorhpic shape for the inputs of
        ``apply_fn``. If ``apply_fn`` is a mapping, ``input_polymorphic_shape``
        must be a mapping of method key to the input polymorphic shape for the
        method.
      input_polymorphic_shape_symbol_values: optional mapping of symbol names
        presented in `input_polymorphic_shape` to possible values (e.g.
        {'batch_size': (1, 2), 'seq_len': (128, 512)}). When there are multiple
        ``apply_fn``s in the form of a flat mapping, this argument must be a
        flat mapping with the same keys (e.g. { 'serving_default': {
        'batch_size': (1, 2), 'seq_len': (128, 512)}). When this argument is
        set, the polymoprhic shape will be concretized to a set of all possible
        concreteized input shape combinations. This is only relevant for export
        model type `constants.ExportModelType.ORBAX_MODEL`
      jax2tf_kwargs: options passed to jax2tf. ``polymorphic_shape`` is inferred
        from ``input_polymorphic_shape`` and should not be set.
        ``with_gradient``, if set, should be consistent with the ``trainable``
        argument above. If ``jax2tf_kwargs`` is unspecified, the default jax2tf
        option will be applied. If ``apply_fn`` is a mapping, `jax2tf_kwargs`
        must either be unspecified or a mapping of method key to the jax2tf
        kwargs for the method. The jax2tf_kwargs is only relevant for TF
        SavedModel export.
      jit_compile: whether to jit compile the jax2tf converted functions. If
        ``apply_fn`` is a mapping, this can either be a boolean applied to all
        functions or a mapping of method key to the jit compile option for the
        method. The jit_compile is only relevant for TF SavedModel export as all
        methods for the Orbax model export are jit compiled.
      pspecs: an optional pytree of PartitionSpecs of the ``params`` in the same
        structure as ``params``. If set, the leaves of ``params`` must be
        jax.Array, and ``JaxModule`` must be created within a DTensor export
        context from ``with maybe_enable_dtensor_export_on(mesh)``. DTensor
        export is only supported for TF SavedModel export.
      allow_multi_axis_sharding_consolidation: Disallowed by default. When set
        to true, it will allow consolidating JAX array multiple axis sharding
        into DTensor single axis sharding during checkpoint conversion. This
        would enable sharding across multiple axis names support for JAX model.
        This is only relevant for TF SavedModel export.
      export_version: The model export version. Either TF_SAVEDMODEL or
        ORBAX_MODEL.
      jax2obm_kwargs: options passed to the Orbax Model export. Accepted
        arguments are 'native_serialization_platforms' which must be a tuple of
        OrbaxNativeSerializationType.

    raises:
      ValueError: If the export version is not supported.
    """
    self._export_version = export_version
    if (
        input_polymorphic_shape_symbol_values is not None
        and export_version != constants.ExportModelType.ORBAX_MODEL
    ):
      raise ValueError(
          '`input_polymorphic_shape_symbol_values` is only supported for'
          ' constants.ExportModelType.ORBAX_MODEL.'
      )

    match export_version:
      case constants.ExportModelType.TF_SAVEDMODEL:
        self._export_module = tensorflow_module.TensorFlowModule(
            params=params,
            apply_fn=apply_fn,
            trainable=trainable,
            input_polymorphic_shape=input_polymorphic_shape,
            jit_compile=jit_compile,
            pspecs=pspecs,
            allow_multi_axis_sharding_consolidation=allow_multi_axis_sharding_consolidation,
            jax2tf_kwargs=jax2tf_kwargs,
            export_version=export_version,
        )
      case constants.ExportModelType.ORBAX_MODEL:
        self._export_module = obm_module.ObmModule(
            params=params,
            apply_fn=apply_fn,
            input_polymorphic_shape=input_polymorphic_shape,
            input_polymorphic_shape_symbol_values=input_polymorphic_shape_symbol_values,
            jax2obm_kwargs=jax2obm_kwargs,
        )
      case _:
        raise ValueError(
            f'Unsupported export version: {export_version}, '
            'must be one of'
            f" {', '.join(c.name for c in constants.ExportModelType)}"
        )

  @property
  def apply_fn_map(self) -> Mapping[str, ApplyFn]:
    """Returns the apply_fn_map."""
    return self._export_module.apply_fn_map

  @property
  def model_params(self) -> PyTree:
    """Returns the model parameters."""
    return self._export_module.model_params

  @property
  def model_param_names(self) -> Sequence[str]:
    """Returns the list of model parameter names.

    The name format matches the one used by JSV to look up parameters in the
    checkpoint (e.g. "params.key.subkey").
    """

    param_names_tree = jax.tree_util.tree_map_with_path(
        lambda path, _: jax.tree_util.keystr(path, simple=True, separator='.'),
        self.model_params,
    )

    return jax.tree.leaves(param_names_tree)

  @property
  def export_version(self) -> constants.ExportModelType:
    """Returns the export version."""
    return self._export_version

  def export_module(self) -> orbax_module_base.OrbaxModuleBase:
    """Returns the export module."""
    return self._export_module

  @property
  def jax2tf_kwargs_map(self) -> Mapping[str, Any]:
    """Returns the jax2tf_kwargs_map."""
    if self._export_version == constants.ExportModelType.ORBAX_MODEL:
      raise TypeError(
          'jax2tf_kwargs_map is not implemented for export version'
          ' ExportModelType.ORBAX_MODEL.'
      )
    return cast(
        tensorflow_module.TensorFlowModule, self._export_module
    ).jax2tf_kwargs_map

  @property
  def input_polymorphic_shape_map(self) -> Mapping[str, PyTree]:
    """Returns the polymorphic shapes."""
    if self._export_version == constants.ExportModelType.ORBAX_MODEL:
      raise TypeError(
          'input_polymorphic_shape_map is not implemented for export version'
          ' ExportModelType.ORBAX_MODEL.'
      )
    return cast(
        tensorflow_module.TensorFlowModule, self._export_module
    ).input_polymorphic_shape_map

  @property
  def with_gradient(self) -> bool:
    """Returns the with_gradient."""
    if self._export_version == constants.ExportModelType.ORBAX_MODEL:
      raise TypeError(
          'with_gradient is not implemented for export version'
          ' ExportModelType.ORBAX_MODEL.'
      )
    return cast(
        tensorflow_module.TensorFlowModule, self._export_module
    ).with_gradient

  def update_variables(self, params: PyTree):
    """Updates the variables associated with self.

    Args:
      params: A PyTree of JAX parameters. The PyTree structure must be the same
        as that of the `params` used to initialize the model. Additionally, the
        shape and dtype of each parameter must be the same as the original
        parameter.
    """
    if self._export_version == constants.ExportModelType.ORBAX_MODEL:
      raise TypeError(
          'update_variables is not implemented for export version'
          ' ExportModelType.ORBAX_MODEL.'
      )
    cast(
        tensorflow_module.TensorFlowModule, self._export_module
    ).update_variables(params)

  def orbax_module(self) -> orbax_module_base.OrbaxModuleBase:
    """Returns the OrbaxModule associated with this JaxModule."""
    return self._export_module

  @property
  def methods(self) -> Mapping[str, Callable[..., Any]]:
    """Named methods in TF context."""
    return self._export_module.methods

  @property
  def jax_methods(self) -> Mapping[str, Callable[..., Any]]:
    """Named methods in JAX context for validation."""
    if self._export_version == constants.ExportModelType.ORBAX_MODEL:
      raise TypeError(
          'jax_methods is not implemented for export version'
          ' ExportModelType.ORBAX_MODEL.'
      )
    return cast(
        tensorflow_module.TensorFlowModule, self._export_module
    ).jax_methods

  def obm_module_to_jax_exported_map(
      self, model_inputs: PyTree
  ) -> Mapping[str, jax_export.Exported]:
    """Converts the orbax.export JaxModule to jax_export.Exported.

    Args:
      model_inputs: The model inputs.

    Returns:
      A mapping from method key to jax_export.Exported.
    """
    if self._export_version == constants.ExportModelType.ORBAX_MODEL:
      raise TypeError(
          'obm_module_to_jax_exported_map is not implemented for export version'
          ' ExportModelType.ORBAX_MODEL.'
      )
    return cast(
        tensorflow_module.TensorFlowModule, self._export_module
    ).obm_module_to_jax_exported_map(model_inputs)
