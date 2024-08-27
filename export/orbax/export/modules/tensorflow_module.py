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
from typing import Any, Optional, Sequence, Union

from absl import logging
import jax
from jax import export as jax_export
from jax.experimental import jax2tf
from orbax.export import config
from orbax.export import constants
from orbax.export import dtensor_utils
from orbax.export import typing as orbax_export_typing
from orbax.export.modules import orbax_module_base
import orbax.export.utils as export_utils
import tensorflow as tf
from tensorflow.experimental import dtensor

PyTree = orbax_export_typing.PyTree
ApplyFn = orbax_export_typing.ApplyFn
obx_export_config = config.config


def _same_keys(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
  return set(a.keys()) == set(b.keys())


def _make_closures(
    params: PyTree, apply_fn_map: Mapping[str, ApplyFn]
) -> Mapping[str, Callable[..., Any]]:
  """Creates closures for apply functions."""

  def bind_params(apply_fn):
    return lambda x: apply_fn(params, x)

  return jax.tree_util.tree_map(bind_params, apply_fn_map)


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
      apply_fn: Union[ApplyFn, Mapping[str, ApplyFn]],
      **kwargs: Any,
  ):
    jax2tf_kwargs = kwargs.get('jax2tf_kwargs', None)
    trainable = kwargs.get('trainable', False)
    input_polymorphic_shape = kwargs.get('input_polymorphic_shape', None)
    jit_compile = kwargs.get('jit_compile', True)
    pspecs = kwargs.get('pspecs', None)
    allow_multi_axis_sharding_consolidation = kwargs.get(
        'allow_multi_axis_sharding_consolidation', None
    )
    self._with_gradient = any(jax.tree_util.tree_leaves(trainable))

    if callable(apply_fn):
      apply_fn_map: dict[str, ApplyFn] = {
          constants.DEFAULT_METHOD_KEY: apply_fn
      }
      input_polymorphic_shape = {
          constants.DEFAULT_METHOD_KEY: input_polymorphic_shape
      }
      jax2tf_kwargs = {constants.DEFAULT_METHOD_KEY: jax2tf_kwargs}
      jit_compile = {constants.DEFAULT_METHOD_KEY: jit_compile}
    else:
      # Check if `apply_fn`, `input_polymorphic_shape` and `jax2tf_kwargs` have
      # the same structure.
      apply_fn_map = apply_fn
      if input_polymorphic_shape is None:
        input_polymorphic_shape = jax.tree_util.tree_map(
            lambda x: None, apply_fn_map
        )
      elif not isinstance(input_polymorphic_shape, Mapping) or not _same_keys(
          input_polymorphic_shape, apply_fn_map
      ):
        raise ValueError(
            '`input_polymorphic_shape` must have the same structure as that of'
            f' `apply_fn`. Got apply_fn={apply_fn_map},'
            f' input_polymorphic_shape={input_polymorphic_shape}.'
        )
      if jax2tf_kwargs is None:
        # OK if it is unspecified, which means `jax2tf_kwargs` is unspecified
        # for all apply functions.
        jax2tf_kwargs = jax.tree_util.tree_map(lambda x: None, apply_fn_map)
      elif not _same_keys(jax2tf_kwargs, apply_fn_map):
        raise ValueError(
            '`jax2tf_kwargs` must either be unspecified or have the same '
            f'structure as that of `apply_fn`. Got apply_fn={apply_fn_map}, '
            f'jax2tf_kwargs={jax2tf_kwargs}.'
        )
      if not isinstance(jit_compile, Mapping) or isinstance(jit_compile, bool):
        jit_compile = jax.tree_util.tree_map(
            lambda x: jit_compile, apply_fn_map
        )
      elif not _same_keys(jit_compile, apply_fn_map):
        raise ValueError(
            '`jit_compile` must either be a boolean or have the same '
            f'structure as that of `apply_fn`. Got apply_fn={apply_fn_map}, '
            f'jit_compile={jit_compile}.'
        )

    if trainable is None:
      trainable = False
    if isinstance(trainable, bool):
      trainable = jax.tree_util.tree_map(lambda x: trainable, params)

    if obx_export_config.obx_export_tf_preprocess_only:  # pytype: disable=attribute-error
      # Skip the heavy jax_params_to_tf_variables() call in TF preprocess only
      # mode.
      self._tf_var_treedef = None
      self._tf_var_leaves = None
      self._methods = dict()
    else:
      tf_vars = self.jax_params_to_tf_variables(
          params, trainable, pspecs, allow_multi_axis_sharding_consolidation
      )
      # Do not attach `tf_vars` to `self` directly, otherwise its structure will
      # be mutated by `tf.Module.__setattr__`.
      self._tf_var_leaves, self._tf_var_treedef = jax.tree_util.tree_flatten(
          tf_vars
      )

      self._methods = jax.tree_util.tree_map(
          self._make_tf_closure,
          apply_fn_map,
          input_polymorphic_shape,
          jax2tf_kwargs,
          jit_compile,
      )

    # # Preserve the original structure of this Metadata object to prevent
    # unintended conversion by TF tf.Module (e.g., Dict to DictWrapper).
    self._nontrackable_metadata = _NonTrackableMetadata(
        apply_fn_map=apply_fn_map,
        tf_var_treedef=self._tf_var_treedef,
        var_trainable=trainable,
        var_pspecs=pspecs,
        model_params=params,
        jax2tf_kwargs_map=jax2tf_kwargs,
        input_polymorphic_shape_map=input_polymorphic_shape,
        allow_multi_axis_sharding_consolidation=allow_multi_axis_sharding_consolidation,
    )

  @property
  def with_gradient(self) -> bool:
    """Returns True if a gradient function is defined."""
    return self._with_gradient

  def jax_params_to_tf_variables(
      self,
      params: PyTree,
      trainable: PyTree,
      pspecs: Optional[PyTree],
      allow_multi_axis_sharding_consolidation: Optional[bool] = None,
  ) -> PyTree:
    """Converts `params` to tf.Variables in the same pytree structure."""
    mesh = dtensor_utils.get_current_mesh()
    default_cpu_device = tf.config.list_logical_devices('CPU')[0]
    if mesh is not None:
      if pspecs is None:
        raise ValueError(
            'DTensor export is enabled but `pspecs` is not specified in'
            ' JaxModule.'
        )
      if not all(
          isinstance(x, jax.Array) for x in jax.tree_util.tree_leaves(params)
      ):
        logging.warning(
            'Some params are not jax.Array, DTensor export will not take'
            ' effect.Falling back to traditional TF export.'
        )
        mesh = None

    if mesh is None and pspecs is not None:
      raise ValueError(
          '`pspecs` is not None but JaxModule is not created within a DTensor'
          ' export context. Please call `initialize_dtensor()` and use `with'
          ' maybe_enable_dtensor_export_on(mesh)` to create a DTensor export'
          ' context.'
      )

    def _to_tf_variable(x, name, trainable, pspec):
      if mesh is not None:
        return dtensor.DVariable(
            dtensor_utils.jax_array_to_dtensor(
                x,
                pspec,
                mesh.dtensor_mesh,
                mesh.jax_mesh,
                allow_multi_axis_sharding_consolidation,
            ),
            trainable=trainable,
            shape=x.shape,
            dtype=x.dtype,
            name=name,
        )

      with tf.device(default_cpu_device):
        return tf.Variable(
            x, trainable=trainable, shape=x.shape, dtype=x.dtype, name=name
        )

    names = export_utils.get_param_names(params)
    if pspecs is None:
      pspecs = jax.tree_util.tree_map(lambda x: None, params)
    return jax.tree_util.tree_map(
        _to_tf_variable, params, names, trainable, pspecs
    )

  def _make_tf_closure(
      self,
      apply_fn: ApplyFn,
      input_polymorphic_shape: Optional[PyTree],
      jax2tf_kwargs: Optional[Mapping[str, Any]],
      jit_compile: bool,
  ) -> Callable[..., Any]:
    """Creates a closure for `apply_fn` in TF context."""
    jax2tf_kwargs = dict(jax2tf_kwargs or {})
    if 'polymorphic_shapes' in jax2tf_kwargs:
      raise ValueError(
          'Do not use `polymorphic_shapes` in `jax2tf_kwargs`, use '
          '`input_polymorphic_shape=...` instead.'
      )

    # All params have static shapes, so the first dimension of
    # polymorphic_shapes is None.
    jax2tf_kwargs['polymorphic_shapes'] = [None, input_polymorphic_shape]

    if 'with_gradient' not in jax2tf_kwargs:
      jax2tf_kwargs['with_gradient'] = self.with_gradient
    elif jax2tf_kwargs['with_gradient'] and not self.with_gradient:
      raise ValueError(
          '`with_gradient=True` is specified in jax2tf_kwargs but '
          'the JaxModule does not contain trainable variables.'
      )
    elif not jax2tf_kwargs['with_gradient'] and self.with_gradient:
      raise ValueError(
          '`with_gradient=False` is specified in jax2tf_kwargs but the '
          'JaxModule contains trainable variables.'
      )

    if logging.vlog_is_on(3):
      logging.vlog(3, 'jax2tf_kwargs=%s', jax2tf_kwargs)

    apply_fn_tf = jax2tf.convert(apply_fn, **jax2tf_kwargs)
    return tf.function(
        lambda x: apply_fn_tf(
            export_utils.get_variable_tree(
                self._tf_var_treedef, self._tf_var_leaves
            ),
            x,
        ),
        jit_compile=jit_compile,
        autograph=False,
    )

  def _get_variable_tree(self) -> PyTree:
    """Returns the PyTree of the tf.Variables associated with self."""
    return jax.tree_util.tree_unflatten(
        self._nontrackable_metadata.tf_var_treedef, self._tf_var_leaves
    )

  @property
  def apply_fn_map(self) -> Mapping[str, ApplyFn]:
    """Returns the apply_fn_map."""
    return self._nontrackable_metadata.apply_fn_map

  @property
  def model_params(self) -> PyTree:
    """Returns the model parameters."""
    return self._nontrackable_metadata.model_params

  @property
  def jax2tf_kwargs_map(self) -> Mapping[str, Any]:
    """Returns the jax2tf_kwargs_map."""
    return self._nontrackable_metadata.jax2tf_kwargs_map

  @property
  def input_polymorphic_shape_map(self) -> Mapping[str, PyTree]:
    """Returns the polymorphic shapes."""
    return self._nontrackable_metadata.input_polymorphic_shape_map

  @property
  def methods(self) -> Mapping[str, Callable[..., Any]]:
    """Named methods in TF context."""
    return self._methods

  @property
  def jax_methods(self) -> Mapping[str, Callable[..., Any]]:
    """Named methods in JAX context for validation."""
    return _make_closures(self.model_params, self.apply_fn_map)

  def update_variables(self, params: PyTree):
    """Updates the variables associated with self.

    Args:
      params: A PyTree of JAX parameters. The PyTree structure must be the same
        as that of the `params` used to initialize the model. Additionally, the
        shape and dtype of each parameter must be the same as the original
        parameter.
    """
    # Update jax model_params
    object.__setattr__(self._nontrackable_metadata, 'model_params', params)

    # Update TF model_params
    _, treedef = jax.tree_util.tree_flatten(params)
    if treedef != self._nontrackable_metadata.tf_var_treedef:
      raise ValueError(
          'The PyTree structure of the updated parameters must be the same as'
          f' that of the original parameters. Got new treedef: {treedef},'
          f' original treedef: {self._nontrackable_metadata.tf_var_treedef}'
      )

    new_vars = self.jax_params_to_tf_variables(
        self._nontrackable_metadata.model_params,
        self._nontrackable_metadata.var_trainable,
        self._nontrackable_metadata.var_pspecs,
        self._nontrackable_metadata.allow_multi_axis_sharding_consolidation,
    )
    jax.tree_util.tree_map(
        lambda v, new_v: v.assign(new_v),
        export_utils.get_variable_tree(
            self._tf_var_treedef,
            self._tf_var_leaves,
        ),
        new_vars,
    )

  def obm_module_to_jax_exported_map(
      self,
      model_inputs: PyTree,
  ) -> Mapping[str, jax_export.Exported]:
    """Convert the orbax.export JaxModule to jax_export.Exported.

    Args:
      model_inputs: The model inputs.

    Returns:
      A mapping from method key to jax_export.Exported.
    """
    apply_fn_map = self.apply_fn_map
    model_params = self.model_params
    input_polymorphic_shape_map = self.input_polymorphic_shape_map
    jax2tf_kwargs_map = self.jax2tf_kwargs_map

    jax_exported_map = {}

    def _symbolic_args_specs(model_inputs, method_key):
      input_polymorphic_shape = input_polymorphic_shape_map[method_key]
      polymorphic_constraints: Sequence[str] = ()
      if 'polymorphic_constraints' in jax2tf_kwargs_map[method_key]:
        polymorphic_constraints = jax2tf_kwargs_map[method_key][
            'polymorphic_constraints'
        ]
      if input_polymorphic_shape is None:
        return model_inputs
      else:
        return jax_export.symbolic_args_specs(
            model_inputs,
            input_polymorphic_shape,
            constraints=polymorphic_constraints,
        )

    symbolic_model_inputs_map = {
        k: _symbolic_args_specs(model_inputs, k)
        for k in input_polymorphic_shape_map.keys()
    }

    def _lowering_platforms(
        jax2tf_kwargs: Any,
    ) -> Optional[Sequence[str]]:
      if jax2tf_kwargs and 'native_serialization_platforms' in jax2tf_kwargs:
        return tuple(jax2tf_kwargs['native_serialization_platforms'])
      else:
        return None

    lowering_platforms_map = {
        k: _lowering_platforms(v) for k, v in jax2tf_kwargs_map.items()
    }

    for method_key, apply_fn in apply_fn_map.items():
      if not hasattr(apply_fn, 'trace'):
        apply_fn = jax.jit(apply_fn)
      if method_key not in input_polymorphic_shape_map:
        raise ValueError(
            f'Method key {method_key} not found in input_polymorphic_shape_map.'
        )
      if method_key not in lowering_platforms_map:
        raise ValueError(
            f'Method key {method_key} not found in lowering_platforms_map.'
        )
      jax_exported = jax_export.export(
          apply_fn, platforms=lowering_platforms_map[method_key]
      )(model_params, symbolic_model_inputs_map[method_key])
      jax_exported_map[method_key] = jax_exported
    return jax_exported_map
