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

import dataclasses
import os
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from absl import logging
import jax
from jax.experimental import jax2tf
from orbax.export import dtensor_utils
from orbax.export import typing as orbax_export_typing
import tensorflow as tf
from tensorflow.experimental import dtensor


PyTree = orbax_export_typing.PyTree
ApplyFn = orbax_export_typing.ApplyFn


def get_obx_export_tf_preprocess_only() -> bool:
  """Returns whether the export is in TF preprocess only mode."""
  # If it is True, the export will only export the
  # servering_config.tf_preprocess instead of the whole model. This mode is
  # majorly used for debugging.
  obx_export_tf_preprocess_only = (
      os.getenv('OBX_EXPORT_TF_PREPROCESS_ONLY') == 'True'
  )
  return obx_export_tf_preprocess_only


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
  allow_multi_axis_sharding_conslidation: Optional[bool]


class JaxModule(tf.Module):
  """An exportable module for JAX functions and parameters.

  Holds tf.Variables converted from JAX parameters, as well as TF functions
  converted from JAX functions and bound with the tf.Variables.
  """

  DEFAULT_METHOD_KEY = 'jax_module_default_method'

  def __init__(
      self,
      params: PyTree,
      apply_fn: Union[ApplyFn, Mapping[str, ApplyFn]],
      trainable: Optional[Union[bool, PyTree]] = None,
      input_polymorphic_shape: Union[PyTree, Mapping[str, PyTree], None] = None,
      jax2tf_kwargs: Optional[Mapping[str, Any]] = None,
      jit_compile: Union[bool, Mapping[str, bool]] = True,
      pspecs: Optional[PyTree] = None,
      allow_multi_axis_sharding_conslidation: Optional[bool] = None,
  ):
    """JaxModule constructor.

    Args:
      params: a pytree of JAX parameters.
      apply_fn: A JAX ``ApplyFn`` (i.e. of signature ``apply_fn(params, x)``),
        or a mapping of method key to ``ApplyFn``. If it is an ``ApplyFn``, it
        will be assigned a key ``JaxModule.DEFAULT_METHOD_KEY`` automatically,
        which can be used to look up the TF function converted from it.
      trainable: a pytree in the same structure as ``params`` and boolean leaves
        to tell if a parameter is trainable. Alternatively, it can be a single
        boolean value to tell if all the parameters are trainable or not. By
        default all parameters are non-trainable. The default value is subject
        to change in the future, thus it is recommended to specify the value
        explicitly.
      input_polymorphic_shape: the polymorhpic shape for the inputs of
        ``apply_fn``. If ``apply_fn`` is a mapping, ``input_polymorphic_shape``
        must be a mapping of method key to the input polymorphic shape for the
        method.
      jax2tf_kwargs: options passed to jax2tf. ``polymorphic_shape`` is inferred
        from ``input_polymorphic_shape`` and should not be set.
        ``with_gradient``, if set, should be consistent with the ``trainable``
        argument above. If ``jax2tf_kwargs`` is unspecified, the default jax2tf
        option will be applied. If ``apply_fn`` is a mapping, `jax2tf_kwargs`
        must either be unspecified or a mapping of method key to the jax2tf
        kwargs for the method.
      jit_compile: whether to jit compile the jax2tf converted functions. If
        ``apply_fn`` is a mapping, this can either be a boolean applied to all
        functions or a mapping of method key to the jit compile option for the
        method.
      pspecs: an optional pytree of PartitionSpecs of the ``params`` in the same
        structure as ``params``. If set, the leaves of ``params`` must be
        jax.Array, and ``JaxModule`` must be created within a DTensor export
        context from ``with maybe_enable_dtensor_export_on(mesh)``.
      allow_multi_axis_sharding_conslidation: Disallowed by default. When set to
        true, it will allow conslidating JAX array multiple axis sharding into
        DTensor single axis sharding during checkpoint conversion. This would
        enable sharding across multiple axis names support for JAX model.
    """
    if callable(apply_fn):
      apply_fn_map: dict[str, ApplyFn] = {self.DEFAULT_METHOD_KEY: apply_fn}
      input_polymorphic_shape = {
          self.DEFAULT_METHOD_KEY: input_polymorphic_shape
      }
      jax2tf_kwargs = {self.DEFAULT_METHOD_KEY: jax2tf_kwargs}
      jit_compile = {self.DEFAULT_METHOD_KEY: jit_compile}
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

    self.with_gradient: bool = any(jax.tree_util.tree_leaves(trainable))

    if get_obx_export_tf_preprocess_only():
      # Skip the heavy jax_params_to_tf_variables() call in TF preprocess only
      # mode.
      tf_var_treedef = None
      self._tf_var_leaves = None
      self._methods = dict()
    else:
      tf_vars = _jax_params_to_tf_variables(
          params, trainable, pspecs, allow_multi_axis_sharding_conslidation
      )
      # Do not attach `tf_vars` to `self` directly, otherwise its structure will
      # be mutated by `tf.Module.__setattr__`.
      self._tf_var_leaves, tf_var_treedef = jax.tree_util.tree_flatten(tf_vars)
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
        tf_var_treedef=tf_var_treedef,
        var_trainable=trainable,
        var_pspecs=pspecs,
        model_params=params,
        jax2tf_kwargs_map=jax2tf_kwargs,
        input_polymorphic_shape_map=input_polymorphic_shape,
        allow_multi_axis_sharding_conslidation=allow_multi_axis_sharding_conslidation,
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
    new_vars = _jax_params_to_tf_variables(
        self._nontrackable_metadata.model_params,
        self._nontrackable_metadata.var_trainable,
        self._nontrackable_metadata.var_pspecs,
        self._nontrackable_metadata.allow_multi_axis_sharding_conslidation,
    )
    jax.tree_util.tree_map(
        lambda v, new_v: v.assign(new_v), self._get_variable_tree(), new_vars
    )

  def _get_variable_tree(self) -> PyTree:
    """Returns the PyTree of the tf.Variables associated with self."""
    return jax.tree_util.tree_unflatten(
        self._nontrackable_metadata.tf_var_treedef, self._tf_var_leaves
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
        lambda x: apply_fn_tf(self._get_variable_tree(), x),
        jit_compile=jit_compile,
        autograph=False,
    )

  @property
  def methods(self) -> Mapping[str, Callable[..., Any]]:
    """Named methods in TF context."""
    return self._methods

  @property
  def jax_methods(self) -> Mapping[str, Callable[..., Any]]:
    """Named methods in JAX context for validation."""
    params = self._nontrackable_metadata.model_params
    apply_fn_map = self._nontrackable_metadata.apply_fn_map
    return _make_closures(params, apply_fn_map)


def _get_key_name(key: Any) -> Union[int, str]:
  """Returns the name of a JAX Key."""
  if isinstance(key, jax.tree_util.SequenceKey):
    return key.idx
  elif isinstance(key, jax.tree_util.DictKey):
    return str(key.key)
  elif isinstance(key, jax.tree_util.GetAttrKey):
    return key.name
  elif isinstance(key, jax.tree_util.FlattenedIndexKey):
    return key.key
  else:
    raise ValueError(f'Unsupported KeyEntry: {type(key)}: "{key}"')


def _get_param_names(params: PyTree) -> PyTree:
  """Gets parameter names for PyTree elements."""

  def _param_name_from_keypath(keypath: Tuple[Any, ...]) -> str:
    name = '.'.join([str(_get_key_name(k)) for k in keypath])
    # '~' is not allowed in variable names but are used by dm-haiku. See
    # https://github.com/google/orbax/issues/420
    return name.replace('~', '_')

  names = jax.tree_util.tree_map_with_path(
      lambda kp, _: _param_name_from_keypath(kp), params
  )

  if jax.tree_util.tree_structure(params) != jax.tree_util.tree_structure(
      names
  ):
    logging.warning(
        (
            'Cannot construct variable names for JAX parameters, which means'
            ' the parameters tree contains customized nodes not registered with'
            ' ``jax.tree_util.register_pytree_with_keys``. Variables will be'
            ' named to `jax_param_<index>` instead. PyTreeDef of params=%s.'
        ),
        jax.tree_util.tree_structure(params),
    )
    flat_params, tree_def = jax.tree_util.tree_flatten(params)
    names = jax.tree_util.tree_unflatten(
        tree_def, [f'jax_param_{i}' for i in range(len(flat_params))]
    )
  return names


def _jax_params_to_tf_variables(
    params: PyTree,
    trainable: PyTree,
    pspecs: Optional[PyTree],
    allow_multi_axis_sharding_conslidation: Optional[bool] = None,
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
          'Some params are not jax.Array, DTensor export will not take effect.'
          'Falling back to traditional TF export.'
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
              allow_multi_axis_sharding_conslidation,
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

  names = _get_param_names(params)
  if pspecs is None:
    pspecs = jax.tree_util.tree_map(lambda x: None, params)
  return jax.tree_util.tree_map(
      _to_tf_variable, params, names, trainable, pspecs
  )
