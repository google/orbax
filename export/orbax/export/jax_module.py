# Copyright 2023 The Orbax Authors.
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
from typing import Any, Callable, Optional, Union, Mapping, Tuple

from absl import logging

import jax
from jax.experimental import jax2tf
from orbax.checkpoint import utils as ckpt_utils
from orbax.export import dtensor_utils
import tensorflow as tf
from tensorflow.experimental import dtensor

PyTree = Any
ApplyFn = Callable[[PyTree, PyTree], PyTree]


def _same_keys(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
  return set(a.keys()) == set(b.keys())


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
      input_polymorphic_shape: Union[PyTree, Mapping[str, PyTree]] = None,
      jax2tf_kwargs: Optional[Mapping[str, Any]] = None,
      jit_compile: Union[bool, Mapping[str, bool]] = True,
      name: Optional[str] = None,
      pspecs: Optional[PyTree] = None,
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
        explictly.
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
      name: the name of the module.
      pspecs: an optional pytree of PartitionSpecs of the ``params`` in the
        same structure as ``params``. If set, the leaves of ``params`` must be
        jax.Array, and ``JaxModule`` must be created within a DTensor export
        context from ``with maybe_enable_dtensor_export_on(mesh)``.
    """
    if callable(apply_fn):
      apply_fn: dict[str, ApplyFn] = {self.DEFAULT_METHOD_KEY: apply_fn}
      input_polymorphic_shape = {
          self.DEFAULT_METHOD_KEY: input_polymorphic_shape
      }
      jax2tf_kwargs = {self.DEFAULT_METHOD_KEY: jax2tf_kwargs}
      jit_compile = {self.DEFAULT_METHOD_KEY: jit_compile}
    else:
      # Check if `apply_fn`, `input_polymorphic_shape` and `jax2tf_kwargs` have
      # the same structure.
      if not isinstance(input_polymorphic_shape, Mapping) or not _same_keys(
          input_polymorphic_shape, apply_fn):
        raise ValueError(
            '`input_polymorphic_shape` must have the same structure as that of '
            f'`apply_fn`. Got apply_fn={apply_fn}, input_polymorphic_shape='
            f'{input_polymorphic_shape}.')
      if jax2tf_kwargs is None:
        # OK if it is unspecified, which means `jax2tf_kwargs` is unspecified
        # for all apply functions.
        jax2tf_kwargs = jax.tree_util.tree_map(lambda x: None, apply_fn)
      elif not _same_keys(jax2tf_kwargs, apply_fn):
        raise ValueError(
            '`jax2tf_kwargs` must either be unspecified or have the same '
            f'structure as that of `apply_fn`. Got apply_fn={apply_fn}, '
            f'jax2tf_kwargs={jax2tf_kwargs}.')
      if not isinstance(jit_compile, Mapping) or isinstance(jit_compile, bool):
        jit_compile = jax.tree_util.tree_map(lambda x: jit_compile, apply_fn)
      elif not _same_keys(jit_compile, apply_fn):
        raise ValueError(
            '`jit_compile` must either be a boolean or have the same '
            f'structure as that of `apply_fn`. Got apply_fn={apply_fn}, '
            f'jit_compile={jit_compile}.')

    if trainable is None:
      trainable = False
    if isinstance(trainable, bool):
      trainable = jax.tree_util.tree_map(lambda x: trainable, params)

    self.with_gradient: bool = any(jax.tree_util.tree_leaves(trainable))
    tf_vars = _jax_params_to_tf_variables(params, trainable, pspecs)
    # Do not attach `tf_vars` to `self` directly, otherwise its structure will
    # be mutated by `tf.Module.__setattr__`.
    self._get_variable_tree = lambda: tf_vars
    self._tf_vars = jax.tree_util.tree_leaves(tf_vars)
    self._methods = jax.tree_util.tree_map(self._make_tf_closure, apply_fn,
                                           input_polymorphic_shape,
                                           jax2tf_kwargs, jit_compile)

    def bind_params(fn: ApplyFn):
      return lambda x: fn(params, x)

    self._jax_methods = jax.tree_util.tree_map(bind_params, apply_fn)

  def _make_tf_closure(self, apply_fn: ApplyFn,
                       input_polymorphic_shape: Optional[PyTree],
                       jax2tf_kwargs: Optional[Mapping[str, Any]],
                       jit_compile: bool) -> Callable[..., Any]:
    """Creates a closure for `apply_fn` in TF context."""
    jax2tf_kwargs = dict(jax2tf_kwargs or {})
    if 'polymorphic_shapes' in jax2tf_kwargs:
      raise ValueError(
          'Do not use `polymorphic_shapes` in `jax2tf_kwargs`, use '
          '`input_polymorphic_shape=...` instead.')

    # All params have static shapes, so the first dimension of
    # polymorphic_shapes is None.
    jax2tf_kwargs['polymorphic_shapes'] = [None, input_polymorphic_shape]

    if 'with_gradient' not in jax2tf_kwargs:
      jax2tf_kwargs['with_gradient'] = self.with_gradient
    elif jax2tf_kwargs['with_gradient'] and not self.with_gradient:
      raise ValueError('`with_gradient=True` is specified in jax2tf_kwargs but '
                       'the JaxModule does not contain trainable variables.')
    elif not jax2tf_kwargs['with_gradient'] and self.with_gradient:
      raise ValueError(
          '`with_gradient=False` is specified in jax2tf_kwargs but the '
          'JaxModule contains trainable variables.')

    apply_fn_tf = jax2tf.convert(apply_fn, **jax2tf_kwargs)
    return tf.function(
        lambda x: apply_fn_tf(self._get_variable_tree(), x),
        jit_compile=jit_compile,
        autograph=False)

  @property
  def methods(self) -> Mapping[str, Callable[..., Any]]:
    """Named methods in TF context."""
    return self._methods

  @property
  def jax_methods(self) -> Mapping[str, Callable[..., Any]]:
    """Named methods in JAX context for validation."""
    return self._jax_methods


def _get_param_names(params: PyTree) -> PyTree:
  """Gets parameter names for PyTree elements."""

  def _param_name_from_keypath(keypath: Tuple[Any, ...]) -> str:
    name = '.'.join([str(ckpt_utils.get_key_name(k)) for k in keypath])
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
    params: PyTree, trainable: PyTree, pspecs: Optional[PyTree]
) -> PyTree:
  """Converts `params` to tf.Variables in the same pytree structure."""
  dmesh = dtensor_utils.get_current_dtensor_mesh()
  if dmesh is not None:
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
      dmesh = None

  if dmesh is None and pspecs is not None:
    raise ValueError(
        '`pspecs` is not None but JaxModule is not created within a DTensor'
        ' export context. Please call `initialize_dtensor()` and use `with'
        ' maybe_enable_dtensor_export_on(mesh)` to create a DTensor export'
        ' context.'
    )

  def _to_tf_variable(x, name, trainable, pspec):
    if dmesh:
      return dtensor.DVariable(
          dtensor_utils.jax_array_to_dtensor(x, pspec, dmesh),
          trainable=trainable,
          shape=x.shape,
          dtype=x.dtype,
          name=name,
      )

    return tf.Variable(
        x, trainable=trainable, shape=x.shape, dtype=x.dtype, name=name
    )
  names = _get_param_names(params)
  if pspecs is None:
    pspecs = jax.tree_util.tree_map(lambda x: None, params)
  return jax.tree_util.tree_map(
      _to_tf_variable, params, names, trainable, pspecs
  )
