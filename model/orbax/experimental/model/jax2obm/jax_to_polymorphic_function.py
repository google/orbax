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

"""Converts JAX function to obm.PolymorphicFunction."""

# pylint: disable=protected-access
from typing import Callable, Dict, List, OrderedDict, Sequence, Tuple, TypeVar, Union

from absl import logging
import jax
from jax import export as jax_export
import jax.numpy as jnp
import numpy as np
from orbax.experimental.model import core as obm
from orbax.experimental.model.jax2obm import constants
from orbax.experimental.model.jax2obm import jax_specific_info
from orbax.experimental.model.jax2obm import utils


def convert_to_tensor(x: jnp.ndarray) -> obm.Tensor:
  """Converts a JAX array to an EM tensor."""
  return obm.Tensor(np.array(x))


def convert_to_variable(x: jnp.ndarray) -> obm.Variable:
  """Converts a JAX array to an EM variable."""
  return obm.Variable(convert_to_tensor(x))


EmVal = obm.tracing.SymbolicTensor


EmValPyTree = TypeVar(
    "EmValPyTree",
    bound=Union[
        EmVal,
        Tuple["EmValPyTree", ...],
        List["EmValPyTree"],
        Dict[str, "EmValPyTree"],
        # For ordered dictionaries
        OrderedDict[str, "EmValPyTree"],
        Tuple[Tuple[str, "EmValPyTree"], ...],
        List[Tuple[str, "EmValPyTree"]],
    ],
)


# TODO(b/332755537): Support argument `with_gradient: bool` in `convert`.
def convert_polymorphic_fn(
    fun_jax: Callable[..., utils.JaxArrayPyTree],
    *,
    polymorphic_shapes: str | None = None,
    polymorphic_constraints: Sequence[str] = (),
    native_serialization_platforms: (
        Sequence[constants.OrbaxNativeSerializationType] | None
    ) = None,
    native_serialization_disabled_checks: Sequence[
        jax_export.DisabledSafetyCheck
    ] = (),
) -> Callable[..., EmValPyTree]:
  """Converts a JAX function to an EM function.

  Args:
    fun_jax: target JAX function to be called. Its arguments and return value
      should be JAX arrays, or nested standard Python containers
      (tuple/list/dict) thereof (pytrees).
    polymorphic_shapes: Specifies input shapes to be treated polymorphically
      during lowering.  .. warning:: The shape-polymorphic lowering is an
      experimental feature. It is meant to be sound, but it is known to reject
      some JAX programs that are shape polymorphic. The details of this feature
      can change.  It should be `None` (all arguments are monomorphic), a single
      PolyShape or string (applies to all arguments), or a tuple/list of the
      same length as the function arguments. For each argument the shape
      specification should be `None` (monomorphic argument), or a Python object
      with the same pytree structure as the argument. See [how optional
      parameters are matched to
      arguments](https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees).
      A shape specification for an array argument should be an object
      `PolyShape(dim0, dim1, ..., dimn)` where each `dim` is a dimension.
    polymorphic_constraints: a sequence of constraints on symbolic dimension
      expressions, of the form `e1 >= e2` or `e1 <= e2`.
    native_serialization_platforms: Optional. Specifies the platform(s) for
      which to lower the code. Must be a tuple of
      enums of type `OrbaxNativeSerializationType'. If not set,
      the JAX default backend on the machine where the lowering is
      done will be used.
    native_serialization_disabled_checks: In conjunction with
      `native_serialization`, disable the specified safety checks. See docstring
      of `DisabledSafetyCheck`.

  Returns:
    A version of `fun_jax` that expects EmVals as arguments (or
    tuple/lists/dicts thereof), and returns EmVals as outputs, and uses
    only EM ops and thus can be called from an EM program.
  """
  create_exported = utils.make_jax_exported_creator(
      fun_jax,
      native_serialization_platforms,
      native_serialization_disabled_checks,
  )

  @obm.tracing.add_variable_support
  def converted_fn(
      *args_em: obm.tracing.TreeOfSymbolicTensors,
      **kwargs_em: obm.tracing.TreeOfSymbolicTensors,
  ) -> Tuple[obm.tracing.TreeOfGraphEdgeTensors, ...]:
    obm.tracing.assert_tracing()

    utils.assert_not_in_jax_transformation()

    def em_to_jax_arg_spec(a: EmVal) -> jax.ShapeDtypeStruct:
      # The shape and JAX dtype for an EM argument
      em_arg_shape = a.shape
      a_jax_dtype = _to_jax_dtype(a.dtype)  # Give JAX a chance to pick the type
      # We count on the fact that jax.ShapeDtypeStruct allows shapes that
      # contain None.
      return jax.ShapeDtypeStruct(em_arg_shape, a_jax_dtype)

    args_jax_specs = jax.tree_util.tree_map(em_to_jax_arg_spec, args_em)
    args_specs = jax_export.symbolic_args_specs(
        args_jax_specs,
        shapes_specs=polymorphic_shapes,
        constraints=polymorphic_constraints,
    )
    # The polymorphic_shapes argument refers to positional arguments only.
    # We assume None for the kwargs.
    kwargs_jax_specs = jax.tree_util.tree_map(em_to_jax_arg_spec, kwargs_em)
    kwargs_specs = jax_export.symbolic_args_specs(
        kwargs_jax_specs, shapes_specs=None
    )
    combined_args_em = (args_em, kwargs_em)
    args_flat_em, _ = jax.tree_util.tree_flatten(combined_args_em)
    args_flat_em: Sequence[EmVal]

    exported = create_exported(*args_specs, **kwargs_specs)

    # TODO(b/332755362): Call _run_exported_as_em with combined_args_em so that
    # the generated obm.ConcreteFunction can have structured input signature.
    outs_flat_em = _run_exported_as_em(exported, args_flat_em)
    outs_tree: jax.tree_util.PyTreeDef = exported.out_tree
    return jax.tree_util.tree_unflatten(outs_tree, outs_flat_em)

  return converted_fn


JaxDimSize = int | str | None
JaxShape = Tuple[JaxDimSize, ...]


def _aval_shape(a) -> JaxShape:
  assert isinstance(a, jax.core.ShapedArray)
  return a.shape


def _run_exported_as_em(
    exported: jax_export.Exported,
    args_flat_em: Sequence[EmVal],
) -> Sequence[obm.tracing.TreeOfGraphEdgeTensors]:
  """Runs an exported JAX function as an EM function.

  Args:
    exported: The exported JAX function information (e.g., input/output shapes,
      MLIR module).
    args_flat_em: The flattened input arguments to the exported function. Each
      `EmVal` corresponds to an element in `exported.in_avals`.

  Returns:
    A sequence of EM tensors representing the corresponding flattened output of
    the executed function.
  """
  args_avals = exported.in_avals

  def _check_shape(val, aval):
    # Check the shape
    assert all(
        d_aval == d_val
        for d_aval, d_val in zip(aval.shape, val.shape)
        if jax.core.is_constant_dim(d_aval)
    ), (aval, val)

  map(_check_shape, args_flat_em, args_avals)

  out_shapes_em = tuple(
      tuple(
          d if jax.core.is_constant_dim(d) else None
          for d in _aval_shape(out_aval)
      )
      for out_aval in exported.out_avals
  )

  out_types = tuple(
      _to_em_dtype(utils._aval_dtype(out_aval))
      for out_aval in exported.out_avals
  )

  shlo_in_sig, jax_in_sig_refinements = (
      jax_specific_info._to_shlo_spec_tree_and_refinement_tuple(
          exported.in_avals, exported.in_shardings_hlo, exported.in_tree
      )
  )
  shlo_out_sig, jax_out_sig_refinements = (
      jax_specific_info._to_shlo_spec_tree_and_refinement_tuple(
          exported.out_avals, exported.out_shardings_hlo, exported.out_tree
      )
  )

  jax_specific_info_ = jax_specific_info.JaxSpecificInfo(
      name=exported.fun_name,
      input_spec_refinements=jax_in_sig_refinements,
      output_spec_refinements=jax_out_sig_refinements,
      nr_devices=exported.nr_devices,
      ordered_effects=exported.ordered_effects,
      unordered_effects=exported.unordered_effects,
      disabled_safety_checks=tuple(exported.disabled_safety_checks),
      uses_shape_polymorphism=exported.uses_global_constants,
  )
  supplemental_info_ = {obm.JAX_SPECIFIC_INFO: jax_specific_info_}

  em_fn = obm.ConcreteFunction(
      input_signature=tuple(
          map(
              lambda tensor, sharding: tensor.spec.with_sharding(sharding),
              args_flat_em,
              exported.in_shardings_hlo,
          )
      ),
      output_signature=tuple(
          map(
              obm.TensorSpec,
              out_shapes_em,
              out_types,
              exported.out_shardings_hlo,
          )
      ),
      base_fn=obm.ShloFunction(
          input_signature=shlo_in_sig,
          output_signature=shlo_out_sig,
          mlir_module_serialized=exported.mlir_module_serialized,
          calling_convention_version=exported.calling_convention_version,
          module_kept_var_idx=exported.module_kept_var_idx,
          lowering_platforms=exported.platforms,
          supplemental_info=supplemental_info_,
          physical_in_dtypes=tuple(
              utils._get_physical_dtype(utils._aval_dtype(v))
              for v in exported.in_avals
          ),
          physical_out_dtypes=tuple(
              utils._get_physical_dtype(utils._aval_dtype(v))
              for v in exported.out_avals
          ),
      ),
  )

  if logging.vlog_is_on(3):
    # We already logged the MLIR module when we exported it.
    logging.vlog(3, "obm.ConcreteFunction %s", str(em_fn))

  return obm.tracing.add_node(input_=args_flat_em, op=em_fn)


# In the EM world, we represent float0 as zeros of this type.
# We pick bool because this is what JAX uses when it lowers float0 to HLO.
_em_np_dtype_for_float0 = np.bool_


def _to_em_dtype(jax_dtype):
  # Note that converting _to_em_dtype and _to_jax_dtype are not inverses,
  # due to float0 and 64-bit behavior.
  try:
    jax_dtype = utils._jax_physical_dtype(jax_dtype)
  except TypeError:
    # `jax_dtype` isn't actually a valid jax dtype (e.g. it is
    # obm.float32), so there is no physical dtype anyway
    pass
  if jax_dtype == jax.float0:
    jax_dtype = _em_np_dtype_for_float0
  return obm.np_dtype_to_dtype(np.dtype(jax_dtype))


def _to_jax_dtype(em_dtype):
  # Note that converting _to_em_dtype and _to_jax_dtype are not inverses,
  # due to float0 and 64-bit behavior.
  dt = jax.dtypes.canonicalize_dtype(obm.dtype_to_np_dtype(em_dtype))
  return dt
