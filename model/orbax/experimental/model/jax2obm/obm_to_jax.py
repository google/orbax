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

"""Utils for converting from OBM back to JAX."""

from typing import Any, List, Sequence, Tuple

import jax
import jax.extend as jex
import numpy as np
from orbax.experimental.model import core as obm
from orbax.experimental.model.jax2obm import jax_supplemental_pb2


def _deserialize_disabled_safety_check(disabled_check):  # pylint: disable=missing-function-docstring
  if disabled_check.HasField("platform"):
    return jax.export.DisabledSafetyCheck.platform()
  elif disabled_check.HasField("custom_call"):
    return jax.export.DisabledSafetyCheck.custom_call(
        disabled_check.custom_call.target_name
    )
  elif disabled_check.HasField("shape_assertions"):
    # shape_assertions has been deprecated in June 2024 (turned into a no-op),
    # and removed in November 2024. We deserialize it to a DisabledSafetyCheck
    # that has no effect.
    # TODO(necula): remove this after June 2025, when we should not have any
    # more serialized artifacts with shape_assertions.
    return jax.export.DisabledSafetyCheck.custom_call("no op")
  else:
    raise ValueError(f"Unknown disabled safety check: {disabled_check}")


_JaxDimSize = int | str | None


def _get_field(proto: Any | None, field_name: str) -> Any | None:
  if proto is None:
    return None
  if proto.HasField(field_name):
    return getattr(proto, field_name)
  else:
    return None


def _refine_dim_size(
    shlo_dim_size: obm.ShloDimSize,
    refinement: str | None,
) -> _JaxDimSize:
  """Refines/converts an OBM dimension size into a JAX dim size.

  Args:
    shlo_dim_size: the OBM dim size.
    refinement: the refinement.

  Returns:
    A JAX dim size.

  Raises:
    ValueError: if the refinement is not valid.
  """
  if refinement is None:
    return shlo_dim_size
  if shlo_dim_size is None:
    try:
      return int(refinement)
    except ValueError:
      return refinement

  def get_exception():
    return ValueError(
        f"The refinement {refinement} is not a valid refinement of the"
        f" ShloDimSize {shlo_dim_size} ."
    )

  try:
    refinement_as_int = int(refinement)
  except ValueError as e:
    raise get_exception() from e
  if refinement_as_int != shlo_dim_size:
    raise get_exception()
  return shlo_dim_size


_JaxShape = Sequence[_JaxDimSize]


def _refine_shape(
    shlo_shape: obm.ShloShape,
    refinement: jax_supplemental_pb2.ShapeRefinement | None,
) -> _JaxShape:
  """Refines/converts an OBM shape into a JAX shape.

  Args:
    shlo_shape: the OBM shape.
    refinement: the refinement.

  Returns:
    A JAX shape.

  Raises:
    ValueError: if the refinement is not valid.
  """
  if shlo_shape is None:
    raise ValueError(
        "Can not convert an unknown-rank shape to JAX because JAX does not"
        " support it."
    )
  if refinement is None:
    return shlo_shape
  dim_size_refinements = refinement.dimension_sizes
  if len(shlo_shape) != len(dim_size_refinements):
    raise ValueError(
        f"`shlo_shape`'s rank {len(shlo_shape)} is not equal to "
        f"`refinement`'s rank {len(dim_size_refinements)} ."
    )
  if not refinement.dimension_sizes:
    return shlo_shape
  return tuple(
      map(
          _refine_dim_size,
          shlo_shape,
          (_get_field(r, "size") for r in dim_size_refinements),
      )
  )


def _refine_dtype(
    shlo_dtype: obm.ShloDType,
    refinement: jax_supplemental_pb2.DTypeRefinement | None,
) -> np.dtype:
  """Refines/converts an OBM dtype into a JAX dtype.

  Args:
    shlo_dtype: the OBM dtype.
    refinement: the refinement.

  Returns:
    A JAX dtype.

  Raises:
    ValueError: if the refinement is not valid.
  """
  if refinement is not None:
    if refinement == jax_supplemental_pb2.DTypeRefinement.f0:
      if shlo_dtype != obm.ShloDType.bool:
        raise ValueError(
            f"The refinement {refinement} is not a valid refinement of the"
            f" ShloDType {shlo_dtype} ."
        )
      return jax.dtypes.float0
    else:
      raise ValueError(f"Unknown DTypeRefinement: {refinement}")
  return obm.shlo_dtype_to_np_dtype(shlo_dtype)


_JaxShapeDtype = Tuple[_JaxShape, np.dtype]


def _refine_tensor_spec(
    shlo_tensor_spec: obm.ShloTensorSpec,
    refinement: jax_supplemental_pb2.ShapeDTypeRefinement | None,
) -> _JaxShapeDtype:
  return (
      _refine_shape(shlo_tensor_spec.shape, _get_field(refinement, "shape")),
      _refine_dtype(shlo_tensor_spec.dtype, _get_field(refinement, "dtype")),
  )


def _get_refinement(
    refinements: jax_supplemental_pb2.ShapeDTypeRefinements | None,
    index: int,
) -> jax_supplemental_pb2.ShapeDTypeRefinement | None:
  if refinements is None:
    return None
  if refinements.HasField("list"):
    return refinements.list.refinements[index]
  elif refinements.HasField("map"):
    return refinements.map.idx_to_refinement.get(index, None)
  else:
    raise ValueError(f"ill-formed ShapeDTypeRefinements: {refinements}")


def _refine_tensor_specs(
    shlo_tensor_specs: Sequence[obm.ShloTensorSpec],
    refinements: jax_supplemental_pb2.ShapeDTypeRefinements | None,
) -> List[_JaxShapeDtype]:
  """Refines/converts a sequence of OBM tensor specs into JAX array specs.

  Args:
    shlo_tensor_specs: the OBM tensor specs.
    refinements: the refinements.

  Returns:
    A list of JAX array specs (i.e. shapes and dtypes).

  Raises:
    ValueError: if the refinements are not valid.
  """
  if (
      refinements is not None
      and refinements.HasField("list")
      and len(refinements.list.refinements) != len(shlo_tensor_specs)
  ):
    raise ValueError(
        f"The number of refinements {len(refinements.list.refinements)} is not"
        f" equal to the number of tensor specs {len(shlo_tensor_specs)} ."
    )
  avals = []
  for i, spec in enumerate(shlo_tensor_specs):
    refinement = _get_refinement(refinements, i)
    avals.append(_refine_tensor_spec(spec, refinement))
  return avals


def _shlo_tensor_spec_to_hlo_sharding(
    shlo_tensor_spec: obm.ShloTensorSpec,
) -> Any | None:
  if shlo_tensor_spec.sharding is None:
    return None
  # `xla_extension.OpSharding` is needed as intermediary, just like the
  # hlo_sharding_to_op_sharding() in
  # orbax/experimental/model/jax2obm/sharding.py.
  return jex.sharding.get_hlo_sharding_from_serialized_proto(
      shlo_tensor_spec.sharding.SerializeToString()
  )


def _restore_spec(
    manifest_signature: obm.type_pb2.FunctionSignature,
    signature_field: str,
    supplemental: jax_supplemental_pb2.Function,
    supplemental_field: str,
) -> Tuple[
    jax.tree_util.PyTreeDef,
    List[_JaxShapeDtype],
    Tuple[Any | None, ...],
]:
  """Restores an input or output signature (and its refinement) into JAX.

  Args:
    manifest_signature: an OBM manifest `FunctionSignature` proto.
    signature_field: the name of a field in the above proto. This should be
      either `"input"` or `"output"`.
    supplemental: a JAX-supplemental `Function` proto.
    supplemental_field: the name of a field in the above proto. This should be
      either `"input_spec_refinements"` or `"output_spec_refinements"`.

  Returns:
    A tuple containing the tree definition, array specs, and shardings of the 
    restored JAX input or output signature.

  Raises:
    ValueError: if the field is not set in the manifest signature.
  """
  if not manifest_signature.HasField(signature_field):
    raise ValueError(
        f"Field `{signature_field}` is not set in `manifest_signature`."
    )
  allowed_supplemental_fields = (
      "input_spec_refinements",
      "output_spec_refinements",
  )
  if supplemental_field not in allowed_supplemental_fields:
    raise ValueError(
        f"Field `{supplemental_field}` is not a valid supplemental field."
        f" Allowed values are: {allowed_supplemental_fields}."
    )
  shlo_tensor_spec_pytree = obm.manifest_type_to_shlo_tensor_spec_pytree(
      getattr(manifest_signature, signature_field)
  )
  shlo_tensor_spec_list, treedef = jax.tree.flatten(shlo_tensor_spec_pytree)
  if not supplemental.HasField(supplemental_field):
    refinements = None
  else:
    refinements = getattr(supplemental, supplemental_field)
  avals = _refine_tensor_specs(shlo_tensor_spec_list, refinements)
  shardings = tuple(
      map(_shlo_tensor_spec_to_hlo_sharding, shlo_tensor_spec_list)
  )
  return treedef, avals, shardings


def _to_jax_shape(
    shape: _JaxShape,
    scope: jax.export.SymbolicScope,
) -> Any:
  """Converts a shape-dtype pair to a JAX (possibly symbolic) shape.

  Args:
    shape: a shape-dtype pair.
    scope: a JAX symbolic-shape scope.

  Returns:
    A JAX (possibly symbolic) shape.

  Raises:
    ValueError: if the shape contains `None` dimensions.
  """
  if any(x is None for x in shape):
    raise ValueError(
        "jax.core.ShapedArray does not allow `None` dimensions."
        f" Got shape: {shape}"
    )
  if any(isinstance(x, str) for x in shape):
    return jax.export.symbolic_shape(
        ",".join(str(x) for x in shape), scope=scope
    )
  return shape


def _to_jax_shaped_array(
    shape_dtype: _JaxShapeDtype,
    scope: jax.export.SymbolicScope,
) -> jax.core.ShapedArray:
  shape, dtype = shape_dtype
  return jax.core.ShapedArray(_to_jax_shape(shape, scope), dtype)


def obm_functions_to_jax_function(
    manifest_function: obm.manifest_pb2.Function,
    jax_supplemental_function: jax_supplemental_pb2.Function,
) -> jax.export.Exported:
  """Converts an OBM function and its supplemental to a JAX `Exported`.

  Note that Jax `effects` only for things like debug printing and debug
  assertions and should be turned off when model exporting.

  Args:
    manifest_function: an OBM manifest `Function` proto.
    jax_supplemental_function: a JAX-supplemental `Function` proto. It should be
      the supplemental of `manifest_function`.

  Returns:
    A JAX `Exported` object.
  """
  in_treedef, in_avals, in_shardings_hlo = _restore_spec(
      manifest_function.signature,
      "input",
      jax_supplemental_function,
      "input_spec_refinements",
  )
  out_treedef, out_avals, out_shardings_hlo = _restore_spec(
      manifest_function.signature,
      "output",
      jax_supplemental_function,
      "output_spec_refinements",
  )

  # TODO(qidichen): investigate if jax effects are really needed,
  # otherwise we need to request visibility from jax._src
  # 

  # TODO(qidichen): Check that `manifest_function.body` is in the
  # `stable_hlo_body` case.

  # Uses the same symbolic-shape scope setup as in `jax.export.deserialize`.
  scope = jax.export.SymbolicScope(())

  exported = jax.export.Exported(
      fun_name=jax_supplemental_function.name,
      in_tree=in_treedef,
      in_avals=tuple(_to_jax_shaped_array(a, scope) for a in in_avals),
      out_tree=out_treedef,
      out_avals=tuple(_to_jax_shaped_array(a, scope) for a in out_avals),
      in_shardings_hlo=in_shardings_hlo,
      out_shardings_hlo=out_shardings_hlo,
      nr_devices=jax_supplemental_function.nr_devices,
      # According to the Jax team, most effects are only for things like debug
      # printing and debug assertions and should be turned off when model
      # exporting.
      ordered_effects=tuple(),
      unordered_effects=tuple(),
      disabled_safety_checks=tuple(
          map(
              _deserialize_disabled_safety_check,
              jax_supplemental_function.disabled_checks,
          )
      ),
      mlir_module_serialized=(
          manifest_function.body.stable_hlo_body.stable_hlo.inlined_bytes
      ),
      calling_convention_version=(
          manifest_function.body.stable_hlo_body.calling_convention_version
      ),
      module_kept_var_idx=tuple(
          manifest_function.body.stable_hlo_body.module_kept_var_idx
      ),
      platforms=tuple(
          manifest_function.body.stable_hlo_body.lowering_platforms
      ),
      uses_global_constants=jax_supplemental_function.uses_shape_polymorphism,
      _get_vjp=None,
  )

  return exported
