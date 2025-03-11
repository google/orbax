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

"""A converter from JAX to ML Exported Model (EM)."""

# pylint: disable=protected-access
from typing import Any, Callable, Dict, Sequence

import jax
from jax import export as jax_export
import numpy as np
import orbax.checkpoint as ocp
from orbax.experimental.model import core as obm
from orbax.experimental.model.jax2obm import constants
from orbax.experimental.model.jax2obm import jax_specific_info
from orbax.experimental.model.jax2obm import sharding
from orbax.experimental.model.jax2obm import utils


def to_shape_dtype_struct(x):
  return jax.ShapeDtypeStruct(x.shape, x.dtype)


def get_shape_dtype_struct(jax_pytree):
  return jax.tree_util.tree_map(to_shape_dtype_struct, jax_pytree)


def jax_exported_to_shlo_fn(
    exported: jax_export.Exported,
    flatten_signature: bool = False,
) -> obm.ShloFunction:
  """Converts a `jax.export.Exported` to an Orbax Model `ShloFunction`."""

  in_shardings_hlo = tuple([
      sharding.hlo_sharding_to_op_sharding(sd)
      for sd in exported.in_shardings_hlo
  ])
  out_shardings_hlo = tuple([
      sharding.hlo_sharding_to_op_sharding(sd)
      for sd in exported.out_shardings_hlo
  ])
  shlo_in_sig, jax_in_sig_refinements = (
      jax_specific_info._to_shlo_spec_tree_and_refinement_tuple(
          exported.in_avals,
          in_shardings_hlo,
          exported.in_tree if not flatten_signature else None,
      )
  )
  shlo_out_sig, jax_out_sig_refinements = (
      jax_specific_info._to_shlo_spec_tree_and_refinement_tuple(
          exported.out_avals,
          out_shardings_hlo,
          exported.out_tree if not flatten_signature else None,
      )
  )

  shlo_func = obm.ShloFunction(
      input_signature=shlo_in_sig,
      output_signature=shlo_out_sig,
      mlir_module_serialized=exported.mlir_module_serialized,
      calling_convention_version=exported.calling_convention_version,
      module_kept_var_idx=exported.module_kept_var_idx,
      lowering_platforms=exported.platforms,
      supplemental_info=jax_specific_info.JaxSpecificInfo(
          name=exported.fun_name,
          input_spec_refinements=jax_in_sig_refinements,
          output_spec_refinements=jax_out_sig_refinements,
          nr_devices=exported.nr_devices,
          ordered_effects=exported.ordered_effects,
          unordered_effects=exported.unordered_effects,
          disabled_safety_checks=tuple(exported.disabled_safety_checks),
          uses_shape_polymorphism=exported.uses_global_constants,
      ),
      physical_in_dtypes=tuple(
          utils._get_physical_dtype(utils._aval_dtype(v))
          for v in exported.in_avals
      ),
      physical_out_dtypes=tuple(
          utils._get_physical_dtype(utils._aval_dtype(v))
          for v in exported.out_avals
      ),
  )
  return shlo_func


# TODO(wangpeng): Rename to `convert_to_obm_fn` or `convert_with_spec` or
#   `convert_with_signature`.
def convert_to_shlo_fn(
    fun_jax: Callable[..., utils.JaxArrayPyTree],
    args_spec: Sequence[Any],
    kwargs_spec: Dict[str, Any],
    native_serialization_platforms: (
        Sequence[constants.OrbaxNativeSerializationType] | None
    ) = None,
    native_serialization_disabled_checks: Sequence[
        jax_export.DisabledSafetyCheck
    ] = (),
    flatten_signature: bool = False,
) -> obm.ShloFunction:
  """Converts a JAX function to an Orbax Model `ShloFunction`.

  For example usage of `convert_to_shlo_fn`, please refer to the
  main_lib_test.py file.

  Args:
    fun_jax: The JAX function to be converted. It should be a jitted function.
      If it's not, we will jit it by simply `jax.jit(fun_jax)`, which means
      `fun_jax` can not contain any sharding annotations that are not
      "FullyReplicated".
    args_spec: A sequence of pytrees of {class}`jax.ShapeDtypeStruct`, or values
      with `.shape` and `.dtype` attributes, or result of calling
      `jax_export.symbolic_args_specs()`. These will be used to trace the
      function in jax export. As a result, the ordering of params and input
      within the specs here should match the jax function that user provides.
    kwargs_spec: A dictionary of keyword arguments to pytrees of
      {class}`jax.ShapeDtypeStruct`, or values with `.shape` and `.dtype`
      attributes, or result of calling `jax_export.symbolic_args_specs()`. These
      will be used to trace the function in jax export.
    native_serialization_platforms: Optional. Specifies the platform(s) for which to lower
      the code. Must be a tuple of OrbaxNativeSerializationType. If not set,
      the JAX default backend will be used. Example can be found in

    native_serialization_disabled_checks: A sequence of safety checks to
      disable. Example can be found in
      https://jax.readthedocs.io/en/latest/_autosummary/jax.export.DisabledSafetyCheck.html#jax.export.DisabledSafetyCheck
    flatten_signature: If True, the input and output signatures of the
      `ShloFunction` will be flattened to tuple. Otherwise, the PyTree structure
      will be kept.

  Returns:
    An Orbax Model `ShloFunction`.
  """
  if not hasattr(fun_jax, "trace"):
    fun_jax = jax.jit(fun_jax)

  create_exported = utils.make_jax_exported_creator(
      fun_jax,
      native_serialization_platforms,
      native_serialization_disabled_checks,
  )

  utils.assert_jax_trace_state_is_clean()

  exported = create_exported(*args_spec, **kwargs_spec)
  return jax_exported_to_shlo_fn(exported, flatten_signature)


# TODO(b/356174487): add more check and support for checkpointing.
def save_checkpoint(
    state: Any,
    path: str,
) -> None:
  """Saves JAX checkpoint to path."""
  if isinstance(state, (np.ndarray, jax.Array)):
    handler = ocp.ArrayCheckpointHandler()
  else:
    handler = ocp.StandardCheckpointHandler()
  checkpointer = ocp.Checkpointer(handler)
  checkpointer.save(path, state)


def convert_path_to_value(
    path: str, mime_type: str | None = None, version: str | None = None
) -> obm.ExternalValue:
  return obm.ExternalValue(
      data=obm.manifest_pb2.UnstructuredData(
          file_system_location=obm.manifest_pb2.FileSystemLocation(
              string_path=path
          ),
          mime_type=mime_type,
          version=version,
      ),
  )
