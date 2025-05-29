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

"""Common utils shared between jax_to_polymorphic_function.py and main_lib.py."""

from typing import Any, Callable, Dict, List, Optional, OrderedDict, Sequence, Tuple, TypeVar, Union

import jax
from jax import export as jax_export
import jax.numpy as jnp
import numpy as np
from orbax.experimental.model import core as obm
from orbax.experimental.model.jax2obm import constants


# TODO(b/338269227): Review JaxArrayPyTree, see if PyTreeDef is a better fit.
JaxArrayPyTree = TypeVar(
    "JaxArrayPyTree",
    bound=Union[
        jnp.ndarray,
        Tuple["JaxArrayPyTree", ...],
        List["JaxArrayPyTree"],
        Dict[str, "JaxArrayPyTree"],
        # For ordered dictionaries
        OrderedDict[str, "JaxArrayPyTree"],
        Tuple[Tuple[str, "JaxArrayPyTree"], ...],
        List[Tuple[str, "JaxArrayPyTree"]],
    ],
)


def _get_physical_dtype(dtype) -> Optional[obm.ShloDType]:
  try:
    if dtype == jax.numpy.bfloat16:
      return obm.ShloDType.bf16
    if dtype == jax.float0:
      return obm.ShloDType.bool
    return obm.np_dtype_to_shlo_dtype(_jax_physical_dtype(dtype))
  except TypeError:
    return None


def assert_jax_trace_state_is_clean():
  # TODO(b/332755487): Find out if there is a better way to check if we are
  #   inside a JAX transformation.
  if not jax.core.trace_state_clean():
    raise ValueError("convert must be used outside all JAX transformations.")


def make_jax_exported_creator(
    fun_jax: Callable[..., JaxArrayPyTree],
    native_serialization_platforms: (
        Sequence[constants.OrbaxNativeSerializationType] | None
    ) = None,
    native_serialization_disabled_checks: Sequence[
        jax_export.DisabledSafetyCheck
    ] = (),
):
  """Calls jax_export.export and returns a callable that returns Exported."""
  if native_serialization_platforms:
    if not isinstance(native_serialization_platforms, (list, tuple)) or not all(
        p in constants.OrbaxNativeSerializationType
        for p in native_serialization_platforms
    ):
      raise ValueError(
          "native_serialization_platforms must be a sequence "
          "containing a subset of constants.OrbaxNativeSerializationType."
          f"Got: {native_serialization_platforms}"
      )
    native_serialization_platforms = tuple(
        p.value for p in native_serialization_platforms
    )

  return jax_export.export(
      fun_jax,
      platforms=native_serialization_platforms,
      disabled_checks=native_serialization_disabled_checks,
  )


JaxDType = np.dtype[Any]


def _aval_dtype(a) -> JaxDType:
  assert isinstance(a, jax.core.UnshapedArray)
  return a.dtype


def _jax_physical_aval(aval: jax.core.ShapedArray) -> jax.core.ShapedArray:
  """Converts JAX avals from logical to physical, if relevant.

  JAX might have avals whose logical vs physical shape/dtype may
  differ, and only the physical view is expected to possibly
  relate to EM. EM impl rules should operate on the physical form.

  A JAX logical aval might even correspond, in principle, to several
  physical avals, but we don't support those here. Instead we assert
  there is only one and return it.

  Args:
    aval: The JAX abstract value to be converted.

  Returns:
    The physical representation of the input aval. If the logical and physical
    representations are the same, the original `aval` is returned.
  """
  # TODO(b/332756743): jax.core doesn't publish physical_aval. Figure out how to
  # access it (or alike). physical_aval = jax.core.physical_aval(aval)
  physical_aval = aval
  assert (
      len(physical_aval.shape) >= len(aval.shape)
      and physical_aval.shape[: len(aval.shape)] == aval.shape
  ), (physical_aval, aval)
  return physical_aval


def _jax_physical_dtype(dtype) -> JaxDType:
  # assuming () is a fine stand-in shape
  return _jax_physical_aval(jax.core.ShapedArray((), dtype)).dtype
