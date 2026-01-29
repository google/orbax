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

"""Class `JaxSpecificInfo` and its companions."""

# pylint: disable=g-importing-member
from collections.abc import Iterable, Sequence
import dataclasses
from typing import Any, TypeVar

import jax
# Somehow JAX requires this import to make `jax.export` available.
from jax import export  # pylint: disable=unused-import
from jax.experimental import layout as jax_layout
import jax.numpy as jnp
import numpy as np
from orbax.experimental.model import core as obm
from orbax.experimental.model.jax2obm import jax_supplemental_pb2
from orbax.experimental.model.jax2obm.jax_supplemental_pb2 import DimensionSizeRefinement
from orbax.experimental.model.jax2obm.jax_supplemental_pb2 import DTypeRefinement
from orbax.experimental.model.jax2obm.jax_supplemental_pb2 import ShapeDTypeRefinement
from orbax.experimental.model.jax2obm.jax_supplemental_pb2 import ShapeDTypeRefinements
from orbax.experimental.model.jax2obm.jax_supplemental_pb2 import ShapeRefinement

from tensorflow.compiler.xla import xla_data_pb2

T1 = TypeVar("T1")
T2 = TypeVar("T2")


def unzip2(
    xys: Iterable[tuple[T1, T2]],
) -> tuple[tuple[T1, ...], tuple[T2, ...]]:
  """Unzip a sequence of tuples into two separate tuples."""
  xs: list[T1] = []
  ys: list[T2] = []
  for x, y in xys:
    xs.append(x)
    ys.append(y)
  return tuple(xs), tuple(ys)


# This function is adapted from the non-public
# `jax._src.layout.Layout._to_xla_layout`. Since there are no plans to make the
# original API public, the logic is replicated here.
def _to_xla_layout_proto(
    layout: jax_layout.Layout,
    dtype: jnp.dtype,
) -> xla_data_pb2.LayoutProto:
  """Converts a `jax.layout.Layout` to an `xla_data_pb2.LayoutProto`."""
  if layout.tiling is None:
    layout_proto = xla_data_pb2.LayoutProto()
    layout_proto.minor_to_major.extend(layout.major_to_minor[::-1])
  else:
    if layout.sub_byte_element_size_in_bits != 0:
      sub_byte_size = layout.sub_byte_element_size_in_bits
    elif jnp.issubdtype(dtype, np.integer):
      sub_byte_size = jnp.iinfo(dtype).bits if jnp.iinfo(dtype).bits < 8 else 0
    else:
      sub_byte_size = 0
    layout_proto = xla_data_pb2.LayoutProto()
    layout_proto.minor_to_major.extend(layout.major_to_minor[::-1])
    layout_proto.tiles.extend(
        xla_data_pb2.TileProto(dimensions=tile) for tile in layout.tiling
    )
    layout_proto.sub_byte_element_size_in_bits = sub_byte_size
  return layout_proto


def _name_leaf(
    path: jax.tree_util.KeyPath, leaf: obm.ShloTensorSpec
) -> obm.ShloTensorSpec:
  """Assigns a name to a leaf node in-place based on its PyTree path.

  The name is a dot-separated string representation of the path keys, extracted
  from dictionary keys, sequence indices, or attribute names. This function is
  designed to be used as the transform function in
  `jax.tree_util.tree_map_with_path`.

  The `leaf` argument's `name` attribute is modified in-place.

  Args:
    path: A tuple of path elements (e.g., DictKey, SequenceKey, GetAttrKey)
      provided by `tree_map_with_path`, representing the path to the leaf in a
      PyTree.
    leaf: The leaf node to be named. It must have a `name` attribute that can be
      set.

  Returns:
    The leaf node with its `name` attribute set to the dot-separated path
    string.

  Raises:
    TypeError: If an unknown key type is encountered in the path.
  """
  path_str_parts = []
  for key in path:
    if isinstance(key, jax.tree_util.DictKey):
      path_str_parts.append(str(key.key))
    elif isinstance(key, jax.tree_util.SequenceKey):
      path_str_parts.append(str(key.idx))
    elif isinstance(key, jax.tree_util.GetAttrKey):
      path_str_parts.append(str(key.name))
    elif isinstance(key, jax.tree_util.FlattenedIndexKey):
      path_str_parts.append(str(key.key))
    else:
      raise TypeError(f"Unknown key type: {type(key)}")
  leaf.name = ".".join(path_str_parts)
  return leaf


def _serialize_effect(eff: jax.core.Effect) -> str:
  """Serializes a JAX Effect to a string.

  Args:
    eff: The JAX Effect to be serialized.

  Returns:
    A string representation of the Effect.
  """
  # Adapted from JAX's `_serialize_effect`.
  try:
    eff_replica = eff.__class__()
  except Exception as exc:
    raise NotImplementedError(
        f"Effect {eff} must have a nullary constructor to be serializable"
    ) from exc
  try:
    hash_eff = hash(eff)
    hash_eff_replica = hash(eff_replica)
  except Exception as exc:
    raise NotImplementedError(
        f"Effect {eff} must be hashable to be serializable"
    ) from exc
  if eff != eff_replica or hash_eff != hash_eff_replica:
    raise NotImplementedError(
        f"Effect {eff} must have a nullary class constructor that produces an "
        "equal effect object."
    )
  return str(eff.__class__)


def _serialize_disabled_safety_check(
    check: jax.export.DisabledSafetyCheck,
) -> jax_supplemental_pb2.DisabledSafetyCheck:
  """Serializes a JAX DisabledSafetyCheck to a proto.

  Args:
    check: The JAX DisabledSafetyCheck to be serialized.

  Returns:
    A proto representation of the DisabledSafetyCheck.
  """
  # Adapted from JAX's `_serialize_disabled_safety_check`.
  proto = jax_supplemental_pb2.DisabledSafetyCheck()
  custom_call_target_str = check.is_custom_call()
  if custom_call_target_str is not None:
    proto.custom_call.target_name = custom_call_target_str
  elif check == jax.export.DisabledSafetyCheck.platform():
    proto.platform.SetInParent()
  else:
    raise ValueError(f"Unrecognized DisabledSafetyCheck: {check}")
  return proto


ShapeDTypeRefinementPair = tuple[ShapeRefinement | None, DTypeRefinement | None]


def _is_useful_refinement_pair(refinement: ShapeDTypeRefinementPair) -> bool:
  shape, dtype = refinement
  return shape is not None or dtype is not None


_MIN_RATIO_TO_USE_REFINEMENT_LIST = 0.5


def _to_shape_dtype_refinements_proto(
    refinements: Sequence[ShapeDTypeRefinementPair] | None,
) -> ShapeDTypeRefinements | None:
  """Converts `ShapeDTypeRefinementPair`s to a `ShapeDTypeRefinements`."""
  if refinements is None:
    return None
  n_useful_refinements = 0
  for refinement in refinements:
    if _is_useful_refinement_pair(refinement):
      n_useful_refinements += 1
  n_refinements = len(refinements)
  if (
      n_refinements == 0
      or n_useful_refinements / n_refinements
      >= _MIN_RATIO_TO_USE_REFINEMENT_LIST
  ):
    return ShapeDTypeRefinements(
        list=jax_supplemental_pb2.ShapeDTypeRefinementList(
            refinements=(
                ShapeDTypeRefinement(shape=shape, dtype=dtype)
                for shape, dtype in refinements
            )
        )
    )
  else:
    return ShapeDTypeRefinements(
        map=jax_supplemental_pb2.ShapeDTypeRefinementMap(
            idx_to_refinement={
                idx: ShapeDTypeRefinement(shape=shape, dtype=dtype)
                for idx, (shape, dtype) in enumerate(refinements)
                if _is_useful_refinement_pair((shape, dtype))
            }
        )
    )

CURRENT_JAX_SUPPLEMENTAL_MIME_TYPE: str = (
    "application/protobuf; type=orbax_model_jax_supplemental.Function"
)
CURRENT_JAX_SUPPLEMENTAL_VERSION: str = "0.0.1"


@dataclasses.dataclass
class JaxSpecificInfo(obm.ShloFunctionSupplementalInfo):
  """JAX-specific information.

  Attributes:
    name: the name of the function.
    uses_shape_polymorphism: bool
  """

  name: str
  input_spec_refinements: Sequence[ShapeDTypeRefinementPair] | None
  output_spec_refinements: Sequence[ShapeDTypeRefinementPair] | None
  nr_devices: int

  ordered_effects: Sequence[jax.core.Effect]
  unordered_effects: Sequence[jax.core.Effect]
  disabled_safety_checks: Sequence[jax.export.DisabledSafetyCheck]

  uses_shape_polymorphism: bool

  def serializable_to_proto(self) -> obm.UnstructuredDataWithExtName:
    """Serializes to an `UnstructuredDataWithExtName`."""
    jax_proto = jax_supplemental_pb2.Function(
        name=self.name,
        input_spec_refinements=_to_shape_dtype_refinements_proto(
            self.input_spec_refinements
        ),
        output_spec_refinements=_to_shape_dtype_refinements_proto(
            self.output_spec_refinements
        ),
        nr_devices=self.nr_devices,
        ordered_effects=map(_serialize_effect, self.ordered_effects),
        unordered_effects=map(_serialize_effect, self.unordered_effects),
        disabled_checks=map(
            _serialize_disabled_safety_check, self.disabled_safety_checks
        ),
        uses_shape_polymorphism=self.uses_shape_polymorphism,
    )
    return obm.UnstructuredDataWithExtName(
        proto=obm.manifest_pb2.UnstructuredData(
            inlined_bytes=jax_proto.SerializeToString(),
            mime_type=CURRENT_JAX_SUPPLEMENTAL_MIME_TYPE,
            version=CURRENT_JAX_SUPPLEMENTAL_VERSION,
        ),
        ext_name="pb",
    )


def _to_shlo_shape_and_refinement(
    jax_shape: Sequence[Any],
) -> tuple[obm.ShloShape, ShapeRefinement | None]:
  """Gets a `ShloShape` and a `ShapeRefinement` from a JAX shape.

  Args:
    jax_shape: a JAX shape.

  Returns:
    A `ShloShape` and an optional `ShapeRefinement`. The
    `ShapeRefinement` will be `None` if no refinement is needed
    (i.e. all the dimensions are integers).
  """
  shlo_dim_sizes: list[obm.ShloDimSize] = []
  dim_refinements: list[str | None] = []
  for dim in jax_shape:
    if dim is None:
      shlo_dim = None
      dim_refinement = None
    else:
      try:
        shlo_dim = int(dim)
        dim_refinement = None
      except Exception:
        shlo_dim = None
        dim_refinement = str(dim)
    shlo_dim_sizes.append(shlo_dim)
    dim_refinements.append(dim_refinement)
  if all(r is None for r in dim_refinements):
    shape_refinement = None
  else:
    shape_refinement = ShapeRefinement(
        dimension_sizes=(
            DimensionSizeRefinement(size=r) for r in dim_refinements
        ),
    )
  return shlo_dim_sizes, shape_refinement


# The StableHLO type to represent float0 (the value will always be zero).
#
# We picked bool because this is what JAX uses when it lowers float0
# to StableHLO.
_shlo_dtype_for_float0 = obm.ShloDType.bool


def _to_shlo_dtype_and_refinement(
    jax_dtype: np.dtype[Any],
) -> tuple[obm.ShloDType, DTypeRefinement | None]:
  if jax_dtype == jax.numpy.int4:
    return obm.ShloDType.i4, None
  if jax_dtype == jax.numpy.uint4:
    return obm.ShloDType.ui4, None
  if jax_dtype == jax.float0:
    return _shlo_dtype_for_float0, DTypeRefinement.f0
  if jax_dtype == jax.numpy.bfloat16:
    return obm.ShloDType.bf16, None
  if not isinstance(jax_dtype, np.dtype):
    raise TypeError(
        f"jax_dtype must be an instance of np.dtype, but got {jax_dtype} of"
        f" type {type(jax_dtype)}"
    )
  return obm.np_dtype_to_shlo_dtype(jax_dtype), None


def _to_shlo_tensor_spec_and_refinement(
    aval: jax.core.AbstractValue,
    sharding: Any,
    layout: jax_layout.Layout | None = None,
) -> tuple[obm.ShloTensorSpec, ShapeDTypeRefinementPair]:
  """Gets a `ShloTensorSpec` and a `ShapeDTypeRefinement` from a `ShapedArray`.

  Args:
    aval: a JAX `ShapedArray`.
    sharding: the sharding of the `ShapedArray`.
    layout: the layout of the tensor.

  Returns:
    A `ShloTensorSpec` and a `ShapeDTypeRefinement`.
  """
  assert isinstance(aval, jax.core.ShapedArray)
  shlo_shape, shape_refinement = _to_shlo_shape_and_refinement(aval.shape)
  shlo_dtype, dtype_refinement = _to_shlo_dtype_and_refinement(aval.dtype)
  xla_layout_proto = (
      _to_xla_layout_proto(layout, aval.dtype) if layout is not None else None
  )
  spec = obm.ShloTensorSpec(
      shape=shlo_shape,
      dtype=shlo_dtype,
      sharding=sharding,
      layout=xla_layout_proto,
  )
  return spec, (shape_refinement, dtype_refinement)


def _to_flat_shlo_specs_and_refinements(
    avals: Sequence[jax.core.AbstractValue],
    shardings: Sequence[Any],
    layouts: Sequence[jax_layout.Layout] | None = None,
) -> tuple[
    tuple[obm.ShloTensorSpec, ...], tuple[ShapeDTypeRefinementPair, ...] | None
]:
  """Converts a sequence of avals to a tuple of ShloTensorSpecs."""
  specs_and_refinements = tuple(
      map(
          _to_shlo_tensor_spec_and_refinement,
          avals,
          shardings,
          layouts if layouts is not None else [None] * len(avals),
      )
  )
  specs, refinements = unzip2(specs_and_refinements)
  if all(shape is None and dtype is None for shape, dtype in refinements):
    refinements = None
  return specs, refinements


def _to_shlo_spec_tree_and_refinement_tuple(
    avals: Sequence[jax.core.AbstractValue],
    shardings: Sequence[Any],
    layouts: Sequence[jax_layout.Layout] | None = None,
    tree_def: jax.tree_util.PyTreeDef | None = None,
    name_leaves: bool = False,
) -> tuple[
    obm.Tree[obm.ShloTensorSpec], tuple[ShapeDTypeRefinementPair, ...] | None
]:
  """Converts a sequence of avals to a tree of ShloTensorSpecs."""
  flat, refinements = _to_flat_shlo_specs_and_refinements(
      avals, shardings, layouts
  )
  if tree_def is None:
    flat: obm.Tree[obm.ShloTensorSpec]  # a tuple is also a tree
    jax_tree = flat
  else:
    jax_tree = jax.tree_util.tree_unflatten(tree_def, flat)

    def assert_leaf(x: Any) -> None:
      if not isinstance(x, obm.ShloTensorSpec):
        raise ValueError(
            f"Leaf needs to be a ShloTensorSpec, but its type is: {type(x)}"
        )
    obm.tree_util.assert_tree(assert_leaf, jax_tree)
    jax_tree: obm.Tree[obm.ShloTensorSpec]
  if name_leaves:
    jax_tree = jax.tree_util.tree_map_with_path(_name_leaf, jax_tree)
  return jax_tree, refinements
