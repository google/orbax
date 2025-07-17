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

"""Export class that implements the save and load abstract class defined in Export Base for use with the Orbax Model export format."""

import functools
import itertools
import os
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, cast

from absl import logging
import jax
from jax import export as jax_export
from orbax.export import constants
from orbax.export import export_base
from orbax.export import jax_module
from orbax.export import obm_configs
from orbax.export import oex_orchestration
from orbax.export import serving_config as osc
from orbax.export import utils
from orbax.export.modules import obm_module
from orbax.export.oex_orchestration import oex_orchestration_pb2
from orbax.export.oex_orchestration import Signature
from orbax.export.typing import PyTree
import tensorflow as tf


class PolymorphicShapeConcretizer:
  """Concretizes a symbolic polymorphic shape to a list of concrete shapes.

  This class is used to concretize a symbolic spec with polymorphic shapes to a
  list of specs with concrete shapes.

  For example,
  Given a symbolic spec with polymorphic shapes
  {'prompt': jax.ShapeDtypeStruct(shape=(b, l), dtype=int32)}

  and the discrete values for the symbols
  {'b': [1, 2], 'l': [3, 4]},

  The result of concretization will be a list of specs with concrete shapes,
  [
      {'prompt': jax.ShapeDtypeStruct(shape=(1, 3), dtype=int32) },
      {'prompt': jax.ShapeDtypeStruct(shape=(1, 4), dtype=int32)},
      {'prompt': jax.ShapeDtypeStruct(shape=(2, 3), dtype=int32)},
      {'prompt': jax.ShapeDtypeStruct(shape=(2, 4), dtype=int32)},
  ]
  """

  def __init__(
      self,
      symbolic_shapes: PyTree,
      symbol_to_values: PyTree,
  ):
    """Initializes the PolymorphicShapeConcretizer class.

    Args:
      symbolic_shapes: The symbolic shapes to concretize. For example,
        {'prompt': jax.ShapeDtypeStruct(shape=(b, l), dtype=int32)}.
      symbol_to_values: A mapping of symbol names to discret values. For
        example, {'b': [1, 2], 'l': [3, 4]}.
    """
    self._validate_inputs(symbolic_shapes, symbol_to_values)
    self._symbolic_shapes = symbolic_shapes
    self._symbol_to_values = symbol_to_values

    self._symbol_names = list(self._symbol_to_values.keys())
    # A list of tuples, where each tuple is a combo of symbol values.
    # For example, if the symbol names are ['b', 'l'] and the values for each
    # symbol are [[1, 2], [3, 4]], then the symbol_value_combos will be
    # [(1, 3), (1, 4), (2, 3), (2, 4)].
    self._symbol_value_combos = list(
        itertools.product(
            *(self._symbol_to_values[name] for name in self._symbol_names)
        )
    )

  def _validate_inputs(self, symbolic_shapes: PyTree, symbol_to_values: PyTree):
    """Validates the types of inputs ."""
    if not isinstance(symbolic_shapes, Mapping):
      raise ValueError(
          f"symbolic_shapes must be a Mapping. Got: {type(symbolic_shapes)}"
      )
    if not all(
        isinstance(v, jax.ShapeDtypeStruct) for v in symbolic_shapes.values()
    ):
      raise ValueError(
          "symbolic_shapes values must be jax.ShapeDtypeStruct instances. Got:"
          f" {symbolic_shapes}"
      )

    if not isinstance(symbol_to_values, Mapping):
      raise ValueError(
          f"symbol_to_values must be a Mapping. Got: {type(symbol_to_values)}"
      )
    if not all(
        isinstance(v, Sequence) and not isinstance(v, (str, bytes))
        for v in symbol_to_values.values()
    ):
      raise ValueError(
          "symbol_to_values values must be Sequences of ints. Got:"
          f" {symbol_to_values}"
      )

  def _concretize_spec(
      self,
      symbolic_spec: jax.ShapeDtypeStruct,
      symbol_to_value: Mapping[str, int],
  ) -> jax.ShapeDtypeStruct:
    concrete_shape = []
    for d in symbolic_spec.shape:
      if isinstance(d, int):
        concrete_shape.append(d)
      else:
        concrete_shape.append(symbol_to_value[str(d)])
    return jax.ShapeDtypeStruct(tuple(concrete_shape), symbolic_spec.dtype)

  def concretize(self) -> Sequence[Mapping[str, jax.ShapeDtypeStruct]]:
    """Concretizes the symbolic shapes to a list of concrete shapes."""

    def _concretize_shapes(symbol_to_value: Mapping[str, int]):
      return {
          k: self._concretize_spec(v, symbol_to_value)
          for k, v in self._symbolic_shapes.items()
      }

    return [
        _concretize_shapes(dict(zip(self._symbol_names, symbol_value_combo)))
        for symbol_value_combo in self._symbol_value_combos
    ]


class ObmExport(export_base.ExportBase):
  """Defines the save and load methods for exporting a model using Orbax Model export."""

  def __init__(
      self,
      module: jax_module.JaxModule,
      serving_configs: Sequence[osc.ServingConfig],
  ):
    """Initializes the ObmExport class."""
    if module.export_version != constants.ExportModelType.ORBAX_MODEL:
      raise ValueError(
          "JaxModule export version is not of type ORBAX_MODEL. Please use the"
          " correct export_version. Expected ORBAX_MODEL, got"
          f" {module.export_version}"
      )

    obm_model_module = module.export_module()

  def save(
      self,
      model_path: str,
      **kwargs: Any,
  ):
    """Saves a Jax model in the Orbax Model export format.

    Args:
      model_path: The path to save the model.
      **kwargs: Additional arguments to pass to the `save` method. Accepted
        arguments are `save_options` and `serving_signatures`.
    """

  def load(self, model_path: str, **kwargs: Any):
    """Loads the model previously saved in the Orbax Model export format."""
    logging.info("Loading model using Orbax Export Model.")
    raise NotImplementedError("ObmExport.load not implemented yet.")

  @property
  def serving_signatures(self) -> Mapping[str, Callable[..., Any]]:
    """Returns a map of signature keys to serving functions."""
    raise NotImplementedError("ObmExport.load not implemented yet.")
