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

"""Functions backed by StableHLO."""

import abc
from dataclasses import dataclass  # pylint: disable=g-importing-member
from typing import Any, Dict, Optional, Tuple

from orbax.experimental.model.core.python import unstructured_data
from orbax.experimental.model.core.python.function import Function
from orbax.experimental.model.core.python.function import ShloDType


class ShloFunctionSupplementalInfo(abc.ABC):
  """Interface for ShloFunction's supplemental_info."""

  @abc.abstractmethod
  def serializable_to_proto(
      self,
  ) -> unstructured_data.UnstructuredDataWithExtName:
    """Serializes it to an `UnstructuredDataWithExtName`."""
    pass

  # TODO(wangpeng): Remove this method once the SaveOptions.version==1
  # path is no longer needed.
  @abc.abstractmethod
  def process_xla_call_module_attrs(
      self, version: int, call_module_attrs: Dict[str, Any]
  ) -> None:
    """Gives it an opportunity to modify the XLA call module attrs.

    When we are serializing to saved_model.pb, we need to generate an
    `XlaCallModule` node for each `ShloFunction`. The
    `ShloFunction.supplemental_info` field may contain some
    information that needs to be stored in the `XlaCallModule` node's
    attributes. This function lets the
    `ShloFunction.supplemental_info` object modify the attributes.

    Args:
      version: the calling convention version of the `XlaCallModule`
        node. It will be the same as
        `ShloFunction.mlir_module_serialization_version`.
      call_module_attrs: the attributes of the `XlaCallModule` node. It is a
        (mutable) dict mapping attribute name to value.
    """
    pass


@dataclass
class ShloFunction(Function):
  """A StableHLO function.

  Attributes:
    mlir_module_serialized: the serialized StableHLO.
    calling_convention_version: the calling convention version of the serialized
      StableHLO. See more versioning details at
      [this](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#native-serialization-versions)
      and
      [this](https://jax.readthedocs.io/en/latest/export/export.html#calling-convention-versions)
      doc.
    module_kept_var_idx: the indices of the arguments that will actually be
      passed to the StableHLO. See this
      [doc](https://jax.readthedocs.io/en/latest/export/export.html#module-calling-convention)
      for its use in the calling convention.
    lowering_platforms: a tuple containing at least one of 'tpu', 'cpu', 'cuda',
      'rocm'. See this
      [doc](https://jax.readthedocs.io/en/latest/export/export.html#module-calling-convention)
      for the calling convention for when there are multiple lowering platforms.
    supplemental_info: an opaque supplemental to the function. It can be used to
      carry e.g. JAX-specific information.
    physical_in_dtypes: physical input dtypes, which (if given) will override
      `input_signature`.
    physical_out_dtypes: physical output dtypes, which (if given) will override
      `output_signature`.
  """

  mlir_module_serialized: bytes
  calling_convention_version: int
  module_kept_var_idx: tuple[int, ...]
  lowering_platforms: Tuple[str, ...]

  supplemental_info: Optional[ShloFunctionSupplementalInfo]

  physical_in_dtypes: Tuple[Optional[ShloDType], ...]
  physical_out_dtypes: Tuple[Optional[ShloDType], ...]
