# Copyright 2026 The Orbax Authors.
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

"""Fake VoxelSpec definition for testing purposes."""

import abc
from collections.abc import Sequence
import dataclasses
import jaxtyping
import numpy as np

from google.protobuf import message


DTypeLike = str | np.dtype | type[np.generic]


@dataclasses.dataclass
class JDSpecBase:
  """JD specification base class for JD modules.

  This is used for Orbax Model (OBM) integration.

  Attributes:
    shape: Tensor dimensions as a tuple of integers.
    dtype: Tensor data type as a `np.dtype`.
  """

  shape: Sequence[int]
  dtype: DTypeLike

  def __post_init__(self):
    object.__setattr__(self, 'shape', tuple(self.shape))
    try:
      dtype = np.dtype(self.dtype)
      object.__setattr__(self, 'dtype', dtype)
    except TypeError as e:
      raise ValueError(
          f'Invalid dtype: {self.dtype!r} cannot be converted to np.dtype.'
      ) from e


class JDModuleBase(abc.ABC):
  """JDModule Abstract Base Class."""

  def __init__(self):
    self._assets: dict[str, str] = {}

  def set_assets(self, assets: dict[str, str]) -> None:
    self._assets = assets

  def export_assets(self) -> dict[str, str]:
    return self._assets

  @abc.abstractmethod
  def get_output_signature(
      self, input_signature: jaxtyping.PyTree
  ) -> jaxtyping.PyTree:
    pass

  @abc.abstractmethod
  def export_plan(self) -> message.Message:
    pass


# Define `__all__` to explicitly declare the public API of this module.
# This controls what `from jd2obm import *` imports and helps linters.
__all__ = [
    'JDSpecBase',
    'JDModuleBase',
]
