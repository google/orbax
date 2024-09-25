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

"""Base class for modules used in Orbax Model export."""

import abc
from collections.abc import Mapping
from typing import Any, Callable, Union

from orbax.export import typing as orbax_export_typing

PyTree = orbax_export_typing.PyTree
ApplyFn = orbax_export_typing.ApplyFn


# TODO: b/365828049 - Explore replacing the base class with a protocol.
class OrbaxModuleBase(abc.ABC):
  """Define the base class API which manages the normalizing and wrapping of model data used in various export paths."""

  @abc.abstractmethod
  def __init__(
      self,
      params: PyTree,
      apply_fn: Union[
          Callable[..., Any], Mapping[str, ApplyFn], dict[str, ApplyFn]
      ],
      **kwargs: Any,
  ):
    """Constructor for creating an export Module."""

  @property
  @abc.abstractmethod
  def apply_fn_map(self) -> Mapping[str, ApplyFn]:
    """Returns the apply_fn_map."""

  @property
  @abc.abstractmethod
  def model_params(self) -> PyTree:
    """Returns the model parameters."""

  @property
  @abc.abstractmethod
  def methods(self) -> Mapping[str, Callable[..., Any]]:
    """Named methods in the context of the chosen export pathway."""

  @property
  @abc.abstractmethod
  def jax_methods(self) -> Mapping[str, Callable[..., Any]]:
    """Named methods in JAX context for validation."""
