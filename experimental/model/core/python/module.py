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

"""The `Module` class.

A `Module` is a dictionary-like container that can contain functions
and variables.
"""

# pylint: disable=g-importing-member
from dataclasses import dataclass
from dataclasses import field
from typing import Dict

from orbax.experimental.model.core.python.concrete_function import ConcreteFunction
from orbax.experimental.model.core.python.concrete_function import is_tree_of_vars
from orbax.experimental.model.core.python.concrete_function import TreeOfVars
from orbax.experimental.model.core.python.function import Function
from orbax.experimental.model.core.python.polymorphic_function import PolymorphicFunction
from orbax.experimental.model.core.python.value import Value


# TODO(wangpeng): This class is nothing more than `Dict[str, VALUE_TYPE]`.
#   Get rid of it?
@dataclass
class Module:
  """A dictionary-like container that can be saved into an ExportedModel."""

  VALUE_TYPE = (
      Function | ConcreteFunction | PolymorphicFunction | Value | TreeOfVars
  )
  DICT_TYPE = Dict[str, VALUE_TYPE]
  _dict: DICT_TYPE = field(default_factory=dict)

  def get_dict(self) -> DICT_TYPE:
    return self._dict

  def add_function(self, name, f: Function) -> None:
    self._dict[name] = f

  def add_concrete_function(self, name, f: ConcreteFunction) -> None:
    self._dict[name] = f

  def add_polymorphic_function(self, name, f: PolymorphicFunction) -> None:
    self._dict[name] = f

  def add_value(self, name, value: Value) -> None:
    self._dict[name] = value

  def add_variables(self, name, variables: TreeOfVars) -> None:
    self._dict[name] = variables

  def __setattr__(self, key, value) -> None:
    if isinstance(value, Function):
      self.add_function(key, value)
    elif isinstance(value, ConcreteFunction):
      self.add_concrete_function(key, value)
    elif isinstance(value, PolymorphicFunction):
      self.add_polymorphic_function(key, value)
    elif isinstance(value, Value):
      self.add_value(key, value)
    elif is_tree_of_vars(value):
      self.add_variables(key, value)
    else:
      super().__setattr__(key, value)

  def __getattr__(self, key):
    # our __getattr__ is called after the default one, so we can just error out
    # on KeyError.

    # Avoid dot access of properties on self (self._pb_to_json_name_mapping),
    # it causes infinite loop while copying in py3 (calling __getattr__ again).
    attribute_mapping = object.__getattribute__(self, '_dict')

    # Catch a key error and raise an AttributeError so deepcopy functions as
    # expected.  Without catching this the getattr on __deepcopy__ will raise
    # a key error because we explicitly override here.
    try:
      return attribute_mapping[key]
    except KeyError as exc:
      raise AttributeError(f'Module has no attribute {key}') from exc
