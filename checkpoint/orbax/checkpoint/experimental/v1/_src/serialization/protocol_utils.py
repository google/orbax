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

"""Utility to work with Protocol."""

from typing import Any, Type

from absl import logging
import typing_extensions


def is_subclass_protocol(cls: Type[Any], protocol: Type[Any]) -> bool:
  """Perform a best-effort check if `cls` is a subclass of protocol.

  This is needed because when the protocol has only attributes (no methods),
  isinstance() doesn't work.  In additional, issubclass also doesn't work with
  Protocols.  This is a best-effort workaround and can only check if the `cls`
  defines all the attributes of the `protocol`.  It does't check the type of the
  attributes.

  Args:
    cls: the cls will be checked if it implements `protocol`.
    protocol: the protocol to be checked.

  Returns:
    True if the class defines all the attributes of the protocol.
  """

  if not typing_extensions.is_protocol(protocol):  # pytype: disable=not-supported-yet
    raise ValueError(f'Protocol {protocol} is not a Protocol.')

  members = typing_extensions.get_protocol_members(protocol)  # pytype: disable=not-supported-yet

  if not members:
    return True  # empty protocol, so it matches any type.

  for attr in members:
    logging.vlog(1, 'attribute: %s', attr)

    # check if the attribute is defined in annotations or members
    if hasattr(cls, '__annotations__'):
      # dataclasses
      if attr not in cls.__annotations__ and not hasattr(cls, attr):
        return False
    elif not hasattr(cls, attr):
      # other types
      return False

  return True
