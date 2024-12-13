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

"""A global registry of checkpoint handler types."""

from typing import Type

from absl import logging
from orbax.checkpoint._src.handlers import checkpoint_handler


CheckpointHandler = checkpoint_handler.CheckpointHandler


class HandlerTypeRegistry:
  """A registry mapping handler type strings to handler types."""

  def __init__(self):
    self._registry = {}

  def add(
      self,
      handler_typestr: str,
      handler_type: Type[CheckpointHandler],
  ) -> None:
    """Adds an entry to the registry."""
    if handler_typestr in self._registry:
      if self._registry[handler_typestr] != handler_type:
        raise ValueError(
            f'Handler type string "{handler_typestr}" already exists in the '
            f'registry with type {self._registry[handler_typestr]}. '
            f'Cannot add type {handler_type}.'
        )
      else:
        logging.info(
            'Handler type string "%s" already exists in the registry with '
            'associated type %s.',
            handler_typestr,
            self._registry[handler_typestr],
        )
        return
    self._registry[handler_typestr] = handler_type

  def get(
      self,
      handler_typestr: str,
  ) -> Type[CheckpointHandler]:
    """Gets an entry from the registry."""
    if handler_typestr not in self._registry:
      raise ValueError(
          f'Handler type string "{handler_typestr}" not found in the registry.'
      )
    return self._registry[handler_typestr]


_GLOBAL_HANDLER_TYPE_REGISTRY = HandlerTypeRegistry()


def register_handler_type(handler_cls):
  _GLOBAL_HANDLER_TYPE_REGISTRY.add(handler_cls.typestr(), handler_cls)
  return handler_cls


def get_handler_type(handler_typestr: str) -> Type[CheckpointHandler]:
  return _GLOBAL_HANDLER_TYPE_REGISTRY.get(handler_typestr)
