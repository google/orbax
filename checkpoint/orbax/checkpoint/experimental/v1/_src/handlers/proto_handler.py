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

"""ProtoHandler class."""

import inspect
import typing
from typing import Any, Awaitable, Type

from google.protobuf import message
from google.protobuf import text_format
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import multihost


_DEFAULT_FILENAME = "proto.pbtxt"


@typing.final
class ProtoHandler(
    handler_types.CheckpointableHandler[message.Message, Type[message.Message]]
):
  """Implementation of :py:class:`.CheckpointableHandler` for protocol buffers.

  ProtoHandler manages the serialization and deserialization of Protocol Buffer
  messages in text format. It utilizes an asynchronous two-tier execution model
  to offload I/O operations, ensuring background writing does not block the main
  process. In distributed environments, it provides multihost coordination to
  ensure that only the primary host performs the write operation.

  **Note: Users are encouraged NEVER to instantiate or use this handler
  directly.** Always use the top-level APIs like `ocp.save_checkpointables` and
  `ocp.load_checkpointables`. Orbax uses this handler by default for standard
  protocol buffer messages.

  To save a custom Protocol Buffer message and aggressively force Orbax to use
  the ProtoHandler (e.g., to specify a custom filename), the recommended
  approach is to use `ocp.Context` with `CheckpointablesOptions`. This allows
  you to bind the handler to a specific dictionary key within the Context scope.

  See :py:class:`~orbax.checkpoint.options.CheckpointablesOptions` for more
  details on handler registration.

  Example Usage:
    Save a protobuf message configuration::

      import orbax.checkpoint as ocp

      # Assuming MyProtoMessage is your compiled protobuf class
      my_proto_msg = MyProtoMessage(config_field="value")

      checkpointables_options = (
          ocp.options.CheckpointablesOptions.create_with_handlers(
              proto_config=ocp.handlers.ProtoHandler(
                  filename="model_config.pbtxt"
              )
          )
      )
      with ocp.Context(checkpointables_options=checkpointables_options):
          ocp.save_checkpointables(path, dict(proto_config=my_proto_msg))

  Attributes:
    filename (str): An optional filename used for saving and loading the
      protobuf data. If not provided, it defaults to a standard internal
      default filename.
  """

  def __init__(
      self,
      filename: str = _DEFAULT_FILENAME,
  ):
    """Initializes :py:class:`.ProtoHandler`."""
    self._filename = filename

  async def _background_save(
      self,
      directory: path_types.PathAwaitingCreation,
      checkpointable: message.Message,
      *,
      primary_host: int | None = None
  ):
    if multihost.is_primary_host(primary_host):
      directory = await directory.await_creation()
      path = directory / self._filename
      str_msg = text_format.MessageToString(checkpointable)
      await async_path.write_text(path, str_msg)

  async def save(
      self,
      directory: path_types.PathAwaitingCreation,
      checkpointable: message.Message,
  ) -> Awaitable[None]:
    context = context_lib.get_context()
    return self._background_save(
        directory,
        checkpointable,
        primary_host=context.multiprocessing_options.primary_host,
    )

  async def _background_load(
      self,
      directory: path_types.Path,
      message_type: Type[message.Message],
  ):
    path = directory / self._filename
    str_msg = await async_path.read_text(path)
    return text_format.Parse(str_msg, message_type())

  async def load(
      self,
      directory: path_types.Path,
      abstract_checkpointable: Type[message.Message] | None = None,
  ) -> Awaitable[message.Message]:
    if abstract_checkpointable is None:
      raise ValueError(
          "abstract_checkpointable must be provided to restore as a Proto."
      )
    return self._background_load(directory, abstract_checkpointable)

  async def metadata(self, directory: path_types.Path) -> Type[message.Message]:
    raise NotImplementedError()

  def is_handleable(self, checkpointable: Any) -> bool:
    return isinstance(checkpointable, message.Message)

  def is_abstract_handleable(self, abstract_checkpointable: Any) -> bool:
    return inspect.isclass(abstract_checkpointable) and issubclass(
        abstract_checkpointable, message.Message
    )
