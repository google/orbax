# Copyright 2023 The Orbax Authors.
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

"""CheckpointArgs base class and registration."""

import dataclasses
import inspect
from typing import Type, Union
from orbax.checkpoint import checkpoint_handler

CheckpointHandler = checkpoint_handler.CheckpointHandler


@dataclasses.dataclass
class CheckpointArgs:
  """Base class for all checkpoint argument dataclasses.

  Subclass this dataclass to define the arguments for your custom
  CheckpointHandler.
  When users use the CheckpointHandler, they will use this `CheckpointArgs` to
  see how to

  Example subclass:
  ```
  import ocp.checkpoint as ocp

  @ocp.args.register_with_handler(MyCheckpointHandler)
  @dataclasses.dataclass
  class MyCheckpointSave(ocp.args.CheckpointArgs):
    item: Any
    options: Any

  @ocp.args.register_with_handler(MyCheckpointHandler)
  @dataclasses.dataclass
  class MyCheckpointRestore(ocp.args.CheckpointArgs):
    options: Any
  ```

  Example usage:
  ```
  ckptr.save(
      path,
      custom_state=MyCheckpointSave(item=..., options=...)
  )


  ckptr.save(
      path,
      custom_state=MyCheckpointRestore(options=...)
  )
  ```
  """

  pass

_CHECKPOINT_ARG_TO_HANDLER = {}


def register_with_handler(handler_cls: Type[CheckpointHandler]):
  """Registers a CheckpointArg subclass with a specific handler.

  This registration is necessary so that when the user passes uses this
  CheckpointArg class with `CompositeCheckpointHandler`, we can automatically
  find the correct Handler to use to save this class.

  Args:
    handler_cls: `CheckpointHandler` to be associated with this `CheckpointArg`.

  Returns:
    Decorator.
  """

  def decorator(cls: Type[CheckpointArgs]):
    if not issubclass(cls, CheckpointArgs):
      raise TypeError(
          f'{cls} must subclass `CheckpointArgs` in order to be registered.'
      )
    _CHECKPOINT_ARG_TO_HANDLER[cls] = handler_cls
    return cls

  return decorator


def get_registered_handler_cls(
    arg: Union[Type[CheckpointArgs], CheckpointArgs]
) -> Type[CheckpointArgs]:
  """Returns the registered CheckpointHandler."""
  if inspect.isclass(arg):
    return _CHECKPOINT_ARG_TO_HANDLER[arg]
  return _CHECKPOINT_ARG_TO_HANDLER[type(arg)]
