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

"""Handler that combines other handlers.

Usage example:

```
import orbax.checkpoint as ocp

# A PyTree of jax.Arrays
my_state = ...
# A dictionary to be serialized as JSON
json_dict = ...

ckpter = ocp.Checkpointer(ocp.CompositeCheckpointHandler(
    'state', 'metadata'))
ckpter.save(
    path,
    args=ocp.args.Composite(
        state=ocp.args.PyTreeSave(my_state),
        metadata=ocp.args.JsonSave(json_dict)
    )
)

restored = ckpter.save(
    path,
    args=ocp.args.Composite(
        state=ocp.args.PyTreeRestore(),
        metadata=ocp.args.JsonRestore()
    )
)
my_state = restored.state
json_dict = restored.metadata
```
"""

from collections.abc import Collection, KeysView
from typing import AbstractSet, Any, Mapping, Optional, Tuple
from orbax.checkpoint import checkpoint_args

CheckpointArgs = checkpoint_args.CheckpointArgs


class CompositeArgs(CheckpointArgs):
  """Args for wrapping multiple checkpoint items together.

  For simplicity, this object is immutable (although objects attached to it
  may be mutable).
  """

  __items__: Mapping[str, CheckpointArgs]

  def __init__(self, **items: CheckpointArgs):
    super().__setattr__('__items__', items)

    reserved_keys = set(dir(self))

    for key, value in items.items():
      # Reserve and prevent users from setting keys that start with '__'. These
      # may be used later to define options for CompositeCheckpointManager.
      if key.startswith('__'):
        raise ValueError(f'Composiite keys cannot start with "__". Got: {key}')
      if key not in reserved_keys:
        # We do not raise an error if the user specifies a key that matches an
        # existing attribute (like 'keys', 'values', 'items'). These can be
        # accessed through self[key], but not self.key.
        super().__setattr__(key, value)

  def __getitem__(self, key: str) -> CheckpointArgs:
    return self.__items__[key]

  def __setattr__(self, key: str, value: Any):
    del key
    del value
    raise ValueError('CompositeArgs is immutable after initialization.')

  def __len__(self) -> int:
    return len(self.__items__)

  def keys(self) -> KeysView[str]:
    return self.__items__.keys()

  def values(self) -> Collection[CheckpointArgs]:
    return self.__items__.values()

  def items(self) -> AbstractSet[Tuple[str, CheckpointArgs]]:
    return self.__items__.items()

  def get(self, key: str, default=None) -> Optional[CheckpointArgs]:
    try:
      return self.__getitem__(key)
    except KeyError:
      return default


# Returned object of CompositeCheckpointHandler is an alias of CompositeArgs.
CompositeResults = CompositeArgs
