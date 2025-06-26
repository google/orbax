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

"""This module provides a PyGrain CheckpointHandler for integration with Orbax."""
import dataclasses
import json
import logging
from typing import Any, Optional
from etils import epath
import jax
import orbax.checkpoint as ocp


# Ipmlements orbax.checkpoint.CheckpointHandler.
class DatasetIteratorCheckpointHandler(ocp.CheckpointHandler):
  """Orbax CheckpointHandler for PyGrain iterators."""

  class DummyIterator:
    """Dummy iterator for testing purposes."""
    data: list[tuple[str, str]]
    current_index: int = 0

    def __init__(self, data: list[tuple[str, str]]):
      self.data = data

    def __iter__(self):
      return self

    def __next__(self):
      result = self.data[self.current_index]
      self.current_index += 1
      return result

    def get_state(self) -> dict[str, int]:
      return {"current_index": self.current_index}

    def set_state(self, state):
      self.current_index = state["current_index"]

  def save(
      self,
      directory: epath.Path,
      # `item` is for backwards compatibility with older Orbax API, see
      # https://orbax.readthedocs.io/en/latest/api_refactor.html.
      item: Optional[DummyIterator] = None,
      args: Any = None,
  ):
    """Saves the given iterator to the checkpoint in `directory`."""
    item = item or args.item  # pytype:disable=attribute-error
    logging.info("args: %s, item: %s", args, item)
    if isinstance(item, DatasetIteratorCheckpointHandler.DummyIterator):
      logging.info("Saving DummyIterator.")
      state = json.dumps(item.get_state(), indent=4)
    else:
      logging.info("Saving DatasetIterator.")
      state = item.get_state().decode()
    process_index, process_count = jax.process_index(), jax.process_count()
    filename = directory / f"process_{process_index}-of-{process_count}.json"
    filename.write_text(state)

  def restore(
      self,
      directory: epath.Path,
      item: Optional[DummyIterator] = None,
      args: Any = None,
  ) -> DummyIterator:
    """Restores the given iterator from the checkpoint in `directory`."""
    item = item or args.item  # pytype:disable=attribute-error
    process_index, process_count = jax.process_index(), jax.process_count()
    filename = directory / f"process_{process_index}-of-{process_count}.json"
    if not filename.exists():
      raise ValueError(f"File {filename} does not exist.")
    state = filename.read_text()
    if isinstance(item, DatasetIteratorCheckpointHandler.DummyIterator):
      state = json.loads(state)
    else:
      state = state.encode()
    item.set_state(state)
    return item

  # Required by interface but not supported by PyGrain checkpoints.
  def structure(self, directory: epath.Path) -> Any:
    del directory
    return None

  # Required by interface.

  def metadata(self, directory: epath.Path) -> Optional[Any]:
    del directory
    return None

  def finalize(self, directory: epath.Path):
    pass

  def close(self):
    pass

  @classmethod
  def typestr(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


@ocp.args.register_with_handler(DatasetIteratorCheckpointHandler, for_save=True)  # pytype:disable=wrong-arg-types
@dataclasses.dataclass
class DatasetIteratorCheckpointSave(ocp.args.CheckpointArgs):
  item: Any


@ocp.args.register_with_handler(DatasetIteratorCheckpointHandler, for_restore=True)  # pytype:disable=wrong-arg-types
@dataclasses.dataclass
class DatasetIteratorCheckpointRestore(ocp.args.CheckpointArgs):
  item: Any

