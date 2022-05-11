# Copyright 2022 The Orbax Authors.
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

"""JsonCheckpointer class. Implementation of Checkpointer interface."""
import json
from typing import Any, Mapping, Optional

from orbax.checkpoint.checkpointer import Checkpointer
import tensorflow as tf


class JsonCheckpointer(Checkpointer):
  """Saves nested dictionary using json."""

  def __init__(self, filename: Optional[str] = None):
    """Initializes JsonCheckpointer.

    Args:
      filename: optional file name given to the written file; defaults to
        'metadata'
    """
    self._filename = filename or 'metadata'

  async def async_save(self, directory: str, item: Mapping[str, Any]):
    raise NotImplementedError('Async save not implemented by JsonCheckpointer.')

  async def async_restore(self,
                          directory: str,
                          item: Optional[bytes] = None) -> bytes:
    raise NotImplementedError(
        'Async restore not implemented by JsonCheckpointer.')

  def save(self, directory: str, item: Mapping[str, Any]):
    """Saves the given item.

    Args:
      directory: save location directory.
      item: nested dictionary.
    """
    path = tf.io.gfile.join(directory, self._filename)
    with tf.io.gfile.GFile(path, mode='w') as f:
      f.write(json.dumps(item))

  def restore(self, directory: str, item: Optional[bytes] = None) -> bytes:
    """Restores json mapping from directory.

    `item` is unused.

    Args:
      directory: restore location directory.
      item: unused

    Returns:
      Binary data read from `directory`.
    """
    del item
    path = tf.io.gfile.join(directory, self._filename)
    with tf.io.gfile.GFile(path, mode='r') as f:
      text = f.read()
    return json.loads(text)
