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

"""DatasetCheckpointer class. Implementation of Checkpointer interface."""

import jax
from jax.experimental import multihost_utils
from orbax.checkpoint.checkpointer import Checkpointer
import tensorflow as tf

_CHECKPOINT_FILE_NAME = 'ckpt'


class DatasetCheckpointer(Checkpointer):
  """A Checkpointer implementation that handles tf.data.Iterator."""

  async def async_save(self, directory: str, item: tf.data.Iterator):
    raise NotImplementedError('Async save not provided by tf.train.Checkpoint.')

  async def async_restore(self, directory: str,
                          item: tf.data.Iterator) -> tf.data.Iterator:
    raise NotImplementedError(
        'Async restore not provided by tf.train.Checkpoint.')

  def save(self, directory: str, item: tf.data.Iterator):
    """Saves the given item.

    In a multihost setting, only saves on host 0.

    Args:
      directory: save location directory.
      item: a tf.data.Iterator to be saved.
    """
    if jax.process_index() == 0:
      ckpt = tf.train.Checkpoint(ds=item)
      ckpt.write(tf.io.gfile.join(directory, _CHECKPOINT_FILE_NAME))
    multihost_utils.sync_global_devices('DatasetCheckpointer:save')

  def restore(self, directory: str, item: tf.data.Iterator) -> tf.data.Iterator:
    """Restores the given item.

    Args:
      directory: restore location directory.
      item: a tf.data.Iterator to be restored.

    Returns:
      a tf.data.Iterator restored from `directory`.
    """
    ckpt = tf.train.Checkpoint(ds=item)
    ckpt.read(tf.io.gfile.join(directory,
                               _CHECKPOINT_FILE_NAME)).assert_consumed()
    return item
