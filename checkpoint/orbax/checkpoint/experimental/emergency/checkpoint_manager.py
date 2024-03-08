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

"""A class providing functionalities for checkpointing to local file system."""

from typing import Union
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import checkpoint_manager

CheckpointManagerOptions = checkpoint_manager.CheckpointManagerOptions


class CheckpointManager(checkpoint_manager.CheckpointManager):
  """A checkpoint manager that stores checkpoint to local storage."""

  def __init__(self, *args, **kwargs):
    super().__init__(primary_host=None, *args, **kwargs)

  def _is_equal_on_all_hosts(self, value: Union[int, float]) -> bool:
    """return true if all `values` are equal on all hosts."""

    global_mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape(
            jax.process_count(), jax.local_device_count()
        ),
        ['x', 'y'],
    )
    pspecs = jax.sharding.PartitionSpec('x', None)

    arr = multihost_utils.host_local_array_to_global_array(
        np.array([value]),
        global_mesh,
        pspecs,
    )

    # calculate the global range, eg. (max - min)
    @jax.jit
    def global_ptp(x):
      return jnp.ptp(x)

    ptp = global_ptp(arr)
    return ptp.addressable_data(0) == 0

  def latest_step(self) -> int | None:
    """Return the latest step if all hosts have the same step, otherwise, None."""
    local_latest = super().latest_step() or -1

    if self._is_equal_on_all_hosts(local_latest):
      return local_latest if local_latest != -1 else None
    else:
      return None
