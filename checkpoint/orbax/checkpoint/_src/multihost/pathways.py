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

"""Pathways-specific multihost utilities."""

import functools
import jax
import numpy as np
from .learning.deepmind.jax.ocean.remote_python import rp


@functools.lru_cache(maxsize=1)
def worker_count() -> int:
  """Gets the number of Pathways workers."""
  fully_replicated_sharding = jax.sharding.NamedSharding(
      jax.sharding.Mesh(jax.devices(), 'x'),
      jax.sharding.PartitionSpec(),
  )

  @rp.stateless_fn
  def _get_worker_count(_) -> jax.Array:
    wc = np.asarray(jax.process_count(), dtype=np.int32)
    return jax.make_array_from_callback(
        (),
        fully_replicated_sharding,
        lambda _: wc,
        dtype=np.int32,
    )

  dummy_input = jax.device_put(
      np.asarray(0, dtype=np.int32),
      device=fully_replicated_sharding,
  )
  _get_worker_count.register_shape_fn(
      lambda _: jax.ShapeDtypeStruct(
          (), dtype=np.int32, sharding=fully_replicated_sharding
      )
  )
  result = _get_worker_count(rp.to_remote_python(dummy_input))
  jax.block_until_ready(result)
  result = rp.from_remote_python(result)
  return result.item()
