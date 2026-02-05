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

"""Small test replicating broadcasting conditions in a multislice environment."""

import functools
import logging
import sys

from absl import flags
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import utils
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint.experimental.emergency import integration_test_utils

from .platforms.xla.megascale.jax import megascale_support
from absl.testing import absltest


class ReproTest(googletest.TestCase):

  def test_main(self):

    host_count = jax.process_count()
    local_device_count = jax.local_device_count()
    logging.info(
        'Device count: %d, host count: %d, local device count: %d',
        jax.device_count(),
        host_count,
        local_device_count,
    )
    flags.FLAGS.experimental_orbax_use_distributed_process_id = True
    if not multihost.is_runtime_to_distributed_ids_initialized():
      multihost.initialize_runtime_to_distributed_ids()

    global_mesh = integration_test_utils.create_global_mesh(
        num_slices=megascale_support.num_slices()
    )
    state = integration_test_utils.create_train_state(global_mesh)
    abstract_state = jax.tree.map(utils.to_shape_dtype_struct, state)
    del state
    shape_dtypes, _ = jax.tree.flatten(abstract_state)
    replica_id = multislice.process_replica_id(
        multihost.process_index(), global_mesh
    )

    def _get_single_replica_sharding(
        mesh: jax.sharding.Mesh,
        pspec: jax.sharding.PartitionSpec,
    ):
      slice_devices = np.asarray([global_mesh.devices[replica_id]])
      slice_mesh = jax.sharding.Mesh(slice_devices, mesh.axis_names)
      ss_sharding = jax.sharding.NamedSharding(slice_mesh, pspec)
      return ss_sharding

    single_replica_shardings = jax.tree.map(
        lambda arr: _get_single_replica_sharding(
            mesh=arr.sharding.mesh,
            pspec=arr.sharding.spec,
        ),
        abstract_state,
    )
    single_replica_shardings_tuple = jax.tree.flatten(single_replica_shardings)[
        0
    ]

    is_restoring_slice = multislice.in_replica(
        multihost.process_index(),
        global_mesh,
        replica_axis_index=0,
        replica_id=0,
    )

    if is_restoring_slice:

      @functools.partial(
          jax.jit,
          static_argnums=0,
          out_shardings=tuple(single_replica_shardings_tuple),
      )
      def create_ones(shape_dtype_tup):
        return jax.tree.map(
            lambda sd: jnp.ones(sd.shape, dtype=sd.dtype), shape_dtype_tup
        )

      ones_pytree = create_ones(tuple(shape_dtypes))
      in_tree = tuple(ones_pytree)

    else:

      @functools.partial(
          jax.jit,
          static_argnums=0,
          out_shardings=tuple(single_replica_shardings_tuple),
      )
      def create_zeros(shape_dtype_tup):
        return jax.tree.map(
            lambda sd: jnp.zeros(sd.shape, dtype=sd.dtype), shape_dtype_tup
        )

      zeros_pytree = create_zeros(tuple(shape_dtypes))
      in_tree = tuple(zeros_pytree)

    _, _ = multislice.broadcast_one_replica_to_all(
        in_tree,
        global_mesh,
        0,
        is_restoring_slice,
    )


if __name__ == '__main__':
  assert any(
      True for _ in filter(lambda x: x.startswith('--megascale'), sys.argv)
  )
  googletest.main()
