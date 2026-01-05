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

"""Utility functions for multi-slice benchmarks."""

from __future__ import annotations

from typing import Any

from absl import logging
from etils import epath
import jax
from orbax.checkpoint import v1 as ocp
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.testing.benchmarks.core import checkpoint_generation


def get_multi_slice_abstract_state(
    context: ocp.Context,
    global_mesh: jax.sharding.Mesh,
    *,
    reference_checkpoint_path: epath.Path,
    reference_sharding_path: epath.Path,
) -> Any:
  """Returns the abstract state for all replicas."""
  with ocp.Context(context=context):
    metadata = ocp.pytree_metadata(reference_checkpoint_path)
    # Abstract tree has shardings on a single replica.
    single_replica_abstract_state = (
        checkpoint_generation.get_abstract_state_from_sharding_config(
            reference_sharding_path,
            metadata.metadata,
            devices=multislice.replica_devices(
                global_mesh, replica_id=0, replica_axis_index=0
            ).tolist(),
        )
    )

    # Blow shardings up to all replicas.
    def _multi_replica_sharding(abstract_arr: jax.ShapeDtypeStruct):
      logging.info(
          "Original (single-replica) sharding: %s", abstract_arr.sharding
      )
      assert isinstance(abstract_arr.sharding, jax.sharding.NamedSharding)
      single_replica_mesh = abstract_arr.sharding.mesh
      single_replica_partition_spec = abstract_arr.sharding.spec
      multi_replica_sharding = jax.sharding.NamedSharding(
          jax.sharding.Mesh(
              devices=global_mesh.devices.reshape(
                  -1, *single_replica_mesh.devices.shape
              ),
              axis_names=["replica", *single_replica_mesh.axis_names],
          ),
          spec=jax.sharding.PartitionSpec(*single_replica_partition_spec),
      )
      logging.info("Multi-replica sharding: %s", multi_replica_sharding)
      return jax.ShapeDtypeStruct(
          shape=abstract_arr.shape,
          dtype=abstract_arr.dtype,
          sharding=multi_replica_sharding,
      )

    return jax.tree.map(
        _multi_replica_sharding,
        single_replica_abstract_state,
    )
