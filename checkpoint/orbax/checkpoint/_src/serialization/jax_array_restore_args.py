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

"""Arguments used when restoring with ArrayHandler."""

import dataclasses
from typing import Any
import jax
from jax.experimental import layout
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.serialization import types

if jax.__version_info__ >= (0, 6, 2):
  Format = layout.Format
else:
  Format = layout.Layout


@dataclasses.dataclass
class ArrayRestoreArgs(types.RestoreArgs):
  """Arguments used when restoring with ArrayHandler.

  restore_type:
    See parent class.
  mesh:
    The device mesh that the array should be restored as. Cannot be None.
  mesh_axes:
    The mesh_axes that the array should be restored as. Cannot be None.
  sharding:
   `jax.sharding.Sharding`, `ShardingMetadata`, or `Layout` object which takes
   precedence over mesh and mesh_axes if provided. Otherwise, mesh and mesh_axes
   will be used to construct a NamedSharding object OR `ShardingMetadata` which
   is an orbax representation of `jax.sharding.Sharding` that stores the same
   properties but does not require accessing real devices.
  global_shape: The global shape that the array should be restored into. If not
    provided, the shape will be restored as written. Presently, arbitrary shape
    transformations are not supported (for example, reshaping to different
    dimensions). Padding and truncating are supported. When the global_shape is
    greater than that of the saved array, 0's will be appended. If the
    global_shape is shorter than that of the saved array, excess elements will
    be dropped from the end of the array.
  shape: Interchangeable with global_shape.
  strict:
    True by default. If True, enforces that the target global shape and the
    origin global shape (as recorded by the saved array) are the same. If False,
    the returned array will be silently truncated or padded to fit the target
    global shape as necessary.
  """

  restore_type: Any | None = jax.Array
  mesh: jax.sharding.Mesh | None = None
  mesh_axes: jax.sharding.PartitionSpec | None = None
  # pyformat: disable
  sharding: jax.sharding.Sharding | sharding_metadata.ShardingMetadata | Format | None = (  # type: ignore[invalid-annotation]
      None
  )
  # pyformat: enable
  global_shape: tuple[int, ...] | None = None
  shape: tuple[int, ...] | None = None
  strict: bool = True

  def __post_init__(self):
    if self.shape is not None and self.global_shape is not None:
      if self.shape != self.global_shape:
        raise ValueError(
            'If `shape` and `global_shape` are both provided, they must match.'
        )
    elif self.shape is None:
      self.shape = self.global_shape
    elif self.global_shape is None:
      self.global_shape = self.shape


@dataclasses.dataclass
class SingleReplicaArrayRestoreArgs(ArrayRestoreArgs):
  """Arguments used when restoring with SingleReplicaArrayHandler.

  In case when training at scale loading checkpoint to all host may be
  very slow especially when checkpoint file is large. To mitigate this
  issue `SingleReplicaArrayHandler` suggests to read the checkpoint only
  on one replica hosts and do broadcasting which should significantly
  improve the training start time at scale.

  single_replica_sharding:
    jax.sharding.NamedSharding object which describes the single replica
    sharding to which current host belongs to.
  """

  single_replica_sharding: jax.sharding.NamedSharding | None = None
