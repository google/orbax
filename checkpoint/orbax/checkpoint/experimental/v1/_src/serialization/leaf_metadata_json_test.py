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

"""Tests `to_json_dict()` of the V1 array and NumPy leaf metadata."""

import json

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint.experimental.v1._src.serialization import (
    array_leaf_handler,
)
from orbax.checkpoint.experimental.v1._src.serialization import (
    numpy_leaf_handler,
)


def _named_sharding() -> sharding_metadata.NamedShardingMetadata:
  # Built directly: rendering must not need the devices that saved it.
  return sharding_metadata.NamedShardingMetadata(
      shape=np.array([2]),
      axis_names=['data'],
      partition_spec=('data',),
      device_mesh=sharding_metadata.DeviceMetadataMesh(
          mesh=[
              sharding_metadata.DeviceMetadata(id=0),
              sharding_metadata.DeviceMetadata(id=1),
          ]
      ),
  )


class LeafMetadataJsonTest(absltest.TestCase):

  def test_array_metadata_to_json_dict(self):
    metadata = array_leaf_handler.ArrayMetadata(
        shape=(8, 16),
        dtype=jnp.dtype(jnp.bfloat16),
        sharding_metadata=_named_sharding(),
        storage_metadata=value_metadata.StorageMetadata(
            chunk_shape=(4, 16), write_shape=(4, 16)
        ),
    )

    rendered = metadata.to_json_dict()

    self.assertEqual(
        rendered,
        {
            'value_type': 'jax.Array',
            'shape': [8, 16],
            'dtype': 'bfloat16',
            'sharding_metadata': {
                'sharding_type': 'NamedSharding',
                'shape': [2],
                'axis_names': ['data'],
                'partition_spec': ['data'],
                'device_mesh': {'mesh': [{'id': 0}, {'id': 1}]},
            },
            'storage_metadata': {
                'chunk_shape': [4, 16],
                'write_shape': [4, 16],
            },
        },
    )
    self.assertEqual(json.loads(json.dumps(rendered)), rendered)

  def test_array_metadata_without_sharding_or_storage(self):
    metadata = array_leaf_handler.ArrayMetadata(
        shape=(),
        dtype=np.dtype('int32'),
        sharding_metadata=None,
        storage_metadata=None,
    )
    self.assertEqual(
        metadata.to_json_dict(),
        {
            'value_type': 'jax.Array',
            'shape': [],
            'dtype': 'int32',
            'sharding_metadata': None,
            'storage_metadata': None,
        },
    )

  def test_numpy_metadata_to_json_dict(self):
    metadata = numpy_leaf_handler.NumpyMetadata(
        shape=(2, 3),
        dtype=np.dtype('float64'),
        storage_metadata=value_metadata.StorageMetadata(chunk_shape=(2, 3)),
    )
    self.assertEqual(
        metadata.to_json_dict(),
        {
            'value_type': 'np.ndarray',
            'shape': [2, 3],
            'dtype': 'float64',
            'storage_metadata': {'chunk_shape': [2, 3], 'write_shape': None},
        },
    )


if __name__ == '__main__':
  absltest.main()
