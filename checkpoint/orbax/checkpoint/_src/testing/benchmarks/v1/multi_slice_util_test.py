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

import json
import os

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import v1 as ocp
from orbax.checkpoint._src.testing.benchmarks.v1 import multi_slice_util


_REQUIRED_DEVICE_COUNT = 16


class MultiSliceUtilTest(parameterized.TestCase):

  def setUp(self):
    self._prev_xla_flags = os.environ.get('XLA_FLAGS')
    os.environ['XLA_FLAGS'] = (
        self._prev_xla_flags or ''
    ) + ' --xla_force_host_platform_device_count=16'
    super().setUp()
    if jax.local_device_count() != _REQUIRED_DEVICE_COUNT:
      self.skipTest(
          f'Test requires {_REQUIRED_DEVICE_COUNT} local devices, but only'
          f' {jax.local_device_count()} are available. Set XLA_FLAGS='
          f'"--xla_force_host_platform_device_count={_REQUIRED_DEVICE_COUNT}"'
          ' before JAX initializes.'
      )
    self.directory = epath.Path(self.create_tempdir().full_path)

  def tearDown(self):
    if self._prev_xla_flags is None:
      os.environ.pop('XLA_FLAGS', None)
    else:
      os.environ['XLA_FLAGS'] = self._prev_xla_flags
    super().tearDown()

  def test_get_multi_slice_abstract_state(self):
    # Setup real checkpoint and sharding config
    pytree = {'a': jnp.arange(32), 'b': {'c': jnp.ones((8, 8))}}
    ref_ckpt_path = self.directory / 'ref_ckpt'
    ocp.save_pytree(ref_ckpt_path, pytree)

    sharding_config = {
        'a': {
            'shape': [32],
            'dtype': 'int32',
            'sharding': {
                'mesh': {'shape': [4], 'axes': ['model']},
                'spec': ['model'],
            },
        },
        'b.c': {
            'shape': [8, 8],
            'dtype': 'float32',
            'sharding': {
                'mesh': {'shape': [4], 'axes': ['model']},
                'spec': [None, 'model'],
            },
        },
    }
    sharding_config_path = self.directory / 'sharding_config.json'
    sharding_config_path.write_text(json.dumps(sharding_config))
    global_mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((4, 4)), ('replica', 'model')
    )

    abstract_pytree = multi_slice_util.get_multi_slice_abstract_state(
        context=ocp.Context(),
        global_mesh=global_mesh,
        reference_checkpoint_path=ref_ckpt_path,
        reference_sharding_path=sharding_config_path,
    )
    self.assertEqual(
        {'replica': 4, 'model': 4}, abstract_pytree['a'].sharding.mesh.shape
    )
    self.assertEqual(
        jax.sharding.PartitionSpec('model'), abstract_pytree['a'].sharding.spec
    )
    self.assertEqual(
        {'replica': 4, 'model': 4},
        abstract_pytree['b']['c'].sharding.mesh.shape,
    )
    self.assertEqual(
        jax.sharding.PartitionSpec(None, 'model'),
        abstract_pytree['b']['c'].sharding.spec,
    )


if __name__ == '__main__':
  absltest.main()
