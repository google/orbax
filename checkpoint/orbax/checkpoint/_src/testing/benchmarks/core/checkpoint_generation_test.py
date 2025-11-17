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

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.checkpointers import checkpointer
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.testing.benchmarks.core import checkpoint_generation
from orbax.checkpoint._src.testing.benchmarks.core import configs


class CheckpointGenerationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='jax_array',
          spec={
              'x': {
                  'dtype': 'float32',
                  'shape': [10, 10],
              }
          },
          expected_type=jax.Array,
          expected_shape=(10, 10),
          expected_dtype=jnp.float32,
      ),
  )
  def test_generate_checkpoint_array_types_succeeds(
      self,
      spec,
      expected_type,
      expected_shape,
      expected_dtype,
  ):
    config = configs.CheckpointConfig(spec=spec)
    pytree = checkpoint_generation.generate_checkpoint(config)

    self.assertIn('x', pytree)
    item = pytree['x']
    self.assertIsInstance(item, expected_type)
    self.assertEqual(item.shape, expected_shape)
    self.assertEqual(item.dtype, expected_dtype)

  @parameterized.named_parameters(
      dict(
          testcase_name='int',
          spec={'x': 'int'},
          expected_type=int,
          expected_value=0,
      ),
      dict(
          testcase_name='str',
          spec={'x': 'str'},
          expected_type=str,
          expected_value='default_string',
      ),
  )
  def test_generate_checkpoint_primitive_types_succeeds(
      self,
      spec,
      expected_type,
      expected_value,
  ):
    config = configs.CheckpointConfig(spec=spec)
    pytree = checkpoint_generation.generate_checkpoint(config)

    self.assertIn('x', pytree)
    item = pytree['x']
    self.assertIsInstance(item, expected_type)
    self.assertEqual(item, expected_value)

  def test_generate_checkpoint_unsupported_string_spec_raises_error(self):
    config = configs.CheckpointConfig(spec={'x': 'unsupported_type'})

    with self.assertRaisesRegex(ValueError, 'Unsupported spec string'):
      checkpoint_generation.generate_checkpoint(config)

  def test_generate_checkpoint_unsupported_spec_type_raises_error(self):
    config = configs.CheckpointConfig(spec={'x': 123})

    with self.assertRaisesRegex(ValueError, 'Unsupported spec type'):
      checkpoint_generation.generate_checkpoint(config)

  def test_generate_checkpoint_array_values_with_no_mesh_succeeds(self):
    spec = {
        'a': {
            'dtype': 'int32',
            'shape': [4, 5],
        }
    }
    init_random_seed = 0
    config = configs.CheckpointConfig(spec=spec, random_seed=init_random_seed)
    np.random.seed(init_random_seed)
    expected_array = jnp.asarray(
        np.random.normal(
            size=spec['a']['shape'], scale=np.prod(spec['a']['shape'])
        ).astype(np.int32)
    )
    pytree = checkpoint_generation.generate_checkpoint(config)

    np.testing.assert_array_equal(np.array(pytree['a']), expected_array)

  def test_generate_checkpoint_with_sharding(self):
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices), 'data')
    spec = {
        'x': {
            'dtype': 'float32',
            'shape': [16, 32],
            'sharding': ['data'],
        }
    }
    init_random_seed = 0
    config = configs.CheckpointConfig(spec=spec, random_seed=init_random_seed)
    expected_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('data')
    )
    np.random.seed(init_random_seed)
    expected_array = np.random.normal(
        size=spec['x']['shape'], scale=np.prod(spec['x']['shape'])
    ).astype(dtype=np.float32)

    pytree = checkpoint_generation.generate_checkpoint(config, mesh=mesh)

    self.assertIn('x', pytree)
    item = pytree['x']
    self.assertIsInstance(item, jax.Array)
    self.assertEqual(item.shape, (16, 32))
    self.assertEqual(item.dtype, jnp.float32)
    self.assertEqual(item.sharding, expected_sharding)
    np.testing.assert_array_equal(np.array(item), expected_array)

  def test_generate_checkpoint_with_replica_sharding(self):
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices).reshape(len(devices)), ('data',))
    spec = {
        'x': {
            'dtype': 'int32',
            'shape': [16, 32],
        }
    }
    init_random_seed = 0
    config = configs.CheckpointConfig(spec=spec, random_seed=init_random_seed)
    np.random.seed(init_random_seed)
    expected_array = np.random.normal(
        size=spec['x']['shape'], scale=np.prod(spec['x']['shape'])
    ).astype(dtype=np.int32)
    expected_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    pytree = checkpoint_generation.generate_checkpoint(config, mesh=mesh)

    self.assertIn('x', pytree)
    item = pytree['x']
    self.assertEqual(item.shape, (16, 32))
    self.assertEqual(item.sharding, expected_sharding)
    np.testing.assert_array_equal(np.array(item), expected_array)

  def test_generate_checkpoint_sharding_no_mesh_raises_error(self):
    spec = {
        'x': {
            'dtype': 'float32',
            'shape': [10, 10],
            'sharding': ['data', None],
        }
    }
    config = configs.CheckpointConfig(spec=spec)

    with self.assertRaisesRegex(
        ValueError, 'Mesh is required when sharding spec is provided.'
    ):
      checkpoint_generation.generate_checkpoint(config, mesh=None)

  def test_load_checkpoint_succeeds(self):
    temp_dir = self.create_tempdir().full_path
    config = configs.CheckpointConfig(
        spec={
            'params': {'dtype': 'float32', 'shape': [2, 2]},
        }
    )
    checkpoint_data = checkpoint_generation.generate_checkpoint(config)
    with checkpointer.Checkpointer(
        pytree_checkpoint_handler.PyTreeCheckpointHandler()
    ) as ckptr:
      ckptr.save(epath.Path(temp_dir) / 'ckpt', checkpoint_data)

    loaded_data = checkpoint_generation.load_checkpoint(
        str(epath.Path(temp_dir) / 'ckpt')
    )

    self.assertKeysEqual(loaded_data.keys(), checkpoint_data.keys())
    self.assertIsInstance(loaded_data['params'], jax.Array)
    self.assertEqual(loaded_data['params'].shape, (2, 2))
    self.assertEqual(loaded_data['params'].dtype, jnp.float32)

  def assertKeysEqual(self, a, b):
    self.assertEqual(set(a), set(b))


if __name__ == '__main__':
  absltest.main()
