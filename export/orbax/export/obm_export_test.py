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

import contextlib
import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import sharding
from jax.experimental import mesh_utils
import jax.numpy as jnp
from orbax.export import constants
from orbax.export import jax_module
from orbax.export import obm_configs
from orbax.export import obm_export
from orbax.export import oex_orchestration
from orbax.export import serving_config as osc
from orbax.export import utils
from orbax.export.oex_orchestration import oex_orchestration_pb2
import tensorflow as tf

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'


# TODO(b/363033166): Remove this function once TF isolation is done.
def _package_jax_module(m: jax_module.JaxModule):
  result = tf.Module()
  result.computation_module = m
  return result


def _assert(b):
  assert b


class PolymorphicShapeConcretizerTest(absltest.TestCase):

  def test_init_fails_for_invalid_symbolic_shapes_type(self):
    with self.assertRaisesRegex(
        ValueError, 'symbolic_shapes must be a Mapping'
    ):
      obm_export.PolymorphicShapeConcretizer(
          # Symbolic shapes should be a PyTree like the following, but it is
          # not, which causes the error.
          # symbolic_shapes={
          #     'input1': jax.ShapeDtypeStruct(shape=(1, 2), dtype=jnp.int32)
          # },
          symbolic_shapes=[jax.ShapeDtypeStruct(shape=(1, 2), dtype=jnp.int32)],
          symbol_to_values={'b': [1, 2]},
      )

  def test_init_fails_for_invalid_symbolic_shapes_value_type(self):
    with self.assertRaisesRegex(
        ValueError,
        'symbolic_shapes values must be jax.ShapeDtypeStruct instances',
    ):
      obm_export.PolymorphicShapeConcretizer(
          # Symbolic shape should be a PyTree with value being
          # jax.ShapeDtypeStruct, but it is not, which causes the error.
          # symbolic_shapes={
          #     'input1': jax.ShapeDtypeStruct(shape=(1, 2), dtype=jnp.int32)
          # },
          symbolic_shapes={'input1': (1, 2)},
          symbol_to_values={'b': [1, 2]},
      )

  def test_init_fails_for_invalid_symbol_to_values_type(self):
    with self.assertRaisesRegex(
        ValueError, 'symbol_to_values must be a Mapping'
    ):
      obm_export.PolymorphicShapeConcretizer(
          symbolic_shapes={
              'input1': jax.ShapeDtypeStruct(shape=(1, 2), dtype=jnp.int32)
          },
          # Symbol to values should be a PyTree like the following, but it is
          # not, which causes the error.
          # symbol_to_values={'b': [1, 2]},
          symbol_to_values=[1, 2],
      )

  def test_invalid_symbol_to_values_value_type(self):
    with self.assertRaisesRegex(
        ValueError, 'symbol_to_values values must be Sequences of ints'
    ):
      obm_export.PolymorphicShapeConcretizer(
          symbolic_shapes={
              'input1': jax.ShapeDtypeStruct(shape=('b',), dtype=jnp.int32)
          },
          # Symbol to values should be a PyTree like the following, but it is
          # not, which causes the error.
          # symbol_to_values={'b': [1, 2]},
          symbol_to_values={'b': '1,2'},
      )

  def test_concretize_succeeds(self):
    symbolic_shapes = {
        'params': jax.ShapeDtypeStruct(shape=(2, 2), dtype=jnp.float32),
        'input': jax.ShapeDtypeStruct(shape=('b', 'l'), dtype=jnp.float32),
    }
    symbol_to_values = {'b': [1, 2], 'l': [3, 4]}
    concretizer = obm_export.PolymorphicShapeConcretizer(
        symbolic_shapes, symbol_to_values
    )
    concrete_shapes_set = concretizer.concretize()
    expected_concrete_shapes_set = [
        {
            'params': jax.ShapeDtypeStruct(shape=(2, 2), dtype=jnp.float32),
            'input': jax.ShapeDtypeStruct(shape=(1, 3), dtype=jnp.float32),
        },
        {
            'params': jax.ShapeDtypeStruct(shape=(2, 2), dtype=jnp.float32),
            'input': jax.ShapeDtypeStruct(shape=(1, 4), dtype=jnp.float32),
        },
        {
            'params': jax.ShapeDtypeStruct(shape=(2, 2), dtype=jnp.float32),
            'input': jax.ShapeDtypeStruct(shape=(2, 3), dtype=jnp.float32),
        },
        {
            'params': jax.ShapeDtypeStruct(shape=(2, 2), dtype=jnp.float32),
            'input': jax.ShapeDtypeStruct(shape=(2, 4), dtype=jnp.float32),
        },
    ]
    self.assertEqual(concrete_shapes_set, expected_concrete_shapes_set)


class ObmExportTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._output_dir = self.create_tempdir().full_path

  def test_incorrect_export_version(self):
    pass


if __name__ == '__main__':
  absltest.main()
