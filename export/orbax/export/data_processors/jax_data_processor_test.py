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

from collections.abc import Mapping
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
from orbax.experimental.model.core.protos import manifest_pb2
from orbax.experimental.model.core.python import value
from orbax.export import obm_configs
from orbax.export.data_processors import jax_data_processor

from absl.testing import absltest
from .testing.pybase import parameterized


class JaxDataProcessorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='obm_function', property_name='obm_function'),
      dict(testcase_name='input_signature', property_name='input_signature'),
      dict(testcase_name='output_signature', property_name='output_signature'),
  )
  def test_property_access_before_prepare_raises_error(self, property_name):
    processor = jax_data_processor.JaxDataProcessor(lambda x: x)
    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        '`prepare()` must be called before accessing this property.',
    ):
      _ = getattr(processor, property_name)

  def test_prepare_fails_with_multiple_calls(self):
    processor = jax_data_processor.JaxDataProcessor(
        lambda x: x, name='identity'
    )
    processor.prepare(
        jax.ShapeDtypeStruct((), jnp.float32),
    )
    with self.assertRaisesWithLiteralMatch(
        RuntimeError, '`prepare()` can only be called once.'
    ):
      processor.prepare(
          jax.ShapeDtypeStruct((), jnp.float32),
      )

  def test_prepare_succeeds(self):
    def add(args: tuple[jax.Array, jax.Array]) -> jax.Array:
      x, y = args
      return x + y

    processor = jax_data_processor.JaxDataProcessor(add, name='add')
    processor.prepare(
        (
            jax.ShapeDtypeStruct((2, 3), jnp.float32),
            jax.ShapeDtypeStruct((2, 3), jnp.float32),
        ),
    )

    self.assertIsNotNone(processor.obm_function)
    self.assertIsNotNone(processor.input_signature)
    self.assertIsNotNone(processor.output_signature)

  def test_prepare_with_params_attributes_are_set(self):
    def add_params(params: Mapping[str, Any], x: jax.Array) -> jax.Array:
      return params['w'] + x

    params = {'w': jnp.ones((2, 3), jnp.float32)}
    processor = jax_data_processor.JaxDataProcessor(
        add_params, name='add_params', params=params
    )
    processor.prepare(
        jax.ShapeDtypeStruct((2, 3), jnp.float32),
    )

    self.assertIsNotNone(processor.obm_function)
    self.assertIsNotNone(processor.save_fn)
    self.assertIsNotNone(processor.input_signature)
    self.assertIsNotNone(processor.output_signature)

  def test_save_fn_returns_external_value(self):
    def add_params(params: Mapping[str, Any], x: jax.Array) -> jax.Array:
      return params['w'] + x

    params = {'w': jnp.ones((2, 3), jnp.float32)}
    weights_name = 'params_add_params'
    processor = jax_data_processor.JaxDataProcessor(
        add_params,
        name='add_params',
        params=params,
        options=obm_configs.Jax2ObmOptions(
            checkpoint_path='processor',
            weights_name=weights_name,
        ),
    )
    processor.prepare(
        jax.ShapeDtypeStruct((2, 3), jnp.float32),
    )
    closure = processor.save_fn
    temp_dir = self.create_tempdir().full_path
    result = closure(temp_dir)

    with self.subTest(name='Result contains params_add_params'):
      self.assertIn(weights_name, result)
    supp = result[weights_name]

    with self.subTest(name='Supplemental is ExternalValue'):
      self.assertIsInstance(supp, value.ExternalValue)
    with self.subTest(name='Supplemental data is UnstructuredData'):
      self.assertIsInstance(supp.data, manifest_pb2.UnstructuredData)
    with self.subTest(name='File system location is correct'):
      self.assertEqual(
          supp.data.file_system_location.string_path, 'processor/add_params'
      )

  def test_save_fn_saves_checkpoint(self):
    def add_params(params: Mapping[str, Any], x: jax.Array) -> jax.Array:
      return params['w'] + x

    params = {'w': jnp.ones((2, 3), jnp.float32)}
    processor = jax_data_processor.JaxDataProcessor(
        add_params,
        name='add_params',
        params=params,
        options=obm_configs.Jax2ObmOptions(checkpoint_path='processor'),
    )
    processor.prepare(
        jax.ShapeDtypeStruct((2, 3), jnp.float32),
    )
    closure = processor.save_fn
    temp_dir = self.create_tempdir().full_path
    closure(temp_dir)
    checkpoint_path = pathlib.Path(temp_dir) / 'processor' / 'add_params'
    self.assertTrue(checkpoint_path.exists())

  def test_prepare_with_platforms_option(self):
    def add(x: jax.Array) -> jax.Array:
      return x + 1.0

    processor = jax_data_processor.JaxDataProcessor(
        add,
        name='add',
        options=obm_configs.Jax2ObmOptions(
            native_serialization_platforms=['cpu', 'tpu']
        ),
    )
    processor.prepare(
        jax.ShapeDtypeStruct((2, 3), jnp.float32),
    )

    self.assertIsNotNone(processor.obm_function)
    self.assertEqual(
        processor.obm_function.lowering_platforms,  # pytype: disable=attribute-error
        ('cpu', 'tpu'),
    )


if __name__ == '__main__':
  googletest.main()
