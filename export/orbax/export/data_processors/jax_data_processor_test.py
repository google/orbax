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
import re
from typing import Any

import jax
import jax.numpy as jnp
from orbax.export import obm_configs
from orbax.export.data_processors import jax_data_processor

from absl.testing import absltest
from .testing.pybase import parameterized
from .third_party.neptune.neptune_model._src.core import value
from .third_party.neptune.protos import manifest_pb2


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

  def test_prepare_with_polymorphic_shapes(self):
    def add(x: jax.Array) -> jax.Array:
      return x + 1.0

    processor = jax_data_processor.JaxDataProcessor(add, name='add')
    processor.prepare(
        jax.ShapeDtypeStruct(('b', 3), jnp.float32),
    )

    self.assertIsNotNone(processor.obm_function)
    self.assertIsNotNone(processor.input_signature)
    self.assertIsNotNone(processor.output_signature)

  def test_prepare_with_polymorphic_shapes_signatures(self):
    def add(x: jax.Array) -> jax.Array:
      return x + 1.0

    processor = jax_data_processor.JaxDataProcessor(add, name='add')
    processor.prepare(
        jax.ShapeDtypeStruct(('b', 3), jnp.float32),
    )

    self.assertEqual(
        processor.input_signature, jax.ShapeDtypeStruct(('b', 3), jnp.float32)
    )

    # Note: The underlying OBM Function signature uses None for dynamic
    # dimensions, regardless of the symbolic string provided by the user.
    out_spec = processor.output_signature
    self.assertEqual(list(out_spec.shape), [None, 3])  # pytype: disable=attribute-error

  def test_prepare_with_polymorphic_shapes_none(self):
    def add(x: jax.Array) -> jax.Array:
      return x + 1.0

    processor = jax_data_processor.JaxDataProcessor(add, name='add')
    processor.prepare(
        jax.ShapeDtypeStruct((None, 3), jnp.float32),
    )

    self.assertEqual(
        processor.input_signature, jax.ShapeDtypeStruct((None, 3), jnp.float32)
    )

    out_spec = processor.output_signature
    self.assertEqual(list(out_spec.shape), [None, 3])  # pytype: disable=attribute-error


class JaxShapeSpecGeneratorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='empty_shape', shape=(), expected='()'),
      dict(testcase_name='one_dim_none', shape=(None,), expected='(b,)'),
      dict(testcase_name='one_dim_int', shape=(4,), expected='(4,)'),
      dict(
          testcase_name='multi_dim_first_none',
          shape=(None, 4),
          expected='(b, 4)',
      ),
      dict(
          testcase_name='multi_dim_second_none',
          shape=(4, None),
          expected='(4, d_0)',
      ),
      dict(
          testcase_name='multi_dim_both_none',
          shape=(None, None),
          expected='(b, d_0)',
      ),
      dict(
          testcase_name='multi_dim_all_none',
          shape=(None, None, None, 256),
          expected='(b, d_0, d_1, 256)',
      ),
      dict(testcase_name='string_dims', shape=('foo', 4), expected='(foo, 4)'),
      dict(
          testcase_name='string_and_none',
          shape=('foo', None),
          expected='(foo, d_0)',
      ),
  )
  def test_jax_shape_spec_generator(self, expected, shape=None):
    spec = jax.ShapeDtypeStruct(shape, jnp.float32)
    generator = jax_data_processor._JaxShapeSpecGenerator()
    self.assertEqual(generator(spec), expected)

  def test_jax_shape_spec_generator_unsupported_type(self):
    spec = object()
    generator = jax_data_processor._JaxShapeSpecGenerator()
    with self.assertRaisesRegex(
        ValueError, f'Unsupported spec type: {re.escape(str(type(spec)))}'
    ):
      generator(spec)

  def test_jax_shape_spec_generator_multiple_calls(self):
    spec1 = jax.ShapeDtypeStruct((None, None), jnp.float32)
    spec2 = jax.ShapeDtypeStruct((None, None, 256), jnp.float32)
    generator = jax_data_processor._JaxShapeSpecGenerator()
    self.assertEqual(generator(spec1), '(b, d_0)')
    self.assertEqual(generator(spec2), '(b, d_1, 256)')

  def test_jax_shape_spec_generator_none_shape(self):
    class NoneShape:
      shape = None

    generator = jax_data_processor._JaxShapeSpecGenerator()
    self.assertEqual(generator(NoneShape()), '...')

  def test_jax_shape_spec_generator_uniterable_shape(self):
    class UniterableShape:
      shape = 4

    generator = jax_data_processor._JaxShapeSpecGenerator()
    with self.assertRaisesRegex(ValueError, 'spec.shape must be iterable'):
      generator(UniterableShape())


if __name__ == '__main__':
  googletest.main()
