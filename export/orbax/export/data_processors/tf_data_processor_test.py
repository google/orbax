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

"""Tests for TfDataProcessor."""

import orbax.experimental.model.core as obm
from orbax.export.data_processors import tf_data_processor
import tensorflow as tf
from absl.testing import absltest
from .util.task.python import error as google_error


class TfDataProcessorTest(googletest.TestCase):

  def test_concrete_function_raises_error_without_calling_prepare(self):
    processor = tf_data_processor.TfDataProcessor(lambda x: x)
    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        '`prepare()` must be called before accessing this property.',
    ):
      _ = processor.concrete_function

  def test_obm_function_raises_error_without_calling_prepare(self):
    processor = tf_data_processor.TfDataProcessor(lambda x: x)
    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        '`prepare()` must be called before accessing this property.',
    ):
      _ = processor.obm_function

  def test_input_signature_raises_error_without_calling_prepare(self):
    processor = tf_data_processor.TfDataProcessor(lambda x: x)
    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        '`prepare()` must be called before accessing this property.',
    ):
      _ = processor.input_signature

  def test_output_signature_raises_error_without_calling_prepare(self):
    processor = tf_data_processor.TfDataProcessor(lambda x: x)
    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        '`prepare()` must be called before accessing this property.',
    ):
      _ = processor.output_signature

  def test_prepare_fails_with_multiple_calls(self):
    processor = tf_data_processor.TfDataProcessor(lambda x: x)
    processor.prepare(
        'add',
        input_signature=(tf.TensorSpec([None, 3], tf.float32),),
    )
    with self.assertRaisesWithLiteralMatch(
        RuntimeError, '`prepare()` can only be called once.'
    ):
      processor.prepare(
          'add',
          input_signature=(tf.TensorSpec([None, 3], tf.float32),),
      )

  def test_prepare_succeeds(self):
    processor = tf_data_processor.TfDataProcessor(
        tf.function(lambda x, y: x + y)
    )
    processor.prepare(
        'add',
        input_signature=(
            tf.TensorSpec([None, 3], tf.float64),
            tf.TensorSpec([None, 3], tf.float64),
        ),
    )

    self.assertIsNotNone(processor.concrete_function)
    self.assertIsNotNone(processor.obm_function)
    self.assertEqual(
        processor.input_signature[0][0],
        obm.ShloTensorSpec(shape=(None, 3), dtype=obm.ShloDType.f64),
    )
    self.assertEqual(
        processor.input_signature[0][1],
        obm.ShloTensorSpec(shape=(None, 3), dtype=obm.ShloDType.f64),
    )
    self.assertEqual(
        processor.output_signature,
        obm.ShloTensorSpec(shape=(None, 3), dtype=obm.ShloDType.f64),
    )

  def test_prepare_polymorphic_function_with_default_input_signature(self):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 4], dtype=tf.float32),
            tf.TensorSpec(shape=[None, 4], dtype=tf.float32),
        ]
    )
    def preprocessor_callable(x, y):
      return x + y

    processor = tf_data_processor.TfDataProcessor(preprocessor_callable)
    processor.prepare(
        'add',
        input_signature=(
            tf.TensorSpec([None, 3], tf.float32),
            tf.TensorSpec([None, 3], tf.float32),
        ),
    )
    self.assertEqual(
        processor.input_signature[0][0],
        obm.ShloTensorSpec(shape=(None, 4), dtype=obm.ShloDType.f32),
    )
    self.assertEqual(
        processor.input_signature[0][1],
        obm.ShloTensorSpec(shape=(None, 4), dtype=obm.ShloDType.f32),
    )
    self.assertEqual(
        processor.output_signature,
        obm.ShloTensorSpec(shape=(None, 4), dtype=obm.ShloDType.f32),
    )

  def test_suppress_x64_output(self):
    processor = tf_data_processor.TfDataProcessor(
        tf.function(
            lambda x, y: tf.cast(x, tf.float64) + tf.cast(y, tf.float64)
        )
    )
    input_signature = (
        tf.TensorSpec([None, 3], tf.float32),
        tf.TensorSpec([None, 3], tf.float32),
    )

    # With suppress_x64_output=True, f64 output is suppressed to f32.
    processor.prepare('add_f64', input_signature, suppress_x64_output=True)
    self.assertEqual(
        processor.output_signature,
        obm.ShloTensorSpec(shape=(None, 3), dtype=obm.ShloDType.f32),
    )

  def test_convert_to_bfloat16(self):
    processor = tf_data_processor.TfDataProcessor(lambda x: 0.5 + x)
    processor.prepare(
        'preprocessor',
        input_signature=(tf.TensorSpec((), tf.float32)),
        bfloat16_options=converter_options_v2_pb2.ConverterOptionsV2(
            bfloat16_optimization_options=converter_options_v2_pb2.BFloat16OptimizationOptions(
                scope=converter_options_v2_pb2.BFloat16OptimizationOptions.ALL,
                skip_safety_checks=True,
            )
        ),
    )
    self.assertEqual(
        processor.output_signature[0],
        obm.ShloTensorSpec(shape=(), dtype=obm.ShloDType.bf16),
    )

  def test_bfloat16_convert_error(self):
    processor = tf_data_processor.TfDataProcessor(lambda x: 0.5 + x)
    with self.assertRaisesRegex(
        google_error.StatusNotOk,
        'Found bfloat16 ops in the model. The model may have been converted'
        ' before. It should not be converted again.',
    ):
      processor.prepare(
          'preprocessor',
          input_signature=(tf.TensorSpec((), tf.bfloat16)),
          bfloat16_options=converter_options_v2_pb2.ConverterOptionsV2(
              bfloat16_optimization_options=converter_options_v2_pb2.BFloat16OptimizationOptions(
                  scope=converter_options_v2_pb2.BFloat16OptimizationOptions.ALL,
              )
          ),
      )

  def test_prepare_with_shlo_bf16_inputs(self):
    processor = tf_data_processor.TfDataProcessor(lambda x: x)
    processor.prepare(
        'identity',
        input_signature=(
            obm.ShloTensorSpec(shape=(1,), dtype=obm.ShloDType.bf16),
        ),
    )
    self.assertEqual(
        processor.concrete_function.structured_input_signature[0][0].dtype,
        tf.bfloat16,
    )
    self.assertEqual(
        processor.input_signature[0][0],
        obm.ShloTensorSpec(shape=(1,), dtype=obm.ShloDType.bf16),
    )


if __name__ == '__main__':
  googletest.main()
