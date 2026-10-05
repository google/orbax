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

import os

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from orbax.export import utils
import tensorflow as tf


TensorSpecWithDefault = utils.TensorSpecWithDefault


@tf.function(
    input_signature=[tf.TensorSpec((None, 2), tf.float32, name='a')],
)
def _tf_simple_pre_processor(a: tf.Tensor) -> tf.Tensor:
  """A simple pre-processor function."""
  return a + 1


var = tf.Variable(100.0)


@tf.function(input_signature=[tf.TensorSpec((8, 2), tf.float32, name='a')])
def _tf_pre_processor(a: tf.Tensor) -> tf.Tensor:
  vocab = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          keys=[
              'v',
          ],
          values=[
              200.0,
          ],
      ),
      default_value=10000.0,
  )
  return tf.cast(
      tf.cast(a**2, tf.float32) + var + vocab.lookup(tf.constant('v')),
      tf.float32,
  )


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.float32, name='float_input'),
        tf.TensorSpec(shape=(), dtype=tf.int8, name='int_input'),
        tf.TensorSpec(shape=(), dtype=tf.string, name='string_input'),
        tf.TensorSpec(
            shape=[None], dtype=tf.int32, name='array_input'
        ),  # None allows for variable-length 1D arrays
    ]
)
def process_mixed_types_with_signature(
    float_val, int_val, string_val, array_val
):
  """A TensorFlow function that processes inputs of various dtypes.

  constrained by a specific input signature.

  Args:
    float_val: A float32 tensor.
    int_val: An int8 tensor.
    string_val: A string tensor.
    array_val: An int32 tensor.

  Returns:
    A dictionary containing processed tensors.
  """
  # The operations inside the function remain the same.
  float_processed = tf.square(float_val)
  int_processed = tf.cast(int_val, tf.int32) * 10
  string_processed = tf.strings.join(['Input string was: ', string_val])
  array_processed = tf.reduce_sum(array_val)

  # tf.print is used for printing values inside a tf.function
  tf.print('--- Inside tf.function (with signature) ---')
  tf.print('Processed float:', float_processed)
  tf.print('Processed int:', int_processed)
  tf.print('Processed string:', string_processed)
  tf.print('Processed array sum:', array_processed)
  tf.print('-------------------------------------------')

  return {
      'float_squared': float_processed,
      'int_multiplied': int_processed,
      'string_joined': string_processed,
      'array_sum': array_processed,
  }


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    base_path = os.path.dirname(os.path.abspath(__file__))
    self._testdata_dir = os.path.join(base_path, 'testdata')

  def _get_testdata_path(self, filename: str) -> str:
    """Returns the full path to a file in the testdata directory."""
    return os.path.join(self._testdata_dir, filename)

  def test_with_default_args(self):
    input_signature = [
        TensorSpecWithDefault(
            tf.TensorSpec([None], tf.int32),
            np.asarray([1, 2]),
        )
    ]
    tf_f = utils.with_default_args(
        tf.reduce_sum,
        input_signature,
    )
    self.assertEqual(tf_f(), 3)

  def test_bad_order(self):
    input_signature = [
        TensorSpecWithDefault(
            tf.TensorSpec([None], tf.int32),
            np.asarray([1, 2]),
        ),
        tf.TensorSpec([None], tf.int32),
    ]
    with self.assertRaisesRegex(
        ValueError,
        'non-default argument follows default argument',
    ):
      utils.with_default_args(lambda x, y: x + y, input_signature)

  def test_missing_default(self):
    input_signature = [[
        TensorSpecWithDefault(
            tf.TensorSpec([None], tf.int32),
            np.asarray([1, 2]),
        ),
        tf.TensorSpec([None], tf.int32),
    ]]
    with self.assertRaisesRegex(
        ValueError,
        'TensorSpecWithDefault must be defined for each tensor in the structure'
        ' for the Python arg',
    ):
      utils.with_default_args(lambda x: x[0] + x[1], input_signature)

  def test_with_default_args_nested(self):
    def f(required_arg, optional_args):
      return (
          required_arg
          + optional_args['foo']
          + optional_args['bar'][0]
          + optional_args['bar'][1]
      )

    input_signature = (
        tf.TensorSpec([2], tf.int32),
        dict(
            foo=TensorSpecWithDefault(
                tf.TensorSpec([2], tf.int32),
                np.asarray([0, 1]),
            ),
            bar=[
                TensorSpecWithDefault(
                    tf.TensorSpec([2], tf.int32),
                    np.asarray([2, 3]),
                ),
                TensorSpecWithDefault(
                    tf.TensorSpec([2], tf.int32),
                    np.asarray([4, 5]),
                ),
            ],
        ),
    )

    tf_f = utils.with_default_args(f, input_signature)
    self.assertAllEqual(tf_f(np.asarray([6, 7])), np.asarray([12, 16]))

  def test_callable_signatures_from_saved_model(self):

    @tf.function(input_signature=[tf.TensorSpec((), tf.float32, name='y')])
    def tf_f(x=tf.constant(1.0, tf.float32)):
      return {'output': x + 1}

    export_dir = self.get_temp_dir()
    tf.saved_model.save(tf.Module(), export_dir, signatures={'f': tf_f})
    loaded = utils.CallableSignatures.from_saved_model(export_dir, ['serve'])
    self.assertAllEqual(loaded.signatures['f'](), {'output': 2})

    # Not like `tf.saved_model.load`, the loaded function by `from_saved_model`
    # take kwargs while the key is the input signature tensor name.
    self.assertAllEqual(
        loaded.signatures['f'](y=tf.constant(2.0, tf.float32)), {'output': 3}
    )

  def test_make_auto_batching_function_simple(self):
    input_signature = (
        tf.TensorSpec([None], tf.int32, name='primary'),
        TensorSpecWithDefault(
            tf.TensorSpec([None], tf.int32, name='optional'), [1]
        ),
    )
    batching_fn = utils.make_auto_batching_function(input_signature)
    self.assertAllEqual(
        batching_fn(tf.constant([0, 0]), tf.constant([1])),
        (np.array([0, 0]), np.array([1, 1])),
    )
    self.assertAllEqual(
        batching_fn(tf.constant([0, 0]), tf.constant([1, 2])),
        (np.array([0, 0]), np.array([1, 2])),
    )

  def test_make_auto_batching_function_nested(self):
    input_signature = (
        tf.TensorSpec([None], tf.int32, name='primary_1'),
        TensorSpecWithDefault(
            tf.TensorSpec([None], tf.int32, name='primary_2'),
            default_val=[1],
            is_primary=True,
        ),
        {
            'optional_1': TensorSpecWithDefault(
                tf.TensorSpec([None], tf.int32, name='optional_1'),
                default_val=[2],
            ),
            'optional_2': TensorSpecWithDefault(
                tf.TensorSpec([None], tf.int32, name='optional_1'),
                default_val=[3],
            ),
        },
    )
    tf_fn = utils.make_auto_batching_function(input_signature)

    tf.nest.map_structure(
        self.assertAllEqual,
        # Non-primary inputs use default values: batch size = 1.
        tf_fn([0, 0], [1, 1]),
        (
            np.array([0, 0]),
            np.array([1, 1]),
            {'optional_1': np.array([2, 2]), 'optional_2': np.array([3, 3])},
        ),
    )

    tf.nest.map_structure(
        self.assertAllEqual,
        # Non-primary inputs batch size = 2.
        tf_fn([0, 0], [1, 1], {'optional_1': [2, 2], 'optional_2': [3, 3]}),
        (
            np.array([0, 0]),
            np.array([1, 1]),
            {'optional_1': np.array([2, 2]), 'optional_2': np.array([3, 3])},
        ),
    )

  def test_make_auto_batching_function_error(self):
    input_signature_no_primary = (
        TensorSpecWithDefault(tf.TensorSpec([None], tf.int32), default_val=[1]),
    )
    with self.assertRaisesRegex(ValueError, 'No primary input tensors'):
      utils.make_auto_batching_function(input_signature_no_primary)

  def test_make_auto_batching_function_runtime_error(self):
    input_signature = (
        tf.TensorSpec([None], tf.int32, name='primary_1'),
        tf.TensorSpec([None], tf.int32, name='primary_2'),
        TensorSpecWithDefault(
            tf.TensorSpec([None], tf.int32, 'optional_1'), default_val=[1]
        ),
    )
    batching_fn = utils.make_auto_batching_function(input_signature)

    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        'All primary input tensors must have the same batch size',
    ):
      batching_fn([1, 2], [3])

    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        'The batch size .* must be 1 or the same as that of the primary'
        ' tensors',
    ):
      batching_fn([1, 2, 3], [4, 5, 6], [7, 8])

  def test_to_bfloat16(self):
    x = {
        'a': np.array([1.0, 2.0], dtype=np.float32),
        'b': tf.constant(['hello', 'world'], dtype=tf.string),
        'c': 3.0,
        'd': jax.ShapeDtypeStruct((2, 2), jnp.float32),
    }
    y = utils.to_bfloat16(x)
    self.assertEqual(y['a'].dtype, jnp.bfloat16)
    self.assertEqual(y['b'].dtype, tf.string)
    self.assertAllEqual(y['b'], x['b'])
    self.assertIsInstance(y['c'], float)
    self.assertEqual(y['d'].dtype, jnp.bfloat16)

if __name__ == '__main__':
  tf.test.main()
