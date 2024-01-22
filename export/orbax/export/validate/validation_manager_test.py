# Copyright 2024 The Orbax Authors.
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

"""Tests for validation_manager."""
import logging

from absl.testing import parameterized
from jax.experimental import jax2tf
import jax.numpy as jnp
import numpy as np
from orbax.export.export_manager import ExportManager
from orbax.export.jax_module import JaxModule
from orbax.export.serving_config import ServingConfig
from orbax.export.validate import validation_manager
from orbax.export.validate.validation_job import ValidationSingleJobResult
from orbax.export.validate.validation_manager import ValidationManager
import tensorflow as tf


def _from_feature_dict(feature_dict):
  return feature_dict['feat']


def _add_output_name(outputs):
  return {'outputs': outputs}


def _linear(params, x, with_bias=False):
  y = x @ params['w']
  if with_bias:
    return y + params['b']
  return y


def convert_to_dict(list_output):
  results = {}
  for i, output in enumerate(list_output):
    results['output_' + str(i)] = output
  return results


def _make_tf_input_signature(inputs):

  def _make_one_array_signature(arg, name):
    return tf.TensorSpec(np.shape(arg), jax2tf.dtype_of_val(arg), name=name)

  input_signature = {}
  for name, value in inputs.items():
    input_signature[name] = _make_one_array_signature(value, name)

  return [input_signature]


class ValidationManagerTest(tf.test.TestCase, parameterized.TestCase):

  def assertAllValidationReportsPass(self, validation_reports):
    if not validation_reports:
      self.fail('validation_reports is empty Dict.')
    for key in validation_reports:
      if validation_reports[key].status.name != 'Pass':
        self.fail(
            f'serving signature `{key}` failed: {validation_reports[key]}')

  def setUp(self):
    super().setUp()
    self._output_dir = self.create_tempdir().full_path

  def test_list_output_format(self):
    params = {'bias': jnp.array(0)}
    batch_input = list(np.arange(16).reshape((16, 1)).astype(np.int32))
    batch_input = [{'feature1': i} for i in batch_input]

    # test if output of JaxModule is List
    serving_configs = [
        ServingConfig(
            'output_is_list',
            input_signature=_make_tf_input_signature(batch_input[0]),
        ),
    ]

    def apply_fn(p, x):
      y = x['feature1'] + p['bias']
      results = []
      for i in range(12):
        results.append(y + i)
      return results

    jax_module = JaxModule(params, apply_fn)
    em = ExportManager(
        jax_module,
        serving_configs,
    )
    em.save(self._output_dir)
    validation_mgr = ValidationManager(jax_module, serving_configs, batch_input)
    loaded_model = tf.saved_model.load(self._output_dir)
    logging.info(
        'loaded_model info = %s',
        loaded_model.signatures['output_is_list'].pretty_printed_signature(),
    )
    validation_reports = validation_mgr.validate(loaded_model)
    self.assertAllValidationReportsPass(validation_reports)

  def test_dict_output_format(self):
    params = {'bias': jnp.array(0)}
    batch_input = list(np.arange(16).reshape((16, 1)).astype(np.int32))
    batch_input = [{'feature1': i} for i in batch_input]

    def apply_fn(p, x):
      y = x['feature1'] + p['bias']
      results = []
      for i in range(12):
        results.append(y + i)
      return results

    jax_module = JaxModule(params, apply_fn)
    serving_configs = [
        ServingConfig(
            'output_is_dict',
            input_signature=_make_tf_input_signature(batch_input[0]),
            tf_postprocessor=convert_to_dict,
        ),
    ]
    em = ExportManager(
        jax_module,
        serving_configs,
    )
    em.save(self._output_dir)
    validation_mgr = ValidationManager(jax_module, serving_configs, batch_input)
    loaded_model = tf.saved_model.load(self._output_dir)
    validation_reports = validation_mgr.validate(loaded_model)
    self.assertAllValidationReportsPass(validation_reports)

  def test_basic(self):

    serving_configs = [
        ServingConfig(
            'without_processors',
            input_signature=[
                {
                    'feature1': tf.TensorSpec(
                        (None,), tf.dtypes.int32, 'feature1'
                    )
                }
            ],
        ),
    ]
    params = {'bias': jnp.array(0)}

    def apply_fn(p, x):
      y = x['feature1'] + p['bias']
      results = {}
      for i in range(9):
        results[str(i)] = y + i
      return results

    jax_module = JaxModule(params, apply_fn, input_polymorphic_shape='b, ...')
    batch_input = list(np.arange(16).reshape((16, 1)).astype(np.int32))
    batch_input = [{'feature1': i} for i in batch_input]
    em = ExportManager(
        jax_module,
        serving_configs,
    )
    em.save(self._output_dir)
    validation_mgr = ValidationManager(jax_module, serving_configs, batch_input)
    loaded_model = tf.saved_model.load(self._output_dir)
    validation_reports = validation_mgr.validate(loaded_model)
    self.assertAllValidationReportsPass(validation_reports)

  @parameterized.named_parameters([
      dict(
          testcase_name='without_xprof',
          serving_configs=[
              ServingConfig(
                  'with_processors',
                  input_signature=[
                      {'feat': tf.TensorSpec((None,), tf.dtypes.int32, 'feat')}
                  ],
                  tf_preprocessor=_from_feature_dict,
                  tf_postprocessor=_add_output_name,
              ),
          ],
          with_xprof=False,
      ),
  ])
  def test_perf(self, serving_configs, with_xprof):
    params = {'bias': jnp.array(1)}
    apply_fn = lambda p, x: x + p['bias']
    jax_module = JaxModule(params, apply_fn, input_polymorphic_shape='b, ...')
    batch_input = list(np.arange(128).reshape((16, 8)).astype(np.int32))
    batch_input = [{'feat': i} for i in batch_input]
    em = ExportManager(
        jax_module,
        serving_configs,
    )
    em.save(self._output_dir)
    validation_mgr = ValidationManager(jax_module, serving_configs, batch_input)
    loaded_model = tf.saved_model.load(self._output_dir)
    validation_reports = validation_mgr.validate(loaded_model, with_xprof)
    self.assertAllValidationReportsPass(validation_reports)

  @parameterized.named_parameters(
      dict(testcase_name='case1', flat_dict={'a': 1}),
      dict(testcase_name='case2', flat_dict={'b': np.zeros((1, 3))}),
      dict(testcase_name='case3', flat_dict={'c': tf.zeros((3, 1))}),
      dict(testcase_name='case4', flat_dict={'d': {1, 2, 3}}),  # python set
      dict(testcase_name='case5', flat_dict={'f': 'one'}),  # str
      dict(
          testcase_name='case6',
          flat_dict={'g': tf.convert_to_tensor(np.array(1.0))}),
  )
  def test_flat_dict_check(self, flat_dict):
    is_flat_dict = validation_manager._is_flat_dict
    self.assertTrue(is_flat_dict(flat_dict))

  @parameterized.named_parameters(
      dict(testcase_name='case1', flat_list=[1]),
      dict(testcase_name='case2', flat_list=['b', np.zeros((1, 3))]),
  )
  def test_flat_sequence_check(self, flat_list):
    is_flat_sequence = validation_manager._is_flat_sequence
    self.assertTrue(is_flat_sequence(flat_list))

  @parameterized.named_parameters(
      dict(testcase_name='case1', nested_dict={'a': {
          'b': 1
      }}),
      dict(testcase_name='case2', nested_dict={'a': [1, 2, 3]}),
      dict(testcase_name='case3', nested_dict={'a': [1]}),
      dict(testcase_name='case4', nested_dict={'a': {
          'b': 1
      }}),
      dict(testcase_name='case5', nested_dict=[1, 2, 3]),
  )
  def test_nested_dict_check(self, nested_dict):
    is_flat_dict = validation_manager._is_flat_dict
    self.assertFalse(is_flat_dict(nested_dict))

  def test_check_input(self):
    flat_dict_inputs = {
        'a': 1,
        'b': np.zeros((1, 3)),
        'c': tf.zeros((3, 1)),
        'd': {1, 2, 3},  #  set
        'f': 'one',
    }
    batch_valid_inputs = [flat_dict_inputs] * 30
    ValidationManager.check_input(batch_valid_inputs, batch_mode=True)

  def test_check_output(self):
    baseline_result = ValidationSingleJobResult(
        outputs=[{
            '1': {
                '2': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 2, 3, 4, 5, 6])
            }
        }],
        latencies=[88.0, 88.0, 90.0, 91.0, 88.5, 88.7, 87.0],
        xprof_url='N/A',
        metadata={})

    candidate_result = ValidationSingleJobResult(
        outputs=[{
            '1': {
                '2': np.array([0.2, 0.4, 0.6, 0.8, 1.0, 0.6, 2, 4, 6, 8, 5, 6])
            }
        }],
        latencies=[88.0, 88.0, 90.0, 91.0, 88.5, 88.7, 87.0],
        xprof_url='N/A',
        metadata={})
    with self.assertRaisesRegex(
        ValueError,
        'Currently ValidationReport only accept  flat dict outputs'):
      ValidationManager.check_output(baseline_result, candidate_result)

    baseline_result = ValidationSingleJobResult(
        outputs=[{
            '1': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 2, 3, 4, 5, 6])
        }],
        latencies=[88.0, 88.0, 90.0, 91.0, 88.5, 88.7, 87.0],
        xprof_url='N/A',
        metadata={})

    candidate_result = ValidationSingleJobResult(
        outputs=[{
            '1': np.array([0.2, 0.4, 0.6, 0.8, 1.0, 0.6, 1, 2, 3, 4, 5, 6]),
            '2': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 2, 3, 4, 5, 6]),
        }],
        latencies=[88.0, 88.0, 90.0, 91.0, 88.5, 88.7, 87.0],
        xprof_url='N/A',
        metadata={})

    with self.assertRaisesRegex(
        ValueError, 'baseline and candidate has different output length'):
      ValidationManager.check_output(baseline_result, candidate_result)


if __name__ == '__main__':
  tf.test.main()
