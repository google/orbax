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
import jax
import jax.numpy as jnp
from orbax.export import constants
from orbax.export.modules import obm_module


@jax.jit
def simple_add(params, inputs):
  return params + inputs


@jax.jit
def simple_subtract(params, inputs):
  return params - inputs


@jax.jit
def _linear(params, inputs):
  return jnp.dot(inputs, params['w']) + params['b']


class ObmModuleTest(parameterized.TestCase):

  def assertHasAttr(self, obj, attr_name):
    self.assertTrue(
        hasattr(obj, attr_name),
        msg=f'Object (of type {type(obj)}) does not have attribute {attr_name}',
    )

  def test_init_raises_error_when_input_poly_shape_symbol_vals_provided(self):
    # Test init raise error when `input_polymorphic_shape_symbol_values` is
    # provided but there is no `input_polymorphic_shape`
    with self.assertRaisesRegex(
        ValueError,
        r'`input_polymorphic_shape` is required when'
        r' `input_polymorphic_shape_symbol_values` is provided.',
    ):
      obm_module.ObmModule(
          params={
              'w': jnp.ones((2, 2), dtype=jnp.float32),
              'b': jnp.ones((2), dtype=jnp.float32),
          },
          apply_fn=_linear,
          # For polymorphic shape, input polymorphic shape could be (b, 2)
          # but set to None to test the error.
          input_polymorphic_shape=None,
          input_polymorphic_shape_symbol_values={'b': (1, 2)},
          jax2obm_kwargs=None,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='ok_model_pos_kwargs',
          input_polymorphic_shape={
              'simple_add': None,
              'simple_subtract': None,
          },
          input_polymorphic_shape_symbol_values=None,
          jax2obm_kwargs={constants.CHECKPOINT_PATH: 'checkpoint_path'},
          apply_fn_map={
              'simple_add': simple_add,
              'simple_subtract': simple_subtract,
          },
          expected_error=None,
      ),
      dict(
          testcase_name=(
              'ok_model_pos_kwargs_with_single_polymorphic_constraint'
          ),
          input_polymorphic_shape={
              'simple_add': None,
              'simple_subtract': None,
          },
          input_polymorphic_shape_symbol_values=None,
          jax2obm_kwargs={
              constants.CHECKPOINT_PATH: 'checkpoint_path',
              constants.POLYMORPHIC_CONSTRAINTS: ('mod(batch, 2) == 0',),
          },
          apply_fn_map={
              'simple_add': simple_add,
              'simple_subtract': simple_subtract,
          },
          expected_error=None,
      ),
      dict(
          testcase_name=(
              'ok_model_pos_kwargs_with_multiple_polymorphic_constraints'
          ),
          input_polymorphic_shape={
              'simple_add': None,
              'simple_subtract': None,
          },
          input_polymorphic_shape_symbol_values=None,
          jax2obm_kwargs={
              constants.CHECKPOINT_PATH: 'checkpoint_path',
              constants.POLYMORPHIC_CONSTRAINTS: {
                  'simple_add': ('mod(batch, 2) == 0',),
                  'simple_subtract': ('mod(batch, 2) == 0',),
              },
          },
          apply_fn_map={
              'simple_add': simple_add,
              'simple_subtract': simple_subtract,
          },
          expected_error=None,
      ),
      dict(
          testcase_name='error_input_poly_shape_not_matching_apply_fn_map',
          input_polymorphic_shape={
              'simple_add': (1, 2),
          },
          input_polymorphic_shape_symbol_values=None,
          jax2obm_kwargs={constants.CHECKPOINT_PATH: 'checkpoint_path'},
          apply_fn_map={
              'simple_add': simple_add,
              'simple_subtract': simple_subtract,
          },
          expected_error=(
              ValueError,
              (
                  r'The size of apply_fn_map:2 should be equal to the size of'
                  r' input_polymorphic_shape_map:1.'
              ),
          ),
      ),
      dict(
          testcase_name='error_input_poly_shape_is_not_mapping',
          input_polymorphic_shape=(1, 2),
          input_polymorphic_shape_symbol_values=None,
          jax2obm_kwargs={constants.CHECKPOINT_PATH: 'checkpoint_path'},
          apply_fn_map={
              'simple_add': simple_add,
              'simple_subtract': simple_subtract,
          },
          expected_error=(
              TypeError,
              (
                  r'When apply_fn is a mapping, input_polymorphic_shape must'
                  r' also be a mapping.'
              ),
          ),
      ),
      dict(
          testcase_name='error_constraints_not_matching_apply_fn_map',
          input_polymorphic_shape={
              'simple_add': None,
              'simple_subtract': None,
          },
          input_polymorphic_shape_symbol_values=None,
          jax2obm_kwargs={
              constants.CHECKPOINT_PATH: 'checkpoint_path',
              constants.POLYMORPHIC_CONSTRAINTS: {
                  'not_simple_add': ('mod(batch, 2) == 0',),
                  'simple_subtract': ('mod(batch, 2) == 0',),
              },
          },
          apply_fn_map={
              'simple_add': simple_add,
              'simple_subtract': simple_subtract,
          },
          expected_error=(
              ValueError,
              (
                  r'The key simple_add is not found in polymorphic_constraints:'
                  r" \{'not_simple_add': \('mod\(batch, 2\) == 0',\),"
                  r" 'simple_subtract': \('mod\(batch, 2\) == 0',\)\}"
              ),
          ),
      ),
      dict(
          testcase_name='error_constraints_size_not_matching_apply_fn_map_size',
          input_polymorphic_shape={
              'simple_add': None,
              'simple_subtract': None,
          },
          input_polymorphic_shape_symbol_values=None,
          jax2obm_kwargs={
              constants.CHECKPOINT_PATH: 'checkpoint_path',
              constants.POLYMORPHIC_CONSTRAINTS: {
                  'not_simple_add': ('mod(batch, 2) == 0',),
                  'simple_add': ('mod(batch, 2) == 0',),
                  'simple_subtract': ('mod(batch, 2) == 0',),
              },
          },
          apply_fn_map={
              'simple_add': simple_add,
              'simple_subtract': simple_subtract,
          },
          expected_error=(
              ValueError,
              (
                  r'The size of polymorphic_constraints:3 should be equal to'
                  r' the size of the apply_fn_map:2.'
              ),
          ),
      ),
      dict(
          testcase_name=(
              'error_constraints_not_mapping_when_apply_fn_is_mapping'
          ),
          input_polymorphic_shape={
              'simple_add': None,
              'simple_subtract': None,
          },
          input_polymorphic_shape_symbol_values=None,
          jax2obm_kwargs={
              constants.CHECKPOINT_PATH: 'checkpoint_path',
              constants.POLYMORPHIC_CONSTRAINTS: 2,
          },
          apply_fn_map={
              'simple_add': simple_add,
              'simple_subtract': simple_subtract,
          },
          expected_error=(
              TypeError,
              (
                  r'If not a Mapping, polymorphic_constraints needs to be a'
                  r" Sequence\. Got type: <class 'int'> \."
              ),
          ),
      ),
      dict(
          testcase_name='error_input_poly_shape_symbol_vals_not_mapping',
          input_polymorphic_shape={
              'simple_add': ('b, 1', 'b, 1'),
              'simple_subtract': ('b, 1', 'b, 1'),
          },
          input_polymorphic_shape_symbol_values=(1, 2, 3),
          jax2obm_kwargs={constants.CHECKPOINT_PATH: 'checkpoint_path'},
          apply_fn_map={
              'simple_add': simple_add,
              'simple_subtract': simple_subtract,
          },
          expected_error=(
              TypeError,
              (
                  r'When apply_fn is a mapping,'
                  r' input_polymorphic_shape_symbol_values must'
                  r' also be a mapping from key to List.'
              ),
          ),
      ),
      dict(
          testcase_name='error_input_poly_shape_symbol_vals_map_key_mismatch',
          input_polymorphic_shape={
              'simple_add': ('b, 1', 'b, 1'),
              'simple_subtract': ('b, 1', 'b, 1'),
          },
          input_polymorphic_shape_symbol_values={
              'mismatch_simple_add': {
                  'b': (1, 2),
              },
              'mismatch_simple_subtract': {
                  'b': (1, 2),
              },
          },
          jax2obm_kwargs={constants.CHECKPOINT_PATH: 'checkpoint_path'},
          apply_fn_map={
              'simple_add': simple_add,
              'simple_subtract': simple_subtract,
          },
          expected_error=(
              ValueError,
              (
                  r'The keys of apply_fn_map and'
                  r' input_polymorphic_shape_symbol_values_map must be the'
                  r' same.'
              ),
          ),
      ),
      dict(
          testcase_name='error_apply_fn_is_empty_map',
          input_polymorphic_shape={
              'simple_add': None,
              'simple_subtract': None,
          },
          input_polymorphic_shape_symbol_values=None,
          jax2obm_kwargs={
              constants.CHECKPOINT_PATH: 'checkpoint_path',
              constants.POLYMORPHIC_CONSTRAINTS: 2,
          },
          apply_fn_map={},
          expected_error=(
              ValueError,
              r'apply_fn_map is empty. Please provide a valid apply_fn_map.',
          ),
      ),
      dict(
          testcase_name='error_input_poly_shape_key_not_matching_apply_fn_map',
          input_polymorphic_shape={
              'simple_subtract': None,
          },
          input_polymorphic_shape_symbol_values=None,
          jax2obm_kwargs={
              constants.CHECKPOINT_PATH: 'checkpoint_path',
              constants.POLYMORPHIC_CONSTRAINTS: 2,
          },
          apply_fn_map={
              'simple_add': simple_add,
          },
          expected_error=(
              ValueError,
              r'The key simple_add is not found in input_polymorphic_shape.',
          ),
      ),
  )
  def test_obm_module_multiple_apply_fns(
      self,
      input_polymorphic_shape,
      input_polymorphic_shape_symbol_values,
      jax2obm_kwargs,
      apply_fn_map,
      expected_error,
  ):

    params_shape = (2, 5)
    params_dtype = jnp.dtype(jnp.float32)
    params = jnp.array(jnp.ones(params_shape, dtype=params_dtype))
    if expected_error is None:
      orbax_model_module = obm_module.ObmModule(
          params=params,
          apply_fn=apply_fn_map,
          input_polymorphic_shape=input_polymorphic_shape,
          input_polymorphic_shape_symbol_values=input_polymorphic_shape_symbol_values,
          jax2obm_kwargs=jax2obm_kwargs,
      )
      expected_weights_name = constants.DEFAULT_WEIGHTS_NAME
      self.assertEqual(
          orbax_model_module.orbax_export_module()[expected_weights_name],
          obm.ExternalValue(
              data=obm.manifest_pb2.UnstructuredData(
                  file_system_location=obm.manifest_pb2.FileSystemLocation(
                      string_path='checkpoint_path'
                  ),
                  mime_type=constants.ORBAX_CHECKPOINT_MIME_TYPE,
              ),
          ),
      )
    else:
      with self.assertRaisesRegex(expected_error[0], expected_error[1]):
        obm_module.ObmModule(
            params=params,
            apply_fn=apply_fn_map,
            input_polymorphic_shape=input_polymorphic_shape,
            input_polymorphic_shape_symbol_values=input_polymorphic_shape_symbol_values,
            jax2obm_kwargs=jax2obm_kwargs,
        )


if __name__ == '__main__':
  absltest.main()
