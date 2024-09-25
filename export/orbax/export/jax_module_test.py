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

from absl.testing import parameterized
import jax
import jax.numpy as jnp
from orbax import export as obx_export
from orbax.export import constants
import tensorflow as tf


DEFAULT_METHOD_KEY = constants.DEFAULT_METHOD_KEY
JaxModule = obx_export.JaxModule


class JaxModuleTest(tf.test.TestCase, parameterized.TestCase):
  def test_jax_module_orbax_model_unimplemented_methods(self):
    def linear(params, x):
      return params['w'] @ x + params['b']

    key_w, key_b = jax.random.split(jax.random.PRNGKey(1234), 2)
    params = {
        'w': jax.random.normal(key_w, shape=(8, 8)),
        'b': jax.random.normal(key_b, shape=(8, 1)),
    }
    j_module = JaxModule(
        params=params,
        apply_fn={'linear': linear},
        export_version=constants.ExportModelType.ORBAX_MODEL,
    )

    self.assertEqual(
        j_module._export_version,
        constants.ExportModelType.ORBAX_MODEL,
    )

    # None of the obm_module methods are implemnted. When the export version is
    # ORBAX_MODEL, the obm_module methods should raise a NotImplementedError.
    with self.assertRaises(NotImplementedError):
      j_module.apply_fn_map  # pylint: disable=pointless-statement

    with self.assertRaises(NotImplementedError):
      j_module.model_params  # pylint: disable=pointless-statement

    with self.assertRaises(NotImplementedError):
      j_module.methods  # pylint: disable=pointless-statement

    with self.assertRaises(TypeError):
      j_module.with_gradient  # pylint: disable=pointless-statement

    with self.assertRaises(TypeError):
      j_module.obm_module_to_jax_exported_map({'x': [1, 2, 3]})

  # Several functions are not supported for ORBAX_MODEL export.
  # This test ensures that the JaxModule raises a TypeError when these
  # functions are called.
  def test_jax_module_orbax_model_unsupported_methods(self):
    def linear(params, x):
      return params['w'] @ x + params['b']

    key_w, key_b = jax.random.split(jax.random.PRNGKey(1234), 2)
    params = {
        'w': jax.random.normal(key_w, shape=(8, 8)),
        'b': jax.random.normal(key_b, shape=(8, 1)),
    }
    j_module = JaxModule(
        params=params,
        apply_fn={'linear': linear},
        export_version=constants.ExportModelType.ORBAX_MODEL,
    )

    self.assertEqual(
        j_module._export_version,
        constants.ExportModelType.ORBAX_MODEL,
    )

    with self.assertRaises(TypeError):
      j_module.update_variables(
          {'w': jax.random.normal(key_w, shape=(8, 8), dtype=jnp.float32)}
      )

    with self.assertRaises(TypeError):
      j_module.jax2tf_kwargs_map  # pylint: disable=pointless-statement

    with self.assertRaises(TypeError):
      j_module.input_polymorphic_shape_map  # pylint: disable=pointless-statement

  def test_jax_module_default_export_version(self):
    j_module = JaxModule(
        params={'w': jnp.array([1, 2, 3])},
        apply_fn=lambda params, x: params['w'] @ x,
    )
    self.assertEqual(
        j_module._export_version,
        constants.ExportModelType.TF_SAVEDMODEL,
    )

  def test_jax_module_export_version_tf_savedmodel(self):
    def linear(params, x):
      return params['w'] @ x + params['b']

    key_w, key_b, key_x = jax.random.split(jax.random.PRNGKey(1234), 3)
    params = {
        'w': jax.random.normal(key_w, shape=(8, 8)),
        'b': jax.random.normal(key_b, shape=(8, 1)),
    }
    jax_module = JaxModule(
        params=params,
        apply_fn=linear,
        jax2tf_kwargs={'with_gradient': True},
        trainable={'w': True, 'b': True},
    )
    self.assertEqual(
        jax_module._export_version,
        constants.ExportModelType.TF_SAVEDMODEL,
    )

    self.assertEqual(
        jax_module.apply_fn_map,
        {DEFAULT_METHOD_KEY: linear},
    )

    self.assertEqual(
        jax_module.model_params,
        params,
    )
    x = jax.random.normal(key_x, shape=(8, 1))
    self.assertAllClose(
        jax_module.methods[DEFAULT_METHOD_KEY](x),
        jax_module.jax_methods[DEFAULT_METHOD_KEY](x),
    )

    self.assertEqual(
        jax_module.input_polymorphic_shape_map,
        {DEFAULT_METHOD_KEY: None},
    )

  def test_jax_module_export_version_tf_savedmodel_update_variables(self):
    def linear(params, x):
      return params['w'] @ x + params['b']

    params = {
        'w': jnp.array([1, 2, 3]),
        'b': jnp.array([3]),
    }

    jax_module = JaxModule(
        params=params,
        apply_fn=linear,)

    jax_module.update_variables(
        {'w': jnp.array([4, 5, 6]), 'b': jnp.array([4])}
    )

    self.assertEqual(
        jax_module.model_params,
        {'w': jnp.array([4, 5, 6]), 'b': jnp.array([4])},
    )

if __name__ == '__main__':
  tf.test.main()
