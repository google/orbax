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

from typing import Mapping
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from orbax.export import constants
from orbax.export import export_manager
from orbax.export import jax_module
from orbax.export import serving_config as sc
from orbax.export.modules import tensorflow_module
import tensorflow as tf

def _from_feature_dict(feature_dict):
  return feature_dict['feat']


def _add_output_name(outputs):
  return {'outputs': outputs}


@jax.jit
def apply_fn(params, x):
  return x + params['bias']


class ExportManagerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._output_dir = self.create_tempdir().full_path

  def test_get_tf_module_tensorflow_export(self):
    serving_config = (
        sc.ServingConfig(
            'serving_default',
            input_signature=[
                tf.TensorSpec((), tf.int32, 'x'),
            ],
        ),
    )
    em = export_manager.ExportManager(
        jax_module.JaxModule(
            params={'bias': jnp.array(1, jnp.int32)},
            apply_fn=apply_fn,
            export_version=constants.ExportModelType.TF_SAVEDMODEL,
        ),
        serving_config,
    )
    self.assertEqual(type(em.tf_module), tf.Module)

  def test_tf_export_module_attributes_tensorflow_export(self):
    serving_config = (
        sc.ServingConfig(
            'serving_default',
            input_signature=[
                tf.TensorSpec((), tf.int32, 'x'),
            ],
        ),
    )
    em = export_manager.ExportManager(
        jax_module.JaxModule(
            params={'bias': jnp.array(1, jnp.int32)},
            apply_fn=apply_fn,
            export_version=constants.ExportModelType.TF_SAVEDMODEL,
        ),
        serving_config,
    )
    self.assertEqual(type(em.tf_module), tf.Module)
    self.assertIsNotNone(em.tf_module.__call__)

  def test_get_tf_module_orbax_model_export(self):
    serving_config = (
        sc.ServingConfig(
            'serving_default',
            input_signature=[
                tf.TensorSpec((), tf.int32, 'x'),
            ],
        ),
    )
    em = export_manager.ExportManager(
        jax_module.JaxModule(
            params={'bias': jnp.array(1, jnp.int32)},
            apply_fn=apply_fn,
            export_version=constants.ExportModelType.ORBAX_MODEL,
        ),
        serving_config,
    )

    with self.assertRaises(TypeError):
      em.tf_module  # pylint: disable=pointless-statement

  def test_get_serving_signatures_tensorflow_export(self):
    serving_config = (
        sc.ServingConfig(
            'serving_default',
            input_signature=[
                tf.TensorSpec((), tf.int32, 'x'),
            ],
        ),
    )
    em = export_manager.ExportManager(
        jax_module.JaxModule(
            params={'bias': jnp.array(1, jnp.int32)},
            apply_fn=apply_fn,
            export_version=constants.ExportModelType.TF_SAVEDMODEL,
        ),
        serving_config,
    )

    self.assertContainsExactSubsequence(
        em.serving_signatures.keys(), ['serving_default']
    )

  def test_get_serving_signatures_orbax_export(self):
    serving_config = (
        sc.ServingConfig(
            'serving_default',
            input_signature=[
                tf.TensorSpec((), tf.int32, 'x'),
            ],
        ),
    )
    em = export_manager.ExportManager(
        jax_module.JaxModule(
            params={'bias': jnp.array(1, jnp.int32)},
            apply_fn=apply_fn,
            export_version=constants.ExportModelType.ORBAX_MODEL,
        ),
        serving_config,
    )

    with self.assertRaises(NotImplementedError):
      em.serving_signatures  # pylint: disable=pointless-statement

  def test_save_model_with_preprocess_output_passthrough_succeeds(self):
    """Tests that the model can be saved with preprocess output passthrough."""
    rng = jax.random.PRNGKey(0)
    params = {
        'w': jax.random.normal(rng, shape=(8, 8)),
        'b': jax.random.normal(rng, shape=(8, 1)),
    }

    @tf.function
    def tf_preprocessor(x: tf.Tensor):
      return ({'pre_out_0': x}, {'pre_out_1': x})

    def jax_func(
        params: Mapping[str, jax.Array], inputs: Mapping[str, jax.Array]
    ):
      outputs = params['w'] @ inputs['pre_out_0'] + params['b']
      return {
          'jax_out_0': outputs,
      }

    @tf.function
    def tf_postprocessor(
        inputs: Mapping[str, tf.Tensor], inputs_extra: Mapping[str, tf.Tensor]
    ):
      return {
          'post_out_0': inputs['jax_out_0'],
          'post_out_1': inputs_extra['pre_out_1'],
      }

    m = jax_module.JaxModule(
        params,
        jax_func,
        input_polymorphic_shape='b, ...',
    )

    serving_configs = [
        sc.ServingConfig(
            'serving_default',
            [tf.TensorSpec((None, 8, 1), tf.float32, name='x')],
            tf_preprocessor=tf_preprocessor,
            tf_postprocessor=tf_postprocessor,
            preprocess_output_passthrough_enabled=True,
        )
    ]

    em = export_manager.ExportManager(m, serving_configs)
    em.save(
        self._output_dir,
    )

    x = jax.random.normal(rng, shape=(8, 8, 1))
    loaded = em.load(self._output_dir)
    self.assertAllClose(
        loaded.signatures['serving_default'](x=tf.convert_to_tensor(x))[
            'post_out_0'
        ],
        params['w'] @ x + params['b'],
        atol=0.05,
        rtol=0.2,
    )
    self.assertAllEqual(
        loaded.signatures['serving_default'](x=tf.convert_to_tensor(x))[
            'post_out_1'
        ],
        x,
    )

  def test_save_jax2tf_model_with_preprocess_output_passthrough_raises_error(
      self,
  ):
    """Tests that the model saving with preprocess output passthrough raises error.

    The error is raised because the preprocessor output doesn't comply with
    the requirements of a tuple of two dicts.
    """
    rng = jax.random.PRNGKey(0)
    params = {
        'w': jax.random.normal(rng, shape=(8, 8)),
        'b': jax.random.normal(rng, shape=(8, 1)),
    }

    @tf.function
    def tf_preprocessor(x: tf.Tensor):
      return ({'pre_out_0': x}, {'pre_out_1': x}, {'error': x})

    def jax_func(
        params: Mapping[str, jax.Array], inputs: Mapping[str, jax.Array]
    ):
      outputs = params['w'] @ inputs['pre_out_0'] + params['b']
      return {
          'jax_out_0': outputs,
      }

    @tf.function
    def tf_postprocessor(
        inputs: Mapping[str, tf.Tensor], inputs_extra: Mapping[str, tf.Tensor]
    ):
      return {
          'post_out_0': inputs['jax_out_0'],
          'post_out_1': inputs_extra['pre_out_1'],
      }

    m = jax_module.JaxModule(
        params,
        jax_func,
        input_polymorphic_shape='b, ...',
    )

    serving_configs = [
        sc.ServingConfig(
            'serving_default',
            [tf.TensorSpec((None, 8, 1), tf.float32, name='x')],
            tf_preprocessor=tf_preprocessor,
            tf_postprocessor=tf_postprocessor,
            preprocess_output_passthrough_enabled=True,
        )
    ]

    em = export_manager.ExportManager(m, serving_configs)
    with self.assertRaises(ValueError):
      em.save(
          self._output_dir,
      )


if __name__ == '__main__':
  tf.test.main()
