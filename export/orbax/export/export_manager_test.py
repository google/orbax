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
    bs = 1

    @tf.function
    def tf_preprocessor(x_float: tf.Tensor, y_str: tf.Tensor):
      # Returns two outputs, one to be passed to the JAX function, the other to
      # be passed to the TF postprocessor.
      return {'pre_out_float': x_float}, {'pre_out_str': y_str}

    def jax_func(
        params: Mapping[str, jax.Array], inputs: Mapping[str, jax.Array]
    ):
      outputs = params['w'] @ inputs['pre_out_float'] + params['b']
      return {
          'jax_out_float': outputs,
      }

    # The TF postprocessor gets two inputs: the JAX function output and the
    # second of the preprocessor output.
    @tf.function
    def tf_postprocessor(
        inputs: Mapping[str, tf.Tensor], inputs_extra: Mapping[str, tf.Tensor]
    ):
      return {
          'post_out_float': inputs['jax_out_float'],
          'post_out_str': inputs_extra['pre_out_str'],
      }

    m = jax_module.JaxModule(
        params,
        jax_func,
    )
    serving_configs = [
        sc.ServingConfig(
            'serving_default',
            [
                tf.TensorSpec((bs, 8, 1), tf.float32, name='x_float'),
                tf.TensorSpec((bs, 8, 1), tf.string, name='y_str'),
            ],
            tf_preprocessor=tf_preprocessor,
            tf_postprocessor=tf_postprocessor,
            preprocess_output_passthrough_enabled=True,
        )
    ]
    em = export_manager.ExportManager(m, serving_configs)
    em.save(
        self._output_dir,
    )

    x_float = jax.random.normal(rng, shape=(bs, 8, 1))
    y_str = tf.constant(['a dummy string'] * bs)
    loaded = em.load(self._output_dir)
    self.assertAllClose(
        loaded.signatures['serving_default'](
            x_float=tf.convert_to_tensor(x_float, dtype=tf.float32),
            y_str=y_str,
        )['post_out_float'],
        params['w'] @ x_float + params['b'],
        atol=0.05,
        rtol=0.2,
    )
    self.assertAllEqual(
        loaded.signatures['serving_default'](
            x_float=tf.convert_to_tensor(x_float),
            y_str=y_str,
        )['post_out_str'],
        y_str,
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
    bs = 1

    @tf.function
    def tf_preprocessor(x_float: tf.Tensor, y_str: tf.Tensor):
      return {'pre_out_float': x_float, 'pre_out_str': y_str}

    def jax_func(
        params: Mapping[str, jax.Array], inputs: Mapping[str, jax.Array]
    ):
      outputs = params['w'] @ inputs['pre_out_float'] + params['b']
      return {
          'jax_out_float': outputs,
      }

    @tf.function
    def tf_postprocessor(
        inputs: Mapping[str, tf.Tensor], inputs_extra: Mapping[str, tf.Tensor]
    ):
      return {
          'post_out_float': inputs['jax_out_float'],
          'post_out_str': inputs_extra['pre_out_str'],
      }

    m = jax_module.JaxModule(
        params,
        jax_func,
    )
    serving_configs = [
        sc.ServingConfig(
            'serving_default',
            [
                tf.TensorSpec((bs, 8, 1), tf.float32, name='x_float'),
                tf.TensorSpec((bs, 8, 1), tf.string, name='y_str'),
            ],
            tf_preprocessor=tf_preprocessor,
            tf_postprocessor=tf_postprocessor,
            preprocess_output_passthrough_enabled=True,
        )
    ]
    em = export_manager.ExportManager(m, serving_configs)
    with self.assertRaisesRegex(
        ValueError,
        'requiring the preprocessor output to be a tuple of two elements',
    ):
      em.save(
          self._output_dir,
      )


if __name__ == '__main__':
  tf.test.main()
