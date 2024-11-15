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
    self.assertTrue(isinstance(em.tf_module.computation_module, tf.Module))

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
        constants.ExportModelType.ORBAX_MODEL,
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
        constants.ExportModelType.ORBAX_MODEL,
    )

    with self.assertRaises(NotImplementedError):
      em.serving_signatures  # pylint: disable=pointless-statement


if __name__ == '__main__':
  tf.test.main()
