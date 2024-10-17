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
import jax.numpy as jnp
from orbax.export import export_manager as em
from orbax.export import jax_module
from orbax.export import serving_config as osc
from orbax.export import tensorflow_export
import tensorflow as tf


class TensorFlowExportTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._output_dir = self.create_tempdir().full_path

  @parameterized.named_parameters(
      dict(
          testcase_name='multiple signatures',
          serving_configs=[
              osc.ServingConfig(
                  'without_processors',
                  input_signature=[tf.TensorSpec((), tf.dtypes.int32)],
              ),
          ],
          expected_keys=['without_processors'],
      ),
  )
  def test_save(self, serving_configs, expected_keys):
    module = tf.Module()
    module.computation_module = jax_module.JaxModule(
        {'bias': jnp.array(1)}, lambda p, x: x + p['bias']
    )
    serving_signatures = {}
    tfe = tensorflow_export.TensorFlowExport()
    em.process_serving_configs(
        serving_configs,
        obx_export_tf_preprocess_only=False,
        module=module,
        serving_signatures=serving_signatures,
    )
    tfe.save(
        module,
        self._output_dir,
        serving_signatures=serving_signatures,
    )
    loaded = tf.saved_model.load(self._output_dir, ['serve'])
    self.assertCountEqual(expected_keys, loaded.signatures.keys())


if __name__ == '__main__':
  tf.test.main()
