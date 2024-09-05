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

import os
from absl.testing import absltest
from orbax.export import config

orbax_export_config = config.config


class ConfigTest(absltest.TestCase):

  def test_config_setting_via_update(self):
    _ = orbax_export_config.define_bool_state(
        name='obx_export_test_bool_config',
        default=True,
        help_str='Configuration only used for tests.',
    )
    # pytype: disable=attribute-error
    self.assertEqual(orbax_export_config.obx_export_test_bool_config, True)
    orbax_export_config.update('obx_export_test_bool_config', False)
    self.assertEqual(orbax_export_config.obx_export_test_bool_config, False)

    orbax_export_config._undefine_state('obx_export_test_bool_config')
    self.assertNotIn('obx_export_test_bool_config', orbax_export_config.values)
    # pytype: enable=attribute-error

  def test_config_setting_via_context(self):
    # The default value of obx_export_tf_preprocess_only is False.
    # pytype: disable=attribute-error
    self.assertEqual(orbax_export_config.obx_export_tf_preprocess_only, False)
    with config.obx_export_tf_preprocess_only(True):
      self.assertEqual(orbax_export_config.obx_export_tf_preprocess_only, True)
    self.assertEqual(orbax_export_config.obx_export_tf_preprocess_only, False)
    # pytype: enable=attribute-error

  def test_config_setting_via_env(self):
    os.environ['OBX_EXPORT_TEST_BOOL_CONFIG'] = 'false'

    # ENV has higher priority here.
    _ = orbax_export_config.define_bool_state(
        name='obx_export_test_bool_config',
        default=True,
        help_str='Configuration only used for tests.',
    )
    self.assertEqual(orbax_export_config.obx_export_test_bool_config, False)  # pytype: disable=attribute-error

    # Attention: update the env variable after `define_bool_state` later will
    # not change the config default value.
    os.environ['OBX_EXPORT_TEST_BOOL_CONFIG'] = 'true'
    self.assertEqual(orbax_export_config.obx_export_test_bool_config, False)  # pytype: disable=attribute-error
    orbax_export_config._undefine_state('obx_export_test_bool_config')


if __name__ == '__main__':
  absltest.main()
