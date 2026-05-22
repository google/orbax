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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import orbax.checkpoint.experimental.v1 as ocp
from orbax.checkpoint.experimental.v1._src.loading import loading
from orbax.checkpoint.experimental.v1._src.metadata import loading as metadata_loading
from orbax.checkpoint.experimental.v1._src.saving import saving


class DeprecationsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='save_pytree',
          alias_func=ocp.save_pytree,
          target_module=saving,
          target_name='save',
      ),
      dict(
          testcase_name='save_pytree_async',
          alias_func=ocp.save_pytree_async,
          target_module=saving,
          target_name='save_async',
      ),
      dict(
          testcase_name='load_pytree',
          alias_func=ocp.load_pytree,
          target_module=loading,
          target_name='load',
      ),
      dict(
          testcase_name='load_pytree_async',
          alias_func=ocp.load_pytree_async,
          target_module=loading,
          target_name='load_async',
      ),
      dict(
          testcase_name='pytree_metadata',
          alias_func=ocp.pytree_metadata,
          target_module=metadata_loading,
          target_name='metadata',
      ),
  )
  def test_deprecated_alias(self, alias_func, target_module, target_name):
    with mock.patch.object(target_module, target_name) as mock_target:
      mock_target.return_value = 'expected_result'

      with self.assertWarnsRegex(
          DeprecationWarning,
          f'Use `{target_name}` instead.',
      ):
        result = alias_func('arg1', kwarg1='val1')

      self.assertEqual(result, 'expected_result')
      mock_target.assert_called_once_with('arg1', kwarg1='val1')


if __name__ == '__main__':
  absltest.main()
