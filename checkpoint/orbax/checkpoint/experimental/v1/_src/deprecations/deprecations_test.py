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

from absl.testing import absltest
from absl.testing import parameterized
from orbax.checkpoint.experimental.v1._src.deprecations import deprecations


# Define a dummy free function and its alias statically
def dummy_free_target(arg, kwarg=None):
  return f'free:{arg}:{kwarg}'


@deprecations.deprecated(new=dummy_free_target)
def dummy_free_alias(*args, **kwargs):
  return dummy_free_target(*args, **kwargs)


# Define a dummy class with a method and its alias statically
class DummyClass:

  def dummy_method_target(self, arg, kwarg=None):
    return f'method:{arg}:{kwarg}'

  @deprecations.deprecated(new=dummy_method_target)
  def dummy_method_alias(self, *args, **kwargs):
    return self.dummy_method_target(*args, **kwargs)


class DeprecationsTest(parameterized.TestCase):

  def test_deprecated_free_function_alias(self):
    # Access the statically defined alias
    with self.assertWarnsRegex(
        DeprecationWarning,
        '`dummy_free_alias` is deprecated, use `dummy_free_target` instead.',
    ):
      result = dummy_free_alias('val', kwarg='ok')

    self.assertEqual(result, 'free:val:ok')

  def test_deprecated_method_alias(self):
    # Access the statically defined method alias
    obj = DummyClass()
    with self.assertWarnsRegex(
        DeprecationWarning,
        '`dummy_method_alias` is deprecated, use `dummy_method_target`'
        ' instead.',
    ):
      result = obj.dummy_method_alias('val', kwarg='ok')

    self.assertEqual(result, 'method:val:ok')


if __name__ == '__main__':
  absltest.main()
