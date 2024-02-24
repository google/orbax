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

"""Tests for Orbax storage constructs."""

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint import storage


class DefaultStepLookupTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='step_dir_test').full_path
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='step',
          existing_step_dirs=[
              '1.1',
              '1.blah_1',
              '1.01',
              '1.blah_01',
              '11.11',
              '1',
              '11',
          ],
          step_prefix=None,
          step_format_fixed_length=None,
          step=1,
          expected_step_dir='1',
      ),
      dict(
          testcase_name='padded',
          existing_step_dirs=[
              '1.1',
              '1.00001',
              '1.blah_00001',
              '1.blah_1',
              '11.00011',
              '00001',
              '000011',
          ],
          step_prefix=None,
          step_format_fixed_length=5,
          step=1,
          expected_step_dir='00001',
      ),
      dict(
          testcase_name='prefix_unpadded',
          existing_step_dirs=[
              '1.1',
              '1.01',
              '1.blah_1',
              'blah_1',
              'bblah_1',
              'blah_11',
              '1',
              '1.blah_01',
              '11.blah_11',
          ],
          step_prefix='blah',
          step_format_fixed_length=None,
          step=1,
          expected_step_dir='blah_1',
      ),
      dict(
          testcase_name='prefix_padded',
          existing_step_dirs=[
              '1.blah_00001',
              'blah_00001',
              'blah_00011',
              'blah_1',
              'bblah_00001',
              '1.00001',
              '1.1',
              '1.blah_1',
              '11.11',
          ],
          step_prefix='blah',
          step_format_fixed_length=5,
          step=1,
          expected_step_dir='blah_00001',
      ),
      dict(
          testcase_name='error.step',
          existing_step_dirs=['1.1', 'blah_1', '01', 'blah_01', '11'],
          step_prefix=None,
          step_format_fixed_length=None,
          step=1,
          expected_step_dir=None,
      ),
      dict(
          testcase_name='error.padded',
          existing_step_dirs=[
              '1',
              '00011',
              '100001',
              'blah_00001',
              'blah_1',
              '00011',
          ],
          step_prefix=None,
          step_format_fixed_length=5,
          step=1,
          expected_step_dir=None,
      ),
      dict(
          testcase_name='error.prefix_unpadded',
          existing_step_dirs=[
              '1',
              '01',
              'bblah_1',
              'blah_01',
              'blah_11',
          ],
          step_prefix='blah',
          step_format_fixed_length=None,
          step=1,
          expected_step_dir=None,
      ),
      dict(
          testcase_name='error.prefix_padded',
          existing_step_dirs=[
              'bblah_00001',
              '00001',
              '1',
              'blah_1',
              '11',
          ],
          step_prefix='blah',
          step_format_fixed_length=5,
          step=1,
          expected_step_dir=None,
      ),
  )
  def test_lookup(
      self,
      existing_step_dirs,
      step_prefix,
      step_format_fixed_length,
      step,
      expected_step_dir,
  ):
    for step_dir in existing_step_dirs:
      (self.directory / step_dir).mkdir(parents=True, exist_ok=True)

    root = storage.root(self.directory)
    if expected_step_dir is None:
      with self.assertRaises(ValueError):
        root.lookup(
            step=step,
            step_lookup=storage.DefaultStepLookup(
                step_prefix=step_prefix,
                step_format_fixed_length=step_format_fixed_length,
            ),
        )
      return

    self.assertEqual(
        self.directory / expected_step_dir,
        root.lookup(
            step=step,
            step_lookup=storage.DefaultStepLookup(
                step_prefix=step_prefix,
                step_format_fixed_length=step_format_fixed_length,
            ),
        ).path,
    )


if __name__ == '__main__':
  absltest.main()
