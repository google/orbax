# Copyright 2022 The Orbax Authors.
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

"""Test for utils.py."""

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint import utils


class UtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='checkpointing_test').full_path
    )

  @parameterized.parameters(
      (3, 'dir', None, None, 'dir/3'),
      (3, 'dir', 'params', None, 'dir/3/params'),
      (3, 'dir', 'params', 'checkpoint', 'dir/checkpoint_3/params'),
  )
  def test_get_save_directory(self, step, directory, name, step_prefix, result):
    self.assertEqual(
        utils.get_save_directory(
            step, directory, name=name, step_prefix=step_prefix
        ),
        epath.Path(result),
    )

  def test_get_save_directory_tmp_dir_override(self):
    self.assertEqual(
        utils.get_save_directory(
            42,
            'path/to/my/dir',
            name='params',
            step_prefix='foobar_',
            override_directory='a/different/dir/path',
        ),
        epath.Path('a/different/dir/path/params'),
    )

  @parameterized.parameters((None,), ('checkpoint_',), ('foobar_',))
  def test_is_tmp_checkpoint(self, step_prefix):
    step_dir = utils.get_save_directory(
        5, self.directory, step_prefix=step_prefix
    )
    step_dir.mkdir(parents=True)
    self.assertFalse(utils.is_tmp_checkpoint(step_dir))
    tmp_step_dir = utils.create_tmp_directory(step_dir)
    self.assertTrue(utils.is_tmp_checkpoint(tmp_step_dir))

    item_dir = utils.get_save_directory(
        10, self.directory, name='params', step_prefix=step_prefix
    )
    item_dir.mkdir(parents=True)
    self.assertFalse(utils.is_tmp_checkpoint(item_dir))
    tmp_item_dir = utils.create_tmp_directory(item_dir)
    self.assertTrue(utils.is_tmp_checkpoint(tmp_item_dir))

  @parameterized.parameters(
      ('0', 0),
      ('0000', 0),
      ('1000', 1000),
      ('checkpoint_0', 0),
      ('checkpoint_0000', 0),
      ('checkpoint_003400', 3400),
      ('foobar_1000', 1000),
      ('0.orbax-checkpoint-tmp-1010101', 0),
      ('0000.orbax-checkpoint-tmp-12323232', 0),
      ('foobar_1.orbax-checkpoint-tmp-12424424', 1),
      ('foobar_000505.orbax-checkpoint-tmp-13124', 505),
      ('checkpoint_16.orbax-checkpoint-tmp-123214324', 16),
  )
  def test_step_from_checkpoint_name(self, name, step):
    self.assertEqual(utils.step_from_checkpoint_name(name), step)

  @parameterized.parameters(
      ('abc',),
      ('checkpoint_',),
      ('checkpoint_1010_',),
      ('_191',),
      ('.orbax-checkpoint-tmp-191913',),
      ('0.orbax-checkpoint-tmp-',),
      ('checkpoint_.orbax-checkpoint-tmp-191913',),
  )
  def test_step_from_checkpoint_name_invalid(self, name):
    with self.assertRaises(ValueError):
      utils.step_from_checkpoint_name(name)


if __name__ == '__main__':
  absltest.main()
