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

import unittest
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint.experimental.v1._src.metadata import serialization


class SerializationTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='serialization_test').full_path
    )

  @parameterized.parameters(
      ({'property': 'value'},),
      ({'property': 123},),
      ({'property': True},),
      ({'property': None},),
      ({'property': []},),
      ({'property': {}},),
      ({'property': [1, 2, 3]},),
      ({'property': {'a': 1, 'b': 2, 'c': 3}},),
      ({},),
  )
  async def test_write_and_read(self, d):
    await serialization.write(self.directory / 'metadata.json', d)
    result = await serialization.read(self.directory / 'metadata.json')
    self.assertDictEqual(result, d)

  @parameterized.parameters(
      ({'property': b'123'},),
      ([],),
  )
  async def test_not_writeable(self, d):
    with self.assertRaises(TypeError):
      await serialization.write(
          self.directory / 'metadata.json',
          d
      )

  async def test_write_no_parent(self):
    with self.assertRaises(FileNotFoundError):
      await serialization.write(
          epath.Path('/foo/bar/metadata.json'), {'property': 'value'}
      )

  async def test_read_no_file(self):
    self.assertIsNone(
        await serialization.read(self.directory / 'metadata.json')
    )


if __name__ == '__main__':
  absltest.main()
