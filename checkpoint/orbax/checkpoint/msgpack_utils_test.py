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

"""Test for utils.py."""
from absl.testing import absltest
from absl.testing import parameterized
from orbax.checkpoint import msgpack_utils


class MsgpackUtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      (3,),
      ('foo',),
      ((1,),),
      (('a', 1, 1.2),),
      ([],),
      ({'a': [2, 'hi', (1, 3, 'bye')], 'b': {'hi': 2}},),
  )
  def test_serialize(self, x):
    self.assertEqual(
        x, msgpack_utils.msgpack_restore(msgpack_utils.msgpack_serialize(x))
    )


if __name__ == '__main__':
  absltest.main()
