# Copyright 2023 The Orbax Authors.
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

"""Tests for CompositeArgHandler."""
from absl.testing import absltest
from orbax.checkpoint import composite_checkpoint_handler


CompositeArgs = composite_checkpoint_handler.CompositeArgs


class CompositeArgsTest(absltest.TestCase):

  def test_args(self):
    args = CompositeArgs(a=1, b=2, d=4)
    self.assertEqual(1, args.a)
    self.assertEqual(2, args.b)
    self.assertEqual({'a', 'b', 'd'}, args.keys())
    self.assertEqual({1, 2, 4}, set(args.values()))
    self.assertEqual(1, args['a'])
    self.assertLen(args, 3)

    with self.assertRaises(KeyError):
      _ = args['c']

    self.assertIsNone(args.get('c'))
    self.assertEqual(4, args.get('c', 4))

  def test_invalid_key(self):
    with self.assertRaisesRegex(ValueError, 'cannot start with'):
      CompositeArgs(__invalid_name=2)

  def test_reserved_keys_are_unchanged(self):
    # To avoid breaking future users, make sure that CompositeArgs
    # only reserves the following attributes:
    self.assertEqual(
        set([x for x in dir(CompositeArgs) if not x.startswith('__')]),
        {'get', 'items', 'keys', 'values'},
    )

  def test_use_reserved_keys(self):
    args = CompositeArgs(keys=3, values=4)

    self.assertNotEqual(3, args.keys)
    self.assertNotEqual(4, args.values)

    self.assertEqual(3, args['keys'])
    self.assertEqual(4, args['values'])

    self.assertEqual({'keys', 'values'}, args.keys())
    self.assertEqual({3, 4}, set(args.values()))

  def test_special_character_key(self):
    args = CompositeArgs(**{
        '.invalid_attribute_but_valid_key': 15,
        'many special characters!': 16})
    self.assertEqual(15, args['.invalid_attribute_but_valid_key'])
    self.assertEqual(16, args['many special characters!'])
    self.assertLen(args, 2)


if __name__ == '__main__':
  absltest.main()
