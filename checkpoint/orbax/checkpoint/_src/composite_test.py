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

from absl.testing import absltest
from absl.testing import parameterized
from orbax.checkpoint._src import composite as composite_lib


class CompositeTest(parameterized.TestCase):

  def test_composite_attributes(self):
    composite = composite_lib.Composite(a=1, b=2, d=4)
    self.assertEqual(1, composite.a)
    self.assertEqual(2, composite.b)
    self.assertEqual(4, composite.d)

  def test_composite_keys(self):
    composite = composite_lib.Composite(a=1, b=2, d=4)
    self.assertEqual({'a', 'b', 'd'}, composite.keys())

  def test_composite_values(self):
    composite = composite_lib.Composite(a=1, b=2, d=4)
    self.assertEqual({1, 2, 4}, set(composite.values()))

  def test_composite_items(self):
    composite = composite_lib.Composite(a=1, b=2, d=4)
    self.assertEqual({('a', 1), ('b', 2), ('d', 4)}, set(composite.items()))

  def test_composite_get(self):
    composite = composite_lib.Composite(a=1, b=2, d=4)
    self.assertEqual(1, composite['a'])
    self.assertEqual(2, composite['b'])
    self.assertEqual(4, composite['d'])

  def test_composite_len(self):
    composite = composite_lib.Composite(a=1, b=2, d=4)
    self.assertLen(composite, 3)

  def test_composite_get_invalid_key(self):
    composite = composite_lib.Composite(a=1, b=2, d=4)
    with self.assertRaises(KeyError):
      _ = composite['c']

  def test_composite_get_default(self):
    composite = composite_lib.Composite(a=1, b=2, d=4)
    self.assertIsNone(composite.get('c'))
    self.assertEqual(4, composite.get('c', 4))

  def test_invalid_key(self):
    with self.assertRaisesRegex(ValueError, 'cannot start with'):
      composite_lib.Composite(__invalid_name=2)

  def test_reserved_keys_are_unchanged(self):
    # To avoid breaking future users, make sure that Composite
    # only reserves the following attributes:
    reserved_keys = {'_is_protocol', '_abc_impl'}
    self.assertEqual(
        set([
            x for x in dir(composite_lib.Composite)
            if not x.startswith('__') and x not in reserved_keys
        ]),
        {'get', 'items', 'keys', 'values'},
    )

  def test_use_reserved_keys(self):
    composite = composite_lib.Composite(keys=3, values=4)

    self.assertNotEqual(3, composite.keys)
    self.assertNotEqual(4, composite.values)

    self.assertEqual(3, composite['keys'])
    self.assertEqual(4, composite['values'])

    self.assertEqual({'keys', 'values'}, composite.keys())
    self.assertEqual({3, 4}, set(composite.values()))

  def test_special_character_key(self):
    composite = composite_lib.Composite(**{
        '.invalid_attribute_but_valid_key': 15,
        'many special characters!': 16,
    })
    self.assertEqual(15, composite['.invalid_attribute_but_valid_key'])
    self.assertEqual(16, composite['many special characters!'])
    self.assertLen(composite, 2)


if __name__ == '__main__':
  absltest.main()
