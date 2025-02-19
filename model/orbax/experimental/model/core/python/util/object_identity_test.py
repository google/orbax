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

# Copyright 2024 The ML Exported Model Authors. All Rights Reserved.
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
# ==============================================================================
"""Unit tests for object_identity."""

from orbax.experimental.model.core.python.util import object_identity
from absl.testing import absltest


class ObjectIdentityWrapperTest(absltest.TestCase):

  def testWrapperNotEqualToWrapped(self):
    class SettableHash(object):

      def __init__(self):
        self.hash_value = 8675309

      def __hash__(self):
        return self.hash_value

    o = SettableHash()
    wrap1 = object_identity._ObjectIdentityWrapper(o)
    wrap2 = object_identity._ObjectIdentityWrapper(o)

    self.assertEqual(wrap1, wrap1)
    self.assertEqual(wrap1, wrap2)
    self.assertEqual(o, wrap1.unwrapped)
    self.assertEqual(o, wrap2.unwrapped)
    with self.assertRaises(TypeError):
      bool(o == wrap1)
    with self.assertRaises(TypeError):
      bool(wrap1 != o)

    self.assertNotIn(o, set([wrap1]))
    o.hash_value = id(o)
    # Since there is now a hash collision we raise an exception
    with self.assertRaises(TypeError):
      bool(o in set([wrap1]))


class ObjectIdentitySetTest(absltest.TestCase):

  def testDifference(self):
    class Element(object):
      pass

    a = Element()
    b = Element()
    c = Element()
    set1 = object_identity.ObjectIdentitySet([a, b])
    set2 = object_identity.ObjectIdentitySet([b, c])
    diff_set = set1.difference(set2)
    self.assertIn(a, diff_set)
    self.assertNotIn(b, diff_set)
    self.assertNotIn(c, diff_set)

  def testDiscard(self):
    a = object()
    b = object()
    set1 = object_identity.ObjectIdentitySet([a, b])
    set1.discard(a)
    self.assertIn(b, set1)
    self.assertNotIn(a, set1)

  def testClear(self):
    a = object()
    b = object()
    set1 = object_identity.ObjectIdentitySet([a, b])
    set1.clear()
    self.assertLen(set1, 0)  # pylint: disable=g-generic-assert


if __name__ == '__main__':
  absltest.main()
