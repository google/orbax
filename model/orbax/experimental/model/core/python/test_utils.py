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

"""Test utilities."""

from collections.abc import Callable
import operator
from typing import Any
from absl.testing import parameterized


class ObmTestCase(parameterized.TestCase):  # pylint: disable=g-doc-args
  """Base class for OBM test cases.

  Provides utility methods for testing OBM code.
  """

  def _assert_tree_equiv(  # pylint: disable=g-doc-args
      self,
      x,
      y,
      is_equiv_leaves: Callable[[Any, Any], bool],
  ):
    """Asserts that two pytrees are equivalent.

    Two pytrees are equivalent if they have the same tree pattern and all
    corresponding tree leaves are equivalent according to `is_equiv_leaves`.
    """
    if isinstance(x, tuple):
      # TODO(b/329306166): Add a knob to decide whether to treat tuple
      #   and list as equivalent.
      self.assertIsInstance(y, tuple)
      self.assertEqual(len(x), len(y))
      for xi, yi in zip(x, y):
        self._assert_tree_equiv(xi, yi, is_equiv_leaves)
    elif isinstance(x, list):
      self.assertIsInstance(y, list)
      self.assertEqual(len(x), len(y))
      for xi, yi in zip(x, y):
        self._assert_tree_equiv(xi, yi, is_equiv_leaves)
    elif isinstance(x, dict):
      self.assertIsInstance(y, dict)
      self.assertEqual(set(x.keys()), set(y.keys()))
      for k in x:
        self._assert_tree_equiv(x[k], y[k], is_equiv_leaves)
    elif x is None:
      self.assertIsNone(y)
    else:
      self.assertTrue(is_equiv_leaves(x, y))

  def assertTreeEquiv(  # pylint: disable=g-doc-args, disable=invalid-name
      self, x, y, is_equiv_leaves: Callable[[Any, Any], bool] = operator.eq
  ):
    """Asserts that two pytrees are equavalent.

    Two pytrees are equivalent if they have the same tree pattern and all
    corresponding tree leaves are equivalent according to `is_equiv_leaves`.
    """
    self._assert_tree_equiv(x, y, is_equiv_leaves)
