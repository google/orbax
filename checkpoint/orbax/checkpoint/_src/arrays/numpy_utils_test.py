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
import numpy as np
from orbax.checkpoint._src.arrays import numpy_utils as np_utils


class PrettySliceTest(parameterized.TestCase):

  @parameterized.parameters([
      (slice(None), ':'),
      (slice(4), ':4'),
      (slice(1, 4), '1:4'),
      (slice(1, 4, 2), '1:4:2'),
  ])
  def test_produces_extended_slice_syntax(self, s: slice, s_string: str):
    self.assertEqual(s_string, np_utils._pretty_slice(s))
    self.assertEqual(s, eval(f'np.s_[{s_string}]'))  # pylint:disable=eval-used


class ResolveDissolveSliceTest(parameterized.TestCase):

  @parameterized.parameters([
      ((3, 4, 5), np.s_[:, :, :], np.s_[0:3:1, 0:4:1, 0:5:1]),
      ((3, 4, 5), np.s_[:2, 1:, ::2], np.s_[0:2:1, 1:4:1, 0:5:2]),
      ((), np.s_[...], np.s_[()]),
  ])
  def test_round_trip(
      self,
      array_shape: np_utils.Shape,
      implicit_slice: np_utils.NdSlice,
      explicit_slice: np_utils.NdSlice,
  ):
    self.assertEqual(
        explicit_slice,
        np_utils.resolve_slice(implicit_slice, array_shape),
    )
    self.assertEqual(
        implicit_slice,
        np_utils.dissolve_slice(explicit_slice, array_shape),
    )

  def test_dissolve_slice_can_remove_redundant_rank(self):
    self.assertEqual(
        (np.s_[1:2],),
        np_utils.dissolve_slice(
            np.s_[1:2, 0:4, 0:5:1], (3, 4, 5), preserve_rank=False
        ),
    )
    self.assertEqual(
        np.s_[:, 1:2],
        np_utils.dissolve_slice(
            np.s_[0:3:1, 1:2:1, 0:5:1], (3, 4, 5), preserve_rank=False
        ),
    )
    self.assertEqual(
        ...,
        np_utils.dissolve_slice(
            np.s_[:3, 0:4, 0:5:1], (3, 4, 5), preserve_rank=False
        ),
    )


class PrettyNdSliceTest(parameterized.TestCase):

  @parameterized.parameters([
      (..., 'np.s_[...]'),
      ((slice(1, 4, 2), slice(None)), 'np.s_[1:4:2, :]'),
  ])
  def test_generally_produces_np_s_syntax(
      self,
      idx: np_utils.Index,
      idx_string: str,
  ):
    self.assertEqual(idx_string, np_utils.pretty_nd_slice(idx))
    self.assertEqual(idx, eval(idx_string))  # pylint:disable=eval-used

  def test_rank_1_slice(self):
    # A tuple of length 1 won't exactly round-trip, but it will produce
    # a value that can be used in the same places, because `a[(s,)]`
    # will do the same as `a[s]`.
    idx = (slice(1, 4, 2),)
    self.assertEqual('np.s_[1:4:2]', np_utils.pretty_nd_slice(idx))

  def test_empty_tuple_is_rendered_as_ellipsis(self):
    # Similarly, an empty won't exactly round-trip, but it will produce
    # a value that can be used in the same places, because `a[...]`
    # will do the same as `a[()]` (unless `a` has rank 0 in which case
    # ellipsis actually does the more useful thing).
    self.assertEqual('np.s_[...]', np_utils.pretty_nd_slice(()))


if __name__ == '__main__':
  absltest.main()
