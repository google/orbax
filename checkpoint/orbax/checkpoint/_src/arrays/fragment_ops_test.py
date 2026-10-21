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
import numpy as np
from orbax.checkpoint._src.arrays import fragment_ops
from orbax.checkpoint._src.arrays import fragments

AbstractFragment = fragments.AbstractFragment
NpFragment = fragments.NpFragment
JaxFragment = fragments.JaxFragment
AbstractFragments = fragments.AbstractFragments
NpFragments = fragments.NpFragments
JaxFragments = fragments.JaxFragments


class UnionFragmentListTest(parameterized.TestCase):

  def test_union_of_no_fragments_is_no_fragments(self):
    self.assertCountEqual(
        [],
        [*fragment_ops.union_fragment_list([])],
    )

  def test_union_of_degenerate_fragment_is_no_fragments(self):
    degenerate = AbstractFragment(index=np.s_[4:4:1, 5:10:1])

    self.assertCountEqual(
        [],
        [*fragment_ops.union_fragment_list([degenerate])],
    )

    with self.subTest('with_duplicate'):
      self.assertCountEqual(
          [],
          [*fragment_ops.union_fragment_list([degenerate, degenerate])],
      )

  def test_union_of_one_fragment_is_that_fragment(self):
    fragment = AbstractFragment(index=np.s_[4:12:1, 5:10:1])
    self.assertCountEqual(
        [fragment],
        [*fragment_ops.union_fragment_list([fragment])],
    )

    with self.subTest('with_duplicate'):
      self.assertCountEqual(
          [fragment],
          [*fragment_ops.union_fragment_list([fragment, fragment])],
      )

  @parameterized.parameters([
      (
          [
              AbstractFragment(index=np.s_[16:20:1, 5:10:1]),
              AbstractFragment(index=np.s_[4:12:1, 5:10:1]),
          ],
      ),
      (
          [
              AbstractFragment(index=np.s_[4:12:1, 15:20:1]),
              AbstractFragment(index=np.s_[4:12:1, 5:10:1]),
          ],
      ),
  ])
  def test_union_of_disjoint_fragments_is_those_fragments(self, fragments_list):
    self.assertCountEqual(
        fragments_list,
        [*fragment_ops.union_fragment_list(fragments_list)],
    )

  def test_union_overlapping_fragments(self):
    u = [
        *fragment_ops.union_fragment_list([
            AbstractFragment(index=np.s_[4:12:1, 5:10:1]),
            AbstractFragment(index=np.s_[8:12:1, 5:15:1]),
        ])
    ]

    self.assertCountEqual(
        [
            AbstractFragment(index=np.s_[4:8:1, 5:10:1]),
            AbstractFragment(index=np.s_[8:12:1, 5:15:1]),
        ],
        u,
    )

  def test_union_overlapping_fragments_3d(self):
    u = [
        *fragment_ops.union_fragment_list([
            AbstractFragment(index=np.s_[4:12:1, 5:10:1, 7:14:1]),
            AbstractFragment(index=np.s_[8:12:1, 5:15:1, 7:14:1]),
            AbstractFragment(index=np.s_[8:12:1, 5:10:1, 7:21:1]),
        ])
    ]

    self.assertCountEqual(
        [
            AbstractFragment(index=np.s_[4:8:1, 5:10:1, 7:14:1]),
            AbstractFragment(index=np.s_[8:12:1, 10:15:1, 7:14:1]),
            AbstractFragment(index=np.s_[8:12:1, 5:10:1, 7:21:1]),
        ],
        u,
    )

  def test_merges_adjacent_fragments(self):
    fragments_list = [
        AbstractFragment(index=np.s_[8:12:1, 5:10:1]),
        AbstractFragment(index=np.s_[4:12:1, 5:10:1]),
        AbstractFragment(index=np.s_[12:16:1, 5:10:1]),
    ]
    self.assertCountEqual(
        [AbstractFragment(index=np.s_[4:16:1, 5:10:1])],
        [*fragment_ops.union_fragment_list(fragments_list)],
    )

    fragments_list = [
        AbstractFragment(index=np.s_[4:8:1, 5:10:1]),
        AbstractFragment(index=np.s_[4:8:1, 15:20:1]),
        AbstractFragment(index=np.s_[4:8:1, 10:15:1]),
    ]
    self.assertCountEqual(
        [AbstractFragment(index=np.s_[4:8:1, 5:20:1])],
        [*fragment_ops.union_fragment_list(fragments_list)],
    )


@parameterized.named_parameters(
    ('np_fragment', NpFragment),
    ('jax_fragment', JaxFragment),
)
class UnionConcreteFragmentListTest(parameterized.TestCase):

  def test_merges_adjacent_fragments(
      self, fragment_t: type[fragments.ConcreteFragment]
  ):
    np_api = fragment_t.NP_API
    fragments_list = [
        fragment_t(
            index=np.s_[4:12:1, 5:10:1],
            value=np_api.full((8, 5), 2),
        ),
        fragment_t(
            index=np.s_[12:16:1, 5:10:1],
            value=np_api.full((4, 5), 3),
        ),
    ]
    actual = [*fragment_ops.union_fragment_list(fragments_list)]
    self.assertLen(actual, 1)
    actual_fragment = actual[0]

    expected_fragment = fragment_t(
        index=np.s_[4:16:1, 5:10:1],
        value=np_api.concatenate(
            [
                np_api.full((8, 5), 2),
                np_api.full((4, 5), 3),
            ],
            axis=0,
        ),
    )

    np.testing.assert_array_equal(
        expected_fragment.value, actual_fragment.value
    )
    np.testing.assert_array_equal(
        expected_fragment.np_index, actual_fragment.np_index
    )

  def test_union_overlapping_fragments_3d(
      self, fragment_t: type[fragments.ConcreteFragment]
  ):
    np_api = fragment_t.NP_API
    fragments_list = [
        fragment_t(
            index=np.s_[4:8:1, 5:10:1, 7:14:1],
            value=np_api.full((4, 5, 7), 1),
        ),
        fragment_t(
            index=np.s_[8:12:1, 5:15:1, 7:14:1],
            value=np_api.full((4, 10, 7), 2),
        ),
        fragment_t(
            index=np.s_[8:16:1, 5:10:1, 7:21:1],
            value=np_api.full((8, 5, 14), 2),
        ),
    ]
    actual = [*fragment_ops.union_fragment_list(fragments_list)]
    self.assertLen(actual, 3)
    expected = [
        fragment_t(
            index=np.s_[4:8:1, 5:10:1, 7:14:1],
            value=np_api.full((4, 5, 7), 1),
        ),
        fragment_t(
            index=np.s_[8:12:1, 10:15:1, 7:14:1],
            value=np_api.full((4, 5, 7), 2),
        ),
        fragment_t(
            index=np.s_[8:16:1, 5:10:1, 7:21:1],
            value=np_api.full((8, 5, 14), 2),
        ),
    ]

    self.assertEqual(len(expected), len(actual))
    for expected_frag in expected:
      matching = [
          a
          for a in actual
          if np.array_equal(a.np_index, expected_frag.np_index)
      ]
      self.assertLen(matching, 1)
      actual_frag = matching[0]
      np.testing.assert_array_equal(expected_frag.value, actual_frag.value)


@parameterized.named_parameters(
    ('np_fragment', NpFragment),
    ('jax_fragment', JaxFragment),
)
class ExtractFragmentTest(parameterized.TestCase):

  def test_extracts(self, fragment_t: type[fragments.ConcreteFragment]):
    np_api = fragment_t.NP_API
    source_fragments = [
        fragment_t(
            index=np.s_[4:8:1, 5:10:1],
            value=np_api.array([
                [405, 406, 407, 408, 409],
                [505, 506, 507, 508, 509],
                [605, 606, 607, 608, 609],
                [705, 706, 707, 708, 709],
            ]),
        ),
        fragment_t(
            index=np.s_[8:12:1, 5:10:1],
            value=np_api.array([
                [805, 806, 807, 808, 809],
                [905, 906, 907, 908, 909],
                [1005, 1006, 1007, 1008, 1009],
                [1105, 1106, 1107, 1108, 1109],
            ]),
        ),
    ]

    required_fragment = AbstractFragment(index=np.s_[6:10:1, 7:9:1])

    expected_fragment = fragment_t(
        index=np.s_[6:10:1, 7:9:1],
        value=np_api.array([
            [607, 608],  # From first fragment
            [707, 708],
            [807, 808],  # From second fragment
            [907, 908],
        ]),
    )

    actual_fragment = fragment_ops.extract_fragment(
        source_fragments, required_fragment, fragment_t
    )

    np.testing.assert_array_equal(
        expected_fragment.value, actual_fragment.value
    )
    np.testing.assert_array_equal(
        expected_fragment.np_index, actual_fragment.np_index
    )


if __name__ == '__main__':
  absltest.main()
