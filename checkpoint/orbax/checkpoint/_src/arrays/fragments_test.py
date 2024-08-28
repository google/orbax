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

import dataclasses

from absl.testing import absltest
import jax
import numpy as np
from orbax.checkpoint._src.arrays import fragments as array_fragments

Fragment = array_fragments.Fragment
Fragments = array_fragments.Fragments


class FragmentTest(absltest.TestCase):

  def test_construction(self):
    with self.subTest('rejects_index_as_ndarray'):
      index = np.s_[1:2:1, 3:4:1]
      with self.assertRaisesRegex(TypeError, 'must be a tuple of slices'):
        _ = Fragment(index=array_fragments._ndarray_from_index(index))  # pytype: disable=wrong-arg-types

    with self.subTest('with_np_index'):
      f = Fragment(index=np.s_[1:2:1, 3:4:1])
      self.assertEqual(f, Fragment(np_index=f.np_index))

    with self.subTest('with_dataclasses_replace_value'):
      f_a = Fragment(index=np.s_[0:2:1, 0:3:1])
      value = np.ones((2, 3))
      f_c = dataclasses.replace(f_a, value=value)
      self.assertEqual(f_c, Fragment(index=f_a.index, value=value))

    with self.subTest('with_dataclasses_replace_np_index'):
      value = np.ones((2, 3))
      old_np_index = array_fragments._ndarray_from_index(np.s_[0:2:1, 0:3:1])
      new_np_index = array_fragments._ndarray_from_index(np.s_[2:4:1, 3:6:1])
      f_old = Fragment(np_index=old_np_index, value=value)
      f_new = dataclasses.replace(f_old, np_index=new_np_index)
      self.assertEqual(f_new, Fragment(np_index=new_np_index, value=value))

    with self.subTest('only_one_of_index_or_np_index_is_allowed'):
      idx = np.s_[1:2:1, 3:4:1]
      with self.assertRaisesRegex(ValueError, 'both index and np_index'):
        _ = Fragment(
            index=idx, np_index=array_fragments._ndarray_from_index(idx)
        )

    with self.subTest('one_of_index_or_np_index_must_be_specified'):
      with self.assertRaisesRegex(ValueError, 'either index or np_index'):
        _ = Fragment()

  def test_equality(self):
    self.assertEqual(
        Fragment(index=np.s_[1:2:1, 3:4:1]),
        Fragment(index=np.s_[1:2:1, 3:4:1]),
    )
    self.assertNotEqual(
        Fragment(index=np.s_[1:2:1, 3:4:1]),
        Fragment(index=np.s_[1:2:1, 4:5:1]),
    )
    self.assertNotEqual(
        Fragment(index=np.s_[1:2:1, 3:4:1], value=np.ones([4, 4]) * 2.0),
        Fragment(index=np.s_[1:2:1, 3:4:1], value=np.ones([4, 4]) * 3.0),
    )

  def test_nd_properties(self):
    f = Fragment(index=np.s_[1:8:2, 3:9:4])
    np.testing.assert_array_equal([1, 3], f.start)
    np.testing.assert_array_equal([8, 9], f.stop)
    np.testing.assert_array_equal([2, 4], f.step)

  def test_rank_zero(self):
    f = Fragment(index=(), value=np.empty([]))
    np.testing.assert_array_equal([], f.start)
    np.testing.assert_array_equal([], f.stop)
    np.testing.assert_array_equal([], f.step)

  def test_repr(self):
    self.assertEqual(
        repr(Fragment(index=np.s_[1:2:1, 3:4:1])),
        'Fragment(index=np.s_[1:2:1, 3:4:1])',
    )
    self.assertEqual(
        repr(Fragment(index=np.s_[1:2:1, 3:4:1], value=np.ones([4, 4]))),
        'Fragment(index=np.s_[1:2:1, 3:4:1], value=...)',
    )

  def test_nbytes_of_concrete_fragment_is_nbytes_of_its_value(self):
    fragment_value = np.full([4, 4], 2.0)
    self.assertEqual(
        fragment_value.nbytes,
        Fragment(index=np.s_[1:5:1, 3:7:1], value=fragment_value).nbytes,
    )

  def test_nbytes_of_abstract_fragment_raises_value_error(self):
    with self.assertRaisesRegex(ValueError, 'nbytes of abstract'):
      _ = Fragment(index=np.s_[1:5:1, 3:7:1]).nbytes

  def test_nbytes_astype_of_concrete_fragment_uses_given_dtype(self):
    fragment_value = np.full([4, 4], 2.0)
    self.assertEqual(
        fragment_value.astype(np.int8).nbytes,
        Fragment(index=np.s_[1:5:1, 3:7:1], value=fragment_value).nbytes_astype(
            np.dtype(np.int8)
        ),
    )

  def test_nbytes_astype_of_abstract_fragment_uses_given_dtype(self):
    self.assertEqual(
        4 * 4 * 2,
        Fragment(index=np.s_[1:5:1, 3:7:1]).nbytes_astype(
            np.dtype(jax.numpy.bfloat16)
        ),
    )


class FragmentsTest(absltest.TestCase):

  def test_can_be_constructed_with_fragment_list(self):
    Fragments(
        shape=(4, 4),
        dtype=np.dtype(np.float32),
        fragments=[Fragment(index=np.s_[1:2:1, 3:4:1])],
    )

  def test_cannot_be_constructed_with_non_fragment_list(self):
    with self.assertRaises(TypeError):
      Fragments(
          shape=(4, 4),
          dtype=np.dtype(np.float32),
          fragments=[np.s_[1:2:1, 3:4:1]],
      )

  def test_non_degenerate_fragments(self):
    fragments = Fragments(
        shape=(5, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            Fragment(index=np.s_[1:2:1, 3:4:1]),
            Fragment(index=np.s_[3:5:1, 0:2:1]),
        ],
    )
    self.assertFalse(fragments.is_degenerate())

  def test_non_degenerate_fragments_due_to_one_non_empty_fragment(self):
    fragments = Fragments(
        shape=(5, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            Fragment(index=np.s_[1:2:1, 3:4:1]),
            Fragment(index=np.s_[3:3:1, 0:2:1]),
        ],
    )
    self.assertFalse(fragments.is_degenerate())

  def test_non_degenerate_fragments_due_to_all_degenerate_fragment(self):
    fragments = Fragments(
        shape=(5, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            Fragment(index=np.s_[2:2:1, 3:4:1]),
            Fragment(index=np.s_[3:5:1, 0:0:1]),
        ],
    )
    self.assertTrue(fragments.is_degenerate())

  def test_nbytes_of_fragments_matches_nbytes_of_same_shaped_array(self):
    fragments = Fragments(
        shape=(5, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            Fragment(index=np.s_[1:2:1, 3:4:1]),
            Fragment(index=np.s_[3:5:1, 0:2:1]),
        ],
    )
    self.assertEqual(np.ones((5, 5), np.float32).nbytes, fragments.nbytes)

  def test_addressable_nbytes_of_abstract_fragments_is_sum_of_individual_fragment_bytes(
      self,
  ):
    fragments = Fragments(
        shape=(5, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            Fragment(index=np.s_[1:2:1, 3:4:1]),
            Fragment(index=np.s_[3:5:1, 0:2:1]),
        ],
    )
    self.assertEqual(((1 * 1) + (2 * 2)) * 4, fragments.addressable_nbytes)


class FragmentsToArrayTest(absltest.TestCase):

  def test_full_fragments_can_be_converted_to_numpy_array(self):
    fragments = Fragments(
        shape=(4, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            Fragment(index=np.s_[0:4:1, 0:2:1], value=np.full([4, 2], 88)),
            Fragment(index=np.s_[0:4:1, 2:5:1], value=np.full([4, 3], 99)),
        ],
    )
    a = np.asarray(fragments)
    np.testing.assert_array_equal(
        np.concatenate([np.full([4, 2], 88), np.full([4, 3], 99)], axis=1), a
    )

  def test_non_full_fragments_raises_exception(self):
    fragments = Fragments(
        shape=(4, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            Fragment(index=np.s_[0:4:1, 0:2:1], value=np.full([4, 2], 88))
        ],
    )
    with self.assertRaisesRegex(
        ValueError,
        r'Attempt to convert non-full Fragments to array',
    ):
      np.asarray(fragments)


class IsFullTest(absltest.TestCase):

  def test_filled_by_one_spanning_fragment(self):
    self.assertTrue(
        array_fragments._is_full(
            Fragments(
                shape=(4, 5),
                dtype=np.dtype(np.float32),
                fragments=[
                    Fragment(index=np.s_[0:4:1, 0:5:1]),
                ],
            )
        )
    )

  def test_filled_by_two_non_spanning_fragments(self):
    self.assertTrue(
        array_fragments._is_full(
            Fragments(
                shape=(4, 5),
                dtype=np.dtype(np.float32),
                fragments=[
                    Fragment(index=np.s_[0:4:1, 0:2:1]),
                    Fragment(index=np.s_[0:4:1, 2:5:1]),
                ],
            )
        )
    )

  def test_not_filled_by_two_non_spanning_fragments(self):
    self.assertFalse(
        array_fragments._is_full(
            Fragments(
                shape=(4, 5),
                dtype=np.dtype(np.float32),
                fragments=[
                    Fragment(index=np.s_[0:4:1, 0:2:1]),
                    Fragment(index=np.s_[2:4:1, 2:5:1]),
                ],
            )
        )
    )

  def test_not_filled_by_no_fragments(self):
    self.assertFalse(
        array_fragments._is_full(
            Fragments(shape=(4, 5), dtype=np.dtype(np.float32), fragments=[])
        )
    )

  def test_rank_0_fragments_not_filled_by_no_fragments(self):
    self.assertFalse(
        array_fragments._is_full(
            Fragments(shape=(), dtype=np.dtype(np.float32), fragments=[])
        )
    )

  def test_dim_0_fragments_filled_by_no_fragments(self):
    self.assertTrue(
        array_fragments._is_full(
            Fragments(shape=(0,), dtype=np.dtype(np.float32), fragments=[])
        )
    )


class AddressableShardsTest(absltest.TestCase):

  def test_unsharded_array_is_fully_replicated(self):
    self.assertEqual(
        array_fragments.addressable_shards(
            jax.ShapeDtypeStruct((4, 5), np.dtype(np.float32))
        ),
        [np.s_[0:4:1, 0:5:1]],
    )


class AbstractFragmentsTest(absltest.TestCase):

  def test_returns_fragments_instance_itself(self):
    fragments = Fragments(
        shape=(2, 3), dtype=np.dtype(np.float32), fragments=[]
    )
    self.assertIs(fragments, array_fragments.abstract_fragments(fragments))

  def test_converts_fully_replicated_shape_dtype_struct(self):
    self.assertEqual(
        Fragments(
            shape=(4, 5),
            dtype=np.dtype(np.float32),
            fragments=[Fragment(index=np.s_[0:4:1, 0:5:1])],
        ),
        array_fragments.abstract_fragments(
            jax.ShapeDtypeStruct((4, 5), np.dtype(np.float32))
        ),
    )


class StackFragmentsTest(absltest.TestCase):

  def test_stacks_fragments_of_same_shape(self):
    fragments = Fragments(
        shape=(24, 35),
        dtype=np.dtype(np.float32),
        fragments=[
            Fragment(index=np.s_[12:16:1, 10:15:1], value=np.full((4, 5), 1.0)),
            Fragment(index=np.s_[16:20:1, 10:15:1], value=np.full((4, 5), 2.0)),
            Fragment(index=np.s_[12:16:1, 10:15:1], value=np.full((4, 5), 3.0)),
            Fragment(index=np.s_[16:20:1, 10:15:1], value=np.full((4, 5), 4.0)),
        ],
    )

    expected_array = np.stack([
        np.full((4, 5), 1.0),
        np.full((4, 5), 2.0),
        np.full((4, 5), 3.0),
        np.full((4, 5), 4.0),
    ])

    actual_array = array_fragments.stack_fragments(fragments)
    np.testing.assert_array_equal(expected_array, actual_array)

  def test_stacks_single_fragment_without_copy(self):
    value = np.arange(20).reshape((4, 5))
    fragments = Fragments(
        shape=(4, 5),
        dtype=np.dtype(np.float32),
        fragments=[Fragment(index=np.s_[0:4:1, 0:5:1], value=value)],
    )

    actual_array = array_fragments.stack_fragments(fragments)
    assert actual_array is not None
    self.assertEqual(actual_array.shape, (1, 4, 5))
    # Result is a view of the original `np.arange(20)` array.
    self.assertIs(actual_array.base, value.base)
    np.testing.assert_array_equal(
        actual_array, np.arange(20).reshape((1, 4, 5))
    )

  def test_returns_none_for_none(self):
    self.assertIsNone(array_fragments.stack_fragments(None))

  def assert_stacking_is_rejected(
      self,
      invalid_fragments: Fragments,
      expected_error_message: str,
  ) -> None:
    with self.assertRaisesRegex(ValueError, expected_error_message):
      array_fragments.validate_fragments_can_be_stacked(invalid_fragments)
    with self.assertRaisesRegex(ValueError, expected_error_message):
      array_fragments.stack_fragments(invalid_fragments)

  def test_rejects_empty_fragments_list(self):
    empty_fragments = Fragments(
        shape=(24, 35), dtype=np.dtype(np.float32), fragments=[]
    )
    self.assert_stacking_is_rejected(empty_fragments, 'No fragments to stack')

  def test_rejects_fragments_of_different_shapes(self):
    fragments = Fragments(
        shape=(24, 35),
        dtype=np.dtype(np.float32),
        fragments=[
            Fragment(index=np.s_[12:16:1, 10:15:1], value=np.full((4, 5), 1.0)),
            Fragment(
                index=np.s_[16:20:1, 10:20:1], value=np.full((4, 10), 2.0)
            ),
        ],
    )
    self.assert_stacking_is_rejected(fragments, 'Differently-shaped fragments')

  def test_rejects_abstract_fragments(self):
    fragments = Fragments(
        shape=(24, 35),
        dtype=np.dtype(np.float32),
        fragments=[
            Fragment(index=np.s_[12:16:1, 10:15:1], value=np.full((4, 5), 1.0)),
            Fragment(index=np.s_[16:20:1, 15:20:1], value=None),
        ],
    )

    self.assert_stacking_is_rejected(fragments, 'Not all fragments have values')


if __name__ == '__main__':
  absltest.main()
