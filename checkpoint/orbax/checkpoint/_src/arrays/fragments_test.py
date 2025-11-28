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

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from orbax.checkpoint._src.arrays import fragments as array_fragments


AnyFragment = array_fragments.NpFragment | array_fragments.JaxFragment
AnyFragments = array_fragments.NpFragments | array_fragments.JaxFragments


@parameterized.named_parameters(
    ('np_fragment', array_fragments.NpFragment),
    ('jax_fragment', array_fragments.JaxFragment),
)
class FragmentTest(parameterized.TestCase):

  def test_construction(self, fragment_t: type[AnyFragment]):
    np_api = fragment_t.NP_API
    with self.subTest('rejects_index_as_ndarray'):
      index = np.s_[1:2:1, 3:4:1]
      with self.assertRaisesRegex(TypeError, 'must be a tuple of slices'):
        _ = fragment_t(index=array_fragments._ndarray_from_index(index))  # pytype: disable=wrong-arg-types

    with self.subTest('with_np_index'):
      f = fragment_t(index=np.s_[1:2:1, 3:4:1])
      self.assertEqual(f, fragment_t(np_index=f.np_index))

    with self.subTest('with_dataclasses_replace_value'):
      f_a = fragment_t(index=np.s_[0:2:1, 0:3:1])
      value = np_api.ones((2, 3))
      f_c = dataclasses.replace(f_a, value=value)
      self.assertEqual(f_c, fragment_t(index=f_a.index, value=value))

    with self.subTest('with_dataclasses_replace_np_index'):
      value = np_api.ones((2, 3))
      old_np_index = array_fragments._ndarray_from_index(np.s_[0:2:1, 0:3:1])
      new_np_index = array_fragments._ndarray_from_index(np.s_[2:4:1, 3:6:1])
      f_old = fragment_t(np_index=old_np_index, value=value)
      f_new = dataclasses.replace(f_old, np_index=new_np_index)
      self.assertEqual(f_new, fragment_t(np_index=new_np_index, value=value))

    with self.subTest('only_one_of_index_or_np_index_is_allowed'):
      idx = np.s_[1:2:1, 3:4:1]
      with self.assertRaisesRegex(ValueError, 'both index and np_index'):
        _ = fragment_t(
            index=idx, np_index=array_fragments._ndarray_from_index(idx)
        )

    with self.subTest('one_of_index_or_np_index_must_be_specified'):
      with self.assertRaisesRegex(ValueError, 'either index or np_index'):
        _ = fragment_t()

  def test_equality(self, fragment_t: type[AnyFragment]):
    np_api = fragment_t.NP_API
    self.assertEqual(
        fragment_t(index=np.s_[1:2:1, 3:4:1]),
        fragment_t(index=np.s_[1:2:1, 3:4:1]),
    )
    self.assertNotEqual(
        fragment_t(index=np.s_[1:2:1, 3:4:1]),
        fragment_t(index=np.s_[1:2:1, 4:5:1]),
    )
    self.assertNotEqual(
        fragment_t(index=np.s_[1:2:1, 3:4:1], value=np_api.ones([4, 4]) * 2.0),
        fragment_t(index=np.s_[1:2:1, 3:4:1], value=np_api.ones([4, 4]) * 3.0),
    )

  def test_nd_properties(self, fragment_t: type[AnyFragment]):
    f = fragment_t(index=np.s_[1:8:2, 3:9:4])
    np.testing.assert_array_equal([1, 3], f.start)
    np.testing.assert_array_equal([8, 9], f.stop)
    np.testing.assert_array_equal([2, 4], f.step)

  def test_rank_zero(self, fragment_t: type[AnyFragment]):
    np_api = fragment_t.NP_API
    f = fragment_t(index=(), value=np_api.empty([]))
    np.testing.assert_array_equal([], f.start)
    np.testing.assert_array_equal([], f.stop)
    np.testing.assert_array_equal([], f.step)

  def test_repr(self, fragment_t: type[AnyFragment]):
    np_api = fragment_t.NP_API
    self.assertEqual(
        repr(fragment_t(index=np.s_[1:2:1, 3:4:1])),
        f'{fragment_t.__name__}(index=np.s_[1:2:1, 3:4:1])',
    )
    self.assertEqual(
        repr(fragment_t(index=np.s_[1:2:1, 3:4:1], value=np_api.ones([4, 4]))),
        f'{fragment_t.__name__}(index=np.s_[1:2:1, 3:4:1], value=...)',
    )

  def test_nbytes_of_concrete_fragment_is_nbytes_of_its_value(
      self, fragment_t: type[AnyFragment]
  ):
    np_api = fragment_t.NP_API
    fragment_value = np_api.full([4, 4], 2.0)
    self.assertEqual(
        fragment_value.nbytes,
        fragment_t(index=np.s_[1:5:1, 3:7:1], value=fragment_value).nbytes,
    )

  def test_nbytes_of_abstract_fragment_raises_value_error(
      self, fragment_t: type[AnyFragment]
  ):
    with self.assertRaisesRegex(ValueError, 'nbytes of abstract'):
      _ = fragment_t(index=np.s_[1:5:1, 3:7:1]).nbytes

  def test_nbytes_astype_of_concrete_fragment_uses_given_dtype(
      self, fragment_t: type[AnyFragment]
  ):
    np_api = fragment_t.NP_API
    fragment_value = np_api.full([4, 4], 2.0)
    self.assertEqual(
        fragment_value.astype(np.int8).nbytes,
        fragment_t(
            index=np.s_[1:5:1, 3:7:1], value=fragment_value
        ).nbytes_astype(np.dtype(np.int8)),
    )

  def test_nbytes_astype_of_abstract_fragment_uses_given_dtype(
      self, fragment_t: type[AnyFragment]
  ):
    self.assertEqual(
        4 * 4 * 2,
        fragment_t(index=np.s_[1:5:1, 3:7:1]).nbytes_astype(
            np.dtype(jax.numpy.bfloat16)
        ),
    )

  def test_slice(self, fragment_t: type[AnyFragment]):
    np_api = fragment_t.NP_API
    full_value = np_api.arange(8 * 9).reshape((8, 9))
    fragment_index = np.s_[4:8:1, 3:9:1]

    f = fragment_t(index=fragment_index, value=full_value[fragment_index])

    with self.subTest('fully_within_fragment_index'):
      slice_index = np.s_[5:7:1, 4:8:1]
      s = f.slice(array_fragments._ndarray_from_index(slice_index))
      self.assertEqual(
          fragment_t(index=np.s_[0:2:1, 0:4:1], value=full_value[slice_index]),
          s,
      )

    with self.subTest('fully_enclosing_fragment_index'):
      slice_index = np.s_[2:10:1, 1:11:1]
      s = f.slice(array_fragments._ndarray_from_index(slice_index))
      self.assertEqual(fragment_t(index=np.s_[2:6:1, 2:8:1], value=f.value), s)

    with self.subTest('spanning_fragment_start'):
      slice_index = np.s_[2:6:1, 2:4:1]
      s = f.slice(array_fragments._ndarray_from_index(slice_index))
      self.assertEqual(
          fragment_t(index=np.s_[2:4:1, 1:2:1], value=f.value[:2, :1]), s
      )

    with self.subTest('spanning_fragment_stop'):
      slice_index = np.s_[6:10:1, 6:10:1]
      s = f.slice(array_fragments._ndarray_from_index(slice_index))
      self.assertEqual(
          fragment_t(index=np.s_[0:2:1, 0:3:1], value=f.value[2:, 3:]), s
      )

    with self.subTest('with_no_overlap'):
      self.assertIsNone(
          f.slice(array_fragments._ndarray_from_index(np.s_[10:12:1, 10:12:1]))
      )
      # This is within the bounds of the fragment but spans no elements.
      self.assertIsNone(
          f.slice(array_fragments._ndarray_from_index(np.s_[6:6:1, 3:9:1]))
      )

    with self.subTest('rank_0'):
      s = fragment_t(index=(), value=np_api.ones([])).slice(
          np.zeros([0, 3], dtype=int)
      )
      self.assertIsNotNone(s)
      self.assertEqual((), s.index)
      self.assertIsInstance(s.value, np_api.ndarray)


@parameterized.named_parameters(
    ('np_fragments', array_fragments.NpFragments),
    ('jax_fragments', array_fragments.JaxFragments),
)
class FragmentsTest(parameterized.TestCase):

  def test_can_be_constructed_with_fragment_list(
      self, fragments_t: type[AnyFragments]
  ):
    fragment_t = fragments_t.FRAGMENT_T
    fragments_t(
        shape=(4, 4),
        dtype=np.dtype(np.float32),
        fragments=[fragment_t(index=np.s_[1:2:1, 3:4:1])],
    )

  def test_cannot_be_constructed_with_non_fragment_list(
      self, fragments_t: type[AnyFragments]
  ):
    with self.assertRaises(TypeError):
      fragments_t(
          shape=(4, 4),
          dtype=np.dtype(np.float32),
          fragments=[np.s_[1:2:1, 3:4:1]],
      )

  def test_non_degenerate_fragments(self, fragments_t: type[AnyFragments]):
    fragment_t = fragments_t.FRAGMENT_T
    fragments = fragments_t(
        shape=(5, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            fragment_t(index=np.s_[1:2:1, 3:4:1]),
            fragment_t(index=np.s_[3:5:1, 0:2:1]),
        ],
    )
    self.assertFalse(fragments.is_degenerate())

  def test_non_degenerate_fragments_due_to_one_non_empty_fragment(
      self, fragments_t: type[AnyFragments]
  ):
    fragment_t = fragments_t.FRAGMENT_T
    fragments = fragments_t(
        shape=(5, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            fragment_t(index=np.s_[1:2:1, 3:4:1]),
            fragment_t(index=np.s_[3:3:1, 0:2:1]),
        ],
    )
    self.assertFalse(fragments.is_degenerate())

  def test_non_degenerate_fragments_due_to_all_degenerate_fragment(
      self, fragments_t: type[AnyFragments]
  ):
    fragment_t = fragments_t.FRAGMENT_T
    fragments = fragments_t(
        shape=(5, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            fragment_t(index=np.s_[2:2:1, 3:4:1]),
            fragment_t(index=np.s_[3:5:1, 0:0:1]),
        ],
    )
    self.assertTrue(fragments.is_degenerate())

  def test_nbytes_of_fragments_matches_nbytes_of_same_shaped_array(
      self, fragments_t: type[AnyFragments]
  ):
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    fragments = fragments_t(
        shape=(5, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            fragment_t(index=np.s_[1:2:1, 3:4:1]),
            fragment_t(index=np.s_[3:5:1, 0:2:1]),
        ],
    )
    self.assertEqual(np_api.ones((5, 5), np.float32).nbytes, fragments.nbytes)

  def test_addressable_nbytes_of_abstract_fragments_is_sum_of_individual_fragment_bytes(
      self, fragments_t: type[AnyFragments]
  ):
    fragment_t = fragments_t.FRAGMENT_T
    fragments = fragments_t(
        shape=(5, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            fragment_t(index=np.s_[1:2:1, 3:4:1]),
            fragment_t(index=np.s_[3:5:1, 0:2:1]),
        ],
    )
    self.assertEqual(((1 * 1) + (2 * 2)) * 4, fragments.addressable_nbytes)

  def test_slice(self, fragments_t: type[AnyFragments]):
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    full_value = np_api.arange(5 * 5).reshape((5, 5))
    fragments = fragments_t(
        shape=(5, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            fragment_t(index=np.s_[1:2:1, 3:4:1], value=full_value[1:2, 3:4]),
            fragment_t(index=np.s_[3:5:1, 0:2:1], value=full_value[3:5, 0:2]),
        ],
    )

    sliced = fragments.slice(np.s_[2:4:1, 0:5:1])
    self.assertEqual(
        fragments_t(
            shape=(2, 5),
            dtype=np.dtype(np.float32),
            fragments=[
                # (The first fragment is entirely sliced away.)
                fragment_t(
                    index=np.s_[1:2:1, 0:2:1], value=full_value[3:4, 0:2]
                ),
            ],
        ),
        sliced,
    )


@parameterized.named_parameters(
    ('np_fragments', array_fragments.NpFragments),
    ('jax_fragments', array_fragments.JaxFragments),
)
class FragmentsToArrayTest(parameterized.TestCase):

  def test_full_fragments_can_be_converted_to_numpy_array(
      self, fragments_t: type[AnyFragments]
  ):
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    fragments = fragments_t(
        shape=(4, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            fragment_t(
                index=np.s_[0:4:1, 0:2:1], value=np_api.full([4, 2], 88)
            ),
            fragment_t(
                index=np.s_[0:4:1, 2:5:1], value=np_api.full([4, 3], 99)
            ),
        ],
    )
    a = np.asarray(fragments)
    np.testing.assert_array_equal(
        np.concatenate(
            [np_api.full([4, 2], 88), np_api.full([4, 3], 99)], axis=1
        ),
        a,
    )

  def test_non_full_fragments_raises_exception(
      self, fragments_t: type[AnyFragments]
  ):
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    fragments = fragments_t(
        shape=(4, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            fragment_t(index=np.s_[0:4:1, 0:2:1], value=np_api.full([4, 2], 88))
        ],
    )
    with self.assertRaisesRegex(
        ValueError,
        r'Attempt to convert non-full Fragments to array',
    ):
      np.asarray(fragments)


@parameterized.named_parameters(
    ('np_fragments', array_fragments.NpFragments),
    ('jax_fragments', array_fragments.JaxFragments),
)
class IsFullTest(parameterized.TestCase):

  def test_filled_by_one_spanning_fragment(
      self, fragments_t: type[AnyFragments]
  ):
    fragment_t = fragments_t.FRAGMENT_T
    self.assertTrue(
        array_fragments._is_full(
            fragments_t(
                shape=(4, 5),
                dtype=np.dtype(np.float32),
                fragments=[
                    fragment_t(index=np.s_[0:4:1, 0:5:1]),
                ],
            )
        )
    )

  def test_filled_by_two_non_spanning_fragments(
      self, fragments_t: type[AnyFragments]
  ):
    fragment_t = fragments_t.FRAGMENT_T
    self.assertTrue(
        array_fragments._is_full(
            fragments_t(
                shape=(4, 5),
                dtype=np.dtype(np.float32),
                fragments=[
                    fragment_t(index=np.s_[0:4:1, 0:2:1]),
                    fragment_t(index=np.s_[0:4:1, 2:5:1]),
                ],
            )
        )
    )

  def test_not_filled_by_two_non_spanning_fragments(
      self, fragments_t: type[AnyFragments]
  ):
    fragment_t = fragments_t.FRAGMENT_T
    self.assertFalse(
        array_fragments._is_full(
            fragments_t(
                shape=(4, 5),
                dtype=np.dtype(np.float32),
                fragments=[
                    fragment_t(index=np.s_[0:4:1, 0:2:1]),
                    fragment_t(index=np.s_[2:4:1, 2:5:1]),
                ],
            )
        )
    )

  def test_not_filled_by_no_fragments(self, fragments_t: type[AnyFragments]):
    self.assertFalse(
        array_fragments._is_full(
            fragments_t(shape=(4, 5), dtype=np.dtype(np.float32), fragments=[])
        )
    )

  def test_rank_0_fragments_not_filled_by_no_fragments(
      self, fragments_t: type[AnyFragments]
  ):
    self.assertFalse(
        array_fragments._is_full(
            fragments_t(shape=(), dtype=np.dtype(np.float32), fragments=[])
        )
    )

  def test_dim_0_fragments_filled_by_no_fragments(
      self, fragments_t: type[AnyFragments]
  ):
    self.assertTrue(
        array_fragments._is_full(
            fragments_t(shape=(0,), dtype=np.dtype(np.float32), fragments=[])
        )
    )


class AddressableShardsTest(parameterized.TestCase):

  def test_unsharded_array_is_fully_replicated(self):
    self.assertEqual(
        array_fragments.addressable_shards(
            jax.ShapeDtypeStruct((4, 5), np.dtype(np.float32))
        ),
        [np.s_[0:4:1, 0:5:1]],
    )


@parameterized.named_parameters(
    ('np_fragments', array_fragments.NpFragments),
    ('jax_fragments', array_fragments.JaxFragments),
)
class AbstractFragmentsTest(parameterized.TestCase):

  def test_returns_fragments_instance_itself(
      self, fragments_t: type[AnyFragments]
  ):
    fragments = fragments_t(
        shape=(2, 3), dtype=np.dtype(np.float32), fragments=[]
    )
    self.assertIs(fragments, array_fragments.abstract_fragments(fragments))

  def test_converts_fully_replicated_shape_dtype_struct(
      self, fragments_t: type[AnyFragments]
  ):
    fragment_t = fragments_t.FRAGMENT_T
    self.assertEqual(
        fragments_t(
            shape=(4, 5),
            dtype=np.dtype(np.float32),
            fragments=[fragment_t(index=np.s_[0:4:1, 0:5:1])],
        ),
        array_fragments.abstract_fragments(
            jax.ShapeDtypeStruct((4, 5), np.dtype(np.float32)),
            fragments_t=fragments_t,
        ),
    )


@parameterized.named_parameters(
    ('np_fragments', array_fragments.NpFragments),
    ('jax_fragments', array_fragments.JaxFragments),
)
class StackFragmentsTest(parameterized.TestCase):

  def test_stacks_fragments_of_same_shape(
      self, fragments_t: type[AnyFragments]
  ):
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    fragments = fragments_t(
        shape=(24, 35),
        dtype=np.dtype(np.float32),
        fragments=[
            fragment_t(
                index=np.s_[12:16:1, 10:15:1], value=np_api.full((4, 5), 1.0)
            ),
            fragment_t(
                index=np.s_[16:20:1, 10:15:1], value=np_api.full((4, 5), 2.0)
            ),
            fragment_t(
                index=np.s_[12:16:1, 10:15:1], value=np_api.full((4, 5), 3.0)
            ),
            fragment_t(
                index=np.s_[16:20:1, 10:15:1], value=np_api.full((4, 5), 4.0)
            ),
        ],
    )

    expected_array = np.stack([
        np_api.full((4, 5), 1.0),
        np_api.full((4, 5), 2.0),
        np_api.full((4, 5), 3.0),
        np_api.full((4, 5), 4.0),
    ])

    actual_array = array_fragments.stack_fragments(fragments)
    np.testing.assert_array_equal(expected_array, actual_array)

  def test_stacks_single_fragment_without_copy(
      self, fragments_t: type[AnyFragments]
  ):
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    value = np_api.arange(20).reshape((4, 5))
    fragments = fragments_t(
        shape=(4, 5),
        dtype=np.dtype(np.float32),
        fragments=[fragment_t(index=np.s_[0:4:1, 0:5:1], value=value)],
    )

    actual_array = array_fragments.stack_fragments(fragments)
    assert actual_array is not None
    self.assertEqual(actual_array.shape, (1, 4, 5))

    # jax.Array has no base attribute.
    if np_api is np:
      assert isinstance(actual_array, np.ndarray)
      assert isinstance(value, np.ndarray)
      with self.subTest('base_is_same_as_original'):
        # Result is a view of the original `np_api.arange(20)` array.
        self.assertIs(actual_array.base, value.base)

    np.testing.assert_array_equal(
        actual_array, np_api.arange(20).reshape((1, 4, 5))
    )

  def test_returns_none_for_none(self, fragments_t: type[AnyFragments]):
    del fragments_t  # Unused.
    self.assertIsNone(array_fragments.stack_fragments(None))

  def assert_stacking_is_rejected(
      self,
      invalid_fragments: AnyFragments,
      expected_error_message: str,
  ) -> None:
    with self.assertRaisesRegex(ValueError, expected_error_message):
      array_fragments.validate_fragments_can_be_stacked(invalid_fragments)
    with self.assertRaisesRegex(ValueError, expected_error_message):
      array_fragments.stack_fragments(invalid_fragments)

  def test_rejects_empty_fragments_list(self, fragments_t: type[AnyFragments]):
    empty_fragments = fragments_t(
        shape=(24, 35), dtype=np.dtype(np.float32), fragments=[]
    )
    self.assert_stacking_is_rejected(empty_fragments, 'No fragments to stack')

  def test_rejects_fragments_of_different_shapes(
      self, fragments_t: type[AnyFragments]
  ):
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    fragments = fragments_t(
        shape=(24, 35),
        dtype=np.dtype(np.float32),
        fragments=[
            fragment_t(
                index=np.s_[12:16:1, 10:15:1], value=np_api.full((4, 5), 1.0)
            ),
            fragment_t(
                index=np.s_[16:20:1, 10:20:1], value=np_api.full((4, 10), 2.0)
            ),
        ],
    )
    self.assert_stacking_is_rejected(fragments, 'Differently-shaped fragments')

  def test_rejects_abstract_fragments(self, fragments_t: type[AnyFragments]):
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    fragments = fragments_t(
        shape=(24, 35),
        dtype=np.dtype(np.float32),
        fragments=[
            fragment_t(
                index=np.s_[12:16:1, 10:15:1], value=np_api.full((4, 5), 1.0)
            ),
            fragment_t(index=np.s_[16:20:1, 15:20:1], value=None),
        ],
    )

    self.assert_stacking_is_rejected(fragments, 'Not all fragments have values')

Fragment = array_fragments.Fragment
Fragments = array_fragments.Fragments


class BackwardsCompatibleTypesTest(absltest.TestCase):

  def test_fragments_and_fragment_type_aliases_still_work(self):

    def _process_fragment(target_fragment: Fragment) -> Fragment:
      return target_fragment

    def _process_fragments(
        target_fragments: Fragments,
    ) -> list[Fragment]:
      """Calculates read chunks to load, covering the given fragments."""
      result: list[Fragment] = []
      for target_fragment in target_fragments.fragments:
        result.append(_process_fragment(target_fragment))
      return result

    del _process_fragments


if __name__ == '__main__':
  absltest.main()
