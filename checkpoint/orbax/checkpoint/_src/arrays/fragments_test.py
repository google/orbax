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

import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.arrays import fragments as array_fragments
from orbax.checkpoint._src.arrays import numpy_utils as np_utils

AbstractFragment = array_fragments.AbstractFragment
AbstractFragments = array_fragments.AbstractFragments
NpFragment = array_fragments.NpFragment
NpFragments = array_fragments.NpFragments
JaxFragment = array_fragments.JaxFragment
JaxFragments = array_fragments.JaxFragments

FragmentT = type[array_fragments.F]
FragmentsT = type[array_fragments.FS]
ConcreteFragmentT = type[array_fragments.Fconcrete]
ConcreteFragmentsT = type[array_fragments.FSconcrete]


class FragmentTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('abstract_fragment', AbstractFragment),
      ('np_fragment', NpFragment),
      ('jax_fragment', JaxFragment),
  )
  def test_construction(self, fragment_t: FragmentT):
    np_api = fragment_t.NP_API
    value = np_api.ones((2, 3)) if np_api is not None else None
    with self.subTest('rejects_index_as_ndarray'):
      index = np.s_[1:2:1, 3:4:1]
      with self.assertRaisesRegex(TypeError, 'must be a tuple of slices'):
        _ = fragment_t(
            index=array_fragments._ndarray_from_index(index),
            value=value
        )  # pytype: disable=wrong-arg-types

    with self.subTest('with_np_index'):
      f = fragment_t(index=np.s_[1:2:1, 3:4:1], value=value)
      self.assertEqual(f, fragment_t(np_index=f.np_index, value=value))

    with self.subTest('with_dataclasses_replace_value'):
      f_a = fragment_t(index=np.s_[0:2:1, 0:3:1], value=value)
      new_value = value + 1 if value is not None else None
      f_c = dataclasses.replace(f_a, value=new_value)
      self.assertEqual(f_c, fragment_t(index=f_a.index, value=new_value))

    with self.subTest('with_dataclasses_replace_np_index'):
      old_np_index = array_fragments._ndarray_from_index(np.s_[0:2:1, 0:3:1])
      new_np_index = array_fragments._ndarray_from_index(np.s_[2:4:1, 3:6:1])
      f_old = fragment_t(np_index=old_np_index, value=value)
      f_new = dataclasses.replace(f_old, np_index=new_np_index)
      self.assertEqual(f_new, fragment_t(np_index=new_np_index, value=value))

    with self.subTest('only_one_of_index_or_np_index_is_allowed'):
      idx = np.s_[1:2:1, 3:4:1]
      with self.assertRaisesRegex(ValueError, 'both index and np_index'):
        _ = fragment_t(
            index=idx, np_index=array_fragments._ndarray_from_index(idx),
            value=value,
        )

    with self.subTest('wrong_type_of_value'):
      wrong_value_type = {
          AbstractFragment: NpFragment,
          NpFragment: JaxFragment,
          JaxFragment: AbstractFragment,
      }[fragment_t]
      wrong_np_api = wrong_value_type.NP_API
      wrong_value = (
          wrong_np_api.array([1.0]) if wrong_np_api is not None else None
      )
      with self.assertRaisesRegex(TypeError, 'Fragment value must be a'):
        _ = fragment_t(index=np.s_[1:2:1, 3:4:1], value=wrong_value)

    with self.subTest('one_of_index_or_np_index_must_be_specified'):
      with self.assertRaisesRegex(ValueError, 'either index or np_index'):
        _ = fragment_t(value=value)

  def test_equality_abstract(self):
    self.assertEqual(
        AbstractFragment(index=np.s_[1:2:1, 3:4:1]),
        AbstractFragment(index=np.s_[1:2:1, 3:4:1]),
    )
    self.assertNotEqual(
        AbstractFragment(index=np.s_[1:2:1, 3:4:1]),
        AbstractFragment(index=np.s_[1:2:1, 4:5:1]),
    )

  @parameterized.named_parameters(
      ('np_fragment', NpFragment),
      ('jax_fragment', JaxFragment),
  )
  def test_equality_concrete(
      self,
      fragment_t: ConcreteFragmentT,
  ):
    np_api = fragment_t.NP_API
    value = np_api.ones([4, 4]) if np_api is not None else None
    self.assertEqual(
        fragment_t(index=np.s_[1:2:1, 3:4:1], value=value),
        fragment_t(index=np.s_[1:2:1, 3:4:1], value=value),
    )
    self.assertNotEqual(
        fragment_t(index=np.s_[1:2:1, 3:4:1], value=value * 2.),
        fragment_t(index=np.s_[1:2:1, 3:4:1], value=value * 3.),
    )

  @parameterized.named_parameters(
      ('abstract_fragment', AbstractFragment),
      ('np_fragment', NpFragment),
      ('jax_fragment', JaxFragment),
  )
  def test_nd_properties(
      self,
      fragment_t: FragmentT,
  ):
    np_api = fragment_t.NP_API
    value = np_api.ones([10, 10]) if np_api is not None else None
    f = fragment_t(index=np.s_[1:8:2, 3:9:4], value=value)
    np.testing.assert_array_equal([1, 3], f.start)
    np.testing.assert_array_equal([8, 9], f.stop)
    np.testing.assert_array_equal([2, 4], f.step)

  @parameterized.named_parameters(
      ('abstract_fragment', AbstractFragment),
      ('np_fragment', NpFragment),
      ('jax_fragment', JaxFragment),
  )
  def test_rank_zero(
      self,
      fragment_t: FragmentT,
  ):
    np_api = fragment_t.NP_API
    f = fragment_t(
        index=(), value=np_api.ones([]) if np_api is not None else None
    )
    np.testing.assert_array_equal([], f.start)
    np.testing.assert_array_equal([], f.stop)
    np.testing.assert_array_equal([], f.step)

  def test_repr_abstract(self):
    self.assertEqual(
        repr(AbstractFragment(index=np.s_[1:2:1, 3:4:1])),
        'AbstractFragment(index=np.s_[1:2:1, 3:4:1])',
    )

  @parameterized.named_parameters(
      ('np_fragment', NpFragment),
      ('jax_fragment', JaxFragment),
  )
  def test_repr_concrete(
      self,
      fragment_t: ConcreteFragmentT,
  ):
    np_api = fragment_t.NP_API
    self.assertEqual(
        repr(fragment_t(index=np.s_[1:2:1, 3:4:1], value=np_api.ones([4, 4]))),
        f'{fragment_t.__name__}(index=np.s_[1:2:1, 3:4:1], value=...)',
    )

  @parameterized.named_parameters(
      ('np_fragment', NpFragment),
      ('jax_fragment', JaxFragment),
  )
  def test_nbytes_of_concrete_fragment_is_nbytes_of_its_value(
      self, fragment_t: ConcreteFragmentT
  ):
    np_api = fragment_t.NP_API
    fragment_value = np_api.full([4, 4], 2.0)
    self.assertEqual(
        fragment_value.nbytes,
        fragment_t(index=np.s_[1:5:1, 3:7:1], value=fragment_value).nbytes,
    )

  @parameterized.named_parameters(
      ('np_fragment', NpFragment),
      ('jax_fragment', JaxFragment),
  )
  def test_nbytes_astype_of_concrete_fragment_uses_given_dtype(
      self,
      fragment_t: ConcreteFragmentT,
  ):
    np_api = fragment_t.NP_API
    fragment_value = np_api.full([4, 4], 2.0)
    self.assertEqual(
        fragment_value.astype(np.int8).nbytes,
        fragment_t(
            index=np.s_[1:5:1, 3:7:1], value=fragment_value
        ).nbytes_astype(np.dtype(np.int8)),
    )

  def test_nbytes_astype_of_abstract_fragment_uses_given_dtype(self):
    self.assertEqual(
        4 * 4 * 2,
        AbstractFragment(
            index=np.s_[1:5:1, 3:7:1]
        ).nbytes_astype(np.dtype(jax.numpy.bfloat16)),
    )

  @parameterized.named_parameters(
      ('np_fragment', NpFragment),
      ('jax_fragment', JaxFragment),
  )
  def test_intersect(
      self,
      fragment_t: ConcreteFragmentT,
  ):
    np_api = fragment_t.NP_API
    full_value = np_api.arange(8 * 9).reshape((8, 9))
    fragment_index = np.s_[4:8:1, 3:9:1]

    f = fragment_t(index=fragment_index, value=full_value[fragment_index])

    with self.subTest('fully_within_fragment_index'):
      bounds = np.s_[5:7:1, 4:8:1]
      s = f.intersect(array_fragments._ndarray_from_index(bounds))
      self.assertEqual(
          fragment_t(index=np.s_[5:7:1, 4:8:1], value=full_value[bounds]),
          s,
      )

    with self.subTest('fully_enclosing_fragment_index'):
      bounds = np.s_[2:10:1, 1:11:1]
      s = f.intersect(array_fragments._ndarray_from_index(bounds))
      self.assertEqual(fragment_t(index=np.s_[4:8:1, 3:9:1], value=f.value), s)

    with self.subTest('spanning_fragment_start'):
      bounds = np.s_[2:6:1, 2:4:1]
      s = f.intersect(array_fragments._ndarray_from_index(bounds))
      self.assertEqual(
          fragment_t(index=np.s_[4:6:1, 3:4:1], value=f.value[:2, :1]), s
      )

    with self.subTest('spanning_fragment_stop'):
      bounds = np.s_[6:10:1, 6:10:1]
      s = f.intersect(array_fragments._ndarray_from_index(bounds))
      self.assertEqual(
          fragment_t(index=np.s_[6:8:1, 6:9:1], value=f.value[2:, 3:]), s
      )

    with self.subTest('with_no_overlap'):
      self.assertIsNone(
          f.intersect(
              array_fragments._ndarray_from_index(np.s_[10:12:1, 10:12:1])
          )
      )
      # This is within the bounds of the fragment but spans no elements.
      self.assertIsNone(
          f.intersect(array_fragments._ndarray_from_index(np.s_[6:6:1, 3:9:1]))
      )

    with self.subTest('rank_0'):
      s = fragment_t(index=(), value=np_api.ones([])).intersect(
          np.zeros([0, 3], dtype=int)
      )
      self.assertIsNotNone(s)
      self.assertEqual((), s.index)
      self.assertIsInstance(s.value, np_api.ndarray)

  @parameterized.named_parameters(
      ('np_fragment', NpFragment),
      ('jax_fragment', JaxFragment),
  )
  def test_slice(
      self,
      fragment_t: ConcreteFragmentT,
  ):
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
      ('np_fragment', NpFragment),
      ('jax_fragment', JaxFragment),
  )
  def test_slice_of_value(
      self,
      fragment_t: ConcreteFragmentT,
  ):
    np_api = fragment_t.NP_API
    full_value = np_api.arange(8 * 9).reshape((8, 9))
    fragment_index = np.s_[4:8:1, 3:9:1]
    fragment = fragment_t(
        index=fragment_index, value=full_value[fragment_index]
    )

    with self.subTest('returns_slice_of_value'):
      np.testing.assert_array_equal(
          full_value[np.s_[5:7:1, 4:8:1]],
          fragment.slice_of_value(
              array_fragments._ndarray_from_index(np.s_[5:7:1, 4:8:1])
          ),
      )

    with self.subTest('raises_if_slice_is_out_of_bounds'):
      with self.assertRaises(ValueError):
        fragment.slice_of_value(
            array_fragments._ndarray_from_index(np.s_[2:6:1, 3:9:1])
        )

      with self.assertRaises(ValueError):
        fragment.slice_of_value(
            array_fragments._ndarray_from_index(np.s_[4:8:1, 8:12:1])
        )


@parameterized.named_parameters(
    ('abstract_fragments', AbstractFragments),
    ('np_fragments', NpFragments),
    ('jax_fragments', JaxFragments),
)
class FragmentsTest(parameterized.TestCase):

  def test_can_be_constructed_with_fragment_list(
      self, fragments_t: FragmentsT
  ):
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    fragment_value = np_api.full([1, 1], 2.0) if np_api is not None else None
    fragments_t(
        shape=(4, 4),
        dtype=np.dtype(np.float32),
        fragments=[fragment_t(index=np.s_[1:2:1, 3:4:1], value=fragment_value)],
    )

  def test_cannot_be_constructed_with_wrong_type_of_fragment(
      self, fragments_t: FragmentsT
  ):
    wrong_fragment_type = {
        AbstractFragment: NpFragment,
        NpFragment: JaxFragment,
        JaxFragment: AbstractFragment,
    }[fragments_t.FRAGMENT_T]
    wrong_np_api = wrong_fragment_type.NP_API
    wrong_fragment = wrong_fragment_type(
        index=np.s_[1:2:1, 3:4:1],
        value=(wrong_np_api.array([1.0]) if wrong_np_api is not None else None),
    )
    with self.assertRaises(TypeError):
      fragments_t(
          shape=(4, 4),
          dtype=np.dtype(np.float32),
          fragments=[wrong_fragment],
      )

  def test_non_degenerate_fragments(self, fragments_t: FragmentsT):
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    fragment_value = (
        [np_api.full([1, 1], 2.0), np_api.full([2, 2], 3.0)]
        if np_api is not None
        else [None, None]
    )
    fragments = fragments_t(
        shape=(5, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            fragment_t(index=np.s_[1:2:1, 3:4:1], value=fragment_value[0]),
            fragment_t(index=np.s_[3:5:1, 0:2:1], value=fragment_value[1]),
        ],
    )
    self.assertFalse(fragments.is_degenerate())

  def test_non_degenerate_fragments_due_to_one_non_empty_fragment(
      self, fragments_t: FragmentsT
  ):
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    fragment_value = (
        [np_api.full([1, 1], 2.0), np_api.full([2, 2], 3.0)]
        if np_api is not None
        else [None, None]
    )
    fragments = fragments_t(
        shape=(5, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            fragment_t(index=np.s_[1:2:1, 3:4:1], value=fragment_value[0]),
            fragment_t(index=np.s_[3:3:1, 0:2:1], value=fragment_value[1]),
        ],
    )
    self.assertFalse(fragments.is_degenerate())

  def test_non_degenerate_fragments_due_to_all_degenerate_fragment(
      self, fragments_t: FragmentsT
  ):
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    fragment_value = (
        [np_api.full([0, 1], 2.0), np_api.full([2, 0], 3.0)]
        if np_api is not None
        else [None, None]
    )
    fragments = fragments_t(
        shape=(5, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            fragment_t(index=np.s_[2:2:1, 3:4:1], value=fragment_value[0]),
            fragment_t(index=np.s_[3:5:1, 0:0:1], value=fragment_value[1]),
        ],
    )
    self.assertTrue(fragments.is_degenerate())

  def test_nbytes_of_fragments_matches_nbytes_of_same_shaped_array(
      self, fragments_t: FragmentsT
  ):
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    fragment_value = (
        [np_api.full([1, 1], 2.0), np_api.full([2, 2], 3.0)]
        if np_api is not None
        else [None, None]
    )
    fragments = fragments_t(
        shape=(5, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            fragment_t(index=np.s_[1:2:1, 3:4:1], value=fragment_value[0]),
            fragment_t(index=np.s_[3:5:1, 0:2:1], value=fragment_value[1]),
        ],
    )
    # Use of `np` rather than `np_api` is deliberate here.
    self.assertEqual(np.ones((5, 5), np.float32).nbytes, fragments.nbytes)

  def test_addressable_nbytes_is_sum_of_individual_fragment_bytes(
      self, fragments_t: FragmentsT
  ):
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    fragment_value = (
        [np_api.full([1, 1], 2.0), np_api.full([2, 2], 3.0)]
        if np_api is not None
        else [None, None]
    )
    fragments = fragments_t(
        shape=(5, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            fragment_t(index=np.s_[1:2:1, 3:4:1], value=fragment_value[0]),
            fragment_t(index=np.s_[3:5:1, 0:2:1], value=fragment_value[1]),
        ],
    )
    self.assertEqual(((1 * 1) + (2 * 2)) * 4, fragments.addressable_nbytes)

  def test_slice(self, fragments_t: FragmentsT):
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    full_value = (
        np_api.arange(5 * 5).reshape((5, 5))
        if np_api is not None
        else [None, None]
    )
    fragments = fragments_t(
        shape=(5, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            fragment_t(
                index=np.s_[1:2:1, 3:4:1],
                value=full_value[1:2, 3:4] if np_api is not None else None,
            ),
            fragment_t(
                index=np.s_[3:5:1, 0:2:1],
                value=full_value[3:5, 0:2] if np_api is not None else None,
            ),
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
                    index=np.s_[1:2:1, 0:2:1],
                    value=full_value[3:4, 0:2] if np_api is not None else None,
                ),
            ],
        ),
        sliced,
    )


@parameterized.named_parameters(
    ('np_fragments', NpFragments),
    ('jax_fragments', JaxFragments),
)
class FragmentsToArrayTest(parameterized.TestCase):

  def test_full_fragments_can_be_converted_to_numpy_array(
      self, fragments_t: ConcreteFragmentsT
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

    expected_array = np.concatenate(
        [np_api.full([4, 2], 88), np_api.full([4, 3], 99)], axis=1
    )

    with self.subTest('with_default_dtype'):
      np.testing.assert_array_equal(expected_array, np.asarray(fragments))

    with self.subTest('with_explicit_dtype'):
      np.testing.assert_array_equal(
          expected_array.astype(int), np.asarray(fragments, dtype=int)
      )

    with self.subTest('with_explicit_copy'):
      a = np.asarray(fragments, copy=True)
      np.testing.assert_array_equal(expected_array, a)
      self.assertIsNone(a.base)
      self.assertFalse(any(a is f.value for f in fragments.fragments))

    with self.subTest('with_explicit_no_copy'):
      with self.assertRaisesRegex(
          ValueError, 'Attempt to convert Fragments to array without copying'
      ):
        np.asarray(fragments, copy=False)

  def test_full_singleton_fragments_can_be_converted_to_numpy_array(
      self, fragments_t: ConcreteFragmentsT
  ):
    # There is a special case for singleton fragments so we need to
    # exercise that separately.

    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    fragments = fragments_t(
        shape=(4, 5),
        dtype=np.dtype(np.float32),
        fragments=[
            fragment_t(
                index=np.s_[0:4:1, 0:5:1], value=np_api.full([4, 5], 88)
            ),
        ],
    )

    with self.subTest('with_explicit_copy'):
      a = np.asarray(fragments, copy=True)
      np.testing.assert_array_equal(fragments.fragments[0].value, a)
      self.assertIsNone(a.base)
      self.assertIsNot(fragments.fragments[0].value, a)

    with self.subTest('with_explicit_no_copy'):
      a = np.asarray(fragments, copy=False)
      if fragments_t is NpFragments:
        self.assertIs(a, fragments.fragments[0].value)
      elif fragments_t is JaxFragments:
        assert (base := a.base) is not None
        self.assertIs(base.obj, fragments.fragments[0].value)
      else:
        raise ValueError(f'Unexpected fragments type: {fragments_t}')

  def test_non_full_fragments_raises_exception(
      self, fragments_t: ConcreteFragmentsT
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


def _fake_sharding(shape) -> jax.sharding.NamedSharding:
  sharding = mock.Mock(spec=jax.sharding.NamedSharding)
  # List each slice twice to all tests to show that they're deduplicated.
  sharding.addressable_devices_indices_map.return_value = dict(
      (mock.Mock(), np.s_[0:2:1, y:y+1:1]) for y in [*range(shape[1])[::2]] * 2
  )
  return sharding


def _fake_jnp_ones(shape, dtype, sharding) -> jax.Array:
  return mock.Mock(
      spec=jax.Array,
      shape=shape,
      dtype=dtype,
      sharding=sharding,
      __getitem__=lambda self, index: jnp.ones(
          np_utils.slice_shape(index), dtype=dtype
      ),
  )


class FragmentsClassMethodsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('np_array', NpFragments),
      ('jnp_array', JaxFragments),
  )
  def test_concrete_fragments_from_all_of(self, fragments_t: FragmentsT):
    shape = (2, 3)
    dtype = np.dtype(np.float32)
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    other_np_api = np if np_api is jnp else jnp

    with self.subTest('with_array'):
      x = np_api.ones(shape, dtype=dtype)
      fs = fragments_t.all_of(x)
      self.assertEqual(fs.shape, shape)
      self.assertEqual(fs.dtype, dtype)
      self.assertEqual(
          fs.fragments, [fragment_t(index=np.s_[0:2:1, 0:3:1], value=x)]
      )

    with self.subTest('with_wrong_array_type_raises'):
      with self.assertRaisesRegex(TypeError, 'Fragment value must be'):
        fragments_t.all_of(other_np_api.ones(shape, dtype=dtype))

    with self.subTest('with_shape_dtype_struct_raises'):
      x = jax.ShapeDtypeStruct(shape, dtype)
      with self.assertRaisesRegex(TypeError, 'Fragment value must be'):
        fragments_t.all_of(x)

  @parameterized.named_parameters(
      ('shape_dtype_struct', jax.ShapeDtypeStruct),
      ('np_array', np.ones),
      ('jnp_array', jnp.ones),
  )
  def test_abstract_fragments_from_all_of(self, value_fn):
    fragments_t = AbstractFragments
    shape = (2, 3)
    dtype = np.dtype(np.float32)
    fragment_t = fragments_t.FRAGMENT_T

    x = value_fn(shape, dtype=dtype)
    fs = fragments_t.all_of(x)
    self.assertEqual(fs.shape, shape)
    self.assertEqual(fs.dtype, dtype)
    self.assertEqual(fs.fragments, [fragment_t(index=np.s_[0:2:1, 0:3:1])])

    with self.subTest('with_none_raises'):
      with self.assertRaisesRegex(TypeError, 'Fragment value must be'):
        fragments_t.all_of(None)  # pytype: disable=wrong-arg-types

  @parameterized.named_parameters(
      ('np_array', NpFragments),
      ('jnp_array', JaxFragments),
  )
  def test_concrete_fragments_from_none_of(self, fragments_t: FragmentsT):
    shape = (2, 3)
    dtype = np.dtype(np.float32)
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    other_np_api = np if np_api is jnp else jnp

    with self.subTest('with_array'):
      x = np_api.ones(shape, dtype=dtype)
      fs = fragments_t.none_of(x)
      self.assertEqual(fs.shape, shape)
      self.assertEqual(fs.dtype, dtype)
      self.assertEqual(fs.fragments, [])

    with self.subTest('with_wrong_array_type_raises'):
      with self.assertRaisesRegex(TypeError, 'Fragment value must be'):
        fragments_t.none_of(other_np_api.ones(shape, dtype=dtype))

    with self.subTest('with_shape_dtype_struct_raises'):
      x = jax.ShapeDtypeStruct(shape, dtype)
      with self.assertRaisesRegex(TypeError, 'Fragment value must be'):
        fragments_t.none_of(x)

  @parameterized.named_parameters(
      ('shape_dtype_struct', jax.ShapeDtypeStruct),
      ('np_array', np.ones),
      ('jnp_array', jnp.ones),
  )
  def test_abstract_fragments_from_none_of(self, value_fn):
    fragments_t = AbstractFragments
    shape = (2, 3)
    dtype = np.dtype(np.float32)

    x = value_fn(shape, dtype=dtype)
    fs = fragments_t.none_of(x)
    self.assertEqual(fs.shape, shape)
    self.assertEqual(fs.dtype, dtype)
    self.assertEqual(fs.fragments, [])

    with self.subTest('with_none_raises'):
      with self.assertRaisesRegex(TypeError, 'Fragment value must be'):
        fragments_t.none_of(None)  # pytype: disable=wrong-arg-types

  def test_np_fragments_from_addressable_shards_of_np_array(self):
    # NumPy arrays aren't sharded so we expect a single fragment spanning the
    # entire array, the same as if we had used `all_of()`.
    fragments_t = NpFragments
    shape = (2, 3)
    dtype = np.dtype(np.float32)
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    other_np_api = np if np_api is jnp else jnp

    with self.subTest('with_array'):
      x = np_api.ones(shape, dtype=dtype)
      fs = fragments_t.addressable_shards_of(x)
      self.assertEqual(fs.shape, shape)
      self.assertEqual(fs.dtype, dtype)
      self.assertEqual(
          fs.fragments, [fragment_t(index=np.s_[0:2:1, 0:3:1], value=x)]
      )

    with self.subTest('with_wrong_array_type_raises'):
      with self.assertRaisesRegex(TypeError, 'Fragment value must be'):
        fragments_t.addressable_shards_of(other_np_api.ones(shape, dtype=dtype))

    with self.subTest('with_shape_dtype_struct_raises'):
      x = jax.ShapeDtypeStruct(shape, dtype)
      with self.assertRaisesRegex(TypeError, 'Fragment value must be'):
        fragments_t.addressable_shards_of(x)

  def test_jax_fragments_from_addressable_shards_of_jnp_array(self):
    fragments_t = JaxFragments
    shape = (2, 3)
    dtype = np.dtype(np.float32)
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    other_np_api = np if np_api is jnp else jnp

    with self.subTest('with_array'):
      x = _fake_jnp_ones(
          shape=shape, dtype=dtype, sharding=_fake_sharding(shape)
      )
      fs = fragments_t.addressable_shards_of(x)
      self.assertEqual(fs.shape, shape)
      self.assertEqual(fs.dtype, dtype)
      self.assertEqual(fs.fragments, [
          fragment_t(index=np.s_[0:2:1, 0:1:1], value=x[0:2:1, 0:1:1]),
          fragment_t(index=np.s_[0:2:1, 2:3:1], value=x[0:2:1, 2:3:1]),
      ])

    with self.subTest('with_wrong_array_type_raises'):
      with self.assertRaisesRegex(TypeError, 'Fragment value must be'):
        fragments_t.addressable_shards_of(other_np_api.ones(shape, dtype=dtype))

    with self.subTest('with_shape_dtype_struct_raises'):
      x = jax.ShapeDtypeStruct(shape, dtype)
      with self.assertRaisesRegex(TypeError, 'Fragment value must be'):
        fragments_t.addressable_shards_of(x)

  def test_abstract_fragments_from_addressable_shards_of_np_array(self):
    # NumPy arrays aren't sharded so we expect a single fragment spanning the
    # entire array, the same as if we had used `all_of()`.
    fragments_t = AbstractFragments
    shape = (2, 3)
    dtype = np.dtype(np.float32)
    fragment_t = fragments_t.FRAGMENT_T

    x = np.ones(shape, dtype=dtype)
    fs = fragments_t.addressable_shards_of(x)
    self.assertEqual(fs.shape, shape)
    self.assertEqual(fs.dtype, dtype)
    self.assertEqual(fs.fragments, [fragment_t(index=np.s_[0:2:1, 0:3:1])])

  @parameterized.named_parameters(
      ('shape_dtype_struct', jax.ShapeDtypeStruct),
      ('jnp_array', _fake_jnp_ones),
  )
  def test_abstract_fragments_from_addressable_shards_of(self, value_fn):
    fragments_t = AbstractFragments
    shape = (2, 3)
    dtype = np.dtype(np.float32)
    fragment_t = fragments_t.FRAGMENT_T

    x = value_fn(shape, dtype=dtype, sharding=_fake_sharding(shape))
    fs = fragments_t.addressable_shards_of(x)
    self.assertEqual(fs.shape, shape)
    self.assertEqual(fs.dtype, dtype)
    self.assertEqual(fs.fragments, [
        fragment_t(index=np.s_[0:2:1, 0:1:1]),
        fragment_t(index=np.s_[0:2:1, 2:3:1]),
    ])

  def test_abstract_fragments_from_addressable_shards_of_shape_dtype_struct_with_no_sharding_raises(
      self,
  ):
    fragments_t = AbstractFragments
    shape = (2, 3)
    dtype = np.dtype(np.float32)

    x = jax.ShapeDtypeStruct(shape, dtype=dtype)
    with self.assertRaisesRegex(
        ValueError, 'Cannot determine addressable shards'
    ):
      fragments_t.addressable_shards_of(x)

  @parameterized.named_parameters(
      ('np_array', NpFragments),
      ('jnp_array', JaxFragments),
  )
  def test_concrete_fragments_of(self, fragments_t: FragmentsT):
    shape = (2, 3)
    dtype = np.dtype(np.float32)
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    other_np_api = np if np_api is jnp else jnp

    with self.subTest('with_array'):
      x = np_api.ones(shape, dtype=dtype)
      fs = fragments_t.of(x, indices=[np.s_[0:2:1, 0:1:1], np.s_[0:2:1, 2:3:1]])
      self.assertEqual(fs.shape, shape)
      self.assertEqual(fs.dtype, dtype)
      self.assertEqual(
          fs.fragments, [
              fragment_t(index=np.s_[0:2:1, 0:1:1], value=x[0:2:1, 0:1:1]),
              fragment_t(index=np.s_[0:2:1, 2:3:1], value=x[0:2:1, 2:3:1]),
          ]
      )

    with self.subTest('with_wrong_array_type_raises'):
      with self.assertRaisesRegex(TypeError, 'Fragment value must be'):
        fragments_t.of(
            other_np_api.ones(shape, dtype=dtype),
            indices=[np.s_[0:2:1, 0:1:1], np.s_[0:2:1, 2:3:1]],
        )

    with self.subTest('with_shape_dtype_struct_raises'):
      x = jax.ShapeDtypeStruct(shape, dtype)
      with self.assertRaisesRegex(TypeError, 'Fragment value must be'):
        fragments_t.of(x, indices=[np.s_[0:2:1, 0:1:1], np.s_[0:2:1, 2:3:1]])

  @parameterized.named_parameters(
      ('shape_dtype_struct', jax.ShapeDtypeStruct),
      ('np_array', np.ones),
      ('jnp_array', jnp.ones),
  )
  def test_abstract_fragments_of(self, value_fn):
    fragments_t = AbstractFragments
    shape = (2, 3)
    dtype = np.dtype(np.float32)
    fragment_t = fragments_t.FRAGMENT_T

    x = value_fn(shape, dtype=dtype)
    fs = fragments_t.of(x, indices=[np.s_[0:2:1, 0:1:1], np.s_[0:2:1, 2:3:1]])
    self.assertEqual(fs.shape, shape)
    self.assertEqual(fs.dtype, dtype)
    self.assertEqual(fs.fragments, [
        fragment_t(index=np.s_[0:2:1, 0:1:1]),
        fragment_t(index=np.s_[0:2:1, 2:3:1]),
    ])


@parameterized.named_parameters(
    ('abstract_fragments', AbstractFragments),
    ('np_fragments', NpFragments),
    ('jax_fragments', JaxFragments),
)
class IsFullTest(parameterized.TestCase):

  def test_filled_by_one_spanning_fragment(
      self, fragments_t: FragmentsT
  ):
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    self.assertTrue(
        array_fragments._is_full(
            fragments_t(
                shape=(4, 5),
                dtype=np.dtype(np.float32),
                fragments=[
                    fragment_t(
                        index=np.s_[0:4:1, 0:5:1],
                        value=np_api.ones([4, 5])
                        if np_api is not None
                        else None,
                    ),
                ],
            )
        )
    )

  def test_filled_by_two_non_spanning_fragments(
      self, fragments_t: FragmentsT
  ):
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    self.assertTrue(
        array_fragments._is_full(
            fragments_t(
                shape=(4, 5),
                dtype=np.dtype(np.float32),
                fragments=[
                    fragment_t(
                        index=np.s_[0:4:1, 0:2:1],
                        value=np_api.ones([4, 2])
                        if np_api is not None
                        else None,
                    ),
                    fragment_t(
                        index=np.s_[0:4:1, 2:5:1],
                        value=np_api.zeros([4, 3])
                        if np_api is not None
                        else None,
                    ),
                ],
            )
        )
    )

  def test_not_filled_by_two_non_spanning_fragments(
      self, fragments_t: FragmentsT
  ):
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    self.assertFalse(
        array_fragments._is_full(
            fragments_t(
                shape=(4, 5),
                dtype=np.dtype(np.float32),
                fragments=[
                    fragment_t(
                        index=np.s_[0:4:1, 0:2:1],
                        value=np_api.ones([4, 2])
                        if np_api is not None
                        else None,
                    ),
                    fragment_t(
                        index=np.s_[2:4:1, 2:5:1],
                        value=np_api.zeros([4, 3])
                        if np_api is not None
                        else None,
                    ),
                ],
            )
        )
    )

  def test_not_filled_by_no_fragments(
      self, fragments_t: FragmentsT
  ):
    self.assertFalse(
        array_fragments._is_full(
            fragments_t(shape=(4, 5), dtype=np.dtype(np.float32), fragments=[])
        )
    )

  def test_rank_0_fragments_not_filled_by_no_fragments(
      self, fragments_t: FragmentsT
  ):
    self.assertFalse(
        array_fragments._is_full(
            fragments_t(shape=(), dtype=np.dtype(np.float32), fragments=[])
        )
    )

  def test_dim_0_fragments_filled_by_no_fragments(
      self, fragments_t: FragmentsT
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


class AbstractFragmentsTest(parameterized.TestCase):

  def test_returns_abstract_fragments_instance_itself(
      self
  ):
    fragments = AbstractFragments(
        shape=(2, 3), dtype=np.dtype(np.float32), fragments=[]
    )
    self.assertIs(fragments, array_fragments.abstract_fragments(fragments))

  @parameterized.named_parameters(
      ('np_fragments', NpFragments),
      ('jax_fragments', JaxFragments),
  )
  def test_converts_concrete_fragments(
      self, fragments_t: ConcreteFragmentsT
  ):
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    concrete_fragments = fragments_t(
        shape=(2, 3),
        dtype=np.dtype(np.float32),
        fragments=[
            fragment_t(index=np.s_[0:2:1, 0:3:1], value=np_api.arange(6)),
        ],
    )
    expected_abstract_fragments = AbstractFragments(
        shape=(2, 3),
        dtype=np.dtype(np.float32),
        fragments=[
            AbstractFragment(
                index=np.s_[0:2:1, 0:3:1], value=None
            ),
        ],
    )
    self.assertEqual(
        expected_abstract_fragments,
        array_fragments.abstract_fragments(concrete_fragments),
    )

  def test_converts_fully_replicated_shape_dtype_struct(self):
    self.assertEqual(
        AbstractFragments(
            shape=(4, 5),
            dtype=np.dtype(np.float32),
            fragments=[AbstractFragment(index=np.s_[0:4:1, 0:5:1])],
        ),
        array_fragments.abstract_fragments(
            jax.ShapeDtypeStruct((4, 5), np.dtype(np.float32))
        ),
    )


class StackFragmentsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('np_fragments', NpFragments),
      ('jax_fragments', JaxFragments),
  )
  def test_stacks_fragments_of_same_shape(
      self, fragments_t: ConcreteFragmentsT
  ):
    fragment_t = fragments_t.FRAGMENT_T
    np_api = fragment_t.NP_API
    full = lambda x: np_api.full((4, 5), x)
    fragments = fragments_t(
        shape=(24, 35),
        dtype=np.dtype(np.float32),
        fragments=[
            fragment_t(index=np.s_[12:16:1, 10:15:1], value=full(1.0)),
            fragment_t(index=np.s_[16:20:1, 10:15:1], value=full(2.0)),
            fragment_t(index=np.s_[12:16:1, 10:15:1], value=full(3.0)),
            fragment_t(index=np.s_[16:20:1, 10:15:1], value=full(4.0)),
        ],
    )

    expected_array = np_api.stack([
        np_api.full((4, 5), 1.0),
        np_api.full((4, 5), 2.0),
        np_api.full((4, 5), 3.0),
        np_api.full((4, 5), 4.0),
    ])

    actual_array = array_fragments.stack_fragments(fragments)
    np.testing.assert_array_equal(expected_array, actual_array)

  @parameterized.named_parameters(
      ('np_fragments', NpFragments),
      ('jax_fragments', JaxFragments),
  )
  def test_stacks_single_fragment_without_copy(
      self, fragments_t: ConcreteFragmentsT
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

    # `jax.Array` has no `base` attribute.
    if np_api is np:
      assert isinstance(actual_array, np.ndarray)
      assert isinstance(value, np.ndarray)
      with self.subTest('base_is_same_as_original'):
        # Result is a view of the original `np_api.arange(20)` array.
        self.assertIs(actual_array.base, value.base)

    np.testing.assert_array_equal(
        actual_array, np_api.arange(20).reshape((1, 4, 5))
    )

  def test_returns_none_for_none(self):
    self.assertIsNone(array_fragments.stack_fragments(None))

  def assert_stacking_is_rejected(
      self,
      invalid_fragments: array_fragments.FSconcrete,
      expected_error_message: str,
  ) -> None:
    with self.assertRaisesRegex(ValueError, expected_error_message):
      array_fragments.validate_fragments_can_be_stacked(invalid_fragments)
    with self.assertRaisesRegex(ValueError, expected_error_message):
      array_fragments.stack_fragments(invalid_fragments)

  @parameterized.named_parameters(
      ('np_fragments', NpFragments),
      ('jax_fragments', JaxFragments),
  )
  def test_rejects_empty_fragments_list(
      self, fragments_t: ConcreteFragmentsT
  ):
    empty_fragments = fragments_t(
        shape=(24, 35), dtype=np.dtype(np.float32), fragments=[]
    )
    self.assert_stacking_is_rejected(empty_fragments, 'No fragments to stack')

  @parameterized.named_parameters(
      ('np_fragments', NpFragments),
      ('jax_fragments', JaxFragments),
  )
  def test_rejects_fragments_of_different_shapes(
      self, fragments_t: ConcreteFragmentsT
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

  def test_rejects_abstract_fragments(self):
    fragments = AbstractFragments(
        shape=(24, 35),
        dtype=np.dtype(np.float32),
        fragments=[
            AbstractFragment(index=np.s_[12:16:1, 10:15:1]),
            AbstractFragment(index=np.s_[16:20:1, 15:20:1]),
        ],
    )

    self.assert_stacking_is_rejected(fragments, 'Not all fragments have values')  # pytype: disable=wrong-arg-types


class BackwardsCompatibleTypesTest(absltest.TestCase):

  def test_fragments_and_fragment_type_aliases_still_work(self):
    def _process_fragment(
        target_fragment: array_fragments.ConcreteFragment,
    ) -> array_fragments.ConcreteFragment:
      return target_fragment

    def _process_fragments(
        target_fragments: array_fragments.ConcreteFragments,
    ) -> list[array_fragments.ConcreteFragment]:
      """Calculates read chunks to load, covering the given fragments."""
      result: list[array_fragments.ConcreteFragment] = []
      for target_fragment in target_fragments.fragments:
        result.append(_process_fragment(target_fragment))
      return result

    del _process_fragments


if __name__ == '__main__':
  absltest.main()
