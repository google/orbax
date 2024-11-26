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

"""Tests for empty values as leafs in the checkpoint tree."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import optax
from orbax.checkpoint._src.metadata import empty_values
from orbax.checkpoint._src.metadata import pytree_metadata_options
from orbax.checkpoint._src.testing import test_tree_utils


class EmptyValuesTest(parameterized.TestCase):

  @parameterized.parameters(
      (1, False, False),
      (np.zeros(0), False, False),
      (dict(), True, True),
      ({}, True, True),
      ({"a": {}}, False, False),
      ([], True, True),
      ([[]], False, False),
      (tuple(), True, True),
      ((tuple(),), False, False),
      (None, True, True),
      ((1, 2), False, False),
      (test_tree_utils.EmptyNamedTuple(), False, True),
      (optax.EmptyState(), False, True),
      (test_tree_utils.MuNu(mu=None, nu=None), False, False),
      (test_tree_utils.NamedTupleWithNestedAttributes(), False, False),
      (
          test_tree_utils.NamedTupleWithNestedAttributes(nested_dict={}),
          False,
          False,
      ),
      (test_tree_utils.MyEmptyChex(), True, True),
      (test_tree_utils.MyChex(), False, False),
      (test_tree_utils.MyChex(my_np_array=np.array([])), False, False),
      (test_tree_utils.MyEmptyClass(), False, False),
      (test_tree_utils.MyClass(), False, False),
      (test_tree_utils.MyClass(a=None, b=None), False, False),
      (test_tree_utils.MyClass(a=None, b=np.zeros(1)), False, False),
      (test_tree_utils.MyEmptyFlax(), False, False),  # TODO: b/378905913 - fix
      (test_tree_utils.MyFlax(), False, False),
      (
          test_tree_utils.MyFlax(
              my_jax_array=None, my_nested_mapping=None, my_sequence=None
          ),
          False,
          False,
      ),
      (test_tree_utils.MyFlax(my_nested_mapping={"a": 1}), False, False),
      (test_tree_utils.MyEmptyDataClass(), False, False),
      (test_tree_utils.MyDataClass(), False, False),
      (
          test_tree_utils.MyDataClass(
              my_jax_array=None,
              my_np_array=None,
              my_empty_dataclass=None,
              my_chex=None,
          ),
          False,
          False,
      ),
  )
  def test_is_supported_empty_value(self, value, expected, expected_rich_type):
    with self.subTest("legacy_metadata"):
      self.assertEqual(
          expected,
          empty_values.is_supported_empty_value(
              value,
              pytree_metadata_options.PyTreeMetadataOptions(
                  support_rich_types=False
              ),
          ),
      )
    with self.subTest("rich_typed_metadata"):
      self.assertEqual(
          expected_rich_type,
          empty_values.is_supported_empty_value(
              value,
              pytree_metadata_options.PyTreeMetadataOptions(
                  support_rich_types=True
              ),
          ),
      )

  @parameterized.parameters(
      (1, ValueError(), ValueError()),
      (np.zeros(0), ValueError(), ValueError()),
      (dict(), empty_values.RESTORE_TYPE_DICT, empty_values.RESTORE_TYPE_DICT),
      ({}, empty_values.RESTORE_TYPE_DICT, empty_values.RESTORE_TYPE_DICT),
      ({"a": {}}, ValueError(), ValueError()),
      ([], empty_values.RESTORE_TYPE_LIST, empty_values.RESTORE_TYPE_LIST),
      ([[]], ValueError(), ValueError()),
      (
          tuple(),
          empty_values.RESTORE_TYPE_TUPLE,
          empty_values.RESTORE_TYPE_TUPLE,
      ),
      ((tuple(),), ValueError(), ValueError()),
      (None, empty_values.RESTORE_TYPE_NONE, empty_values.RESTORE_TYPE_NONE),
      ((1, 2), ValueError(), ValueError()),
      (
          test_tree_utils.EmptyNamedTuple(),
          ValueError(),
          empty_values.RESTORE_TYPE_NAMED_TUPLE,
      ),
      (optax.EmptyState(), ValueError(), empty_values.RESTORE_TYPE_NAMED_TUPLE),
      (test_tree_utils.MuNu(mu=None, nu=None), ValueError(), ValueError()),
      (
          test_tree_utils.NamedTupleWithNestedAttributes(),
          ValueError(),
          ValueError(),
      ),
      (
          test_tree_utils.NamedTupleWithNestedAttributes(nested_dict={}),
          ValueError(),
          ValueError(),
      ),
      (
          test_tree_utils.MyEmptyChex(),
          empty_values.RESTORE_TYPE_DICT,
          empty_values.RESTORE_TYPE_DICT,
      ),
      (test_tree_utils.MyChex(), ValueError(), ValueError()),
      (
          test_tree_utils.MyChex(my_np_array=np.array([])),
          ValueError(),
          ValueError(),
      ),
      (test_tree_utils.MyEmptyClass(), ValueError(), ValueError()),
      (test_tree_utils.MyClass(), ValueError(), ValueError()),
      (test_tree_utils.MyClass(a=None, b=None), ValueError(), ValueError()),
      (
          test_tree_utils.MyClass(a=None, b=np.zeros(1)),
          ValueError(),
          ValueError(),
      ),
      (
          test_tree_utils.MyEmptyFlax(),  # TODO: b/378905913 - fix
          ValueError(),
          ValueError(),
      ),
      (test_tree_utils.MyFlax(), ValueError(), ValueError()),
      (
          test_tree_utils.MyFlax(
              my_jax_array=None, my_nested_mapping=None, my_sequence=None
          ),
          ValueError(),
          ValueError(),
      ),
      (
          test_tree_utils.MyFlax(my_nested_mapping={"a": 1}),
          ValueError(),
          ValueError(),
      ),
      (test_tree_utils.MyEmptyDataClass(), ValueError(), ValueError()),
      (test_tree_utils.MyDataClass(), ValueError(), ValueError()),
      (
          test_tree_utils.MyDataClass(
              my_jax_array=None,
              my_np_array=None,
              my_empty_dataclass=None,
              my_chex=None,
          ),
          ValueError(),
          ValueError(),
      ),
  )
  def test_get_empty_value_typestr(self, value, expected, expected_rich_type):
    with self.subTest("legacy_metadata"):
      if isinstance(expected, ValueError):
        with self.assertRaises(ValueError):
          empty_values.get_empty_value_typestr(
              value,
              pytree_metadata_options.PyTreeMetadataOptions(
                  support_rich_types=False
              ),
          )
      else:
        self.assertEqual(
            expected,
            empty_values.get_empty_value_typestr(
                value,
                pytree_metadata_options.PyTreeMetadataOptions(
                    support_rich_types=False
                ),
            ),
        )
    with self.subTest("rich_typed_metadata"):
      if isinstance(expected_rich_type, ValueError):
        with self.assertRaises(ValueError):
          empty_values.get_empty_value_typestr(
              value,
              pytree_metadata_options.PyTreeMetadataOptions(
                  support_rich_types=True
              ),
          )
      else:
        self.assertEqual(
            expected_rich_type,
            empty_values.get_empty_value_typestr(
                value,
                pytree_metadata_options.PyTreeMetadataOptions(
                    support_rich_types=True
                ),
            ),
        )

  @parameterized.parameters(
      (empty_values.RESTORE_TYPE_NONE, None, None),
      (empty_values.RESTORE_TYPE_DICT, {}, {}),
      (empty_values.RESTORE_TYPE_LIST, [], []),
      (empty_values.RESTORE_TYPE_TUPLE, tuple(), tuple()),
      (
          empty_values.RESTORE_TYPE_NAMED_TUPLE,
          None,
          empty_values.OrbaxEmptyNamedTuple(),
      ),
      (empty_values.RESTORE_TYPE_UNKNOWN, ValueError(), ValueError()),
      ("blah", ValueError(), ValueError()),
  )
  def test_get_empty_value_from_typestr(
      self, typestr, expected, expected_rich_type
  ):
    with self.subTest("legacy_metadata"):
      if isinstance(expected, ValueError):
        with self.assertRaises(ValueError):
          empty_values.get_empty_value_from_typestr(
              typestr,
              pytree_metadata_options.PyTreeMetadataOptions(
                  support_rich_types=False
              ),
          )
      else:
        self.assertEqual(
            expected,
            empty_values.get_empty_value_from_typestr(
                typestr,
                pytree_metadata_options.PyTreeMetadataOptions(
                    support_rich_types=False
                ),
            ),
        )
    with self.subTest("rich_typed_metadata"):
      if isinstance(expected_rich_type, ValueError):
        with self.assertRaises(ValueError):
          empty_values.get_empty_value_from_typestr(
              typestr,
              pytree_metadata_options.PyTreeMetadataOptions(
                  support_rich_types=True
              ),
          )
      else:
        self.assertEqual(
            expected_rich_type,
            empty_values.get_empty_value_from_typestr(
                typestr,
                pytree_metadata_options.PyTreeMetadataOptions(
                    support_rich_types=True
                ),
            ),
        )


if __name__ == "__main__":
  absltest.main()
