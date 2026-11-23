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
from etils import epath
import jax
import numpy as np
from orbax.checkpoint._src.handlers import param_info_utils
from orbax.checkpoint._src.metadata import empty_values
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.serialization import type_handlers


class BuildParamInfosTest(absltest.TestCase):

  def test_populates_tree_derived_fields(self):
    directory = epath.Path(self.create_tempdir().full_path)
    item = {'a': np.arange(4), 'b': {'c': 1.0, 'd': None}}

    param_infos = param_info_utils.build_param_infos(item, directory)

    self.assertEqual(
        jax.tree.structure(item, is_leaf=lambda x: x is None),
        jax.tree.structure(param_infos, is_leaf=lambda x: x is None),
    )
    a_info = param_infos['a']
    self.assertEqual(a_info.name, 'a')
    self.assertEqual(a_info.keypath, (jax.tree_util.DictKey('a'),))
    self.assertEqual(a_info.parent_dir, directory)
    self.assertTrue(a_info.is_ocdbt_checkpoint)
    self.assertIsNotNone(a_info.ts_context)
    self.assertEqual(a_info.value_typestr, 'np.ndarray')
    self.assertFalse(a_info.skip_deserialize)
    self.assertEqual(param_infos['b']['c'].name, 'b.c')
    self.assertEqual(
        param_infos['b']['d'].value_typestr, empty_values.RESTORE_TYPE_NONE
    )

  def test_uses_provided_names(self):
    directory = epath.Path(self.create_tempdir().full_path)
    item = {'a': np.arange(4)}

    param_infos = param_info_utils.build_param_infos(
        item, directory, names={'a': 'custom_name'}
    )

    self.assertEqual(param_infos['a'].name, 'custom_name')
    self.assertEqual(param_infos['a'].path, directory / 'custom_name')

  def test_skip_deserialize(self):
    directory = epath.Path(self.create_tempdir().full_path)
    item = {
        'placeholder': type_handlers.PLACEHOLDER,
        'metadata': tree_metadata.ValueMetadataEntry(
            value_type='np.ndarray', skip_deserialize=True
        ),
    }

    param_infos = param_info_utils.build_param_infos(item, directory)

    self.assertTrue(param_infos['placeholder'].skip_deserialize)
    self.assertTrue(param_infos['metadata'].skip_deserialize)

  def test_forwards_param_info_kwargs(self):
    directory = epath.Path(self.create_tempdir().full_path)
    item = {'a': np.arange(4)}

    param_infos = param_info_utils.build_param_infos(
        item,
        directory,
        use_ocdbt=False,
        use_zarr3=True,
        ocdbt_target_data_file_size=123,
    )

    self.assertFalse(param_infos['a'].is_ocdbt_checkpoint)
    self.assertTrue(param_infos['a'].use_zarr3)
    self.assertEqual(param_infos['a'].ocdbt_target_data_file_size, 123)


class ToValueMetadataTreeTest(absltest.TestCase):

  def test_converts_param_infos(self):
    directory = epath.Path(self.create_tempdir().full_path)
    item = {'a': np.arange(4), 'b': {'c': None, 'd': {}}}
    param_infos = param_info_utils.build_param_infos(item, directory)
    param_infos['a'] = param_infos['a'].replace(write_shape=(2,))

    value_metadata_tree = param_info_utils.to_value_metadata_tree(param_infos)

    self.assertEqual(
        value_metadata_tree,
        {
            'a': tree_metadata.ValueMetadataEntry(
                value_type='np.ndarray',
                skip_deserialize=False,
                write_shape=(2,),
            ),
            'b': {
                'c': tree_metadata.ValueMetadataEntry(
                    value_type=empty_values.RESTORE_TYPE_NONE,
                    skip_deserialize=True,
                ),
                'd': tree_metadata.ValueMetadataEntry(
                    value_type=empty_values.RESTORE_TYPE_DICT,
                    skip_deserialize=True,
                ),
            },
        },
    )

  def test_raises_without_value_typestr(self):
    directory = epath.Path(self.create_tempdir().full_path)
    param_infos = param_info_utils.build_param_infos(
        {'a': np.arange(4)}, directory
    )
    param_infos['a'] = param_infos['a'].replace(value_typestr=None)

    with self.assertRaisesRegex(AssertionError, 'value_typestr'):
      param_info_utils.to_value_metadata_tree(param_infos)


if __name__ == '__main__':
  absltest.main()
