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

from typing import Any

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.serialization import types
from orbax.checkpoint._src.testing import test_tree_utils
from orbax.checkpoint._src.tree import utils as tree_utils


def _to_param_infos(
    tree: Any,
    pytree_metadata_options: tree_metadata.PyTreeMetadataOptions,
):
  return jax.tree.map(
      # Other properties are not relevant.
      lambda x: types.ParamInfo(
          value_typestr=types.get_param_typestr(
              x,
              type_handlers.GLOBAL_TYPE_HANDLER_REGISTRY,
              pytree_metadata_options,
          )
      ),
      tree,
      is_leaf=tree_utils.is_empty_or_leaf,
  )


class InternalTreeMetadataEntryTest(parameterized.TestCase):

  @parameterized.product(
      test_pytree=test_tree_utils.TEST_PYTREES,
      pytree_metadata_options=[
          tree_metadata.PyTreeMetadataOptions(support_rich_types=False),
          tree_metadata.PyTreeMetadataOptions(support_rich_types=True),
      ],
  )
  def test_as_nested_tree(
      self,
      test_pytree: test_tree_utils.TestPyTree,
      pytree_metadata_options: tree_metadata.PyTreeMetadataOptions,
  ):
    tree = test_pytree.provide_tree()
    original_internal_tree_metadata = tree_metadata.InternalTreeMetadata.build(
        param_infos=_to_param_infos(tree, pytree_metadata_options),
        pytree_metadata_options=pytree_metadata_options,
    )
    json_object = original_internal_tree_metadata.to_json()
    restored_internal_tree_metadata = (
        tree_metadata.InternalTreeMetadata.from_json(
            json_object, pytree_metadata_options
        )
    )

    if pytree_metadata_options.support_rich_types:
      expected_tree_metadata = (
          test_pytree.expected_nested_tree_metadata_with_rich_types
      )
    else:
      expected_tree_metadata = test_pytree.expected_nested_tree_metadata
    restored_tree_metadata = restored_internal_tree_metadata.as_nested_tree()
    logging.info('expected_tree_metadata: \n%s', expected_tree_metadata)
    logging.info('restored_tree_metadata: \n%s', restored_tree_metadata)
    self.assertEqual(
        jax.tree.structure(
            expected_tree_metadata, is_leaf=tree_utils.is_empty_or_leaf
        ),
        jax.tree.structure(
            restored_tree_metadata, is_leaf=tree_utils.is_empty_or_leaf
        ),
    )


if __name__ == '__main__':
  absltest.main()
