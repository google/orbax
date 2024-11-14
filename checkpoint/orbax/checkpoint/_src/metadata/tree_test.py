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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from orbax.checkpoint._src.metadata import tree as tree_metadata_lib
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.testing import test_tree_utils
from orbax.checkpoint._src.tree import utils as tree_utils


class InternalTreeMetadataTest(parameterized.TestCase):

  def _to_param_infos(self, tree: Any):
    return jax.tree.map(
        # Other properties are not relevant.
        lambda x: type_handlers.ParamInfo(
            value_typestr=type_handlers.get_param_typestr(
                x, type_handlers.GLOBAL_TYPE_HANDLER_REGISTRY
            )
        ),
        tree,
        is_leaf=tree_utils.is_empty_or_leaf,
    )

  @parameterized.named_parameters(
      test_tree_utils.TEST_PYTREES_FOR_NAMED_PARAMETERS
  )
  def test_json_conversion(self, test_pytree: test_tree_utils.TestPyTree):
    tree = test_pytree.provide_tree()

    internal_tree_metadata = tree_metadata_lib.InternalTreeMetadata.build(
        self._to_param_infos(tree),
    )
    internal_tree_metadata_json = internal_tree_metadata.to_json()

    # Round trip check for json conversion.
    self.assertCountEqual(
        internal_tree_metadata.tree_metadata_entries,
        (
            tree_metadata_lib.InternalTreeMetadata.from_json(
                internal_tree_metadata_json
            )
        ).tree_metadata_entries,
    )

    # Specifically check _TREE_METADATA_KEY.
    self.assertDictEqual(
        test_pytree.expected_tree_metadata_key_json,
        internal_tree_metadata_json[tree_metadata_lib._TREE_METADATA_KEY],
    )


if __name__ == '__main__':
  absltest.main()
