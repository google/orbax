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

"""Tests for pytree metadata."""

import json

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.path import format_utils


PyTreeCheckpointHandler = test_utils.PyTreeCheckpointHandler
PyTreeSaveArgs = pytree_checkpoint_handler.PyTreeSaveArgs


# TODO: b/438823853 - This test should accompany an independent PyTree metadata
# writer, which is currently tightly coupled to PyTreeCheckpointHandler.
class PyTreeMetadataTest(parameterized.TestCase):

  @parameterized.product(use_ocdbt=(True, False), use_zarr3=(True, False))
  def test_metadata_properties(self, use_ocdbt: bool, use_zarr3: bool):
    directory = epath.Path(self.create_tempdir().full_path)
    handler = PyTreeCheckpointHandler(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3)
    item = {'a': np.array([1, 2, 3]), 'b': {'c': 'test'}}
    custom_metadata = {'key1': 'value1', 'key2': 123}
    handler.save(
        directory, args=PyTreeSaveArgs(item, custom_metadata=custom_metadata)
    )

    metadata_path = directory / format_utils.PYTREE_METADATA_FILE
    self.assertTrue(metadata_path.exists())

    metadata_json = json.loads(metadata_path.read_text())
    internal_metadata = tree_metadata.InternalTreeMetadata.from_json(
        metadata_json
    )
    self.assertEqual(internal_metadata.custom_metadata, custom_metadata)
    self.assertEqual(internal_metadata.use_ocdbt, use_ocdbt)
    self.assertEqual(internal_metadata.use_zarr3, use_zarr3)

  @parameterized.product(use_ocdbt=(True, False), use_zarr3=(True, False))
  def test_as_custom_metadata(self, use_ocdbt: bool, use_zarr3: bool):
    directory = epath.Path(self.create_tempdir().full_path)
    item = {'a': np.array([1, 2, 3]), 'b': {'c': 'test'}}
    custom_metadata = {'key1': 'value1', 'key2': 123}
    handler = PyTreeCheckpointHandler(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3)
    handler.save(
        directory,
        args=PyTreeSaveArgs(item, custom_metadata=custom_metadata),
    )

    metadata_path = directory / format_utils.PYTREE_METADATA_FILE
    metadata_json = json.loads(metadata_path.read_text())
    internal_metadata = tree_metadata.InternalTreeMetadata.from_json(
        metadata_json
    )

    metadata_tree = internal_metadata.as_custom_metadata(
        directory, handler._handler_impl._type_handler_registry
    )
    self.assertEqual(metadata_tree['a'].shape, (3,))
    self.assertEqual(metadata_tree['b']['c'].name, 'b.c')


if __name__ == '__main__':
  absltest.main()
