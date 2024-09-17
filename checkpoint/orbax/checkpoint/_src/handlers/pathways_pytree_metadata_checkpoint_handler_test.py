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

from unittest import mock

from absl.testing import absltest
from etils import epath
from orbax.checkpoint._src.handlers import pathways_pytree_metadata_checkpoint_handler


class PathwaysPytreeMetadataCheckpointHandlerTest(absltest.TestCase):

  def test_save_and_restore(self):
    """Test that the handler can save and restore a dictionary of various types.
    """
    handler = (
        pathways_pytree_metadata_checkpoint_handler.PathwaysPyTreeCheckpointHandler()
    )
    directory = epath.Path(
        '/tmp/pathways_pytree_metadata_checkpoint_handler_test'
    )
    directory.mkdir(parents=True, exist_ok=True)

    test_data = {
        'str': 'hello',
        'int': 123,
        'list': [1, 2, 3],
        'tuple': (4, 5, 6),
        'nested': {
            'a': 'world',
            'b': 789,
            'c': ['a', 'b', ['c', ['d']]]
        },
    }

    # Mock the super().save and super().restore methods.
    with mock.patch.object(
        pathways_pytree_metadata_checkpoint_handler.BasePyTreeCheckpointHandler,
        'save'
    ) as mock_save, mock.patch.object(
        pathways_pytree_metadata_checkpoint_handler.BasePyTreeCheckpointHandler,
        'restore'
    ) as mock_restore:
      # Directly return the value passed to save in restore
      mock_restore.side_effect = lambda dir, args: pathways_pytree_metadata_checkpoint_handler.BasePyTreeRestoreArgs(
          item=mock_save.call_args.args[1].item)

      # Save the data
      handler.save(directory, test_data)

      # Restore the data
      restored_data = handler.restore(directory)

    mock_save.assert_called_once()
    mock_restore.assert_called_once()
    self.assertEqual(restored_data, test_data)

if __name__ == '__main__':
  absltest.main()
