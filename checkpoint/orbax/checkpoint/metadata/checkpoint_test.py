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

"""Tests for orbax.checkpoint.metadata.checkpoint.py."""

import pickle
import time
from absl.testing import absltest
from etils import epath
from orbax.checkpoint.metadata import checkpoint


class CheckpointMetadataStoreTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='checkpoint_metadata_test').full_path
    )
    self.write_enabled_store = checkpoint.checkpoint_metadata_store(
        enable_write=True
    )
    self.read_only_store = checkpoint.checkpoint_metadata_store(
        enable_write=False
    )

  def test_read_unknown_path(self):
    self.assertIsNone(
        self.write_enabled_store.read(checkpoint_path='unknown_checkpoint_path')
    )

  def test_write_unknown_path(self):
    with self.assertRaisesRegex(ValueError, 'Checkpoint path does not exist'):
      self.write_enabled_store.write(
          checkpoint_path='unknown_checkpoint_path',
          checkpoint_metadata=checkpoint.CheckpointMetadata(),
      )

  def test_read_default_values(self):
    metadata = checkpoint.CheckpointMetadata()
    self.write_enabled_store.write(
        checkpoint_path=self.directory, checkpoint_metadata=metadata
    )
    self.assertEqual(
        self.write_enabled_store.read(checkpoint_path=self.directory), metadata
    )

  def test_read_with_values(self):
    metadata = checkpoint.CheckpointMetadata(
        init_timestamp_nsecs=time.time_ns(),
        commit_timestamp_nsecs=time.time_ns() + 1,
    )
    self.write_enabled_store.write(
        checkpoint_path=self.directory, checkpoint_metadata=metadata
    )
    self.assertEqual(
        self.write_enabled_store.read(checkpoint_path=self.directory), metadata
    )

  def test_read_corrupt_json_data(self):
    metadata_file = checkpoint._metadata_file_path(self.directory)
    metadata_file.touch()
    self.assertIsNone(
        self.write_enabled_store.read(checkpoint_path=self.directory)
    )

  def test_update_without_prior_data(self):
    self.write_enabled_store.update(
        checkpoint_path=self.directory,
        init_timestamp_nsecs=1,
        commit_timestamp_nsecs=2,
    )
    self.assertEqual(
        self.write_enabled_store.read(checkpoint_path=self.directory),
        checkpoint.CheckpointMetadata(
            init_timestamp_nsecs=1,
            commit_timestamp_nsecs=2,
        ),
    )

  def test_update_with_prior_data(self):
    metadata = checkpoint.CheckpointMetadata(init_timestamp_nsecs=1)
    self.write_enabled_store.write(
        checkpoint_path=self.directory, checkpoint_metadata=metadata
    )
    self.write_enabled_store.update(
        checkpoint_path=self.directory,
        commit_timestamp_nsecs=2,
    )
    self.assertEqual(
        self.write_enabled_store.read(checkpoint_path=self.directory),
        checkpoint.CheckpointMetadata(
            init_timestamp_nsecs=1,
            commit_timestamp_nsecs=2,
        ),
    )

  def test_update_with_unknown_kwargs(self):
    with self.assertRaisesRegex(
        TypeError, "got an unexpected keyword argument 'blah'"
    ):
      self.write_enabled_store.update(
          checkpoint_path=self.directory,
          init_timestamp_nsecs=1,
          blah=2,
      )

  def test_write_with_read_only_store_is_no_op(self):
    self.read_only_store.write(
        checkpoint_path=self.directory,
        checkpoint_metadata=checkpoint.CheckpointMetadata(),
    )
    self.assertIsNone(self.read_only_store.read(checkpoint_path=self.directory))
    self.assertIsNone(
        self.write_enabled_store.read(checkpoint_path=self.directory)
    )

  def test_pickle(self):
    with self.assertRaisesRegex(TypeError, 'cannot pickle'):
      _ = pickle.dumps(self.write_enabled_store)
    with self.assertRaisesRegex(TypeError, 'cannot pickle'):
      _ = pickle.dumps(self.read_only_store)


if __name__ == '__main__':
  absltest.main()
