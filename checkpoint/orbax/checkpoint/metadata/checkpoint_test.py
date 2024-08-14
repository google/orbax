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
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint.metadata import checkpoint


class CheckpointMetadataStoreTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path)
    checkpoint._CHECKPOINT_METADATA_STORE_FOR_WRITES = (
        checkpoint._BlockingCheckpointMetadataStore(enable_write=True)
    )
    checkpoint._CHECKPOINT_METADATA_STORE_FOR_READS = (
        checkpoint._BlockingCheckpointMetadataStore(enable_write=False)
    )
    self._non_blocking_store_for_writes = checkpoint.checkpoint_metadata_store(
        enable_write=True, blocking_write=False
    )
    checkpoint._CHECKPOINT_METADATA_STORE_NON_BLOCKING_FOR_READS = (
        checkpoint._NonBlockingCheckpointMetadataStore(enable_write=False)
    )

  def tearDown(self):
    super().tearDown()
    checkpoint._CHECKPOINT_METADATA_STORE_FOR_WRITES.close()
    checkpoint._CHECKPOINT_METADATA_STORE_FOR_READS.close()
    self._non_blocking_store_for_writes.close()
    checkpoint._CHECKPOINT_METADATA_STORE_NON_BLOCKING_FOR_READS.close()

  def write_metadata_store(
      self, blocking_write: bool
  ) -> checkpoint.CheckpointMetadataStore:
    if blocking_write:
      return checkpoint.checkpoint_metadata_store(
          enable_write=True, blocking_write=True
      )
    return self._non_blocking_store_for_writes

  def read_metadata_store(
      self, blocking_write: bool
  ) -> checkpoint.CheckpointMetadataStore:
    return checkpoint.checkpoint_metadata_store(
        enable_write=False, blocking_write=blocking_write
    )

  @parameterized.parameters(True, False)
  def test_read_unknown_path(self, blocking_write: bool):
    self.assertIsNone(
        self.write_metadata_store(blocking_write).read(
            checkpoint_path='unknown_checkpoint_path'
        )
    )

  @parameterized.parameters(True, False)
  def test_write_unknown_path(self, blocking_write: bool):
    if blocking_write:
      with self.assertRaisesRegex(ValueError, 'Checkpoint path does not exist'):
        self.write_metadata_store(blocking_write).write(
            checkpoint_path='unknown_checkpoint_path',
            checkpoint_metadata=checkpoint.CheckpointMetadata(),
        )
    else:
      self.write_metadata_store(blocking_write).write(
          checkpoint_path='unknown_checkpoint_path',
          checkpoint_metadata=checkpoint.CheckpointMetadata(),
      )
      try:
        self.write_metadata_store(blocking_write).wait_until_finished()
      except ValueError:
        # We don't want to fail the test because above write's future.result()
        # will raise 'ValueError: Checkpoint path does not exist ...'.
        pass
      self.assertIsNone(
          self.read_metadata_store(blocking_write).read(
              checkpoint_path='unknown_checkpoint_path'
          )
      )

  @parameterized.parameters(True, False)
  def test_read_default_values(self, blocking_write: bool):
    metadata = checkpoint.CheckpointMetadata()

    self.write_metadata_store(blocking_write).write(
        checkpoint_path=self.directory, checkpoint_metadata=metadata
    )

    self.write_metadata_store(blocking_write).wait_until_finished()

    self.assertEqual(
        self.write_metadata_store(blocking_write).read(
            checkpoint_path=self.directory
        ),
        metadata,
    )

  @parameterized.parameters(True, False)
  def test_read_with_values(self, blocking_write: bool):
    metadata = checkpoint.CheckpointMetadata(
        init_timestamp_nsecs=time.time_ns(),
        commit_timestamp_nsecs=time.time_ns() + 1,
    )
    self.write_metadata_store(blocking_write).write(
        checkpoint_path=self.directory, checkpoint_metadata=metadata
    )

    self.write_metadata_store(blocking_write).wait_until_finished()

    self.assertEqual(
        self.write_metadata_store(blocking_write).read(
            checkpoint_path=self.directory
        ),
        metadata,
    )

  @parameterized.parameters(True, False)
  def test_read_corrupt_json_data(self, blocking_write: bool):
    metadata_file = checkpoint.metadata_file_path(self.directory)
    metadata_file.touch()

    self.assertIsNone(
        self.write_metadata_store(blocking_write).read(
            checkpoint_path=self.directory
        )
    )

  @parameterized.parameters(True, False)
  def test_update_without_prior_data(self, blocking_write: bool):
    self.write_metadata_store(blocking_write).update(
        checkpoint_path=self.directory,
        init_timestamp_nsecs=1,
        commit_timestamp_nsecs=2,
    )

    self.write_metadata_store(blocking_write).wait_until_finished()

    self.assertEqual(
        self.write_metadata_store(blocking_write).read(
            checkpoint_path=self.directory
        ),
        checkpoint.CheckpointMetadata(
            init_timestamp_nsecs=1,
            commit_timestamp_nsecs=2,
        ),
    )

  @parameterized.parameters(True, False)
  def test_update_with_prior_data(self, blocking_write: bool):
    metadata = checkpoint.CheckpointMetadata(init_timestamp_nsecs=1)
    self.write_metadata_store(blocking_write).write(
        checkpoint_path=self.directory, checkpoint_metadata=metadata
    )

    self.write_metadata_store(blocking_write).update(
        checkpoint_path=self.directory,
        commit_timestamp_nsecs=2,
    )

    self.write_metadata_store(blocking_write).wait_until_finished()

    self.assertEqual(
        self.write_metadata_store(blocking_write).read(
            checkpoint_path=self.directory
        ),
        checkpoint.CheckpointMetadata(
            init_timestamp_nsecs=1,
            commit_timestamp_nsecs=2,
        ),
    )

  @parameterized.parameters(True, False)
  def test_update_with_unknown_kwargs(self, blocking_write: bool):
    with self.assertRaisesRegex(
        TypeError, "got an unexpected keyword argument 'blah'"
    ):
      self.write_metadata_store(blocking_write).update(
          checkpoint_path=self.directory,
          init_timestamp_nsecs=1,
          blah=2,
      )

  @parameterized.parameters(True, False)
  def test_write_with_read_only_store_is_no_op(self, blocking_write: bool):
    self.assertIsNone(
        self.read_metadata_store(blocking_write).read(
            checkpoint_path=self.directory
        )
    )

    self.read_metadata_store(blocking_write).write(
        checkpoint_path=self.directory,
        checkpoint_metadata=checkpoint.CheckpointMetadata(),
    )

    self.assertIsNone(
        self.read_metadata_store(blocking_write).read(
            checkpoint_path=self.directory
        )
    )
    self.assertIsNone(
        self.write_metadata_store(blocking_write).read(
            checkpoint_path=self.directory
        )
    )

  def test_non_blocking_write_request_enables_writes(self):
    # setup some data with blocking store.
    self.write_metadata_store(blocking_write=True).write(
        checkpoint_path=self.directory,
        checkpoint_metadata=checkpoint.CheckpointMetadata(
            init_timestamp_nsecs=1
        ),
    )

    self.assertEqual(
        self.read_metadata_store(blocking_write=False).read(
            checkpoint_path=self.directory
        ),
        checkpoint.CheckpointMetadata(init_timestamp_nsecs=1),
    )

    # write validations
    self.write_metadata_store(blocking_write=False).write(
        checkpoint_path=self.directory,
        checkpoint_metadata=checkpoint.CheckpointMetadata(
            init_timestamp_nsecs=2, commit_timestamp_nsecs=3
        ),
    )
    self.write_metadata_store(blocking_write=False).wait_until_finished()
    self.assertEqual(
        self.read_metadata_store(blocking_write=False).read(
            checkpoint_path=self.directory
        ),
        checkpoint.CheckpointMetadata(
            init_timestamp_nsecs=2, commit_timestamp_nsecs=3
        ),
    )
    self.assertEqual(
        self.write_metadata_store(blocking_write=False).read(
            checkpoint_path=self.directory
        ),
        checkpoint.CheckpointMetadata(
            init_timestamp_nsecs=2, commit_timestamp_nsecs=3
        ),
    )

    # update validations
    self.write_metadata_store(blocking_write=False).update(
        checkpoint_path=self.directory, commit_timestamp_nsecs=7
    )
    self.write_metadata_store(blocking_write=False).wait_until_finished()
    self.assertEqual(
        self.read_metadata_store(blocking_write=False).read(
            checkpoint_path=self.directory
        ),
        checkpoint.CheckpointMetadata(
            init_timestamp_nsecs=2, commit_timestamp_nsecs=7
        ),
    )
    self.assertEqual(
        self.write_metadata_store(blocking_write=False).read(
            checkpoint_path=self.directory
        ),
        checkpoint.CheckpointMetadata(
            init_timestamp_nsecs=2, commit_timestamp_nsecs=7
        ),
    )

  @parameterized.parameters(True, False)
  def test_pickle(self, blocking_write: bool):
    with self.assertRaisesRegex(TypeError, 'cannot pickle'):
      _ = pickle.dumps(self.write_metadata_store(blocking_write))
    _ = pickle.dumps(self.read_metadata_store(blocking_write))


if __name__ == '__main__':
  absltest.main()
