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

"""Initialization test for multi-tier checkpointing."""

import tempfile
from unittest import mock

from absl.testing import absltest
from etils import epath
import jax
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import initialization
import yaml


class MultiTierCheckpointingInitializationTest(
    absltest.TestCase,
):
  """Tests for multi-tier checkpointing initialization."""

  def test_wait_for_replicator_file_to_disappear_success(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
      replicator_file = epath.Path(tmp_dir) / initialization._REPLICATOR_FILE
      self.assertFalse(replicator_file.exists())
      initialization._wait_for_replicator_file_to_disappear(
          epath.Path(tmp_dir), timeout_seconds=5
      )

  def test_wait_for_replicator_file_to_disappear_timeout(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
      replicator_file = epath.Path(tmp_dir) / initialization._REPLICATOR_FILE
      replicator_file.write_text("replicator.yaml")
      self.assertTrue(replicator_file.exists())
      with self.assertRaises(TimeoutError):
        initialization._wait_for_replicator_file_to_disappear(
            epath.Path(tmp_dir), timeout_seconds=1
        )

  def test_create_replicator_file(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
      replicator_file = epath.Path(tmp_dir) / initialization._REPLICATOR_FILE
      self.assertFalse(replicator_file.exists())
      initialization._create_replicator_file(
          epath.Path(tmp_dir),
          run_name="test-run",
          num_nodes=1,
          num_slices=1,
          node_rank=0,
          peer_ranks=[1],
          backup_interval_minutes=10,
      )
      expected_replicator_data = {
          "job-name": "test-run",
          "framework": "orbax",
          "assume-data-parallelism": 1,
          "node-rank": 0,
          "nodes": 1,
          "peer-ranks": [1],
          "backup-interval-minutes": 10,
      }

      self.assertTrue(replicator_file.exists())
      replicator_data = dict(yaml.safe_load(replicator_file.read_text()))
      self.assertDictEqual(replicator_data, expected_replicator_data)

  def test_block_and_process_restore_dir_success(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
      restore_dir = epath.Path(tmp_dir) / "test-run-s1-n0-w0.restore"
      restore_dir.write_text("restore_dir")
      self.assertTrue(restore_dir.exists())
      initialization._block_and_process_restore_dir(epath.Path(tmp_dir))
      self.assertFalse(restore_dir.exists())
      step_dir = epath.Path(tmp_dir) / "1"
      self.assertTrue(step_dir.exists())

  def test_block_and_process_restore_dir_timeout(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
      restore_dir = epath.Path(tmp_dir) / "test-run-s0-n0-w0.restore"
      self.assertFalse(restore_dir.exists())
      with self.assertRaises(TimeoutError):
        initialization._block_and_process_restore_dir(
            epath.Path(tmp_dir), timeout_seconds=1
        )

  def test_jax_init_info_file_exists(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
      jax_init_info_file = (
          epath.Path(tmp_dir) / initialization._JAX_INIT_INFO_FILE
      )
      jax_init_info_file.write_text("0\ncoordinator_address")
      self.assertTrue(jax_init_info_file.exists())
      process_id, coordinator_address = initialization._retrieve_jax_init_info(
          epath.Path(tmp_dir), timeout_seconds=1
      )
      self.assertEqual(process_id, "0")
      self.assertEqual(coordinator_address, "coordinator_address")

  def test_jax_init_info_file_not_exists_timeout(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
      with self.assertRaises(TimeoutError):
        initialization._retrieve_jax_init_info(
            epath.Path(tmp_dir), timeout_seconds=1
        )

  def test_jax_init_info_file_has_empty_values(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
      jax_init_info_file = (
          epath.Path(tmp_dir) / initialization._JAX_INIT_INFO_FILE
      )
      jax_init_info_file.write_text("")
      self.assertTrue(jax_init_info_file.exists())
      with self.assertRaises(ValueError):
        initialization._retrieve_jax_init_info(epath.Path(tmp_dir))

  def test_initialize_multi_tier_checkpointing_incorrect_jax_init_info(
      self,
  ):
    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp_dir_path = epath.Path(tmp_dir)
      jax_init_info_file = (
          tmp_dir_path / initialization._JAX_INIT_INFO_FILE
      )
      jax_init_info_file.write_text("0\n")
      self.assertTrue(jax_init_info_file.exists())
      with self.assertRaises(ValueError):
        initialization.initialize_multi_tier_checkpointing(
            local_checkpoint_directory=tmp_dir_path,
            num_slices=2,
            run_name="test-run",
        )

  @mock.patch.object(
      initialization, "_wait_for_replicator_file_to_disappear", autospec=True
  )
  @mock.patch.object(
      initialization, "_create_replicator_file", autospec=True
  )
  @mock.patch.object(
      initialization, "_retrieve_jax_init_info", autospec=True
  )
  @mock.patch.object(jax.distributed, "initialize", autospec=True)
  @mock.patch.object(
      multihost, "initialize_runtime_to_distributed_ids", autospec=True
  )
  @mock.patch.object(
      multihost, "initialize_distributed_to_device_ids", autospec=True
  )
  @mock.patch.object(
      multihost, "runtime_to_distributed_ids", autospec=True
  )
  def test_initialize_multi_tier_checkpointing_success(
      self,
      mock_runtime_to_distributed_ids,
      mock_initialize_distributed_to_device_ids,
      mock_initialize_runtime_to_distributed_ids,
      mock_jax_distributed_initialize,
      mock_retrieve_jax_init_info,
      mock_create_replicator_file,
      mock_wait_for_replicator_file_to_disappear,
  ):
    mock_runtime_to_distributed_ids.return_value = [0, 1]
    mock_retrieve_jax_init_info.return_value = ["0", "coordinator_address"]
    mock_jax_distributed_initialize.return_value = None
    mock_initialize_runtime_to_distributed_ids.return_value = [None, None]
    mock_initialize_distributed_to_device_ids.return_value = None
    mock_create_replicator_file.return_value = [None, None]
    mock_wait_for_replicator_file_to_disappear.return_value = False

    with tempfile.TemporaryDirectory() as tmp_dir:
      epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
      replicator_file = epath.Path(tmp_dir) / initialization._REPLICATOR_FILE
      replicator_file.write_text("replicator.yaml")
      self.assertTrue(replicator_file.exists())

      jax_init_info_file = (
          epath.Path(tmp_dir) / initialization._JAX_INIT_INFO_FILE
      )
      jax_init_info_file.write_text("0\ncoordinator_address")
      self.assertTrue(jax_init_info_file.exists())
      restore_dir = epath.Path(tmp_dir) / "test-run-s1-n0-w0.restore"
      restore_dir.write_text("restore_dir")
      self.assertTrue(restore_dir.exists())

      initialization.initialize_multi_tier_checkpointing(
          epath.Path(tmp_dir),
          num_slices=1,
          run_name="test-run",
      )
      mock_jax_distributed_initialize.assert_called_once_with(
          process_id=0,
          coordinator_address="coordinator_address",
          initialization_timeout=900,
      )
      mock_initialize_runtime_to_distributed_ids.assert_called_once()
      mock_initialize_distributed_to_device_ids.assert_called_once()
      self.assertEqual(mock_wait_for_replicator_file_to_disappear.call_count, 2)
      mock_create_replicator_file.assert_called_once()
      expected_restore_dir = epath.Path(tmp_dir) / "1"
      self.assertTrue(expected_restore_dir.exists())

  @mock.patch.object(
      initialization, "_wait_for_replicator_file_to_disappear", autospec=True
  )
  @mock.patch.object(
      initialization, "_create_replicator_file", autospec=True
  )
  @mock.patch.object(
      initialization, "_retrieve_jax_init_info", autospec=True
  )
  @mock.patch.object(jax.distributed, "initialize", autospec=True)
  @mock.patch.object(
      multihost, "initialize_runtime_to_distributed_ids", autospec=True
  )
  @mock.patch.object(
      multihost, "initialize_distributed_to_device_ids", autospec=True
  )
  @mock.patch.object(
      multihost, "runtime_to_distributed_ids", autospec=True
  )
  def test_initialize_multi_tier_checkpointing_run_name_not_set(
      self,
      mock_runtime_to_distributed_ids,
      mock_initialize_distributed_to_device_ids,
      mock_initialize_runtime_to_distributed_ids,
      mock_jax_distributed_initialize,
      mock_retrieve_jax_init_info,
      mock_create_replicator_file,
      mock_wait_for_replicator_file_to_disappear,
  ):
    mock_runtime_to_distributed_ids.return_value = [0, 1]
    mock_retrieve_jax_init_info.return_value = ["0", "coordinator_address"]
    mock_jax_distributed_initialize.return_value = None
    mock_initialize_runtime_to_distributed_ids.return_value = [None, None]
    mock_initialize_distributed_to_device_ids.return_value = None
    mock_create_replicator_file.return_value = None
    mock_wait_for_replicator_file_to_disappear.return_value = False

    with tempfile.TemporaryDirectory() as tmp_dir:
      epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
      replicator_file = epath.Path(tmp_dir) / initialization._REPLICATOR_FILE
      replicator_file.write_text("replicator.yaml")
      self.assertTrue(replicator_file.exists())

      jax_init_info_file = (
          epath.Path(tmp_dir) / initialization._JAX_INIT_INFO_FILE
      )
      jax_init_info_file.write_text("0\ncoordinator_address")
      self.assertTrue(jax_init_info_file.exists())
      restore_dir = epath.Path(tmp_dir) / "test-run-s1-n0-w0.restore"
      restore_dir.write_text("restore_dir")
      self.assertTrue(restore_dir.exists())

      with self.assertRaises(ValueError):
        initialization.initialize_multi_tier_checkpointing(
            epath.Path(tmp_dir),
            num_slices=1,
            run_name="",
        )
      mock_jax_distributed_initialize.assert_called_once_with(
          process_id=0,
          coordinator_address="coordinator_address",
          initialization_timeout=900,
      )
      mock_initialize_runtime_to_distributed_ids.assert_called_once()
      mock_initialize_distributed_to_device_ids.assert_called_once()
      self.assertEqual(mock_wait_for_replicator_file_to_disappear.call_count, 1)


if __name__ == "__main__":
  absltest.main()
