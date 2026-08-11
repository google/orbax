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

"""Initialization test for multi-tier checkpointing."""

import os
from unittest import mock

from absl.testing import absltest
from etils import epath
import jax
import numpy as np
from orbax.checkpoint._src.futures import signaling_client
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import initialization
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import pathways_topology
import yaml


class MultiTierCheckpointingInitializationTest(
    absltest.TestCase,
):
  """Tests for multi-tier checkpointing initialization."""

  def test_wait_for_replicator_file_to_disappear_success(self):
    tmp_dir = self.create_tempdir().full_path
    epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    replicator_file = epath.Path(tmp_dir) / initialization._REPLICATOR_FILE
    self.assertFalse(replicator_file.exists())
    initialization._wait_for_replicator_file_to_disappear(
        epath.Path(tmp_dir), timeout_seconds=5
    )

  def test_wait_for_replicator_file_to_disappear_timeout(self):
    tmp_dir = self.create_tempdir().full_path
    epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    replicator_file = epath.Path(tmp_dir) / initialization._REPLICATOR_FILE
    replicator_file.write_text("replicator.yaml")
    self.assertTrue(replicator_file.exists())
    with self.assertRaises(TimeoutError):
      initialization._wait_for_replicator_file_to_disappear(
          epath.Path(tmp_dir), timeout_seconds=1
      )

  def test_wait_for_replicator_file_to_disappear_fails_on_replicator_failed_file(
      self,
  ):
    tmp_dir = self.create_tempdir().full_path
    root = epath.Path(tmp_dir)
    root.mkdir(parents=True, exist_ok=True)
    failed_file = root / initialization._REPLICATOR_FAILED_FILE
    failed_file.write_text("replicator failed before disappearing")

    with self.assertRaisesRegex(
        RuntimeError,
        "Replicator fatal errors: replicator failed before disappearing",
    ):
      initialization._wait_for_replicator_file_to_disappear(
          root, timeout_seconds=5
      )
    self.assertFalse(failed_file.exists())

  def test_wait_for_replicator_file_to_disappear_ignores_errors_if_check_for_errors_is_false(
      self,
  ):
    tmp_dir = self.create_tempdir().full_path
    root = epath.Path(tmp_dir)
    root.mkdir(parents=True, exist_ok=True)
    failed_file = root / initialization._REPLICATOR_FAILED_FILE
    failed_file.write_text("replicator failed before disappearing")

    # Should not raise exception
    initialization._wait_for_replicator_file_to_disappear(
        root, timeout_seconds=5, check_for_errors=False
    )
    # The file is not processed so it should still exist
    self.assertTrue(failed_file.exists())

  def test_create_replicator_file(self):
    tmp_dir = self.create_tempdir().full_path
    epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    replicator_file = epath.Path(tmp_dir) / initialization._REPLICATOR_FILE
    self.assertFalse(replicator_file.exists())
    initialization._create_replicator_file(
        epath.Path(tmp_dir),
        run_name="test-run",
        num_nodes=2,
        data_parallelism=1,
        node_rank=0,
        peer_ranks=[1],
        backup_interval_minutes=10,
        backup_interval_steps=None,
    )
    expected_replicator_data = {
        "job-name": "test-run",
        "framework": "orbax",
        "assume-data-parallelism": 1,
        "node-rank": 0,
        "nodes": 2,
        "peer-ranks": [1],
        "backup-interval-minutes": 10,
    }

    self.assertTrue(replicator_file.exists())
    replicator_data = dict(yaml.safe_load(replicator_file.read_text()))
    self.assertDictEqual(replicator_data, expected_replicator_data)

  def test_create_replicator_file_steps(self):
    tmp_dir = self.create_tempdir().full_path
    epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    replicator_file = epath.Path(tmp_dir) / initialization._REPLICATOR_FILE
    self.assertFalse(replicator_file.exists())
    initialization._create_replicator_file(
        epath.Path(tmp_dir),
        run_name="test-run",
        num_nodes=2,
        data_parallelism=1,
        node_rank=0,
        peer_ranks=[1],
        backup_interval_minutes=None,
        backup_interval_steps=100,
    )
    expected_replicator_data = {
        "job-name": "test-run",
        "framework": "orbax",
        "assume-data-parallelism": 1,
        "node-rank": 0,
        "nodes": 2,
        "peer-ranks": [1],
        "backup-interval-steps": 100,
    }

    self.assertTrue(replicator_file.exists())
    replicator_data = dict(yaml.safe_load(replicator_file.read_text()))
    self.assertDictEqual(replicator_data, expected_replicator_data)

  def test_create_replicator_file_rejects_both_intervals_set(self):
    tmp_dir = self.create_tempdir().full_path
    epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    with self.assertRaisesRegex(
        ValueError,
        "Exactly one of backup_interval_minutes or backup_interval_steps",
    ):
      initialization._create_replicator_file(
          epath.Path(tmp_dir),
          run_name="test-run",
          num_nodes=2,
          data_parallelism=1,
          node_rank=0,
          peer_ranks=[1],
          backup_interval_minutes=10,
          backup_interval_steps=100,
      )

  def test_create_replicator_file_rejects_neither_interval_set(self):
    tmp_dir = self.create_tempdir().full_path
    epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    with self.assertRaisesRegex(
        ValueError,
        "Exactly one of backup_interval_minutes or backup_interval_steps",
    ):
      initialization._create_replicator_file(
          epath.Path(tmp_dir),
          run_name="test-run",
          num_nodes=2,
          data_parallelism=1,
          node_rank=0,
          peer_ranks=[1],
          backup_interval_minutes=None,
          backup_interval_steps=None,
      )

  def test_create_replicator_file_rejects_invalid_node_rank(self):
    tmp_dir = self.create_tempdir().full_path
    epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    with self.assertRaisesRegex(ValueError, "Invalid node_rank=-1"):
      initialization._create_replicator_file(
          epath.Path(tmp_dir),
          run_name="test-run",
          num_nodes=2,
          data_parallelism=1,
          node_rank=-1,
          peer_ranks=[1],
          backup_interval_minutes=10,
          backup_interval_steps=None,
      )

  def test_create_replicator_file_rejects_invalid_peer_rank(self):
    tmp_dir = self.create_tempdir().full_path
    epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    with self.assertRaisesRegex(ValueError, "Invalid peer_ranks"):
      initialization._create_replicator_file(
          epath.Path(tmp_dir),
          run_name="test-run",
          num_nodes=2,
          data_parallelism=1,
          node_rank=0,
          peer_ranks=[2],
          backup_interval_minutes=10,
          backup_interval_steps=None,
      )

  def test_block_and_process_restore_dir_success(self):
    tmp_dir = self.create_tempdir().full_path
    epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    restore_dir = epath.Path(tmp_dir) / "test-run-s1-n0-w0.restore"
    restore_dir.write_text("restore_dir")
    self.assertTrue(restore_dir.exists())
    self.assertTrue(
        initialization._block_and_process_restore_dir(
            epath.Path(tmp_dir), timeout_seconds=10
        )
    )
    self.assertFalse(restore_dir.exists())
    step_dir = epath.Path(tmp_dir) / "1"
    self.assertTrue(step_dir.exists())

  def test_block_and_process_restore_dir_keeps_process_metadata(self):
    tmp_dir = self.create_tempdir().full_path
    root = epath.Path(tmp_dir)
    root.mkdir(parents=True, exist_ok=True)
    restore_dir = root / "test-run-s1-n0-w0.restore"
    restore_dir.mkdir(parents=True, exist_ok=True)
    per_step_process_metadata = restore_dir / "process_metadata"
    per_step_process_metadata.mkdir(parents=True, exist_ok=True)
    (per_step_process_metadata / "mesh.json").write_text("metadata")

    self.assertTrue(
        initialization._block_and_process_restore_dir(
            root, timeout_seconds=10
        )
    )

    stable_process_metadata = root / "process_metadata"
    self.assertFalse(stable_process_metadata.exists())
    self.assertEqual(
        (root / "1" / "process_metadata" / "mesh.json").read_text(),
        "metadata",
    )

  def test_block_and_process_restore_dir_timeout(self):
    tmp_dir = self.create_tempdir().full_path
    epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    restore_dir = epath.Path(tmp_dir) / "test-run-s0-n0-w0.restore"
    self.assertFalse(restore_dir.exists())
    with self.assertRaises(TimeoutError):
      initialization._block_and_process_restore_dir(
          epath.Path(tmp_dir), timeout_seconds=1
      )

  def test_block_and_process_restore_dir_fails_on_replicator_failed_file(
      self,
  ):
    tmp_dir = self.create_tempdir().full_path
    root = epath.Path(tmp_dir)
    root.mkdir(parents=True, exist_ok=True)
    failed_file = root / initialization._REPLICATOR_FAILED_FILE
    failed_file.write_text("replicator daemon failed to start")

    with self.assertRaisesRegex(
        RuntimeError,
        "Replicator fatal errors: replicator daemon failed to start",
    ):
      initialization._block_and_process_restore_dir(root, timeout_seconds=1)
    self.assertFalse(failed_file.exists())

  def test_check_for_replicator_errors_processes_non_fatal_errors_file(self):
    tmp_dir = self.create_tempdir().full_path
    root = epath.Path(tmp_dir)
    root.mkdir(parents=True, exist_ok=True)
    errors_file = root / initialization._REPLICATOR_ERRORS_FILE
    errors_file.write_text("transient replicator error warning")

    initialization._check_for_replicator_errors(root)
    self.assertFalse(errors_file.exists())

  @mock.patch.object(initialization.epath, "Path")
  def test_read_replicator_error_file_handles_oserror(self, mock_path_cls):
    mock_path_obj = mock_path_cls.return_value
    mock_path_obj.read_text.side_effect = OSError("read error")

    result = initialization._read_replicator_error_file(
        epath.Path("dummy_path")
    )
    self.assertIsNone(result)
    mock_path_obj.read_text.assert_called_once()

  @mock.patch.object(initialization.epath, "Path")
  def test_cleanup_replicator_error_file_handles_oserror(self, mock_path_cls):
    mock_path_obj = mock_path_cls.return_value
    mock_path_obj.unlink.side_effect = OSError("unlink error")

    # Should not raise exception
    initialization._cleanup_replicator_error_file(epath.Path("dummy_path"))
    mock_path_obj.unlink.assert_called_once()

  def test_block_and_process_restore_dir_accepts_no_checkpoint_marker(self):
    tmp_dir = self.create_tempdir().full_path
    root = epath.Path(tmp_dir)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "test-run-s0-n0-w0.restore"
    marker.write_text("no checkpoint")

    self.assertTrue(
        initialization._block_and_process_restore_dir(
            root,
            timeout_seconds=1,
        )
    )
    self.assertFalse(marker.exists())
    self.assertFalse((root / "0").exists())

  def test_block_and_process_restore_dir_treats_step_zero_symlink_as_restore(
      self,
  ):
    tmp_dir = self.create_tempdir().full_path
    root = epath.Path(tmp_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "source-checkpoint"
    target.mkdir(parents=True, exist_ok=True)
    marker = root / "test-run-s0-n0-w0.restore"
    os.symlink(os.fspath(target), os.fspath(marker), target_is_directory=True)

    self.assertTrue(
        initialization._block_and_process_restore_dir(
            root,
            timeout_seconds=1,
        )
    )
    self.assertFalse(marker.exists())
    self.assertTrue((root / "0").exists())

  def test_validate_node_rank_by_process_index_rejects_negative_rank(self):
    with self.assertRaisesRegex(ValueError, "invalid entries"):
      initialization._validate_node_rank_by_process_index(
          [0, -1], num_nodes=2
      )

  def test_extract_step_rejects_malformed_restore_name(self):
    with self.assertRaisesRegex(
        ValueError, "Unexpected restore artifact name"
    ):
      initialization._extract_step("malformed.restore")

  def test_jax_init_info_file_exists(self):
    tmp_dir = self.create_tempdir().full_path
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
    tmp_dir = self.create_tempdir().full_path
    epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    with self.assertRaises(TimeoutError):
      initialization._retrieve_jax_init_info(
          epath.Path(tmp_dir), timeout_seconds=1
      )

  def test_jax_init_info_file_has_empty_values(self):
    tmp_dir = self.create_tempdir().full_path
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
    tmp_dir = self.create_tempdir().full_path
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

    tmp_dir = self.create_tempdir().full_path
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

    with (
        mock.patch.object(
            initialization.jax, "process_count", return_value=2
        ),
        mock.patch.object(
            initialization.jax, "process_index", return_value=0
        ),
    ):
      initialization.initialize_multi_tier_checkpointing(
          epath.Path(tmp_dir),
          num_slices=1,
          run_name="test-run",
          data_parallelism=1,
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

    tmp_dir = self.create_tempdir().full_path
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

    mock_jax_distributed_initialize.assert_not_called()
    mock_initialize_runtime_to_distributed_ids.assert_not_called()
    mock_initialize_distributed_to_device_ids.assert_not_called()
    self.assertEqual(mock_wait_for_replicator_file_to_disappear.call_count, 0)

  @mock.patch.object(initialization, "_initialize_mtc_colocated", autospec=True)
  @mock.patch.object(jax.distributed, "initialize", autospec=True)
  def test_initialize_multi_tier_checkpointing_colocated_success(
      self,
      mock_jax_distributed_initialize,
      mock_init_mtc_colocated,
  ):
    tmp_dir = self.create_tempdir().full_path
    tmp_dir_path = epath.Path(tmp_dir)

    initialization.initialize_multi_tier_checkpointing(
        tmp_dir_path,
        num_slices=1,
        run_name="test-colocated-run",
        data_parallelism=1,
        use_colocated_python=True,
        backup_interval_minutes=15,
        backup_interval_steps=None,
        devices=None,
    )

    # Verify colocated Python path is taken
    mock_init_mtc_colocated.assert_called_once_with(
        local_checkpoint_directory=tmp_dir_path,
        backup_interval_minutes=15,
        backup_interval_steps=None,
        num_slices=1,
        run_name="test-colocated-run",
        data_parallelism=1,
        timeout_seconds=900,
        devices=None,
    )

    # Verify standard multi-controller JAX init is bypassed
    mock_jax_distributed_initialize.assert_not_called()

  @mock.patch.object(initialization, "_initialize_mtc_colocated", autospec=True)
  @mock.patch.object(jax.distributed, "initialize", autospec=True)
  def test_initialize_multi_tier_checkpointing_colocated_uses_devices(
      self,
      mock_jax_distributed_initialize,
      mock_init_mtc_colocated,
  ):
    tmp_dir_path = epath.Path(self.create_tempdir().full_path)
    active_devices = (
        mock.Mock(spec=jax.Device),
        mock.Mock(spec=jax.Device),
    )

    initialization.initialize_multi_tier_checkpointing(
        tmp_dir_path,
        num_slices=1,
        run_name="test-colocated-run",
        data_parallelism=1,
        use_colocated_python=True,
        devices=active_devices,
    )

    mock_init_mtc_colocated.assert_called_once_with(
        local_checkpoint_directory=tmp_dir_path,
        backup_interval_minutes=30,
        backup_interval_steps=None,
        num_slices=1,
        run_name="test-colocated-run",
        data_parallelism=1,
        timeout_seconds=900,
        devices=active_devices,
    )
    mock_jax_distributed_initialize.assert_not_called()

  def test_initialize_multi_tier_checkpointing_rejects_devices_without_colocated_python(
      self,
  ):
    tmp_dir_path = epath.Path(self.create_tempdir().full_path)

    with self.assertRaisesRegex(
        ValueError, "`devices` is only supported when use_colocated_python=True"
    ):
      initialization.initialize_multi_tier_checkpointing(
          tmp_dir_path,
          run_name="test-run",
          devices=(mock.Mock(spec=jax.Device),),
      )

  @mock.patch.object(initialization.multislice, "slice_count", autospec=True)
  @mock.patch.object(initialization, "_initialize_mtc_colocated", autospec=True)
  def test_initialize_multi_tier_checkpointing_infers_defaults_when_none(
      self,
      mock_init_mtc_colocated,
      mock_slice_count,
  ):
    tmp_dir_path = epath.Path(self.create_tempdir().full_path)
    mock_slice_count.return_value = 8

    initialization.initialize_multi_tier_checkpointing(
        tmp_dir_path,
        run_name="test-colocated-run",
        num_slices=None,
        data_parallelism=None,
        use_colocated_python=True,
    )

    mock_init_mtc_colocated.assert_called_once_with(
        local_checkpoint_directory=tmp_dir_path,
        backup_interval_minutes=30,
        backup_interval_steps=None,
        num_slices=8,
        run_name="test-colocated-run",
        data_parallelism=8,
        timeout_seconds=900,
        devices=None,
    )

  @mock.patch.object(initialization.multislice, "slice_count", autospec=True)
  @mock.patch.object(initialization, "_initialize_mtc_colocated", autospec=True)
  def test_initialize_multi_tier_checkpointing_infers_defaults_when_zero_or_negative(
      self,
      mock_init_mtc_colocated,
      mock_slice_count,
  ):
    tmp_dir_path = epath.Path(self.create_tempdir().full_path)
    mock_slice_count.return_value = 8

    initialization.initialize_multi_tier_checkpointing(
        tmp_dir_path,
        run_name="test-colocated-run",
        num_slices=0,
        data_parallelism=-1,
        use_colocated_python=True,
    )

    mock_init_mtc_colocated.assert_called_once_with(
        local_checkpoint_directory=tmp_dir_path,
        backup_interval_minutes=30,
        backup_interval_steps=None,
        num_slices=8,
        run_name="test-colocated-run",
        data_parallelism=8,
        timeout_seconds=900,
        devices=None,
    )

  @mock.patch.object(initialization.multislice, "slice_count", autospec=True)
  @mock.patch.object(initialization, "_initialize_mtc_colocated", autospec=True)
  def test_initialize_multi_tier_checkpointing_infers_data_parallelism_from_num_slices(
      self,
      mock_init_mtc_colocated,
      mock_slice_count,
  ):
    tmp_dir_path = epath.Path(self.create_tempdir().full_path)
    mock_slice_count.return_value = 8

    initialization.initialize_multi_tier_checkpointing(
        tmp_dir_path,
        run_name="test-colocated-run",
        num_slices=2,
        data_parallelism=None,
        use_colocated_python=True,
    )

    mock_init_mtc_colocated.assert_called_once_with(
        local_checkpoint_directory=tmp_dir_path,
        backup_interval_minutes=30,
        backup_interval_steps=None,
        num_slices=2,
        run_name="test-colocated-run",
        data_parallelism=2,
        timeout_seconds=900,
        devices=None,
    )

  @mock.patch.object(initialization.multislice, "slice_count", autospec=True)
  @mock.patch.object(initialization, "_initialize_jax_from_mtc", autospec=True)
  @mock.patch.object(
      multihost, "initialize_runtime_to_distributed_ids", autospec=True
  )
  def test_initialize_multi_tier_checkpointing_infers_defaults_when_none_in_standard_path(
      self,
      mock_initialize_runtime_to_distributed_ids,
      unused_mock_initialize_jax_from_mtc,
      mock_slice_count,
  ):
    mock_slice_count.return_value = 8
    mock_initialize_runtime_to_distributed_ids.side_effect = (
        RuntimeError("stop test early")
    )

    with self.assertRaisesRegex(RuntimeError, "stop test early"):
      initialization.initialize_multi_tier_checkpointing(
          epath.Path(self.create_tempdir().full_path),
          num_slices=None,
          use_mtc_process_ids=True,
          run_name="test-run",
      )
    mock_slice_count.assert_called_once()

  @mock.patch.object(initialization.jax, "make_array_from_callback")
  @mock.patch.object(initialization.jax, "block_until_ready")
  @mock.patch.object(initialization.time, "time")
  @mock.patch.object(initialization, "_block_and_process_restore_dir")
  @mock.patch.object(initialization, "_wait_for_replicator_file_to_disappear")
  @mock.patch.object(initialization, "_create_replicator_file")
  @mock.patch.object(initialization.dispatchers, "get_dummy_input_array")
  @mock.patch.object(initialization.colocated_python, "colocated_python")
  @mock.patch.object(
      initialization.colocated_transport,
      "install_pathways_colocated_serialization_patch",
  )
  @mock.patch.object(initialization.jax, "devices")
  @mock.patch.object(initialization.pathways_topology.Topology, "from_devices")
  @mock.patch.object(initialization.jax, "device_count", return_value=8)
  @mock.patch.object(initialization.jax, "process_index", return_value=0)
  @mock.patch.object(initialization.jax, "process_count", return_value=1)
  def test_initialize_mtc_colocated_marks_sidecar_runtime(
      self,
      mock_process_count,
      mock_process_index,
      mock_device_count,
      mock_topology_from_devices,
      mock_devices,
      mock_install_patch,
      mock_colocated_python,
      mock_get_dummy_input_array,
      mock_create_replicator_file,
      mock_wait_for_replicator_file_to_disappear,
      mock_block_and_process_restore_dir,
      mock_time,
      mock_block_until_ready,
      mock_make_array_from_callback,
  ):
    # Suppress unused argument warnings
    self.assertIsNotNone(mock_process_count)
    self.assertIsNotNone(mock_process_index)
    self.assertIsNotNone(mock_device_count)

    dummy_in = mock.Mock(
        spec=jax.core.ShapedArray, shape=(), sharding="dummy-sharding"
    )
    worker_rank_in = np.asarray([1], dtype=np.int32)
    topology = mock.Mock(spec=pathways_topology.Topology)
    topology.worker_cpu_devices.return_value = (
        mock.Mock(spec=jax.Device, id=7),
        mock.Mock(spec=jax.Device, id=8),
        mock.Mock(spec=jax.Device, id=9),
        mock.Mock(spec=jax.Device, id=10),
    )
    topology.worker_rank_array.return_value = worker_rank_in
    topology.num_workers = 4
    topology.workers = (
        mock.Mock(spec=pathways_topology.Worker, key=(0, 0), device_ids=(0, 1)),
        mock.Mock(spec=pathways_topology.Worker, key=(1, 0), device_ids=(2, 3)),
        mock.Mock(spec=pathways_topology.Worker, key=(0, 1), device_ids=(4, 5)),
        mock.Mock(spec=pathways_topology.Worker, key=(1, 1), device_ids=(6, 7)),
    )
    topology.peer_ranks_by_worker_rank.return_value = (
        (2,),
        (3,),
        (0,),
        (1,),
    )
    mock_topology_from_devices.return_value = topology
    mock_get_dummy_input_array.return_value = dummy_in
    active_devices = (
        mock.Mock(spec=jax.Device),
        mock.Mock(spec=jax.Device),
    )
    mock_devices.return_value = ("stale_tpu0", "stale_tpu1")
    mock_make_array_from_callback.return_value = np.asarray(True)
    mock_time.return_value = 100.0

    def _wrap_setup(fn):
      closure_contents = tuple(
          cell.cell_contents for cell in fn.__closure__ or ()
      )
      self.assertNotIn(topology, closure_contents)
      self.assertNotIn(
          topology.worker_cpu_devices.return_value, closure_contents
      )

      class _Wrapped:
        def specialize(self, *, out_specs_fn):
          del out_specs_fn
          return fn

      return _Wrapped()

    mock_colocated_python.side_effect = _wrap_setup

    with mock.patch.object(
        signaling_client,
        "mark_pathways_colocated_runtime_active",
        autospec=True,
    ) as mock_mark_sidecar_runtime:
      initialization._initialize_mtc_colocated(
          local_checkpoint_directory=epath.Path("/tmp/mtc"),
          backup_interval_minutes=15,
          backup_interval_steps=None,
          num_slices=2,
          run_name="test-run",
          data_parallelism=1,
          timeout_seconds=900,
          devices=active_devices,
      )

    mock_install_patch.assert_called_once_with()
    mock_devices.assert_not_called()
    mock_topology_from_devices.assert_called_once_with(active_devices)
    topology.worker_cpu_devices.assert_called_once_with()
    topology.worker_rank_array.assert_called_once_with(
        topology.worker_cpu_devices.return_value
    )
    topology.peer_ranks_by_worker_rank.assert_called_once_with(2)
    mock_mark_sidecar_runtime.assert_called_once_with()
    mock_create_replicator_file.assert_called_once()
    self.assertEqual(
        mock_create_replicator_file.call_args.kwargs["node_rank"], 1
    )
    self.assertEqual(
        mock_create_replicator_file.call_args.kwargs["peer_ranks"], [3]
    )
    mock_wait_for_replicator_file_to_disappear.assert_called_once_with(
        epath.Path("/tmp/mtc"),
        timeout_seconds=600,
    )
    mock_block_and_process_restore_dir.assert_called_once_with(
        epath.Path("/tmp/mtc"),
        timeout_seconds=900,
    )
    mock_block_until_ready.assert_called_once()

  @mock.patch.object(
      initialization, "_wait_for_replicator_file_to_disappear", autospec=True
  )
  @mock.patch.object(initialization, "_create_replicator_file", autospec=True)
  @mock.patch.object(jax.distributed, "initialize", autospec=True)
  @mock.patch.object(
      multihost, "initialize_runtime_to_distributed_ids", autospec=True
  )
  @mock.patch.object(
      multihost, "initialize_distributed_to_device_ids", autospec=True
  )
  @mock.patch.object(multihost, "runtime_to_distributed_ids", autospec=True)
  def test_initialize_multi_tier_checkpointing_skip_init_info(
      self,
      mock_runtime_to_distributed_ids,
      mock_initialize_distributed_to_device_ids,
      mock_initialize_runtime_to_distributed_ids,
      mock_jax_distributed_initialize,
      mock_create_replicator_file,
      mock_wait_for_replicator_file_to_disappear,
  ):
    mock_runtime_to_distributed_ids.return_value = [0, 1]
    mock_jax_distributed_initialize.return_value = None
    mock_initialize_runtime_to_distributed_ids.return_value = [None, None]
    mock_initialize_distributed_to_device_ids.return_value = None
    mock_create_replicator_file.return_value = [None, None]
    mock_wait_for_replicator_file_to_disappear.return_value = False

    tmp_dir = self.create_tempdir().full_path
    epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    replicator_file = epath.Path(tmp_dir) / initialization._REPLICATOR_FILE
    replicator_file.write_text("replicator.yaml")
    self.assertTrue(replicator_file.exists())

    restore_dir = epath.Path(tmp_dir) / "test-run-s1-n0-w0.restore"
    restore_dir.write_text("restore_dir")
    self.assertTrue(restore_dir.exists())

    with (
        mock.patch.object(
            initialization.jax, "process_count", return_value=2
        ),
        mock.patch.object(
            initialization.jax, "process_index", return_value=0
        ),
    ):
      initialization.initialize_multi_tier_checkpointing(
          epath.Path(tmp_dir),
          num_slices=1,
          run_name="test-run",
          data_parallelism=1,
          use_mtc_process_ids=False,
      )
    mock_jax_distributed_initialize.assert_called_once_with(
        initialization_timeout=900,
    )
    mock_initialize_runtime_to_distributed_ids.assert_called_once()
    mock_initialize_distributed_to_device_ids.assert_called_once()
    self.assertEqual(mock_wait_for_replicator_file_to_disappear.call_count, 2)
    mock_create_replicator_file.assert_called_once()
    expected_restore_dir = epath.Path(tmp_dir) / "1"
    self.assertTrue(expected_restore_dir.exists())


if __name__ == "__main__":
  absltest.main()
