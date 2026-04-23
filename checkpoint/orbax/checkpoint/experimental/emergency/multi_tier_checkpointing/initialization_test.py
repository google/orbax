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

import tempfile
from unittest import mock

from absl.testing import absltest
from etils import epath
import jax
import numpy as np
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
          data_parallelism=1,
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
      initialization._block_and_process_restore_dir(
          epath.Path(tmp_dir), timeout_seconds=10
      )
      self.assertFalse(restore_dir.exists())
      step_dir = epath.Path(tmp_dir) / "1"
      self.assertTrue(step_dir.exists())

  def test_block_and_process_restore_dir_keeps_process_metadata(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = epath.Path(tmp_dir)
      root.mkdir(parents=True, exist_ok=True)
      restore_dir = root / "test-run-s1-n0-w0.restore"
      restore_dir.mkdir(parents=True, exist_ok=True)
      per_step_process_metadata = restore_dir / "process_metadata"
      per_step_process_metadata.mkdir(parents=True, exist_ok=True)
      (per_step_process_metadata / "mesh.json").write_text("metadata")

      initialization._block_and_process_restore_dir(
          root, timeout_seconds=10
      )

      stable_process_metadata = root / "process_metadata"
      self.assertFalse(stable_process_metadata.exists())
      self.assertEqual(
          (root / "1" / "process_metadata" / "mesh.json").read_text(),
          "metadata",
      )

  def test_block_and_process_restore_dir_timeout(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
      restore_dir = epath.Path(tmp_dir) / "test-run-s0-n0-w0.restore"
      self.assertFalse(restore_dir.exists())
      with self.assertRaises(TimeoutError):
        initialization._block_and_process_restore_dir(
            epath.Path(tmp_dir), timeout_seconds=1
        )

  def test_extract_step_rejects_malformed_restore_name(self):
    with self.assertRaisesRegex(
        ValueError, "Unexpected restore artifact name"
    ):
      initialization._extract_step("malformed.restore")

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
    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp_dir_path = epath.Path(tmp_dir)

      initialization.initialize_multi_tier_checkpointing(
          tmp_dir_path,
          num_slices=1,
          run_name="test-colocated-run",
          data_parallelism=1,
          use_colocated_python=True,
          backup_interval_minutes=15,
      )

      # Verify colocated Python path is taken
      mock_init_mtc_colocated.assert_called_once_with(
          local_checkpoint_directory=tmp_dir_path,
          backup_interval_minutes=15,
          num_slices=1,
          run_name="test-colocated-run",
          data_parallelism=1,
          timeout_seconds=900,
      )

      # Verify standard multi-controller JAX init is bypassed
      mock_jax_distributed_initialize.assert_not_called()

  @mock.patch.object(initialization.jax, "make_array_from_callback")
  @mock.patch.object(initialization.jax, "block_until_ready")
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
  @mock.patch.object(
      initialization.colocated_transport, "unique_colocated_cpu_devices"
  )
  @mock.patch.object(initialization.jax, "device_count", return_value=8)
  @mock.patch.object(initialization.jax, "process_index", return_value=0)
  @mock.patch.object(initialization.jax, "process_count", return_value=1)
  def test_initialize_mtc_colocated_marks_sidecar_runtime(
      self,
      mock_process_count,
      mock_process_index,
      mock_device_count,
      mock_unique_colocated_cpu_devices,
      mock_devices,
      mock_install_patch,
      mock_colocated_python,
      mock_get_dummy_input_array,
      mock_create_replicator_file,
      mock_wait_for_replicator_file_to_disappear,
      mock_block_and_process_restore_dir,
      mock_block_until_ready,
      mock_make_array_from_callback,
  ):
    # Suppress unused argument warnings
    self.assertIsNotNone(mock_process_count)
    self.assertIsNotNone(mock_process_index)
    self.assertIsNotNone(mock_device_count)

    dummy_in = mock.Mock(shape=(), sharding="dummy-sharding")
    mock_get_dummy_input_array.return_value = dummy_in
    mock_devices.return_value = ["tpu0"]
    mock_unique_colocated_cpu_devices.return_value = (
        mock.Mock(id=7, process_index=0),
    )
    mock_make_array_from_callback.return_value = np.asarray(True)

    def _wrap_setup(fn):
      class _Wrapped:
        def specialize(self, *, out_specs_fn):
          del out_specs_fn
          return fn

      return _Wrapped()

    mock_colocated_python.side_effect = _wrap_setup

    with mock.patch(
        "orbax.checkpoint._src.futures.signaling_client.mark_pathways_colocated_runtime_active"
    ) as mock_mark_sidecar_runtime:
      initialization._initialize_mtc_colocated(
          local_checkpoint_directory=epath.Path("/tmp/mtc"),
          backup_interval_minutes=15,
          num_slices=1,
          run_name="test-run",
          data_parallelism=1,
          timeout_seconds=30,
      )

    mock_install_patch.assert_called_once_with()
    mock_unique_colocated_cpu_devices.assert_called_once_with(("tpu0",))
    mock_mark_sidecar_runtime.assert_called_once_with()
    mock_create_replicator_file.assert_called_once()
    mock_wait_for_replicator_file_to_disappear.assert_called_once()
    mock_block_and_process_restore_dir.assert_called_once()
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

    with tempfile.TemporaryDirectory() as tmp_dir:
      epath.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
      replicator_file = epath.Path(tmp_dir) / initialization._REPLICATOR_FILE
      replicator_file.write_text("replicator.yaml")
      self.assertTrue(replicator_file.exists())

      restore_dir = epath.Path(tmp_dir) / "test-run-s1-n0-w0.restore"
      restore_dir.write_text("restore_dir")
      self.assertTrue(restore_dir.exists())

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
