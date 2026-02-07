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

"""Tests for P2P CheckpointManager."""

from unittest import mock

from absl.testing import absltest
from etils import epath
import jax
import numpy as np
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint.experimental.emergency.p2p import args as p2p_args_lib
from orbax.checkpoint.experimental.emergency.p2p import checkpoint_manager as p2p_cm
from orbax.checkpoint.experimental.emergency.p2p import local
from orbax.checkpoint.experimental.emergency.p2p import peer_selector
from orbax.checkpoint.experimental.emergency.p2p import persistent
from orbax.checkpoint.experimental.emergency.p2p import protocol
from orbax.checkpoint.experimental.emergency.p2p import service

Mesh = jax.sharding.Mesh


class MockDevice:

  def __init__(self, process_index):
    self.process_index = process_index

  def __repr__(self):
    return f'MockDevice(pi={self.process_index})'


class CheckpointManagerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(multihost, 'get_jax_distributed_client')
    )
    self.enter_context(mock.patch.object(jax, 'process_count', return_value=4))
    if not multihost.is_runtime_to_distributed_ids_initialized():
      multihost.initialize_runtime_to_distributed_ids()
    self.local_dir = epath.Path(self.create_tempdir('local').full_path)
    self.persistent_dir = epath.Path(
        self.create_tempdir('persistent').full_path
    )

    devices = np.array([
        [MockDevice(0), MockDevice(1)],
        [MockDevice(2), MockDevice(3)],
    ])
    self.mesh = mock.Mock(
        spec=jax.sharding.Mesh,
        devices=devices,
        axis_names=('x',),
        shape={'x': devices.shape[0]},
        shape_tuple=devices.shape,
    )
    self.abstract_state = {'a': 1}

    self.mock_local = self.enter_context(
        mock.patch.object(local, 'LocalCheckpointManager')
    )
    self.mock_persistent = self.enter_context(
        mock.patch.object(
            target=persistent,
            attribute='PersistentCheckpointManager',
            autospec=True,
        )
    )
    self.mock_p2p_node = self.enter_context(
        mock.patch.object(service, 'P2PNode', autospec=True)
    )
    self.mock_peer_selector = self.enter_context(
        mock.patch.object(peer_selector, 'PeerSelector', autospec=True)
    )
    self.mock_sync_global_data = self.enter_context(
        mock.patch(
            'orbax.checkpoint.experimental.emergency.path.sync_global_data',
            autospec=True,
        )
    )
    self.mock_socket = self.enter_context(
        mock.patch('socket.gethostbyname', return_value='127.0.0.1')
    )
    self.enter_context(
        mock.patch('socket.gethostname', return_value='localhost')
    )
    self.mock_global_max = self.enter_context(
        mock.patch.object(multihost, 'global_max', return_value=[0])
    )

    # Mock instances returned by constructors
    self.local_manager_instance = self.mock_local.return_value
    self.persistent_manager_instance = self.mock_persistent.return_value
    self.p2p_node_instance = self.mock_p2p_node.return_value
    self.peer_selector_instance = self.mock_peer_selector.return_value

    self.p2p_node_instance.ip = '127.0.0.1'
    self.p2p_node_instance.port = 12345
    self.peer_selector_instance.get_latest_complete_step.return_value = None

  @mock.patch.object(multihost, 'process_index', return_value=0)
  def test_init_starts_p2p_and_discovery(self, _):
    self.local_manager_instance.scan_stored_steps.return_value = (0, [])
    self.mock_sync_global_data.return_value = []

    manager = p2p_cm.CheckpointManager(
        self.mesh,
        self.abstract_state,
        self.local_dir,
    )

    self.mock_local.assert_called_once()
    self.mock_persistent.assert_not_called()
    self.mock_p2p_node.assert_called_once()
    self.p2p_node_instance.start.assert_called_once()
    self.mock_peer_selector.assert_called_once()

    self.local_manager_instance.scan_stored_steps.assert_called_once()
    self.mock_sync_global_data.assert_called_once()
    manager.close()

  @mock.patch.object(multihost, 'process_index', return_value=0)
  def test_init_with_persistent(self, _):
    self.local_manager_instance.scan_stored_steps.return_value = (0, [])
    self.mock_sync_global_data.return_value = []
    manager = p2p_cm.CheckpointManager(
        self.mesh,
        self.abstract_state,
        self.local_dir,
        persistent_directory=self.persistent_dir,
    )
    self.mock_local.assert_called_once()
    self.mock_persistent.assert_called_once()
    self.assertIsNotNone(manager._persistent_manager)
    manager.close()

  @mock.patch.object(multihost, 'process_index', return_value=0)
  def test_save_without_persistent(self, _):
    self.local_manager_instance.scan_stored_steps.return_value = (0, [])
    self.mock_sync_global_data.return_value = []
    manager = p2p_cm.CheckpointManager(
        self.mesh,
        self.abstract_state,
        self.local_dir,
    )
    args = p2p_args_lib.Composite(state={'a': 1})
    manager.save(1, args=args)
    self.local_manager_instance.save.assert_called_once_with(
        1, args=args, force=False
    )
    self.persistent_manager_instance.save.assert_not_called()
    manager.close()

  @mock.patch.object(multihost, 'process_index', return_value=0)
  def test_save_with_persistent(self, _):
    self.local_manager_instance.scan_stored_steps.return_value = (0, [])
    self.mock_sync_global_data.return_value = []
    manager = p2p_cm.CheckpointManager(
        self.mesh,
        self.abstract_state,
        self.local_dir,
        persistent_directory=self.persistent_dir,
    )
    args = p2p_args_lib.Composite(state={'a': 1})
    manager.save(1, args=args)
    self.local_manager_instance.save.assert_called_once_with(
        1, args=args, force=False
    )
    self.persistent_manager_instance.save.assert_called_once_with(
        1, args=args, force=False
    )
    manager.close()

  @mock.patch.object(multihost, 'process_index', return_value=0)
  def test_restore_strategy_a_local_found(self, _):
    self.local_manager_instance.scan_stored_steps.return_value = (0, [1])
    self.local_manager_instance.all_steps.return_value = [1]
    self.local_manager_instance.restore.return_value = {'a': 1}
    self.mock_sync_global_data.return_value = []

    manager = p2p_cm.CheckpointManager(
        self.mesh,
        self.abstract_state,
        self.local_dir,
    )

    result = manager.restore(1)

    self.assertEqual(result, {'a': 1})
    self.local_manager_instance.restore.assert_called_once_with(1)
    self.p2p_node_instance.fetch_shard_from_peer.assert_not_called()
    manager.close()

  @mock.patch.object(multihost, 'process_index', return_value=0)
  def test_restore_strategy_b_p2p_fetch(self, process_index):
    self.local_manager_instance.scan_stored_steps.return_value = (0, [])
    self.local_manager_instance.all_steps.return_value = []
    self.mock_sync_global_data.return_value = []
    # Peer has step 1
    self.peer_selector_instance.get_source_peer.return_value = (
        protocol.PeerDiscoveryInfo(
            ip='1.2.3.4', port=5678, process_index=1, steps=[1]
        )
    )
    self.peer_selector_instance.get_latest_complete_step.return_value = 1
    self.p2p_node_instance.fetch_shard_from_peer.return_value = True
    self.local_manager_instance.restore.return_value = {'a': 1}

    manager = p2p_cm.CheckpointManager(
        self.mesh,
        self.abstract_state,
        self.local_dir,
    )

    result = manager.restore(1)
    self.assertEqual(result, {'a': 1})
    self.local_manager_instance.all_steps.assert_called()
    self.peer_selector_instance.get_source_peer.assert_called_once_with(
        1, process_index.return_value
    )
    self.p2p_node_instance.fetch_shard_from_peer.assert_called_once_with(
        '1.2.3.4', 5678, 1, 1
    )
    self.local_manager_instance.restore.assert_called_once()
    self.assertIn('directory', self.local_manager_instance.restore.call_args[1])
    manager.close()

  @mock.patch.object(multihost, 'process_index', return_value=0)
  def test_restore_strategy_c_persistent_fallback(self, process_index):
    self.local_manager_instance.scan_stored_steps.return_value = (0, [])
    self.local_manager_instance.all_steps.return_value = []
    self.mock_sync_global_data.return_value = []
    # P2P fetch fails
    self.peer_selector_instance.get_source_peer.return_value = None
    self.persistent_manager_instance.restore.return_value = {'a': 1}
    self.mock_global_max.return_value = [1]

    manager = p2p_cm.CheckpointManager(
        self.mesh,
        self.abstract_state,
        self.local_dir,
        persistent_directory=self.persistent_dir,
    )

    result = manager.restore(1)
    self.assertEqual(result, {'a': 1})
    self.local_manager_instance.all_steps.assert_called_once()
    self.peer_selector_instance.get_source_peer.assert_called_once_with(
        1, process_index.return_value
    )
    self.p2p_node_instance.fetch_shard_from_peer.assert_not_called()
    self.persistent_manager_instance.restore.assert_called_once_with(
        1, args=mock.ANY
    )
    self.assertIsInstance(
        self.persistent_manager_instance.restore.call_args[1]['args'],
        p2p_args_lib.Composite,
    )
    manager.close()

  @mock.patch.object(multihost, 'process_index', return_value=0)
  def test_restore_all_fail(self, _):
    self.local_manager_instance.scan_stored_steps.return_value = (0, [])
    self.local_manager_instance.all_steps.return_value = []
    self.mock_sync_global_data.return_value = []
    # P2P fetch fails
    self.peer_selector_instance.get_source_peer.return_value = None

    manager = p2p_cm.CheckpointManager(
        self.mesh,
        self.abstract_state,
        self.local_dir,
    )
    with self.assertRaises(FileNotFoundError):
      manager.restore(1)
    manager.close()

  @mock.patch.object(multihost, 'process_index', return_value=0)
  def test_restore_coordinated_fallback_peer_failed(self, _):
    """Tests that we fall back to persistent if a peer fails, even if we succeeded locally."""
    # 1. Setup: Local restore succeeds
    self.local_manager_instance.scan_stored_steps.return_value = (0, [1])
    self.local_manager_instance.all_steps.return_value = [1]
    self.local_manager_instance.restore.return_value = {'a': 1}  # Local success
    self.mock_sync_global_data.return_value = []

    # 2. Setup: Persistent manager returns a different value so we can verify
    # fallback was used
    self.persistent_manager_instance.restore.return_value = {'a': 999}

    # 3. Setup: global_max returns 1, indicating SOMEONE failed
    self.mock_global_max.return_value = [1]

    manager = p2p_cm.CheckpointManager(
        self.mesh,
        self.abstract_state,
        self.local_dir,
        persistent_directory=self.persistent_dir,
    )

    # 4. Action
    result = manager.restore(1)

    # 5. Verification
    # Should use persistent result
    self.assertEqual(result, {'a': 999})

    # Local restore WAS attempted
    self.local_manager_instance.restore.assert_called_once_with(1)

    # Persistent restore WAS called
    self.persistent_manager_instance.restore.assert_called_once_with(
        1, args=mock.ANY
    )

    # global_max was called with [0] because WE succeeded (my_failure=0)
    self.mock_global_max.assert_called_once_with([0])

    manager.close()

  @mock.patch.object(multihost, 'process_index', return_value=0)
  def test_restore_coordinated_fallback_local_failed(self, _):
    """Tests that we fall back to persistent if we fail locally."""
    # 1. Setup: Local/P2P fail
    self.local_manager_instance.scan_stored_steps.return_value = (0, [])
    self.local_manager_instance.all_steps.return_value = []
    self.mock_sync_global_data.return_value = []
    self.peer_selector_instance.get_source_peer.return_value = None  # P2P fails

    self.persistent_manager_instance.restore.return_value = {'a': 999}
    self.mock_global_max.return_value = [1]  # Everyone knows someone failed

    manager = p2p_cm.CheckpointManager(
        self.mesh,
        self.abstract_state,
        self.local_dir,
        persistent_directory=self.persistent_dir,
    )

    # 4. Action
    result = manager.restore(1)

    # 5. Verification
    self.assertEqual(result, {'a': 999})
    self.persistent_manager_instance.restore.assert_called_once()

    # global_max was called with [1] because WE failed (my_failure=1)
    self.mock_global_max.assert_called_once_with([1])

    manager.close()

  @mock.patch.object(multihost, 'process_index', return_value=0)
  def test_restore_no_step_in_p2p_but_in_persistent(self, _):
    """Tests fallback to persistent step if P2P has no step."""
    self.local_manager_instance.scan_stored_steps.return_value = (0, [])
    self.mock_sync_global_data.return_value = []

    # P2P has no latest complete step
    self.peer_selector_instance.get_latest_complete_step.return_value = None

    # Persistent has step 100
    self.persistent_manager_instance.latest_step.return_value = 100
    self.persistent_manager_instance.restore.return_value = {'a': 100}

    manager = p2p_cm.CheckpointManager(
        self.mesh,
        self.abstract_state,
        self.local_dir,
        persistent_directory=self.persistent_dir,
    )

    # Reset mock to ensure we only check calls during restore.
    self.mock_global_max.reset_mock()

    result = manager.restore(None)

    self.assertEqual(result, {'a': 100})
    self.persistent_manager_instance.latest_step.assert_called_once()
    self.persistent_manager_instance.restore.assert_called_once_with(
        100, args=mock.ANY
    )
    # Persistent storage is trusted; no global sync needed.
    self.mock_global_max.assert_not_called()

    manager.close()

  @mock.patch.object(p2p_cm.shutil, 'rmtree', autospec=True)
  @mock.patch.object(multihost, 'process_index', return_value=0)
  def test_restore_p2p_cleanup(self, unused_process_index, mock_rmtree):
    """Tests that P2P restore directory is cleaned up after restore."""
    self.local_manager_instance.scan_stored_steps.return_value = (0, [])
    self.local_manager_instance.all_steps.return_value = []
    self.mock_sync_global_data.return_value = []

    # P2P fetch succeeds
    self.peer_selector_instance.get_source_peer.return_value = (
        protocol.PeerDiscoveryInfo(
            ip='1.2.3.4', port=5678, process_index=1, steps=[1]
        )
    )
    self.p2p_node_instance.fetch_shard_from_peer.return_value = True
    self.local_manager_instance.restore.return_value = {'a': 1}

    manager = p2p_cm.CheckpointManager(
        self.mesh,
        self.abstract_state,
        self.local_dir,
    )

    # Make p2p_restore_dir exist so cleanup is triggered
    p2p_restore_dir = self.local_dir / service.constants.P2P_RESTORE_DIR_NAME
    p2p_restore_dir.mkdir()

    manager.restore(1)

    mock_rmtree.assert_called_once_with(str(p2p_restore_dir))
    manager.close()


if __name__ == '__main__':
  absltest.main()
