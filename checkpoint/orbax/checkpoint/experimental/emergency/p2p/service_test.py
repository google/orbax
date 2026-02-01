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

"""Unit tests for P2PNode service."""

import functools
import threading
from unittest import mock

from absl.testing import absltest
from etils import epath
from orbax.checkpoint.experimental.emergency.p2p import service


class NodeHandlerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_request = mock.Mock()
    self.mock_node_service = mock.create_autospec(
        service.P2PNode, instance=True
    )
    mock_server = mock.Mock(spec=service._ThreadingTCPServer)
    # Mock handle during __init__ because BaseRequestHandler calls it.
    with mock.patch.object(service.NodeHandler, 'handle'):
      self.handler = service.NodeHandler(
          self.mock_request,
          ('client_addr', 1234),
          mock_server,
          service=self.mock_node_service,
      )

  @mock.patch.object(service.protocol, 'optimize_socket', autospec=True)
  def test_setup(self, mock_optimize_socket):
    mock_optimize_socket.reset_mock()
    self.handler.setup()
    mock_optimize_socket.assert_called_once_with(self.mock_request)

  @mock.patch.object(service.protocol.TCPMessage, 'recv', autospec=True)
  @mock.patch.object(service.protocol.TCPMessage, 'send_json', autospec=True)
  def test_handle_get_manifest(self, mock_send_json, mock_recv):
    mock_recv.return_value = (service.protocol.OP_GET_MANIFEST, {'step': 1})
    self.mock_node_service.handle_get_manifest.return_value = [
        {'rel_path': 'foo'}
    ]

    service.NodeHandler.handle(self.handler)

    mock_recv.assert_called_once_with(self.mock_request)
    self.mock_node_service.handle_get_manifest.assert_called_once_with(
        {'step': 1}
    )
    mock_send_json.assert_called_once_with(
        self.mock_request,
        service.protocol.OP_RESPONSE_JSON,
        [{'rel_path': 'foo'}],
    )

  @mock.patch.object(service.protocol.TCPMessage, 'recv', autospec=True)
  def test_handle_download_file(self, mock_recv):
    mock_recv.return_value = (
        service.protocol.OP_DOWNLOAD_FILE,
        {'rel_path': 'foo'},
    )

    service.NodeHandler.handle(self.handler)

    mock_recv.assert_called_once_with(self.mock_request)
    self.mock_node_service.handle_download.assert_called_once_with(
        self.mock_request, {'rel_path': 'foo'}
    )

  @mock.patch.object(
      service.protocol.TCPMessage, 'recv', side_effect=ValueError, autospec=True
  )
  def test_handle_error(self, _):
    with self.assertLogs(level='ERROR') as log_output:
      service.NodeHandler.handle(self.handler)
      self.assertIn('P2P Handshake Error', log_output[0][0].message)


class P2PNodeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = epath.Path(self.create_tempdir().full_path)
    self.mock_server_cls = self.enter_context(
        mock.patch.object(service, '_ThreadingTCPServer', autospec=True)
    )
    self.mock_server = self.mock_server_cls.return_value
    self.mock_server.server_address = ('localhost', 12345)
    self.mock_process_index = self.enter_context(
        mock.patch.object(service.multihost, 'process_index', return_value=0)
    )
    self.mock_getaddrinfo = self.enter_context(
        mock.patch.object(
            service.socket,
            'getaddrinfo',
            return_value=[(service.socket.AF_INET, 0, 0, '', ('127.0.0.1', 0))],
        )
    )

    self.node = service.P2PNode(directory=self.temp_dir)

  def test_init_and_properties(self):
    self.assertEqual(self.node.ip, '127.0.0.1')
    self.assertEqual(self.node.port, 12345)
    self.assertEqual(self.node.directory, self.temp_dir)
    self.assertEqual(self.node.process_index, 0)
    self.mock_server_cls.assert_called_once()
    args, _ = self.mock_server_cls.call_args
    self.assertEqual(args[0], ('0.0.0.0', 0))
    self.assertIsInstance(args[1], functools.partial)
    self.assertEqual(args[1].func, service.NodeHandler)
    self.assertEqual(args[1].keywords, {'service': self.node})

  @mock.patch.object(threading, 'Thread', autospec=True)
  def test_ipv6_init_and_properties(self, mock_thread):
    self.mock_server_cls.reset_mock()
    self.mock_getaddrinfo.return_value = [(
        service.socket.AF_INET6,
        0,
        0,
        '',
        ('::1', 0),
    )]
    node = service.P2PNode(directory=self.temp_dir)
    self.assertEqual(node.ip, '::1')
    self.mock_server_cls.assert_called_once()
    args, _ = self.mock_server_cls.call_args
    self.assertEqual(args[0], ('::', 0))
    node.start()
    mock_thread.assert_called_once_with(
        target=node.server.serve_forever, daemon=True
    )
    mock_thread.return_value.start.assert_called_once()
    self.assertEqual(node._thread, mock_thread.return_value)

    node.stop()
    self.mock_server.shutdown.assert_called_once()
    self.mock_server.server_close.assert_called_once()
    mock_thread.return_value.join.assert_called_once()
    self.assertIsNone(node._thread)

  @mock.patch.object(threading, 'Thread', autospec=True)
  def test_start_stop(self, mock_thread):
    self.assertIsNone(self.node._thread)
    self.node.start()
    mock_thread.assert_called_once_with(
        target=self.mock_server.serve_forever, daemon=True
    )
    mock_thread.return_value.start.assert_called_once()
    self.assertEqual(self.node._thread, mock_thread.return_value)

    # check that start is idempotent
    self.node.start()
    mock_thread.assert_called_once()
    mock_thread.return_value.start.assert_called_once()

    self.node.stop()
    self.mock_server.shutdown.assert_called_once()
    self.mock_server.server_close.assert_called_once()
    mock_thread.return_value.join.assert_called_once()
    self.assertIsNone(self.node._thread)

    # check that stop is idempotent
    self.node.stop()
    self.mock_server.shutdown.assert_called_once()

  def test_handle_get_manifest_no_step(self):
    self.assertEqual(self.node.handle_get_manifest({}), [])

  def test_handle_get_manifest_empty(self):
    self.assertEqual(self.node.handle_get_manifest({'step': 1}), [])

  def test_handle_get_manifest_no_shard_dir(self):
    step_dir = self.temp_dir / '1'
    step_dir.mkdir()
    (step_dir / 'state').mkdir()
    self.assertEmpty(
        self.node.handle_get_manifest({'step': 1, 'process_index': 10})
    )

  def test_handle_get_manifest_success(self):
    step_dir = self.temp_dir / '1'
    shard_dir = step_dir / 'state' / 'ocdbt.process_10'
    shard_dir.mkdir(parents=True)
    (shard_dir / 'file1').write_text('foo')
    (shard_dir / 'subdir').mkdir()
    (shard_dir / 'subdir' / 'file2').write_text('bar_baz')

    manifest = self.node.handle_get_manifest({'step': 1, 'process_index': 10})
    self.assertLen(manifest, 2)
    expected_files = [
        {'rel_path': '1/state/ocdbt.process_10/file1', 'size': 3},
        {'rel_path': '1/state/ocdbt.process_10/subdir/file2', 'size': 7},
    ]
    self.assertCountEqual(manifest, expected_files)

    # stored process index 10 does not match requested 11
    manifest = self.node.handle_get_manifest({'step': 1, 'process_index': 11})
    self.assertEmpty(manifest)

  @mock.patch.object(service.protocol.TCPMessage, 'send_file', autospec=True)
  def test_handle_download_unsafe_path(self, mock_send_file):
    sock = mock.Mock()
    self.node.handle_download(sock, {'rel_path': '../unsafe'})
    mock_send_file.assert_called_once_with(sock, epath.Path('__INVALID__'))

    mock_send_file.reset_mock()
    self.node.handle_download(sock, {'rel_path': '/unsafe'})
    mock_send_file.assert_called_once_with(sock, epath.Path('__INVALID__'))

  @mock.patch.object(service.protocol.TCPMessage, 'send_file', autospec=True)
  def test_handle_download_missing_file(self, mock_send_file):
    sock = mock.Mock()
    self.node.handle_download(sock, {'rel_path': '1/missing'})
    mock_send_file.assert_called_once_with(sock, epath.Path('__MISSING__'))

  @mock.patch.object(service.protocol.TCPMessage, 'send_file', autospec=True)
  def test_handle_download_success(self, mock_send_file):
    sock = mock.Mock()
    step_dir = self.temp_dir / '1'
    step_dir.mkdir()
    (step_dir / 'file1').write_text('foo')
    self.node.handle_download(sock, {'rel_path': '1/file1'})
    mock_send_file.assert_called_once_with(sock, self.temp_dir / '1/file1')

  @mock.patch.object(service.shutil, 'rmtree', autospec=True)
  @mock.patch.object(service.shutil, 'move', autospec=True)
  @mock.patch.object(service.time, 'time', autospec=True)
  @mock.patch.object(service.protocol.TCPClient, 'request', autospec=True)
  @mock.patch.object(service.protocol.TCPClient, 'download', autospec=True)
  def test_fetch_shard_from_peer_no_manifest(
      self,
      unused_mock_download,
      mock_request,
      unused_mock_time,
      unused_mock_move,
      unused_mock_rmtree,
  ):
    mock_request.return_value = []
    self.assertFalse(self.node.fetch_shard_from_peer('peer', 123, 1, 10))
    mock_request.assert_called_once_with(
        'peer',
        123,
        service.protocol.OP_GET_MANIFEST,
        {'step': 1, 'process_index': 10},
    )

  @mock.patch.object(service.shutil, 'rmtree', autospec=True)
  @mock.patch.object(service.shutil, 'move', autospec=True)
  @mock.patch.object(service.time, 'time', autospec=True)
  @mock.patch.object(service.protocol.TCPClient, 'request', autospec=True)
  @mock.patch.object(service.protocol.TCPClient, 'download', autospec=True)
  def test_fetch_shard_from_peer_incomplete_download(
      self,
      mock_download,
      mock_request,
      mock_time,
      unused_mock_move,
      mock_rmtree,
  ):
    mock_request.return_value = [{'rel_path': '1/file1', 'size': 10}]
    mock_download.return_value = 0  # 0 bytes downloaded for file of size 10
    mock_time.return_value = 0
    self.assertFalse(self.node.fetch_shard_from_peer('peer', 123, 1, 10))
    mock_download.assert_called_once()
    stage_dir = self.temp_dir / 'stage_1_10'
    mock_rmtree.assert_called_with(str(stage_dir), ignore_errors=True)

  @mock.patch.object(service.shutil, 'rmtree', autospec=True)
  @mock.patch.object(service.shutil, 'move', autospec=True)
  @mock.patch.object(
      service.time,
      'time',
      side_effect=[100.0, 102.0, 103.0, 104.0],
      autospec=True,
  )
  @mock.patch.object(service.protocol.TCPClient, 'request', autospec=True)
  @mock.patch.object(service.protocol.TCPClient, 'download', autospec=True)
  def test_fetch_shard_from_peer_success(
      self,
      mock_download,
      mock_request,
      unused_mock_time,
      mock_move,
      mock_rmtree,
  ):
    manifest = [
        {'rel_path': '1/file1', 'size': 10},
        {'rel_path': '1/subdir/file2', 'size': 20},
    ]
    mock_request.return_value = manifest

    def download_side_effect(unused_ip, unused_port, rel_path, dest_path):
      dest_path.parent.mkdir(parents=True, exist_ok=True)
      if rel_path == '1/file1':
        dest_path.write_text('0123456789')
        return 10
      elif rel_path == '1/subdir/file2':
        dest_path.write_text('01234567890123456789')
        return 20
      return 0

    mock_download.side_effect = download_side_effect

    self.assertTrue(self.node.fetch_shard_from_peer('peer', 123, 1, 10))

    stage_dir = self.temp_dir / 'stage_1_10'
    mock_download.assert_has_calls(
        [
            mock.call('peer', 123, '1/file1', stage_dir / '1/file1'),
            mock.call(
                'peer', 123, '1/subdir/file2', stage_dir / '1/subdir/file2'
            ),
        ],
        any_order=True,
    )

    final_dir = self.temp_dir / service.constants.P2P_RESTORE_DIR_NAME / '1'
    mock_move.assert_called_once_with(str(stage_dir / '1'), str(final_dir))
    mock_rmtree.assert_called_with(str(stage_dir), ignore_errors=True)

  @mock.patch.object(service.shutil, 'rmtree', autospec=True)
  @mock.patch.object(service.shutil, 'move', autospec=True)
  @mock.patch.object(service.time, 'time', autospec=True)
  @mock.patch.object(service.protocol.TCPClient, 'request', autospec=True)
  @mock.patch.object(service.protocol.TCPClient, 'download', autospec=True)
  def test_fetch_shard_from_peer_exception_cleanup(
      self,
      mock_download,
      mock_request,
      unused_mock_time,
      unused_mock_move,
      mock_rmtree,
  ):
    """Tests that stage_dir is cleaned up if an exception occurs."""
    mock_request.return_value = [{'rel_path': '1/file1', 'size': 10}]
    mock_download.side_effect = OSError('Download failed')

    self.assertFalse(self.node.fetch_shard_from_peer('peer', 123, 1, 10))

    stage_dir = self.temp_dir / 'stage_1_10'
    mock_rmtree.assert_called_with(str(stage_dir), ignore_errors=True)


if __name__ == '__main__':
  absltest.main()
