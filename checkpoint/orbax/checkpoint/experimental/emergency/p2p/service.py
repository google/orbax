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

"""Defines the internal P2P Node service for serving checkpoint shards."""

import concurrent.futures
import shutil
import socket
import socketserver
import threading
import time
from typing import Any, final

from absl import logging
from etils import epath
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint.experimental.emergency.p2p import constants
from orbax.checkpoint.experimental.emergency.p2p import protocol


class _ThreadingTCPServer(socketserver.ThreadingTCPServer):
  """A ThreadingTCPServer that holds a reference to a P2PNode."""

  service: 'P2PNode'


@final
class NodeHandler(socketserver.BaseRequestHandler):
  """Handles incoming Data Plane requests."""

  server: _ThreadingTCPServer

  def setup(self):
    protocol.optimize_socket(self.request)

  def handle(self):
    try:
      opcode, payload = protocol.TCPMessage.recv(self.request)

      if opcode == protocol.OP_GET_MANIFEST:
        resp = self.server.service.handle_get_manifest(payload)
        protocol.TCPMessage.send_json(
            self.request, protocol.OP_RESPONSE_JSON, resp
        )

      elif opcode == protocol.OP_DOWNLOAD_FILE:
        self.server.service.handle_download(self.request, payload)

    except (OSError, ValueError) as e:
      logging.error('P2P Handshake Error [%s]: %s', self.client_address, e)


@final
class P2PNode:
  """Internal sidecar service for managing local data and peer transfers.

  This node runs a TCP server that can serve checkpoint files to other peers
  on-demand. It can also act as a client to fetch checkpoint files from other
  peers.

  The P2P node can be used to restore a checkpoint from a peer that has the
  checkpoint saved locally, in case the checkpoint is not available in CNS.
  """

  def __init__(
      self,
      directory: epath.PathLike,
  ):
    """Initializes P2PNode.

    Args:
      directory: The directory containing checkpoint shards.
    """
    self.directory = epath.Path(directory)
    self.process_index = multihost.process_index()

    self.server = _ThreadingTCPServer(('0.0.0.0', 0), NodeHandler)
    self.server.service = self

    self.ip = socket.getaddrinfo(socket.gethostname(), None)[0][4][0]
    # Capture the actual port assigned by the OS
    self.port = self.server.server_address[1]

    self._thread: threading.Thread | None = None

  def start(self):
    """Starts the P2P server in a background thread."""
    if self._thread is not None:
      return
    self._thread = threading.Thread(
        target=self.server.serve_forever, daemon=True
    )
    self._thread.start()
    logging.info(
        'P2P node %d listening on %s:%d',
        self.process_index,
        self.ip,
        self.port,
    )

  def stop(self):
    """Stops the P2P server."""
    if not self._thread:
      return
    logging.info('Stopping P2P node %d...', self.process_index)
    self.server.shutdown()
    self.server.server_close()
    self._thread.join(timeout=2.0)
    self._thread = None

  def _get_stored_process_index(self, step_path: epath.Path) -> int | None:
    """Returns the process index of the shard stored in the given step path."""
    item_path = step_path / constants.STATE_SUBDIR
    if item_path.exists():
      for path in item_path.glob(f'{constants.PROCESS_SUBDIR_PREFIX}*'):
        if path.is_dir():
          # Format: ocdbt.process_0, ocdbt.process_12, etc.
          return int(path.name.split('_')[-1])
    return None

  def handle_get_manifest(
      self, payload: dict[str, Any]
  ) -> list[dict[str, Any]]:
    """Handles GET_MANIFEST request.

    Args:
      payload: The request payload, containing step and process_index.

    Returns:
      A list of file metadata dicts, containing rel_path and size.
    """
    step = payload.get('step')
    req_process_index = payload.get('process_index')
    if step is None or req_process_index is  None:
      return []

    step_dir = self.directory / str(step)
    if not step_dir.exists():
      return []

    stored_process_index = self._get_stored_process_index(step_dir)

    # If process_index is specified, only return manifest if it matches.
    if req_process_index != stored_process_index:
      return []

    files = []
    for root, _, filenames in step_dir.walk():
      for filename in filenames:
        abs_path = root / filename
        rel_path = abs_path.relative_to(self.directory)
        files.append({
            'rel_path': str(rel_path),
            'size': abs_path.stat().length,
        })
    return files

  def handle_download(self, sock, payload: dict[str, Any]):
    """Handles DOWNLOAD_FILE request.

    Sends the requested file to the client if it exists and is safe to send.

    Args:
      sock: The socket to send the file to.
      payload: The request payload, containing rel_path of file to download.
    """
    rel_path_str = payload.get('rel_path')

    if not rel_path_str or '..' in rel_path_str or rel_path_str.startswith('/'):
      logging.warning('Blocked unsafe P2P path request: %s', rel_path_str)
      protocol.TCPMessage.send_file(sock, epath.Path('__INVALID__'))
      return

    full_path = self.directory / rel_path_str
    if full_path.exists() and full_path.is_file():
      protocol.TCPMessage.send_file(sock, full_path)
    else:
      logging.warning('Requested file not found: %s', full_path)
      protocol.TCPMessage.send_file(sock, epath.Path('__MISSING__'))

  def fetch_shard_from_peer(
      self, ip: str, port: int, step: int, stored_process_index: int
  ) -> bool:
    """Fetches checkpoint shard from a peer.

    Args:
      ip: The IP address of the peer to fetch from.
      port: The port of the peer to fetch from.
      step: The checkpoint step to fetch.
      stored_process_index: The process index of the shard to fetch. This is
        the process index whose checkpoint data is stored on the peer and is
        being requested.

    Returns:
      True if the shard was fetched successfully, False otherwise.
    """
    logging.info('Requesting manifest from %s:%d for step %d', ip, port, step)

    manifest = protocol.TCPClient.request(
        ip,
        port,
        protocol.OP_GET_MANIFEST,
        {'step': step, 'process_index': stored_process_index},
    )

    if not manifest:
      logging.warning('Failed to get manifest from %s', ip)
      return False

    # TODO(exlin): Remove this directory once the transfer is globally completed
    # to save memory space.
    stage_dir = self.directory / f'stage_{step}_{stored_process_index}'
    if stage_dir.exists():
      shutil.rmtree(str(stage_dir))
    stage_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    total_bytes = 0

    try:
      with concurrent.futures.ThreadPoolExecutor(max_workers=8) as exc:
        futures = []
        for f_meta in manifest:
          rel_path_str = f_meta['rel_path']
          dest_path = stage_dir / rel_path_str
          futures.append(
              exc.submit(
                  protocol.TCPClient.download, ip, port, rel_path_str, dest_path
              )
          )

        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        if any(r == 0 and m['size'] > 0 for r, m in zip(results, manifest)):
          logging.error('Incomplete download from peer. Aborting.')
          return False

        total_bytes = sum(results)

      final_dir = self.directory / constants.P2P_RESTORE_DIR_NAME / str(step)
      if final_dir.exists():
        shutil.rmtree(str(final_dir))
      final_dir.parent.mkdir(parents=True, exist_ok=True)

      source_step_dir = stage_dir / str(step)
      if source_step_dir.exists():
        shutil.move(str(source_step_dir), str(final_dir))
      else:
        shutil.move(str(stage_dir), str(final_dir))

      duration = time.time() - start_time
      bw_mbps = (total_bytes / 1024 / 1024) / duration if duration > 0 else 0.0

      logging.info(
          'P2P transfer complete: %.2f MB in %.2fs (%.2f MB/s) | Step: %d',
          total_bytes / 1024 / 1024,
          duration,
          bw_mbps,
          step,
      )
      return True

    except (OSError, ValueError) as e:
      logging.exception('P2P restore failed: %s', e)
      return False
    finally:
      if stage_dir.exists():
        shutil.rmtree(str(stage_dir), ignore_errors=True)
