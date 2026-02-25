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

"""Defines the internal wire protocol for P2P checkpoint transfer."""

import dataclasses
import json
import socket
import struct
from typing import Any, Final
from absl import logging
from etils import epath
from orbax.checkpoint.experimental.emergency.p2p import constants
from typing_extensions import Self

_HEADER_STRUCT = struct.Struct('!B I')
_HEADER_SIZE = _HEADER_STRUCT.size  # pylint: disable=invalid-name

OP_ERROR: Final = 0
OP_GET_MANIFEST: Final = 1
OP_DOWNLOAD_FILE: Final = 2
OP_RESPONSE_JSON: Final = 3
OP_FILE_STREAM: Final = 4


@dataclasses.dataclass(frozen=True)
class PeerDiscoveryInfo:
  """Immutable value object representing a peer's state."""

  ip: str
  port: int
  process_index: int
  # Checkpoint steps discovered to be present in this peer's local storage
  # stored by a process identified by process_index field.
  steps: list[int] = dataclasses.field(default_factory=list)
  replica_id: int = -1
  local_process_index: int = -1

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> Self:
    return cls(
        ip=data['ip'],
        port=data['port'],
        process_index=data['process_index'],
        steps=data.get('steps', []),
    )


def optimize_socket(sock: socket.socket) -> None:
  try:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(
        socket.SOL_SOCKET, socket.SO_RCVBUF, constants.SOCKET_BUFFER_SIZE
    )
    sock.setsockopt(
        socket.SOL_SOCKET, socket.SO_SNDBUF, constants.SOCKET_BUFFER_SIZE
    )
  except OSError as e:
    logging.error('Failed to optimize socket: %s', e)


class TCPMessage:
  """Framing utility."""

  @staticmethod
  def send_json(sock: socket.socket, opcode: int, data: Any = None) -> None:
    """Sends a JSON-serialized message over the socket.

    Args:
      sock: The socket to send the message over.
      opcode: The message opcode.
      data: The JSON-serializable data to send.
    """
    payload = json.dumps(data).encode('utf-8') if data else b''
    header = _HEADER_STRUCT.pack(opcode, len(payload))
    sock.sendall(header + payload)

  @staticmethod
  def send_file(sock: socket.socket, filepath: epath.Path) -> None:
    """Sends a file over the socket.

    If the file exists, it sends an OP_FILE_STREAM message with the file size,
    followed by the file content. If the file does not exist, it sends
    OP_FILE_STREAM with size 0.

    Args:
      sock: The socket to send the file over.
      filepath: The path to the file to send.
    """
    try:
      filesize = filepath.stat().length
    except OSError as e:
      logging.error('Failed to stat file %s, sending size 0: %s', filepath, e)
      header = _HEADER_STRUCT.pack(OP_FILE_STREAM, 8)
      sock.sendall(header + struct.pack('!Q', 0))
      return

    header = _HEADER_STRUCT.pack(OP_FILE_STREAM, 8)
    size_payload = struct.pack('!Q', filesize)
    sock.sendall(header + size_payload)

    with filepath.open('rb') as f:
      try:
        sock.sendfile(f)
      except (BrokenPipeError, ConnectionResetError) as e:
        logging.error(
            'Connection closed while sending file %s: %s', filepath, e
        )

  @staticmethod
  def recv(sock: socket.socket) -> tuple[int, Any]:
    """Receives a message from the socket.

    Reads the message header to determine the opcode and payload length.
    If opcode is OP_FILE_STREAM, it reads and returns the file size.
    Otherwise, it reads the JSON payload and returns it.

    Args:
      sock: The socket to receive message from.

    Returns:
      A tuple of (opcode, data). If an error occurs, (OP_ERROR, None) is
      returned.
    """
    try:
      peer = str(sock.getpeername())
    except OSError:
      peer = 'unknown'
    header_data = TCPMessage._recv_exact(sock, _HEADER_SIZE)
    if not header_data:
      logging.error('Failed to receive header from peer=%s', peer)
      return OP_ERROR, None
    try:
      opcode, length = _HEADER_STRUCT.unpack(header_data)
    except struct.error as e:
      logging.error('Failed to unpack header from peer=%s: %s', peer, e)
      return OP_ERROR, None

    if opcode == OP_FILE_STREAM:
      size_data = TCPMessage._recv_exact(sock, 8)
      if not size_data:
        logging.error('Failed to receive filesize from peer=%s', peer)
        return OP_ERROR, None
      try:
        return opcode, struct.unpack('!Q', size_data)[0]
      except struct.error as e:
        logging.error('Failed to unpack filesize from peer=%s: %s', peer, e)
        return OP_ERROR, None

    payload_data = TCPMessage._recv_exact(sock, length)
    if not payload_data:
      logging.error(
          'Failed to receive payload of length %d from peer=%s for opcode=%d',
          length,
          peer,
          opcode,
      )
      return OP_ERROR, None

    if length == 0:
      return opcode, None

    try:
      return opcode, json.loads(payload_data.decode('utf-8'))
    except json.JSONDecodeError as e:
      logging.error(
          'Failed to decode JSON payload from peer=%s for opcode=%d: %s',
          peer,
          opcode,
          e,
      )
      return OP_ERROR, None

  @staticmethod
  def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Reads exactly n bytes from the socket.

    Args:
      sock: The socket to read from.
      n: The number of bytes to read.

    Returns:
      The read bytes, or None if the connection is closed before n bytes are
      received.
    """
    data = bytearray(n)
    view = memoryview(data)
    pos = 0
    while pos < n:
      read = sock.recv_into(view[pos:], n - pos)
      if read == 0:
        return None
      pos += read
    return bytes(data)


class TCPClient:
  """TCP client for P2P communication."""

  @staticmethod
  def request(host: str, port: int, opcode: int, payload: Any = None) -> Any:
    """Sends a request to the given host and port and returns the response.

    Args:
      host: The host to connect to.
      port: The port to connect to.
      opcode: The message opcode.
      payload: The JSON-serializable payload to send.

    Returns:
      The JSON response from the server, or None if an error occurs.
    """
    try:
      with socket.create_connection(
          (host, port), timeout=constants.CONNECT_TIMEOUT_SECONDS
      ) as sock:
        optimize_socket(sock)
        TCPMessage.send_json(sock, opcode, payload)
        resp_op, resp_data = TCPMessage.recv(sock)
        if resp_op == OP_RESPONSE_JSON:
          return resp_data
        else:
          logging.error(
              'Received unexpected opcode %d from %s:%d in response to'
              ' opcode=%d',
              resp_op,
              host,
              port,
              opcode,
          )
          return None
    except OSError as e:
      logging.error(
          'Failed to connect to or communicate with %s:%d for opcode=%d: %s',
          host,
          port,
          opcode,
          e,
      )
      return None

  @staticmethod
  def download(
      host: str, port: int, rel_path: str, dest_path: epath.Path
  ) -> int:
    """Downloads a file from the given host and port.

    Args:
      host: The host to connect to.
      port: The port to connect to.
      rel_path: The relative path of the file to download.
      dest_path: The destination path to save the file to.

    Returns:
      The number of bytes downloaded, or 0 if an error occurs.
    """
    try:
      with socket.create_connection(
          (host, port), timeout=constants.TRANSFER_TIMEOUT_SECONDS
      ) as sock:
        optimize_socket(sock)
        TCPMessage.send_json(sock, OP_DOWNLOAD_FILE, {'rel_path': rel_path})
        opcode, filesize = TCPMessage.recv(sock)

        if opcode != OP_FILE_STREAM:
          logging.error(
              'Received unexpected opcode %d from %s:%d when downloading %s',
              opcode,
              host,
              port,
              rel_path,
          )
          return 0
        if filesize == 0:
          logging.error(
              'Peer %s:%d reported filesize=0 for %s. This may indicate the'
              ' file is missing or invalid on the peer.',
              host,
              port,
              rel_path,
          )
          return 0

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with dest_path.open('wb') as f:
          received = 0
          buf = bytearray(constants.CHUNK_SIZE)
          view = memoryview(buf)
          while received < filesize:
            to_read = min(constants.CHUNK_SIZE, filesize - received)
            nbytes = sock.recv_into(view[:to_read])
            if nbytes == 0:
              logging.error(
                  'Connection closed prematurely by peer %s:%d while'
                  ' downloading %s. Received %d/%d bytes.',
                  host,
                  port,
                  rel_path,
                  received,
                  filesize,
              )
              break
            f.write(view[:nbytes])
            received += nbytes

        if received != filesize:
          logging.error(
              'Failed to download %s from %s:%d. Expected %d bytes, received'
              ' %d.',
              rel_path,
              host,
              port,
              filesize,
              received,
          )
          return 0
        return received
    except OSError as e:
      logging.error(
          'Failed to download %s from %s:%d: %s', rel_path, host, port, e
      )
      return 0
