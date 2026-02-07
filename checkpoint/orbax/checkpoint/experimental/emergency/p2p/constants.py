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

"""Shared constants for P2P checkpointing."""

# Directory names
P2P_RESTORE_DIR_NAME = 'p2p_restore'
STATE_SUBDIR = 'state'
PROCESS_SUBDIR_PREFIX = 'ocdbt.process_'

# Tuning for high-throughput networks (16MB buffers)
SOCKET_BUFFER_SIZE = 16 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024

# Timeouts
CONNECT_TIMEOUT_SECONDS = 5
TRANSFER_TIMEOUT_SECONDS = 60
