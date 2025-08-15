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

"""Synchronization utilities."""

import threading
from typing import Sequence
from absl import logging
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.futures import synchronization
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.synchronization import signaling_client

HandlerAwaitableSignal = synchronization.HandlerAwaitableSignal
OperationIdGenerator = synchronization.OperationIdGenerator
_get_unique_barrier_key = (
    future.AwaitableSignalsContract.get_unique_awaitable_singal_key
)


async def _get_awaitable_signals_from_contract(
    client: signaling_client.SignalingClient,
    operation_id: str,
) -> Sequence[HandlerAwaitableSignal]:
  """Gets the awaitable signals for the current operation id."""
  barrier_key = _get_unique_barrier_key(
      HandlerAwaitableSignal.AWAITABLE_SIGNALS_CONTRACT,
      operation_id,
  )
  values_str = await client.key_value_try_get(barrier_key)
  if values_str is None:
    # If the key is not found, then there are no awaitable signals.
    return []
  return [HandlerAwaitableSignal(value) for value in values_str.split(',')]


async def await_contracted_signals(operation_id: str):
  """Waits for the contracted signals to be set.

  This function may be called from a background thread, but it assumes that all
  signals that may be awaited must have previously been added to the contract.
  In other words, signals must be added to the contract synchronously, before
  calling this function.

  Args:
    operation_id: The operation id to use for the barrier key. This should be
      obtained on the main thread, synchronously.
  """
  client = signaling_client.get_signaling_client()

  timeout_secs = multihost.coordination_timeout()
  receive_signals = await _get_awaitable_signals_from_contract(
      client, operation_id
  )
  for signal in receive_signals:
    logging.vlog(
        1,
        '[process=%d][thread=%s] Waiting for <%s> timeout: %d secs to be set',
        multihost.process_index(),
        threading.current_thread().name,
        signal.value,
        timeout_secs,
    )
    barrier_key = _get_unique_barrier_key(signal, operation_id)
    await client.blocking_key_value_get(barrier_key, timeout_secs)
