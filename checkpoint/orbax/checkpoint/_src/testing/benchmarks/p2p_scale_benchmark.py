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

"""Benchmarks P2P checkpoint restore after host loss, at scale.

One run of `P2pScaleBenchmark`, on every process:

1. Saves step 0 to the local and the persistent tier.
2. Simulates a job restart in which the scenario's processes come back on new
   hosts: closes the manager and deletes their local checkpoint directory.
3. Creates a new manager and restores step 0.
4. Checks the restored values.

Which tier served the restore is in the P2P manager's log ("Restoration
finished using ..."): its own disk if the process kept its data, a peer if it
lost it, or persistent storage if some shard has no copy left anywhere.

Scenarios (`scenario`), as the set of processes that lose their local data:

* `all_local`: none.
* `lost_hosts`: processes of replica 1, picked by `lost_hosts_selection`
  (`one_host`, `replica_minus_one`, `spread`); every shard keeps a copy.
* `lost_replica_shard`: relative id 0 in every replica; that shard has no copy
  left, so every process falls back to persistent storage.
"""

from collections.abc import Iterator, Sequence
import contextlib
import dataclasses
import itertools
import json
import tempfile
from typing import Any
import zlib

from absl import logging
from etils import epath
import jax
from orbax.checkpoint import checkpoint_utils
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.testing.benchmarks.core import backends
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib
from orbax.checkpoint._src.testing.benchmarks.core import pytree_utils
from orbax.checkpoint._src.tree import utils as tree_utils
from orbax.checkpoint.experimental.emergency import path as emergency_path
from orbax.checkpoint.experimental.emergency.p2p import args as p2p_args
from orbax.checkpoint.experimental.emergency.p2p import checkpoint_manager as p2p_manager
from orbax.checkpoint.experimental.emergency.p2p import options as p2p_options

ALL_LOCAL = 'all_local'
LOST_HOSTS = 'lost_hosts'
LOST_REPLICA_SHARD = 'lost_replica_shard'
ONE_HOST = 'one_host'
REPLICA_MINUS_ONE = 'replica_minus_one'
SPREAD = 'spread'

_STEP = 0
_LOST_REPLICA = 1  # The replica `lost_hosts` takes processes from.
_KV_TIMEOUT_MS = 10 * 60 * 1000
_kv_exchanges = itertools.count()


# --- SimDevice: P2P's host exchanges through the jax.distributed KV store ---


def _kv_allgather(value: Any, name: str) -> list[Any]:
  """Every process's JSON `value`, in process order, via the KV store.

  Process 0 packs all values under one compressed key, so each process reads
  one key rather than one per process (at thousands of processes the
  coordination service otherwise cancels RPCs).

  Args:
    value: This process's JSON-serializable value.
    name: Names the exchange; the same on every process.

  Returns:
    Every process's value, in process order.
  """
  count = jax.process_count()
  if count == 1:
    return [value]
  client = multihost.get_jax_distributed_client()
  prefix = f'p2p_scale_kv/{next(_kv_exchanges)}_{name}'
  client.key_value_set(
      f'{prefix}/{multihost.process_index()}', json.dumps(value)
  )
  client.wait_at_barrier(prefix, _KV_TIMEOUT_MS)
  if multihost.process_index() == 0:
    by_index = {
        int(k.rsplit('/', 1)[1]): v for k, v in client.key_value_dir_get(prefix)
    }
    packed = '[' + ','.join(by_index[i] for i in range(count)) + ']'
    client.key_value_set_bytes(f'{prefix}_all', zlib.compress(packed.encode()))
  packed = client.blocking_key_value_get_bytes(f'{prefix}_all', _KV_TIMEOUT_MS)
  return json.loads(zlib.decompress(packed))


@contextlib.contextmanager
def _kv_host_exchanges() -> Iterator[None]:
  """On SimDevice, routes P2P's host-value exchanges through the KV store.

  SimDevice fabricates the result of device collectives, which P2P uses to
  share its peer registry (`sync_global_data`) and to agree on the persistent
  fallback (`global_max`).
  """
  if not backends.is_simdevice():
    yield
    return

  def sync_global_data(local_data: dict[str, Any]) -> list[dict[str, Any]]:
    return _kv_allgather(local_data, 'sync')

  def global_max(values: list[int]) -> list[int]:
    return [max(column) for column in zip(*_kv_allgather(values, 'max'))]

  originals = emergency_path.sync_global_data, multihost.global_max
  emergency_path.sync_global_data = sync_global_data
  multihost.global_max = global_max
  try:
    yield
  finally:
    emergency_path.sync_global_data, multihost.global_max = originals


# --- Options and scenarios ---


@dataclasses.dataclass(frozen=True)
class P2pScaleBenchmarkOptions(benchmarks_core.BenchmarkOptions):
  """Options for `P2pScaleBenchmark`.

  Attributes:
    scenario: `all_local`, `lost_hosts` or `lost_replica_shard`.
    lost_hosts_selection: For `lost_hosts`: `one_host` (the first process of
      replica 1), `replica_minus_one` (all of replica 1 but its last process) or
      `spread` (`lost_hosts_fraction` of all processes, spread over the replicas
      so that every shard keeps a copy).
    lost_hosts_fraction: The fraction of processes `spread` takes.
    replica_axis_index: The mesh axis that indexes replicas.
    use_tmpdir_for_local: Without `--local_directory`, keep the local tier under
      `$TMPDIR` (on Borg, the task's RAM file system) instead of under the
      output directory.
  """

  scenario: str | Sequence[str] = LOST_HOSTS
  lost_hosts_selection: str | Sequence[str] = ONE_HOST
  lost_hosts_fraction: float | Sequence[float] = 0.25
  replica_axis_index: int | Sequence[int] = 0
  use_tmpdir_for_local: bool | Sequence[bool] = False

  def is_valid(self) -> bool:
    return self.scenario in (ALL_LOCAL, LOST_HOSTS, LOST_REPLICA_SHARD) and (
        self.lost_hosts_selection in (ONE_HOST, REPLICA_MINUS_ONE, SPREAD)
    )


def replica_topology(
    mesh: jax.sharding.Mesh, replica_axis_index: int
) -> list[list[int]]:
  """`topology[replica][relative_id]` is a process index, as P2P numbers them."""
  topology = []
  for replica in range(
      multislice.replica_count(mesh, replica_axis_index=replica_axis_index)
  ):
    devices = multislice.replica_devices(
        mesh, replica_axis_index=replica_axis_index, replica_id=replica
    )
    processes = []
    for device in devices.flatten():
      if device.process_index not in processes:
        processes.append(device.process_index)
    topology.append(processes)
  return topology


def lost_processes(
    topology: list[list[int]], options: P2pScaleBenchmarkOptions
) -> set[int]:
  """The processes that lose their local data; the same on every process."""
  replicas, per_replica = len(topology), len(topology[0])
  processes = list(itertools.chain.from_iterable(topology))
  if len(set(processes)) != len(processes) or any(
      len(r) != per_replica for r in topology
  ):
    raise ValueError(
        'P2P needs equal replicas and each process in exactly one replica (put'
        f' the replica axis across processes); got {topology}.'
    )
  if options.scenario == ALL_LOCAL:
    return set()
  if options.scenario == LOST_REPLICA_SHARD:
    return {replica[0] for replica in topology}
  if replicas < 2:
    raise ValueError('lost_hosts needs at least two replicas.')
  if options.lost_hosts_selection == ONE_HOST:
    return {topology[_LOST_REPLICA][0]}
  if options.lost_hosts_selection == REPLICA_MINUS_ONE:
    return set(topology[_LOST_REPLICA][:-1])
  # SPREAD: walk the relative ids, moving to the next replica on every lap, so
  # no relative id is lost in every replica.
  count = max(1, round(options.lost_hosts_fraction * len(processes)))  # pyrefly: ignore[unsupported-operation]
  if count > per_replica * (replicas - 1):
    raise ValueError(f'spread cannot lose {count} processes and keep a copy.')
  return {
      topology[(_LOST_REPLICA + k % per_replica + k // per_replica) % replicas][
          k % per_replica
      ]
      for k in range(count)
  }


def _directory_bytes(directory: epath.Path) -> int:
  total = 0
  if directory.exists():
    for root, _, names in directory.walk():
      total += sum((root / name).stat().length for name in names)
  return total


def _record(metrics: metric_lib.Metrics, key: str, value: Any, unit: str):
  metrics.results[f'p2p_restore::p2p/{key}'] = (value, unit)


@benchmarks_core.benchmark_options(P2pScaleBenchmarkOptions)
class P2pScaleBenchmark(benchmarks_core.BenchmarksGenerator):
  """Saves, simulates host loss, restores and validates on every process."""

  def _process_local_directory(
      self,
      context: benchmarks_core.TestContext,
      options: P2pScaleBenchmarkOptions,
  ) -> epath.Path:
    """Returns this process's local tier directory."""
    if context.local_path is not None:
      base = epath.Path(context.local_path)
    elif options.use_tmpdir_for_local:
      base = epath.Path(tempfile.gettempdir()) / 'orbax_p2p_scale'
      base /= f'{context.name}_{context.repeat_index}'
    else:
      base = context.path / 'local'
    return base / f'process_{multihost.process_index()}'

  def _create_manager(
      self,
      local_directory: epath.Path,
      persistent_directory: epath.Path,
      mesh: jax.sharding.Mesh,
      abstract_pytree: Any,
      options: P2pScaleBenchmarkOptions,
  ) -> p2p_manager.CheckpointManager:
    """Creates a P2P manager that keeps one step in each tier."""
    every_step = dict(save_interval_steps=1, max_to_keep=1)
    return p2p_manager.CheckpointManager(
        local_directory=local_directory,
        persistent_directory=persistent_directory,
        global_mesh=mesh,
        abstract_state=abstract_pytree,
        options=p2p_options.CheckpointManagerOptions(
            local=p2p_options.LocalCheckpointOptions(**every_step),
            persistent=p2p_options.PersistentCheckpointOptions(**every_step),
            replica_axis_index=options.replica_axis_index,  # pyrefly: ignore[bad-argument-type]
        ),
    )

  def test_fn(
      self, context: benchmarks_core.TestContext
  ) -> benchmarks_core.TestResult:
    options = context.options
    assert isinstance(options, P2pScaleBenchmarkOptions)
    mesh, pytree = context.mesh, context.pytree
    if mesh is None:
      raise ValueError('P2pScaleBenchmark needs a mesh.')
    if not multihost.is_runtime_to_distributed_ids_initialized():
      multihost.initialize_runtime_to_distributed_ids()
    if not multihost.is_distributed_to_device_ids_initialized():
      multihost.initialize_distributed_to_device_ids()

    metrics = metric_lib.Metrics()
    process = multihost.process_index()
    topology = replica_topology(mesh, options.replica_axis_index)  # pyrefly: ignore[bad-argument-type]
    is_lost = process in lost_processes(topology, options)
    abstract_pytree = jax.tree.map(tree_utils.to_shape_dtype_struct, pytree)
    restore_args = p2p_args.Composite(
        state=pytree_checkpoint_handler.PyTreeRestoreArgs(
            restore_args=checkpoint_utils.construct_restore_args(
                abstract_pytree
            ),
            item=abstract_pytree,
        )
    )
    local_directory = self._process_local_directory(context, options)
    persistent_directory = context.path / 'persistent'

    def new_manager():
      return self._create_manager(
          local_directory, persistent_directory, mesh, abstract_pytree, options
      )

    if process == 0:
      persistent_directory.rmtree(missing_ok=True)
      persistent_directory.mkdir(parents=True)
    local_directory.rmtree(missing_ok=True)
    local_directory.mkdir(parents=True)
    multihost.sync_global_processes('p2p_scale:directories')
    try:
      with _kv_host_exchanges():
        # 1. Save step 0 to both tiers.
        manager = new_manager()
        save_args = p2p_args.Composite(
            state=pytree_checkpoint_handler.PyTreeSaveArgs(pytree)
        )
        with metrics.measure('save_blocking'):
          manager.save(_STEP, args=save_args)
        with metrics.measure('save_wait_local'):
          manager._local_manager.wait_until_finished()  # pylint: disable=protected-access
        with metrics.measure('save_wait_persistent'):
          manager.wait_until_finished()
        manager.close()
        step_bytes = _directory_bytes(local_directory / str(_STEP))

        # 2. Host replacement: a new host starts with an empty local disk.
        if is_lost:
          local_directory.rmtree()
          local_directory.mkdir(parents=True)
        multihost.sync_global_processes('p2p_scale:hosts_replaced')

        # 3. Restart (peer discovery, background fetch) and restore.
        with metrics.measure('restart_and_restore'):
          with metrics.measure('restart'):
            manager = new_manager()
          with metrics.measure('restore'):
            restored = manager.restore(_STEP, args=restore_args)['state']
        manager.close()
    finally:
      if not str(local_directory).startswith(str(context.path)):
        local_directory.rmtree(missing_ok=True)  # Don't fill the RAM disk.

    # 4. Check and record.
    with metrics.measure('validate'):
      pytree_utils.assert_pytree_equal(pytree, restored)  # Shapes on SimDevice.
    _record(metrics, 'lost_host', int(is_lost), '')
    _record(metrics, 'step_bytes', step_bytes, 'bytes')
    logging.info('Process %d: lost its local step=%s.', process, is_lost)
    return benchmarks_core.TestResult(metrics=metrics)
