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

"""Handles peer discovery logic, shard selection, and topology visualization."""

import collections
import dataclasses
from typing import Any, final

from absl import logging
import jax
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint.experimental.emergency.p2p import protocol
from orbax.checkpoint.experimental.emergency.p2p import registry


@final
class PeerSelector:
  """Manages the global P2P registry and selects peers for shard retrieval."""

  def __init__(
      self,
      global_mesh: jax.sharding.Mesh,
      replica_axis_index: int,
      raw_metadata_list: list[dict[str, Any]],
  ):
    self._global_mesh = global_mesh
    self._replica_axis_index = replica_axis_index
    self._registry = registry.PeerRegistry()
    self._process_map = self._build_topology_map()
    for data in raw_metadata_list:
      try:
        peer = protocol.PeerDiscoveryInfo.from_dict(data)

        # Enrich peer data with topology info if available locally
        if peer.process_index in self._process_map:
          meta = self._process_map[peer.process_index]
          peer = dataclasses.replace(
              peer,
              replica_id=meta["replica"],
              local_process_index=meta["relative"],
          )
        self._registry.register_peer(peer)
      except (KeyError, ValueError) as e:
        logging.warning("Failed to parse peer metadata: %s", e)

  def _build_topology_map(self) -> dict[int, dict[str, int]]:
    """Maps Process Index -> {replica_id, relative_shard_id}."""
    mapping = {}
    num_replicas = multislice.replica_count(
        self._global_mesh, replica_axis_index=self._replica_axis_index
    )

    for rid in range(num_replicas):
      replica_devices = multislice.replica_devices(
          self._global_mesh,
          replica_axis_index=self._replica_axis_index,
          replica_id=rid,
      )

      # Flatten in case of multi-dimensional local meshes
      flat_devices = replica_devices.flatten()

      # Extract unique process indices for this replica
      seen_set = set()
      seen_procs = []
      for d in flat_devices:
        if d.process_index not in seen_set:
          seen_procs.append(d.process_index)
          seen_set.add(d.process_index)

      # Assign relative IDs based on sorted order within the replica
      for relative_id, pid in enumerate(seen_procs):
        mapping[pid] = {"replica": rid, "relative": relative_id}

    return mapping

  def get_latest_complete_step(self) -> int | None:
    """Finds the highest step where at least one full copy exists."""
    if not self._process_map:
      return None

    # We need every 'relative_id' (0 to N) to be present in the cluster
    required_relative_ids = set(
        m["relative"] for m in self._process_map.values()
    )
    complete_steps = []

    for step in self._registry.iter_steps():
      shard_map = self._registry.get_shard_map(step)

      # Determine which logical shards are available across ALL peers
      available_relative_ids = set()
      for pid in shard_map.keys():
        if pid in self._process_map:
          available_relative_ids.add(self._process_map[pid]["relative"])

      if required_relative_ids.issubset(available_relative_ids):
        complete_steps.append(step)

    return max(complete_steps) if complete_steps else None

  def get_source_peer(
      self, step: int, target_process_index: int
  ) -> protocol.PeerDiscoveryInfo | None:
    """Finds a peer holding the data, preferring Local Replica > Deterministic LB."""
    if not self._registry.has_step(step):
      return None

    # 1. Try Direct Match (Fastest/Simplest)
    direct_peer = self._registry.get_peer(step, target_process_index)
    if direct_peer:
      return direct_peer

    # 2. Topology Match (Find a different process holding the same data)
    target_meta = self._process_map.get(target_process_index)
    if not target_meta:
      return None

    target_relative_id = target_meta["relative"]
    target_replica = target_meta["replica"]

    shard_map = self._registry.get_shard_map(step)
    candidates = []

    for pid, peer in shard_map.items():
      cand_meta = self._process_map.get(pid)
      if not cand_meta:
        continue

      # If it's the same logical shard (relative_id),
      # but a different physical process
      if cand_meta["relative"] == target_relative_id:
        candidates.append(peer)

    if not candidates:
      return None

    # Deterministic Selection for Load Balancing
    # Sort by replica_id to ensure every node sees the same list order
    candidates.sort(key=lambda p: p.replica_id)

    # Consumer from Replica N prefers Provider from Replica N (or N mod M)
    idx = target_replica % len(candidates)
    return candidates[idx]

  def visualize_topology(self, step: int) -> str:
    """Renders a clean, tabular visualization of the P2P mesh state."""

    def print_table(headers: list[str], rows: list[list[str]]) -> list[str]:
      if not rows:
        return []
      col_widths = [len(h) for h in headers]
      for row in rows:
        for i, cell in enumerate(row):
          col_widths[i] = max(col_widths[i], len(str(cell)))

      fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])
      separator = "-+-".join(["-" * w for w in col_widths])

      out = []
      out.append(fmt.format(*headers))
      out.append(separator)
      for row in rows:
        out.append(fmt.format(*row))
      return out

    output = []
    output.append(f"P2P TOPOLOGY VISUALIZATION (Step: {step})")
    output.append("=" * 60)

    # 1. Topology Map
    output.append("\n[1] Physical Topology Map")
    topo_rows = []
    for pid, meta in sorted(self._process_map.items()):
      topo_rows.append([str(pid), str(meta["replica"]), str(meta["relative"])])
    output.extend(
        print_table(
            ["Process ID", "Replica Group", "Shard ID (Relative)"], topo_rows
        )
    )

    # 2. Availability Matrix
    output.append("\n[2] Data Availability Matrix")
    replicas = collections.defaultdict(dict)
    max_rel = 0
    if self._process_map:
      max_rel = max(m["relative"] for m in self._process_map.values())

    for pid, meta in self._process_map.items():
      replicas[meta["replica"]][meta["relative"]] = pid

    matrix_header = ["Replica"] + [f"Shard {i}" for i in range(max_rel + 1)]
    matrix_rows = []

    for rid in sorted(replicas.keys()):
      row = [f"Group {rid}"]
      for rel in range(max_rel + 1):
        pid = replicas[rid].get(rel)
        if pid is not None:
          candidates = self._registry.get_peer(step, pid)
          if candidates:
            row.append(f"[OK] P{pid}")
          else:
            row.append(f"[--] P{pid}")
        else:
          row.append("    ")
      matrix_rows.append(row)

    output.extend(print_table(matrix_header, matrix_rows))

    # 3. Fetch Plan
    output.append("\n[3] Recovery Plan")
    plan_rows = []

    for pid in sorted(self._process_map.keys()):
      if not self._registry.get_peer(step, pid):
        meta = self._process_map[pid]
        source = self.get_source_peer(step, pid)

        consumer = f"P{pid} (Rep {meta['replica']})"
        if source:
          src_pid = source.process_index
          src_meta = self._process_map.get(src_pid, {"replica": "?"})
          ip_short = source.ip.split(".")[-1]

          plan_rows.append([
              consumer,
              "FETCH FROM ->",
              f"P{src_pid} (Rep {src_meta['replica']}) @ ..{ip_short}",
              "READY",
          ])
        else:
          plan_rows.append(
              [consumer, "CANNOT FETCH", "NO PEER FOUND", "CRITICAL"]
          )

    if not plan_rows:
      output.append("No active transfers required. All data present locally.")
    else:
      output.extend(
          print_table(
              ["Consumer Node", "Action", "Source Node", "Status"], plan_rows
          )
      )

    output.append("-" * 60)
    return "\n".join(output)
