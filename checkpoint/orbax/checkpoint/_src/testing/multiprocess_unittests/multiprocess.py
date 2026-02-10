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

"""Launches pytest tests across multiple simulated JAX processes.

This script sets up a distributed environment using JAX's distributed
variables and runs the specified pytest tests in multiple subprocesses,
each acting as a distinct JAX process.
"""

import contextlib
import os
import socket
import subprocess
import sys


def find_free_port():
  with contextlib.closing(
      socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  ) as s:
    s.bind(("", 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    return s.getsockname()[1]


def main():
  # Configuration
  num_processes = 2

  # 1. Get the script to run (e.g., "my_script.py")
  if len(sys.argv) < 2:
    print("Usage: python launch_multihost.py <script_to_run.py>")
    sys.exit(1)
  script_to_run = sys.argv[1]

  # 2. Setup Network
  coordinator_port = find_free_port()
  coordinator_address = f"localhost:{coordinator_port}"

  slicebuilder_ports = [
      find_free_port() for _ in range(num_processes)
  ]
  slicebuilder_addresses = ",".join(
      f"localhost:{port}" for port in slicebuilder_ports
  )

  print(f"🚀 Starting {num_processes} JAX processes...")
  print(f"📍 Coordinator: {coordinator_address}")
  print(f"📄 Running script: {script_to_run}")

  processes = []
  tpu_chips_per_process = 4
  for rank in range(num_processes):
    # Copy environment
    env = os.environ.copy()

    # JAX Distributed Setup
    env["JAX_COORDINATOR_ADDRESS"] = coordinator_address
    env["JAX_NUM_PROCESSES"] = str(num_processes)
    env["JAX_PROCESS_ID"] = str(rank)

    # Force CPU backend and simulate devices
    env["KERAS_BACKEND"] = "jax"
    # This makes JAX see 4 devices total (2 per process)
    # env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    # if rank == 0:
    #   env["TPU_VISIBLE_DEVICES"] = "0,1,2,3"
    # else:
    #   env["TPU_VISIBLE_DEVICES"] = "4,5,6,7"

    device_ids = range(
        rank * tpu_chips_per_process, (rank + 1) * tpu_chips_per_process
    )
    env["CLOUD_TPU_TASK_ID"] = str(rank)
    env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "2,2,1"
    env["TPU_PROCESS_BOUNDS"] = "1,2,1"
    env["TPU_PROCESS_ADDRESSES"] = slicebuilder_addresses
    env["TPU_PROCESS_PORT"] = str(slicebuilder_ports[rank])
    env["TPU_VISIBLE_CHIPS"] = ",".join(map(str, device_ids))
    env["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "1"

    # COMMAND: python my_script.py
    cmd = [sys.executable, script_to_run]

    # Launch
    p = subprocess.Popen(cmd, env=env)
    processes.append(p)

  # Wait for completion
  exit_codes = [p.wait() for p in processes]

  if any(c != 0 for c in exit_codes):
    print("\n❌ Some processes failed.")
    sys.exit(1)
  else:
    print("\n✅ All processes finished successfully.")


if __name__ == "__main__":
  main()
