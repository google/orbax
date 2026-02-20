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

"""Launches and bootstraps tests across multiple simulated JAX processes."""

import argparse
import contextlib
import os
import runpy
import socket
import subprocess
import sys

from absl import logging
import jax
import pytest


def find_free_port():
  with contextlib.closing(
      socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  ) as s:
    s.bind(("", 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    return s.getsockname()[1]


def run_worker_and_command(command):
  """Worker Mode: Initializes JAX explicitly, then executes the target command."""

  coordinator_address = os.environ.get("JAX_COORDINATOR_ADDRESS")
  num_processes = os.environ.get("JAX_NUM_PROCESSES")
  process_id = os.environ.get("JAX_PROCESS_ID")

  if coordinator_address is None:
    raise ValueError(
        "Environment variables for JAX distributed not found. "
        "Did you use launch_multihost.py?"
    )

  # Explicit Initialization
  jax.distributed.initialize(
      coordinator_address=coordinator_address,
      num_processes=int(num_processes),
      process_id=int(process_id),
  )

  print(f"[Rank {process_id}] JAX Initialized. Executing: {' '.join(command)}")
  print(f"[Rank {process_id}] JAX devices: {jax.devices()}")

  # Clean up 'python' from the command if the user accidentally included it
  if command[0] == "python" or command[0] == "python3":
    command = command[1:]

  cmd_name = command[0]

  # Execute the requested script/tool inside this initialized process
  if cmd_name == "pytest":
    sys.exit(pytest.main(command[1:]))

  elif cmd_name.endswith(".py"):
    # Overwrite sys.argv so the target script sees its expected arguments
    sys.argv = command
    runpy.run_path(cmd_name, run_name="__main__")

  else:
    # Fallback for arbitrary shell commands
    sys.exit(subprocess.call(command))


def main():
  # 1. Parse arguments meant for launch.py
  parser = argparse.ArgumentParser(description="JAX Multihost Launcher")
  parser.add_argument(
      "--worker_mode", action="store_true", help=argparse.SUPPRESS
  )
  parser.add_argument(
      "--num_processes", type=int, default=2, help="Number of simulated hosts"
  )
  parser.add_argument(
      "--tpu_chips_per_process", type=int, default=4, help="TPU chips per host"
  )

  # `args` gets the launcher configs, `command` gets everything else
  args, command = parser.parse_known_args()

  # 2. WORKER MODE
  if args.worker_mode:
    if not command:
      raise ValueError("No command provided for the worker to execute.")
    run_worker_and_command(command)
    return

  # 3. LAUNCHER MODE
  if not command:
    logging.error(
        "Usage: python %s [LAUNCH_ARGS] <script.py> [SCRIPT_ARGS]",
        os.path.basename(__file__),
    )
    sys.exit(1)

  coordinator_port = find_free_port()
  coordinator_address = f"localhost:{coordinator_port}"

  slicebuilder_ports = [find_free_port() for _ in range(args.num_processes)]
  slicebuilder_addresses = ",".join(
      f"localhost:{port}" for port in slicebuilder_ports
  )

  logging.info(
      "🚀 Starting %s JAX processes (%s chips/process)...",
      args.num_processes,
      args.tpu_chips_per_process,
  )
  logging.info("📍 Coordinator: %s", coordinator_address)

  tpu_chips_per_process = args.tpu_chips_per_process
  num_tpu_chips = args.num_processes * args.tpu_chips_per_process
  if num_tpu_chips == 0:
    tpu_host_bounds = ""
    tpu_chips_per_host_bounds = ""
  elif num_tpu_chips == 1:
    assert tpu_chips_per_process == 1
    tpu_host_bounds = "1,1,1"
    tpu_chips_per_host_bounds = "1,1,1"
  elif num_tpu_chips == 4:
    if tpu_chips_per_process == 1:
      tpu_host_bounds = "2,2,1"
      tpu_chips_per_host_bounds = "1,1,1"
    elif tpu_chips_per_process == 2:
      tpu_host_bounds = "2,1,1"
      tpu_chips_per_host_bounds = "1,2,1"
    elif tpu_chips_per_process == 4:
      tpu_host_bounds = "1,1,1"
      tpu_chips_per_host_bounds = "2,2,1"
    else:
      raise ValueError(
          "Invalid number of TPU chips per worker {}".format(
              tpu_chips_per_process
          )
      )
  elif num_tpu_chips == 8:
    if tpu_chips_per_process == 1:
      tpu_host_bounds = "4,2,1"
      tpu_chips_per_host_bounds = "1,1,1"
    elif tpu_chips_per_process == 2:
      tpu_host_bounds = "2,2,1"
      tpu_chips_per_host_bounds = "1,2,1"
    elif tpu_chips_per_process == 4:
      # Note: this branch assumes we are using 2x4 v6e LitePod, and will not
      # work with 4x2 v5e LitePod.
      tpu_host_bounds = "1,2,1"
      tpu_chips_per_host_bounds = "2,2,1"
    elif tpu_chips_per_process == 8:
      tpu_host_bounds = "1,1,1"
      tpu_chips_per_host_bounds = "2,4,1"
    else:
      # TODO(phawkins): implement other cases.
      raise ValueError(
          "Invalid number of TPU chips per worker {}".format(
              tpu_chips_per_process
          )
      )
  else:
    raise ValueError(f"Invalid number of TPU chips {num_tpu_chips}")

  processes = []
  for rank in range(args.num_processes):
    env = os.environ.copy()

    # JAX Distributed Setup
    env["JAX_COORDINATOR_ADDRESS"] = coordinator_address
    env["JAX_NUM_PROCESSES"] = str(args.num_processes)
    env["JAX_PROCESS_ID"] = str(rank)

    device_ids = range(
        rank * args.tpu_chips_per_process,
        (rank + 1) * args.tpu_chips_per_process,
    )

    # Simulated TPU Setup
    env["CLOUD_TPU_TASK_ID"] = str(rank)
    env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = tpu_chips_per_host_bounds
    env["TPU_PROCESS_BOUNDS"] = tpu_host_bounds
    env["TPU_PROCESS_ADDRESSES"] = slicebuilder_addresses
    env["TPU_PROCESS_PORT"] = str(slicebuilder_ports[rank])
    env["TPU_VISIBLE_CHIPS"] = ",".join(map(str, device_ids))
    env["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "1"

    # Format the user's command to inject the current process rank where {rank}
    # is used
    worker_cmd = [c.format(rank=rank) for c in command]

    # Spawn THIS script again, triggering worker_mode
    cmd = [sys.executable, __file__, "--worker_mode"] + worker_cmd

    p = subprocess.Popen(cmd, env=env)
    processes.append(p)

  exit_codes = [p.wait() for p in processes]

  if any(c != 0 for c in exit_codes):
    logging.error("\n❌ Some processes failed.")
    sys.exit(1)
  else:
    logging.info("\n✅ All processes finished successfully.")


if __name__ == "__main__":
  main()
