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

"""Test script for JAX distributed."""

import os
import jax


def run_workload():
  """Initializes JAX distributed environment and prints device information."""
  coordinator_address = os.environ.get("JAX_COORDINATOR_ADDRESS")
  num_processes = os.environ.get("JAX_NUM_PROCESSES")
  process_id = os.environ.get("JAX_PROCESS_ID")

  # Check if they exist (to avoid cryptic errors if run standalone)
  if coordinator_address is None:
    raise ValueError(
        "Environment variables for JAX distributed not found. Did you use"
        " launch_multihost.py?"
    )

  # 2. Pass them directly to initialize
  jax.distributed.initialize(
      coordinator_address=coordinator_address,
      num_processes=int(num_processes),
      process_id=int(process_id),
  )

  # ... rest of your code ...
  print(f"[Rank {jax.process_index()}] Successfully initialized!")

  # Verify devices
  print(
      f"[Rank {jax.process_index()}] Local devices: {jax.local_device_count()}"
  )
  print(f"[Rank {jax.process_index()}] Global devices: {jax.device_count()}")
  print(f"[Rank {jax.process_index()}] Devices: {jax.devices()}")


if __name__ == "__main__":
  run_workload()
