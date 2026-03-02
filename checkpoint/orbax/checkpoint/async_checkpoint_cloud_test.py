# Copyright 2024 The Orbax Authors.
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

#!/usr/bin/python3
import os
import sys

from absl import flags
import jax
from jax._src.mesh import Mesh
from jax._src.partition_spec import PartitionSpec
from jax.experimental import multihost_utils
from jax.experimental.pjit import pjit
import numpy as np
import orbax.checkpoint as orbax

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["JAX_USE_PJRT_C_API_ON_TPU"] = "1"

FLAGS = flags.FLAGS
bucket_path_flag = flags.DEFINE_string(
    "bucket_path", None, "GCS bucket path, e.g. gs://my-bucket"
)
dir_name_flag = flags.DEFINE_string(
    "ckpt_dir", None, "GCS cloud bucket directory"
)

FLAGS(sys.argv)  # parses the flags.

print("orbax_async_checkpoint_test: Done parsing flags", flush=True)
bucket_path = bucket_path_flag.value
dir_name = dir_name_flag.value
ckpt_dir = os.path.join(bucket_path, dir_name)
print(f"orbax_async_checkpoint_test: ckpt_dir={ckpt_dir}", flush=True)

print(
    "orbax_async_checkpoint_test: Attempting to initialize the jax distributed"
    " system...",
    flush=True,
)

jax.distributed.initialize()

print(
    "orbax_async_checkpoint_test: Jax distributed system initialized!",
    flush=True,
)
# You must initialize the jax distributed system before interacting with the
# device backend.
print(
    f"orbax_async_checkpoint_test: Devices found. number {len(jax.devices())},"
    f" details: {jax.devices()}",
    flush=True,
)

mngr = orbax.CheckpointManager(
    ckpt_dir,
    orbax.AsyncCheckpointer(orbax.PyTreeCheckpointHandler()),
    options=orbax.CheckpointManagerOptions(create=True),
)
print(
    "orbax_async_checkpoint_test: Created Async CheckpointManager", flush=True
)

BATCH = 512
SIZE = 128


def gen_data():
  return jax.process_index() * jax.numpy.ones(
      (BATCH, SIZE, SIZE), dtype=jax.numpy.bfloat16
  )


# Define array lengths based on device count
num_devices = len(jax.devices())
mesh_shape = [len(jax.devices())]
devices = np.asarray(jax.devices()).reshape(*mesh_shape)
mesh = Mesh(devices, "x")
gen_data_sharded = pjit(
    gen_data, in_shardings=None, out_shardings=PartitionSpec("x")
)


with Mesh(mesh.devices, mesh.axis_names):
  presharded_X = jax.block_until_ready(gen_data_sharded())
  to_save = {"my_X": presharded_X}
  print("orbax_async_checkpoint_test: Attempting save...", flush=True)
  mngr.save(0, to_save)
  print("orbax_async_checkpoint_test: Save successful!", flush=True)

# Try to restore now
print("orbax_async_checkpoint_test: Attempting restore...!", flush=True)

s2 = mngr.restore(0)
print("orbax_async_checkpoint_test: Restore successful!", flush=True)

multihost_utils.sync_global_devices("Script finished")
print("orbax_async_checkpoint_test: Test finished successfully", flush=True)
