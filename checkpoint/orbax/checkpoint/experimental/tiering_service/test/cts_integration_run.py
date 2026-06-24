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

"""Integration test script using TieringClient directly with CheckpointManager."""

import asyncio
from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from orbax.checkpoint.experimental.tiering_service import client as cts_client

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "mode", None, "Execution mode: save or restore.", required=True
)
flags.DEFINE_integer("step", 1, "Step number to save or restore.")
flags.DEFINE_string(
    "server_address", "localhost:50051", "Address of the CTS gRPC server."
)
flags.DEFINE_string(
    "checkpoint_dir",
    "/tmp/lustre-mount/checkpoints",
    "Base directory for checkpoints.",
)


async def _run_save(
    client: cts_client.TieringClient, checkpoint_dir: str, step: int
) -> None:
  """Reserves tiering path, saves checkpoint via CheckpointManager, and finalizes."""
  logging.info("Reserving asset for path: %s", checkpoint_dir)
  asset_uuid, reserved_path = await client.reserve(path=checkpoint_dir)
  logging.info("Reserved asset_uuid: %s at path: %s", asset_uuid, reserved_path)

  mngr = ocp.CheckpointManager(reserved_path)
  try:
    logging.info("Starting JAX save for step %d...", step)
    x = jnp.arange(10, dtype=jnp.int32)
    y = jnp.ones((2, 5), dtype=jnp.float32)
    save_args = ocp.args.Composite(
        x=ocp.args.ArraySave(x),
        y=ocp.args.ArraySave(y),
    )
    mngr.save(step, args=save_args)
    logging.info("Saved step %d successfully.", step)
  finally:
    mngr.close()

  logging.info("Finalizing asset_uuid: %s", asset_uuid)
  await client.finalize(asset_uuid)
  logging.info("Finalized asset %s successfully.", asset_uuid)


async def _run_restore(
    client: cts_client.TieringClient, checkpoint_dir: str, step: int
) -> None:
  """Prefetches asset via client, restores via CheckpointManager, and releases."""
  logging.info("Prefetching asset for path: %s", checkpoint_dir)
  restored_path = await client.prefetch(path=checkpoint_dir)
  logging.info("Prefetched path: %s", restored_path)

  mngr = ocp.CheckpointManager(restored_path)
  try:
    logging.info("Starting JAX restore for step %d...", step)
    abstract_x = jax.ShapeDtypeStruct((10,), jnp.int32)
    abstract_y = jax.ShapeDtypeStruct((2, 5), jnp.float32)
    restore_args = ocp.args.Composite(
        x=ocp.args.ArrayRestore(abstract_x),
        y=ocp.args.ArrayRestore(abstract_y),
    )
    restored = mngr.restore(step, args=restore_args)

    if not jnp.array_equal(restored.x, jnp.arange(10, dtype=jnp.int32)):
      raise ValueError(
          f"Restored X mismatch: expected {jnp.arange(10)}, got {restored.x}"
      )
    if not jnp.array_equal(restored.y, jnp.ones((2, 5), dtype=jnp.float32)):
      raise ValueError(f"Restored Y mismatch: expected ones, got {restored.y}")

    print("INTEGRATION_TEST_SUCCESS")
    logging.info("Restored step %d successfully and verified content.", step)
  finally:
    mngr.close()
    await client.release_path(restored_path)


async def _async_main() -> None:
  """Connects TieringClient and runs save/restore operation based on mode flag."""
  client = cts_client.TieringClient(
      server_address=FLAGS.server_address, secure=False
  )
  await client.connect()
  try:
    if FLAGS.mode == "save":
      await _run_save(client, FLAGS.checkpoint_dir, FLAGS.step)
    elif FLAGS.mode == "restore":
      await _run_restore(client, FLAGS.checkpoint_dir, FLAGS.step)
    else:
      raise ValueError(f"Unknown mode: {FLAGS.mode}")
  finally:
    await client.close()


def main(argv):
  """Entry point for the absl application executing _async_main."""
  del argv
  asyncio.run(_async_main())


if __name__ == "__main__":
  app.run(main)
