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

"""Tiering Service Database CLI."""

import asyncio
from collections.abc import Sequence
import sys

import fire
from orbax.checkpoint.experimental.tiering_service import db_lib
from orbax.checkpoint.experimental.tiering_service import server_config
import uvloop


class DbCli:
  """Tiering Service Database CLI."""

  async def initialize_db(self, yaml_path: str) -> None:
    """Initializes or verifies the Tiering Service database.

    Args:
      yaml_path: Path to the YAML configuration file.

    Raises:
      ValueError: If the configuration file cannot be opened or existing
      database
        entries do not match the configuration.
    """
    try:
      config = server_config.load_config(yaml_path)
    except OSError as e:
      raise ValueError(f"Failed to open configuration file: {e}") from e
    if not await db_lib.async_is_db_initialized(config):
      await db_lib.async_initialize_db(config)
    else:
      await db_lib.async_verify_db(config)


def main(argv: Sequence[str] | None = None) -> None:
  """Main entry point for db_cli."""
  if argv is None:
    argv = sys.argv
  uvloop.install()
  try:
    asyncio.get_event_loop()
  except RuntimeError:
    # Create the high-performance uvloop instead
    loop = uvloop.new_event_loop()
    asyncio.set_event_loop(loop)
  fire.Fire(DbCli, command=argv[1:])


if __name__ == "__main__":
  main()
