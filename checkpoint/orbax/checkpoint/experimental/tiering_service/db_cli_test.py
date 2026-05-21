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

import asyncio
import textwrap
import unittest

from absl.testing import absltest
import aiosqlite  # pylint: disable=unused-import
import greenlet  # pylint: disable=unused-import
from orbax.checkpoint.experimental.tiering_service import db_cli
from orbax.checkpoint.experimental.tiering_service import db_lib


class DbCliTest(absltest.TestCase, unittest.IsolatedAsyncioTestCase):

  async def test_initialize_db_success_uninitialized(self):
    tmp_db = self.create_tempfile()
    db_url = f"sqlite+aiosqlite:///{tmp_db.full_path}"
    yaml_content = textwrap.dedent(f"""
        db_connection_str: {db_url}
        storage_backends:
          - level: 0
            backend_type: BACKEND_TYPE_LUSTRE
            prefix: /mnt/lustre
            zone: us-central1-a
    """)
    tmp_yaml = self.create_tempfile(content=yaml_content)

    # Fire is going to run on its own event loop.
    await asyncio.to_thread(
        db_cli.main,
        ["db_cli.py", "initialize_db", "--yaml_path", tmp_yaml.full_path],
    )

    config = db_cli.server_config.load_config(tmp_yaml.full_path)
    self.assertTrue(await db_lib.async_is_db_initialized(config))

  def test_initialize_db_success_already_initialized(self):
    tmp_db = self.create_tempfile()
    db_url = f"sqlite+aiosqlite:///{tmp_db.full_path}"
    yaml_content = textwrap.dedent(f"""
        db_connection_str: {db_url}
        storage_backends:
          - level: 0
            backend_type: BACKEND_TYPE_LUSTRE
            prefix: /mnt/lustre
            zone: us-central1-a
    """)
    tmp_yaml = self.create_tempfile(content=yaml_content)

    db_cli.main(
        ["db_cli.py", "initialize_db", "--yaml_path", tmp_yaml.full_path]
    )

    # Re-initialize the DB from the same config should succeed.
    db_cli.main(
        ["db_cli.py", "initialize_db", "--yaml_path", tmp_yaml.full_path]
    )

  def test_missing_file_raises(self):
    with self.assertRaisesRegex(
        ValueError, "Failed to open configuration file:"
    ):
      db_cli.main([
          "db_cli.py",
          "initialize_db",
          "--yaml_path",
          "/nonexistent/path/to/config.yaml",
      ])


if __name__ == "__main__":
  absltest.main()
