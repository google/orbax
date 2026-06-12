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

import datetime
import textwrap
import unittest

from absl.testing import absltest
import aiosqlite  # pylint: disable=unused-import
import greenlet  # pylint: disable=unused-import
from orbax.checkpoint.experimental.tiering_service import db_lib
from orbax.checkpoint.experimental.tiering_service import db_schema
from orbax.checkpoint.experimental.tiering_service import server_config
from sqlalchemy import exc as sqlalchemy_exc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
import yaml


class DbLibTest(absltest.TestCase, unittest.IsolatedAsyncioTestCase):

  async def test_initialize_db_from_yaml(self):
    tmp_file = self.create_tempfile()
    db_url = f"sqlite+aiosqlite:///{tmp_file.full_path}"
    yaml_content = textwrap.dedent(f"""\
        db_connection_str: {db_url}
        storage_backends:
          - level: 0
            backend_type: BACKEND_TYPE_LUSTRE
            prefix: /mnt/lustre
            zone: us-central1-a
          - level: 0
            backend_type: BACKEND_TYPE_LUSTRE
            prefix: /mnt/lustre2
            zone: us-central1-b
          - level: 1
            backend_type: BACKEND_TYPE_GCS
            prefix: gs://my-bucket
            region: us-central1
          - level: 1
            backend_type: BACKEND_TYPE_GCS
            prefix: gs://my-bucket2
            region: us-west1
          - level: 2
            backend_type: BACKEND_TYPE_GCS
            prefix: gs://my-bucket3
            multi_regions: [us-central1, us-east1]
    """)
    config_dict = yaml.safe_load(yaml_content)
    config = server_config.parse_config(config_dict)

    await db_lib.async_initialize_db(config)
    await db_lib.async_verify_db(config)

  async def test_verify_db_mismatch_raises(self):
    tmp_file = self.create_tempfile()
    db_url = f"sqlite+aiosqlite:///{tmp_file.full_path}"
    yaml_content = textwrap.dedent(f"""\
        db_connection_str: {db_url}
        storage_backends:
          - level: 1
            backend_type: BACKEND_TYPE_GCS
            prefix: gs://my-bucket
            region: us-central1
    """)
    config_dict = yaml.safe_load(yaml_content)
    config = server_config.parse_config(config_dict)
    await db_lib.async_initialize_db(config)

    # Modify config to expect a different region
    yaml_content_mod = textwrap.dedent(f"""\
        db_connection_str: {db_url}
        storage_backends:
          - level: 1
            backend_type: BACKEND_TYPE_GCS
            prefix: gs://my-bucket
            region: us-east1
    """)
    config_mod = server_config.parse_config(yaml.safe_load(yaml_content_mod))
    with self.assertRaisesRegex(
        ValueError,
        "Configuration expects StorageBackend with key",
    ):
      await db_lib.async_verify_db(config_mod)

    # Modify config to expect a different prefix in the same region
    yaml_content_prefix = textwrap.dedent(f"""\
        db_connection_str: {db_url}
        storage_backends:
          - level: 1
            backend_type: BACKEND_TYPE_GCS
            prefix: gs://other-bucket
            region: us-central1
    """)
    config_prefix = server_config.parse_config(
        yaml.safe_load(yaml_content_prefix)
    )
    with self.assertRaisesRegex(
        ValueError,
        "Backend with key .* mismatch prefix",
    ):
      await db_lib.async_verify_db(config_prefix)

  async def test_initialize_db_missing_location_rejected(self):
    tmp_file = self.create_tempfile()
    db_url = f"sqlite+aiosqlite:///{tmp_file.full_path}"
    yaml_content = textwrap.dedent(f"""\
        db_connection_str: {db_url}
        storage_backends:
          - level: 1
            backend_type: BACKEND_TYPE_GCS
            prefix: gs://my-bucket
    """)
    config_dict = yaml.safe_load(yaml_content)
    config = server_config.parse_config(config_dict)

    with self.assertRaisesRegex(
        sqlalchemy_exc.IntegrityError,
        "check_mutually_exclusive_locations",
    ):
      await db_lib.async_initialize_db(config)

  async def test_is_db_initialized(self):
    tmp_file = self.create_tempfile()
    db_url = f"sqlite+aiosqlite:///{tmp_file.full_path}"
    yaml_content = textwrap.dedent(f"""\
        db_connection_str: {db_url}
        storage_backends:
          - level: 1
            backend_type: BACKEND_TYPE_GCS
            prefix: gs://my-bucket
            region: us-central1
    """)
    config_dict = yaml.safe_load(yaml_content)
    config = server_config.parse_config(config_dict)

    self.assertFalse(await db_lib.async_is_db_initialized(config))

    await db_lib.async_initialize_db(config)

    self.assertTrue(await db_lib.async_is_db_initialized(config))

  async def test_sqlite_url_translation(self):
    tmp_file = self.create_tempfile()
    # Pass standard sqlite:// instead of sqlite+aiosqlite://
    db_url = f"sqlite:///{tmp_file.full_path}"
    yaml_content = textwrap.dedent(f"""\
        db_connection_str: {db_url}
        storage_backends:
          - level: 1
            backend_type: BACKEND_TYPE_GCS
            prefix: gs://my-bucket
            region: us-central1
    """)
    config_dict = yaml.safe_load(yaml_content)
    config = server_config.parse_config(config_dict)

    # This will use get_async_engine, which should translate sqlite:// to
    # sqlite+aiosqlite://
    engine = db_lib.get_async_engine(config)
    self.assertEqual(
        str(engine.url), f"sqlite+aiosqlite:///{tmp_file.full_path}"
    )
    await engine.dispose()

  async def test_acquire_and_get_active_jobs(self):
    tmp_file = self.create_tempfile()
    db_url = f"sqlite+aiosqlite:///{tmp_file.full_path}"
    yaml_content = textwrap.dedent(f"""\
        db_connection_str: {db_url}
        storage_backends:
          - level: 1
            backend_type: BACKEND_TYPE_GCS
            prefix: gs://my-bucket
            region: us-central1
    """)
    config_dict = yaml.safe_load(yaml_content)
    config = server_config.parse_config(config_dict)
    await db_lib.async_initialize_db(config)

    engine = db_lib.get_async_engine(config)

    async_session = sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )
    async with async_session() as session:
      # Create dependencies for valid AssetJob payload
      asset = db_schema.Asset(
          asset_uuid="dummy-uuid",
          path="/experiment/dummy-path",
          user="testuser",
      )
      backend = db_schema.StorageBackend(
          level=1,
          zone="us-central1-a",
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
          prefix="gs://my-bucket",
      )
      session.add_all([asset, backend])
      await session.commit()

      tier_path = db_schema.TierPath(
          asset_uuid="dummy-uuid",
          storage_backend_id=backend.id,
          path="/path1",
      )
      session.add(tier_path)
      await session.commit()

      # Create valid job
      job = db_schema.AssetJob(
          asset_uuid="dummy-uuid",
          request_type=db_schema.RequestType.REQUEST_TYPE_COPY,
          status=db_schema.JobStatus.JOB_STATUS_QUEUED,
          request_id="req-123",
          target_tier_path_id=tier_path.id,
      )
      session.add(job)
      await session.commit()

      # Acquire
      acquired_job = await db_lib.acquire_next_job(
          session=session,
          backend_id=backend.id,
          lease_duration=datetime.timedelta(minutes=5),
          hostname="test-host",
          pid=1234,
          max_active=10,
      )
      self.assertIsNotNone(acquired_job)
      self.assertEqual(acquired_job.request_id, "req-123")
      self.assertEqual(
          acquired_job.status, db_schema.JobStatus.JOB_STATUS_PROCESSING
      )
      self.assertEqual(acquired_job.worker_host, "test-host")
      await session.commit()

      # Get active jobs
      active_jobs = await db_lib.get_active_jobs(
          session=session, hostname="test-host", pid=1234
      )
      self.assertLen(active_jobs, 1)
      self.assertEqual(active_jobs[0].request_id, "req-123")


if __name__ == "__main__":
  absltest.main()
