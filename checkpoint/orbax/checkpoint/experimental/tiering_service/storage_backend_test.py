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

import unittest
from absl.testing import absltest
from orbax.checkpoint.experimental.tiering_service import db_schema
from orbax.checkpoint.experimental.tiering_service import storage_backend
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine


class StorageBackendDbTest(absltest.TestCase, unittest.IsolatedAsyncioTestCase):

  async def asyncSetUp(self) -> None:
    await super().asyncSetUp()
    tmp_file = self.create_tempfile()
    self.engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_file.full_path}"
    )
    async with self.engine.begin() as conn:
      await conn.run_sync(db_schema.Base.metadata.create_all)
    self.session_maker = async_sessionmaker(self.engine, expire_on_commit=False)

  async def asyncTearDown(self) -> None:
    await self.engine.dispose()
    await super().asyncTearDown()

  async def _setup_backends(self, session):
    b1 = db_schema.StorageBackend(
        level=0,
        backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
        prefix="/mnt/lustre1",
        zone="us-central1-a",
    )
    b2 = db_schema.StorageBackend(
        level=0,
        backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
        prefix="/mnt/lustre2",
        zone="us-central1-b",
    )
    b3 = db_schema.StorageBackend(
        level=1,
        backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
        prefix="gs://bucket",
        region="us-central1",
    )
    session.add_all([b1, b2, b3])
    await session.commit()

  async def test_find_backends_by_level(self):
    async with self.session_maker() as session:
      await self._setup_backends(session)

      backends = await storage_backend.find_backends_by_level(session, level=0)
      self.assertLen(backends, 2)

      backends_l1 = await storage_backend.find_backends_by_level(
          session, level=1
      )
      self.assertLen(backends_l1, 1)
      self.assertEqual(backends_l1[0].prefix, "gs://bucket")

      backends_all = await storage_backend.find_backends_by_level(
          session, level=None
      )
      self.assertLen(backends_all, 3)

  async def test_locate_closest_backend(self):
    async with self.session_maker() as session:
      await self._setup_backends(session)

      backends = await storage_backend.find_backends_by_level(session, level=0)
      self.assertLen(backends, 2)

      match = storage_backend.locate_closest_backend(
          backends, "us-central1-a", None
      )
      self.assertIsNotNone(match)
      self.assertEqual(match.prefix, "/mnt/lustre1")

      match_fallback = storage_backend.locate_closest_backend(
          backends, "us-central1-c", "us-central1"
      )
      self.assertIsNone(match_fallback)


class StorageBackendPathTest(absltest.TestCase):

  def test_get_storage_path_safe_prefix(self):
    backend = db_schema.StorageBackend(
        level=1,
        backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
        prefix="gs://my-bucket",
    )
    self.assertEqual(
        storage_backend.get_storage_path(backend, "gs://my-bucket/file"),
        "gs://my-bucket/file",
    )
    self.assertEqual(
        storage_backend.get_storage_path(backend, "gs://my-bucket"),
        "gs://my-bucket",
    )
    self.assertEqual(
        storage_backend.get_storage_path(backend, "gs://my-bucket-2/file"),
        "gs://my-bucket/gs://my-bucket-2/file",
    )
    self.assertEqual(
        storage_backend.get_storage_path(backend, "file"),
        "gs://my-bucket/file",
    )

  def test_get_storage_path_with_trailing_slash(self):
    backend = db_schema.StorageBackend(
        level=1,
        backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
        prefix="gs://my-bucket/",
    )
    self.assertEqual(
        storage_backend.get_storage_path(backend, "gs://my-bucket/file"),
        "gs://my-bucket/file",
    )


if __name__ == "__main__":
  absltest.main()
