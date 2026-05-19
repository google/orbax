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

from absl.testing import absltest
from orbax.checkpoint.experimental.tiering_service import server_config
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2


class ServerConfigTest(absltest.TestCase):

  def test_parse_timedelta(self):
    self.assertEqual(
        server_config._parse_timedelta("1s"),
        datetime.timedelta(seconds=1),
    )
    self.assertEqual(
        server_config._parse_timedelta("30m"),
        datetime.timedelta(minutes=30),
    )
    self.assertEqual(
        server_config._parse_timedelta("1h"),
        datetime.timedelta(hours=1),
    )
    with self.assertRaisesRegex(
        ValueError, "Invalid duration format for client_keep_alive_interval:"
    ):
      server_config._parse_timedelta("invalid")
    with self.assertRaisesRegex(
        ValueError, "Invalid duration type for client_keep_alive_interval:"
    ):
      server_config._parse_timedelta(123)  # type: ignore

  def test_parse_client_keep_alive(self):
    config = tiering_service_pb2.ServerConfig()
    server_config._parse_client_keep_alive(
        {"client_keep_alive_interval_seconds": 600}, config
    )
    self.assertEqual(config.client_keep_alive_interval_seconds, 600)

    config = tiering_service_pb2.ServerConfig()
    server_config._parse_client_keep_alive(
        {"client_keep_alive_interval": "15m"}, config
    )
    self.assertEqual(config.client_keep_alive_interval_seconds, 900)

    config = tiering_service_pb2.ServerConfig()
    server_config._parse_client_keep_alive(
        {"client_keep_alive_interval": 300}, config
    )
    self.assertEqual(config.client_keep_alive_interval_seconds, 300)

    config = tiering_service_pb2.ServerConfig()
    server_config._parse_client_keep_alive({}, config)
    self.assertEqual(config.client_keep_alive_interval_seconds, 1800)

  def test_parse_db_connection(self):
    config = tiering_service_pb2.ServerConfig()
    server_config._parse_db_connection(
        {"db_connection_str": "sqlite:///test.db"}, config
    )
    self.assertEqual(config.db_connection_str, "sqlite:///test.db")

    config = tiering_service_pb2.ServerConfig()
    server_config._parse_db_connection({}, config)
    self.assertEqual(config.db_connection_str, "sqlite+aiosqlite:///:memory:")

  def test_parse_storage_backend_valid(self):
    backend = tiering_service_pb2.StorageBackend()
    server_config._parse_storage_backend(
        {
            "level": 0,
            "backend_type": "Lustre",
            "prefix": "/mnt/lustre",
            "zone": "us-central1-a",
        },
        backend,
    )
    self.assertEqual(backend.level, 0)
    self.assertEqual(
        backend.backend_type, tiering_service_pb2.BACKEND_TYPE_LUSTRE
    )
    self.assertEqual(backend.prefix, "/mnt/lustre")
    self.assertEqual(backend.zone, "us-central1-a")

    backend = tiering_service_pb2.StorageBackend()
    server_config._parse_storage_backend(
        {
            "level": 1,
            "backend_type": "GCS",
            "prefix": "gs://bucket",
            "region": "us-east1",
        },
        backend,
    )
    self.assertEqual(backend.level, 1)
    self.assertEqual(backend.backend_type, tiering_service_pb2.BACKEND_TYPE_GCS)
    self.assertEqual(backend.prefix, "gs://bucket")
    self.assertEqual(backend.region, "us-east1")

    backend = tiering_service_pb2.StorageBackend()
    server_config._parse_storage_backend(
        {
            "level": 2,
            "backend_type": tiering_service_pb2.BACKEND_TYPE_GCS,
            "prefix": "gs://multi",
            "multi_regions": ["us-central1", "us-east1"],
        },
        backend,
    )
    self.assertEqual(backend.level, 2)
    self.assertEqual(
        list(backend.multi_regions.regions), ["us-central1", "us-east1"]
    )

  def test_parse_storage_backend_errors(self):
    backend = tiering_service_pb2.StorageBackend()
    with self.assertRaisesRegex(
        ValueError, "StorageBackend configuration missing required key: 'level'"
    ):
      server_config._parse_storage_backend(
          {"backend_type": "GCS", "prefix": "gs://b"},
          backend,
      )

    with self.assertRaisesRegex(
        ValueError,
        "StorageBackend configuration missing required key: 'backend_type'",
    ):
      server_config._parse_storage_backend(
          {"level": 1, "prefix": "gs://b"}, backend
      )

    with self.assertRaisesRegex(
        ValueError,
        "StorageBackend configuration missing required key: 'prefix'",
    ):
      server_config._parse_storage_backend(
          {"level": 1, "backend_type": "GCS"}, backend
      )

    with self.assertRaisesRegex(ValueError, "Unknown storage backend_type:"):
      server_config._parse_storage_backend(
          {"level": 1, "backend_type": "invalid", "prefix": "gs://b"}, backend
      )

    with self.assertRaisesRegex(ValueError, "Invalid multi_regions format:"):
      server_config._parse_storage_backend(
          {
              "level": 1,
              "backend_type": "GCS",
              "prefix": "gs://b",
              "multi_regions": 123,
          },
          backend,
      )

  def test_parse_storage_backends(self):
    config = tiering_service_pb2.ServerConfig()
    server_config._parse_storage_backends(
        {
            "storage_backends": [{
                "level": 0,
                "backend_type": "Lustre",
                "prefix": "/mnt/lustre",
                "zone": "us-central1-a",
            }]
        },
        config,
    )
    self.assertLen(config.storage_backends, 1)
    self.assertEqual(config.storage_backends[0].prefix, "/mnt/lustre")

  def test_load_config(self):
    tmp_file = self.create_tempfile(content=textwrap.dedent("""
        client_keep_alive_interval: 10m
        db_connection_str: sqlite:///load.db
        storage_backends:
          - level: 0
            backend_type: Lustre
            prefix: /mnt/lustre
            zone: us-central1-a
    """))
    config = server_config.load_config(tmp_file.full_path)
    self.assertEqual(config.client_keep_alive_interval_seconds, 600)
    self.assertEqual(config.db_connection_str, "sqlite:///load.db")
    self.assertLen(config.storage_backends, 1)


if __name__ == "__main__":
  absltest.main()
