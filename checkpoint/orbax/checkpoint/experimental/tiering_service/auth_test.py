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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import grpc
from orbax.checkpoint.experimental.tiering_service import auth
from orbax.checkpoint.experimental.tiering_service import db_schema


class AuthTest(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):

  async def test_get_oauth_token_success(self):
    context = mock.create_autospec(
        grpc.aio.ServicerContext, instance=True, spec_set=True
    )
    context.invocation_metadata = mock.AsyncMock(
        return_value=(("authorization", "Bearer valid-token"),)
    )
    token = await auth.get_oauth_token(context)
    self.assertEqual(token, "valid-token")

  async def test_get_oauth_token_no_header(self):
    context = mock.create_autospec(
        grpc.aio.ServicerContext, instance=True, spec_set=True
    )
    context.invocation_metadata = mock.AsyncMock(return_value=())
    token = await auth.get_oauth_token(context)
    self.assertIsNone(token)

  async def test_get_oauth_token_malformed_header(self):
    context = mock.create_autospec(
        grpc.aio.ServicerContext, instance=True, spec_set=True
    )
    context.invocation_metadata = mock.AsyncMock(
        return_value=(("authorization", "not-bearer-token"),)
    )
    token = await auth.get_oauth_token(context)
    self.assertIsNone(token)

  @parameterized.named_parameters(
      (
          "gcs_with_token",
          db_schema.BackendType.BACKEND_TYPE_GCS,
          "valid-token",
          True,
      ),
      (
          "gcs_no_token",
          db_schema.BackendType.BACKEND_TYPE_GCS,
          None,
          False,
      ),
      (
          "lustre_with_token",
          db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          "valid-token",
          True,
      ),
      (
          "lustre_no_token",
          db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          None,
          False,
      ),
  )
  async def test_has_write_permission(self, backend_type, token, expected):
    backend = db_schema.StorageBackend(
        backend_type=backend_type,
        prefix=(
            "gs://bucket"
            if backend_type == db_schema.BackendType.BACKEND_TYPE_GCS
            else "/mnt/lustre"
        ),
    )
    result = await auth.has_write_permission(token, backend, "path")
    self.assertEqual(result, expected)

  async def test_has_write_permission_unknown_backend(self):
    backend = mock.create_autospec(db_schema.StorageBackend, instance=True)
    backend.backend_type = "UNKNOWN"
    result = await auth.has_write_permission("token", backend, "path")
    self.assertFalse(result)


if __name__ == "__main__":
  absltest.main()
