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
      dict(
          testcase_name="gcs_with_token",
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
          prefix="gs://bucket",
          token="valid-token",
          expected=True,
      ),
      dict(
          testcase_name="gcs_no_token",
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
          prefix="gs://bucket",
          token=None,
          expected=False,
      ),
      dict(
          testcase_name="lustre_with_token",
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre",
          token="valid-token",
          expected=True,
      ),
      dict(
          testcase_name="lustre_no_token",
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre",
          token=None,
          expected=False,
      ),
  )
  async def test_has_write_permission(
      self, backend_type, prefix, token, expected
  ):
    backend = db_schema.StorageBackend(backend_type=backend_type, prefix=prefix)
    result = await auth.has_write_permission(
        token, backend=backend, path="path"
    )
    self.assertEqual(result, expected)

  async def test_has_write_permission_unknown_backend(self):
    backend = mock.create_autospec(db_schema.StorageBackend, instance=True)
    backend.backend_type = "UNKNOWN"
    result = await auth.has_write_permission(
        "token", backend=backend, path="path"
    )
    self.assertIs(result, False)

  @parameterized.named_parameters(
      dict(
          testcase_name="gcs_with_token",
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
          prefix="gs://bucket",
          token="valid-token",
          expected=True,
      ),
      dict(
          testcase_name="gcs_no_token",
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
          prefix="gs://bucket",
          token=None,
          expected=False,
      ),
      dict(
          testcase_name="lustre_with_token",
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre",
          token="valid-token",
          expected=True,
      ),
      dict(
          testcase_name="lustre_no_token",
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre",
          token=None,
          expected=False,
      ),
  )
  async def test_has_read_permission(
      self, backend_type, prefix, token, expected
  ):
    backend = db_schema.StorageBackend(backend_type=backend_type, prefix=prefix)
    result = await auth.has_read_permission(token, backend=backend, path="path")
    self.assertEqual(result, expected)

  async def test_has_read_permission_unknown_backend(self):
    backend = mock.create_autospec(db_schema.StorageBackend, instance=True)
    backend.backend_type = "UNKNOWN"
    result = await auth.has_read_permission(
        "token", backend=backend, path="path"
    )
    self.assertIs(result, False)


if __name__ == "__main__":
  absltest.main()
