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

"""Unit tests for GCP Storage Clients."""

import os
import unittest
from unittest import mock
import httpx
from orbax.checkpoint.experimental.tiering_service import gcp_storage_client


class GCPStorageClientTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.mock_creds = mock.MagicMock()
    self.mock_creds.valid = True
    self.mock_creds.token = "dummy_auth_token"
    self.mock_default_auth = mock.patch(
        "google.auth.default", return_value=(self.mock_creds, "dummy-project")
    )
    self.mock_default_auth.start()

    # Mock httpx.AsyncClient
    self.mock_client = mock.AsyncMock(spec=httpx.AsyncClient)
    self.mock_client_patcher = mock.patch(
        "httpx.AsyncClient", return_value=self.mock_client
    )
    self.mock_client_patcher.start()

  def tearDown(self):
    self.mock_default_auth.stop()
    self.mock_client_patcher.stop()
    super().tearDown()

  async def test_gcs_to_gcs_trigger_copy_success(self):
    client = gcp_storage_client.GcsToGcsClient(project="test-project")

    # 1. Mock the first POST call to create transfer job
    mock_post_resp_1 = mock.MagicMock(spec=httpx.Response)
    mock_post_resp_1.status_code = 200
    mock_post_resp_1.json.return_value = {"name": "transferJobs/job-123"}

    # 2. Mock the second POST call to run transfer job
    mock_post_resp_2 = mock.MagicMock(spec=httpx.Response)
    mock_post_resp_2.status_code = 200
    mock_post_resp_2.json.return_value = {"name": "transferOperations/op-456"}

    self.mock_client.post.side_effect = [mock_post_resp_1, mock_post_resp_2]

    op_name = await client.trigger_copy(
        request_id="req-1",
        source_path="gs://src-bucket/path/to/src",
        destination_path="gs://dest-bucket/path/to/dest",
    )

    self.assertEqual(op_name, "transferOperations/op-456")
    self.assertEqual(self.mock_client.post.call_count, 2)

  async def test_gcs_to_gcs_trigger_copy_sts_fail(self):
    client = gcp_storage_client.GcsToGcsClient(project="test-project")

    mock_post_resp = mock.MagicMock(spec=httpx.Response)
    mock_post_resp.status_code = 400
    mock_post_resp.text = "Invalid arguments"
    self.mock_client.post.return_value = mock_post_resp

    with self.assertRaises(RuntimeError) as ctx:
      await client.trigger_copy(
          request_id="req-1",
          source_path="gs://src-bucket/path",
          destination_path="gs://dest-bucket/path",
      )
    self.assertIn("Failed to create Storage Transfer Job", str(ctx.exception))

  async def test_gcs_to_gcs_poll_operation_in_progress(self):
    client = gcp_storage_client.GcsToGcsClient(project="test-project")

    mock_get_resp = mock.MagicMock(spec=httpx.Response)
    mock_get_resp.status_code = 200
    mock_get_resp.json.return_value = {
        "done": False,
        "metadata": {
            "counters": {
                "bytesTransferredToSink": "500",
                "bytesFoundToTransfer": "1000",
            }
        },
    }
    self.mock_client.get.return_value = mock_get_resp

    result = await client.poll_operation("transferOperations/op-456")
    self.assertEqual(
        result.status, gcp_storage_client.OperationStatus.IN_PROGRESS
    )
    self.assertEqual(result.detail_info["bytes_copied"], 500)
    self.assertEqual(result.detail_info["total_bytes"], 1000)

  async def test_gcs_to_gcs_poll_operation_success(self):
    client = gcp_storage_client.GcsToGcsClient(project="test-project")

    mock_get_resp = mock.MagicMock(spec=httpx.Response)
    mock_get_resp.status_code = 200
    mock_get_resp.json.return_value = {
        "done": True,
        "metadata": {"status": "SUCCESS"},
    }
    self.mock_client.get.return_value = mock_get_resp

    result = await client.poll_operation("transferOperations/op-456")
    self.assertEqual(result.status, gcp_storage_client.OperationStatus.SUCCESS)

  async def test_gcs_to_gcs_poll_operation_failed(self):
    client = gcp_storage_client.GcsToGcsClient(project="test-project")

    mock_get_resp = mock.MagicMock(spec=httpx.Response)
    mock_get_resp.status_code = 200
    mock_get_resp.json.return_value = {
        "done": True,
        "metadata": {"status": "FAILED"},
    }
    self.mock_client.get.return_value = mock_get_resp

    result = await client.poll_operation("transferOperations/op-456")
    self.assertEqual(result.status, gcp_storage_client.OperationStatus.FAILED)
    self.assertIn("error", result.detail_info)

  @mock.patch.dict(
      os.environ,
      {
          "CTS_LUSTRE_LOCATION": "us-central1-a",
          "CTS_LUSTRE_INSTANCE": "lustre-1",
      },
  )
  async def test_gcs_to_lustre_trigger_copy(self):
    client = gcp_storage_client.GcsToLustreClient(project="test-project")

    mock_post_resp = mock.MagicMock(spec=httpx.Response)
    mock_post_resp.status_code = 200
    mock_post_resp.json.return_value = {"name": "operations/import-123"}
    self.mock_client.post.return_value = mock_post_resp

    op_name = await client.trigger_copy(
        request_id="req-1",
        source_path="gs://src-bucket/path",
        destination_path="/lustre/path",
    )
    self.assertEqual(op_name, "operations/import-123")
    self.mock_client.post.assert_called_once()

  @mock.patch.dict(
      os.environ,
      {
          "CTS_LUSTRE_LOCATION": "us-central1-a",
          "CTS_LUSTRE_INSTANCE": "lustre-1",
      },
  )
  async def test_lustre_to_gcs_trigger_copy(self):
    client = gcp_storage_client.LustreToGcsClient(project="test-project")

    mock_post_resp = mock.MagicMock(spec=httpx.Response)
    mock_post_resp.status_code = 200
    mock_post_resp.json.return_value = {"name": "operations/export-123"}
    self.mock_client.post.return_value = mock_post_resp

    op_name = await client.trigger_copy(
        request_id="req-1",
        source_path="/lustre/path",
        destination_path="gs://dest-bucket/path",
    )
    self.assertEqual(op_name, "operations/export-123")
    self.mock_client.post.assert_called_once()

  @mock.patch.dict(
      os.environ,
      {
          "CTS_LUSTRE_LOCATION": "us-central1-a",
          "CTS_LUSTRE_INSTANCE": "lustre-1",
      },
  )
  async def test_lustre_poll_operation_done_success(self):
    client = gcp_storage_client.GcsToLustreClient(project="test-project")

    mock_get_resp = mock.MagicMock(spec=httpx.Response)
    mock_get_resp.status_code = 200
    mock_get_resp.json.return_value = {
        "done": True,
        "response": {"some_metadata": "val"},
    }
    self.mock_client.get.return_value = mock_get_resp

    result = await client.poll_operation("operations/import-123")
    self.assertEqual(result.status, gcp_storage_client.OperationStatus.SUCCESS)
    self.assertEqual(result.detail_info, {"some_metadata": "val"})

  @mock.patch.dict(
      os.environ,
      {
          "CTS_LUSTRE_LOCATION": "us-central1-a",
          "CTS_LUSTRE_INSTANCE": "lustre-1",
      },
  )
  async def test_lustre_poll_operation_done_fail(self):
    client = gcp_storage_client.GcsToLustreClient(project="test-project")

    mock_get_resp = mock.MagicMock(spec=httpx.Response)
    mock_get_resp.status_code = 200
    mock_get_resp.json.return_value = {
        "done": True,
        "error": {"message": "import failed"},
    }
    self.mock_client.get.return_value = mock_get_resp

    result = await client.poll_operation("operations/import-123")
    self.assertEqual(result.status, gcp_storage_client.OperationStatus.FAILED)
    self.assertEqual(result.detail_info["error"], {"message": "import failed"})

  @mock.patch.dict(
      os.environ,
      {
          "CTS_LUSTRE_LOCATION": "us-central1-a",
          "CTS_LUSTRE_INSTANCE": "lustre-1",
      },
  )
  async def test_lustre_poll_operation_in_progress(self):
    client = gcp_storage_client.GcsToLustreClient(project="test-project")

    mock_get_resp = mock.MagicMock(spec=httpx.Response)
    mock_get_resp.status_code = 200
    mock_get_resp.json.return_value = {
        "done": False,
        "metadata": {"percent_complete": 42},
    }
    self.mock_client.get.return_value = mock_get_resp

    result = await client.poll_operation("operations/import-123")
    self.assertEqual(
        result.status, gcp_storage_client.OperationStatus.IN_PROGRESS
    )
    self.assertEqual(result.detail_info, {"percent_complete": 42})

  @mock.patch("google.auth.impersonated_credentials.Credentials")
  async def test_service_account_token_impersonation(
      self, mock_impersonated_creds_class
  ):
    mock_imp_creds = mock.MagicMock()
    mock_imp_creds.valid = True
    mock_imp_creds.token = "impersonated_token"
    mock_impersonated_creds_class.return_value = mock_imp_creds

    client = gcp_storage_client.GcsToGcsClient(
        project="test-project",
        service_account="sa@test.iam.gserviceaccount.com",
    )

    token, project = await client._get_token_and_project()
    self.assertEqual(token, "impersonated_token")
    self.assertEqual(project, "test-project")

    # Verify that impersonated credentials were constructed correctly
    mock_impersonated_creds_class.assert_called_once_with(
        source_credentials=self.mock_creds,
        target_principal="sa@test.iam.gserviceaccount.com",
        target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )


if __name__ == "__main__":
  unittest.main()
