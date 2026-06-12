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

"""GCP storage clients for Checkpoint Tiering Service (CTS)."""

import abc
import asyncio
import dataclasses
import datetime
import enum
import os
from typing import Any
import google.auth
from google.auth import exceptions as auth_exceptions
from google.auth import impersonated_credentials
from google.auth import transport
import httpx


class OperationStatus(enum.Enum):
  """The status of a job or operation."""

  IN_PROGRESS = "IN_PROGRESS"
  SUCCESS = "SUCCESS"
  FAILED = "FAILED"


@dataclasses.dataclass
class Result:
  status: OperationStatus
  detail_info: dict[str, Any]


@dataclasses.dataclass
class TransferContext:
  job_request_id: str
  source_path: str
  destination_path: str
  transfer_status: dict[str, Any]


class HttpxRequest(transport.Request):
  """A google-auth compatible transport request using HTTPX (sync)."""

  def __init__(self, client: httpx.Client):
    self._client = client

  def __call__(
      self,
      url: str,
      method: str = "GET",
      body: bytes | None = None,
      headers: dict[str, str] | None = None,
      timeout: float | None = None,
      **kwargs,
  ) -> transport.Response:
    try:
      response = self._client.request(
          method=method,
          url=url,
          headers=headers,
          content=body,
          timeout=timeout,
          **kwargs,
      )

      class HttpxResponse(transport.Response):
        """A google-auth compatible transport response using HTTPX."""

        def __init__(self, resp):
          self._resp = resp

        @property
        def status(self) -> int:
          return self._resp.status_code

        @property
        def headers(self) -> dict[str, str]:
          return dict(self._resp.headers)

        @property
        def data(self) -> bytes:
          return self._resp.content

      return HttpxResponse(response)
    except httpx.TimeoutException as e:
      raise auth_exceptions.TransportError(f"Timeout: {e}")
    except httpx.RequestError as e:
      raise auth_exceptions.TransportError(f"Request error: {e}")


class GCPStorageClient(abc.ABC):
  """Client interface to interact with GCP storage backend (e.g.

  Lustre, GCS).
  """

  def __init__(
      self,
      project: str | None = None,
      location: str | None = None,
      instance: str | None = None,
      service_account: str | None = None,
  ):
    self.project = project
    self.location = location
    self.instance = instance
    self.service_account = service_account
    self._credentials = None
    self._async_client = None

  @property
  def async_client(self) -> httpx.AsyncClient:
    if self._async_client is None:
      self._async_client = httpx.AsyncClient()
    return self._async_client

  async def close(self):
    if self._async_client is not None:
      await self._async_client.aclose()
      self._async_client = None

  async def __aenter__(self):
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.close()

  async def _get_token_and_project(self) -> tuple[str, str]:
    """Gets authentication credentials and projects."""
    if not self._credentials:
      base_credentials, detected_project = await asyncio.to_thread(
          google.auth.default,
          scopes=["https://www.googleapis.com/auth/cloud-platform"],
      )
      if not self.project:
        self.project = detected_project

      if self.service_account:
        self._credentials = impersonated_credentials.Credentials(
            source_credentials=base_credentials,
            target_principal=self.service_account,
            target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
      else:
        self._credentials = base_credentials

    if not self._credentials.valid:
      with httpx.Client() as client:
        await asyncio.to_thread(self._credentials.refresh, HttpxRequest(client))

    if not self.project:
      raise ValueError("GCP Project ID must be specified or auto-detected.")

    return self._credentials.token, self.project

  @abc.abstractmethod
  async def trigger_copy(
      self,
      request_id: str,
      source_path: str,
      destination_path: str,
  ) -> str:
    """Triggers copy and returns operation name."""
    pass

  @abc.abstractmethod
  async def poll_operation(
      self,
      operation_name: str,
      context: TransferContext | None = None,
  ) -> Result:
    """Polls operation status and returns a Result object."""
    pass


def _parse_gcs_path(gcs_path: str) -> tuple[str, str]:
  """Parses a GCS path like gs://bucket/prefix/file into (bucket, prefix)."""
  path_no_scheme = gcs_path.replace("gs://", "")
  parts = path_no_scheme.split("/", 1)
  bucket = parts[0]
  prefix = parts[1] if len(parts) > 1 else ""
  return bucket, prefix


class GcsToGcsClient(GCPStorageClient):
  """Client implementation for GCS-to-GCS operations using Storage Transfer Service."""

  def __init__(
      self,
      project: str | None = None,
      service_account: str | None = None,
  ):
    super().__init__(project=project, service_account=service_account)

  async def trigger_copy(
      self,
      request_id: str,
      source_path: str,
      destination_path: str,
  ) -> str:
    """Triggers GCS-to-GCS transfer using GCP Storage Transfer Service (STS)."""
    token, project = await self._get_token_and_project()
    url = "https://storagetransfer.googleapis.com/v1/transferJobs"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    src_bucket, src_prefix = _parse_gcs_path(source_path)
    dest_bucket, dest_prefix = _parse_gcs_path(destination_path)

    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "projectId": project,
        "transferSpec": {
            "gcsDataSource": {
                "bucketName": src_bucket,
                "path": src_prefix,
            },
            "gcsDataSink": {
                "bucketName": dest_bucket,
                "path": dest_prefix,
            },
            "transferOptions": {
                "overwriteObjectsAlreadyExistingInSink": True,
            },
        },
        "schedule": {
            "scheduleStartDate": {
                "year": now.year,
                "month": now.month,
                "day": now.day,
            },
        },
        "status": "DISABLED",
    }

    response = await self.async_client.post(url, json=payload, headers=headers)

    if response.status_code != 200:
      raise RuntimeError(
          f"Failed to create Storage Transfer Job: {response.status_code} -"
          f" {response.text}"
      )

    job_name = response.json()["name"]

    # Run the job immediately. This returns the operation name directly.
    run_url = f"https://storagetransfer.googleapis.com/v1/{job_name}:run"
    run_payload = {"projectId": project}
    run_response = await self.async_client.post(
        run_url, json=run_payload, headers=headers
    )

    if run_response.status_code != 200:
      raise RuntimeError(
          f"Failed to run Storage Transfer Job {job_name}:"
          f" {run_response.status_code} - {run_response.text}"
      )

    operation_name = run_response.json()["name"]
    return operation_name

  async def poll_operation(
      self,
      operation_name: str,
      context: TransferContext | None = None,
  ) -> Result:
    """Polls Storage Transfer Service operation status."""
    token, _ = await self._get_token_and_project()
    url = f"https://storagetransfer.googleapis.com/v1/{operation_name}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    response = await self.async_client.get(url, headers=headers)
    if response.status_code != 200:
      raise RuntimeError(
          f"Failed to poll Storage Transfer operation: {response.status_code} -"
          f" {response.text}"
      )

    data = response.json()
    done = data.get("done", False)

    if not done:
      metadata = data.get("metadata", {})
      counters = metadata.get("counters", {})
      bytes_transferred = int(counters.get("bytesTransferredToSink", 0))
      bytes_found = int(counters.get("bytesFoundToTransfer", 0))
      return Result(
          status=OperationStatus.IN_PROGRESS,
          detail_info={
              "bytes_copied": bytes_transferred,
              "total_bytes": bytes_found,
          },
      )

    if "error" in data:
      return Result(
          status=OperationStatus.FAILED,
          detail_info={"error": data["error"]},
      )

    metadata = data.get("metadata", {})
    op_status = metadata.get("status")

    if op_status == "SUCCESS":
      return Result(
          status=OperationStatus.SUCCESS,
          detail_info={},
      )
    else:
      error_msg = f"STS Operation ended with status: {op_status}"
      return Result(
          status=OperationStatus.FAILED,
          detail_info={"error": error_msg},
      )


class GcpLustreBaseClient(GCPStorageClient):
  """Base client interface to interact with GCP Managed Lustre API via REST."""

  def __init__(
      self,
      project: str | None = None,
      location: str | None = None,
      instance: str | None = None,
      service_account: str | None = None,
  ):
    location = location or os.environ.get("CTS_LUSTRE_LOCATION")
    instance = instance or os.environ.get("CTS_LUSTRE_INSTANCE")

    if not location or not instance:
      raise ValueError("Lustre location and instance must be specified.")

    super().__init__(
        project=project,
        location=location,
        instance=instance,
        service_account=service_account,
    )

  async def poll_operation(
      self,
      operation_name: str,
      context: TransferContext | None = None,
  ) -> Result:
    """Polls operation status and returns a Result object."""
    del context  # Unused for Lustre
    token, _ = await self._get_token_and_project()
    url = f"https://lustre.googleapis.com/v1/{operation_name}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    response = await self.async_client.get(url, headers=headers)
    if response.status_code != 200:
      raise RuntimeError(
          f"Failed to poll operation: {response.status_code} - {response.text}"
      )
    data = response.json()
    done = data.get("done", False)
    if done:
      if "error" in data:
        return Result(
            status=OperationStatus.FAILED,
            detail_info={"error": data["error"]},
        )
      else:
        return Result(
            status=OperationStatus.SUCCESS,
            detail_info=data.get("response", {}),
        )
    else:
      return Result(
          status=OperationStatus.IN_PROGRESS,
          detail_info=data.get("metadata", {}),
      )


class GcsToLustreClient(GcpLustreBaseClient):
  """Client implementation to trigger GCS-to-Lustre imports."""

  async def trigger_copy(
      self,
      request_id: str,
      source_path: str,
      destination_path: str,
  ) -> str:
    """Triggers import from GCS to Lustre and returns the Operation name."""
    token, project = await self._get_token_and_project()
    url = (
        f"https://lustre.googleapis.com/v1/projects/{project}"
        f"/locations/{self.location}/instances/{self.instance}:importData"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "gcsPath": {"uri": source_path},
        "lustrePath": {"path": destination_path},
        "requestId": request_id,
    }
    response = await self.async_client.post(url, json=payload, headers=headers)
    if response.status_code != 200:
      raise RuntimeError(
          f"Failed to trigger import: {response.status_code} - {response.text}"
      )
    return response.json()["name"]


class LustreToGcsClient(GcpLustreBaseClient):
  """Client implementation to trigger Lustre-to-GCS exports."""

  async def trigger_copy(
      self,
      request_id: str,
      source_path: str,
      destination_path: str,
  ) -> str:
    """Triggers export from Lustre to GCS and returns the Operation name."""
    token, project = await self._get_token_and_project()
    url = (
        f"https://lustre.googleapis.com/v1/projects/{project}"
        f"/locations/{self.location}/instances/{self.instance}:exportData"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "lustrePath": {"path": source_path},
        "gcsPath": {"uri": destination_path},
        "requestId": request_id,
    }
    response = await self.async_client.post(url, json=payload, headers=headers)
    if response.status_code != 200:
      raise RuntimeError(
          f"Failed to trigger export: {response.status_code} - {response.text}"
      )
    return response.json()["name"]
