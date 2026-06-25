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
from typing import Any, Callable

import google.auth
from google.auth import exceptions as auth_exceptions
from google.auth import impersonated_credentials
from google.auth import transport
from google.protobuf import json_format
import httpx
from orbax.checkpoint.experimental.tiering_service import db_schema
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2


class OperationStatus(enum.Enum):
  """The status of a job or operation."""

  IN_PROGRESS = "IN_PROGRESS"
  SUCCESS = "SUCCESS"
  FAILED = "FAILED"


@dataclasses.dataclass
class Result:
  status: OperationStatus
  detail_info: dict[str, Any]


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
      zone: str | None = None,
      instance: str | None = None,
      service_account: str | None = None,
  ):
    if not zone or not instance:
      raise ValueError("Lustre zone and instance must be specified.")

    super().__init__(
        project=project,
        instance=instance,
        service_account=service_account,
    )
    self.zone = zone

  async def poll_operation(
      self,
      operation_name: str,
  ) -> Result:
    """Polls operation status and returns a Result object."""
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
        f"/locations/{self.zone}/instances/{self.instance}:importData"
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
        f"/locations/{self.zone}/instances/{self.instance}:exportData"
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


def serialize_transfer_status(
    status: tiering_service_pb2.TransferStatus,
) -> dict[str, Any]:
  """Serializes a TransferStatus proto to a flat database-compatible dict."""
  data = json_format.MessageToDict(status, preserving_proto_field_name=True)
  detail_info = data.pop("detail_info", {})
  res = {**data, **detail_info}
  if "bytes_copied" in res:
    res["bytes_copied"] = int(res["bytes_copied"])
  if "total_bytes" in res:
    res["total_bytes"] = int(res["total_bytes"])
  return res


def deserialize_transfer_status(
    data: dict[str, Any],
) -> tiering_service_pb2.TransferStatus:
  """Deserializes a flat database dict into a TransferStatus proto."""
  known_keys = {
      "request_id",
      "status",
      "client_type",
      "bytes_copied",
      "total_bytes",
      "zone",
  }
  status_data = {k: v for k, v in data.items() if k in known_keys}
  detail_info = {k: v for k, v in data.items() if k not in known_keys}
  status_data["detail_info"] = detail_info

  status_pb = tiering_service_pb2.TransferStatus()
  json_format.ParseDict(status_data, status_pb)
  return status_pb


_CLIENT_BUILDERS: dict[
    tuple[db_schema.BackendType, db_schema.BackendType],
    Callable[
        [db_schema.TierPath, db_schema.TierPath, str | None, str | None],
        GCPStorageClient,
    ],
] = {}


def register_client(
    source_type: db_schema.BackendType,
    target_type: db_schema.BackendType,
):
  """Decorator to register a client builder for a specific backend pair."""
  def decorator(builder_func):
    _CLIENT_BUILDERS[(source_type, target_type)] = builder_func
    return builder_func
  return decorator


@register_client(
    db_schema.BackendType.BACKEND_TYPE_GCS,
    db_schema.BackendType.BACKEND_TYPE_GCS,
)
def _build_gcs_to_gcs(
    source_tp: db_schema.TierPath,
    target_tp: db_schema.TierPath,
    project: str | None,
    service_account: str | None,
) -> GcsToGcsClient:
  """Builds GcsToGcsClient from TierPaths."""
  del source_tp, target_tp  # Unused
  return GcsToGcsClient(project=project, service_account=service_account)


@register_client(
    db_schema.BackendType.BACKEND_TYPE_LUSTRE,
    db_schema.BackendType.BACKEND_TYPE_GCS,
)
def _build_lustre_to_gcs(
    source_tp: db_schema.TierPath,
    target_tp: db_schema.TierPath,
    project: str | None,
    service_account: str | None,
) -> LustreToGcsClient:
  """Builds LustreToGcsClient from TierPaths."""
  del target_tp  # Unused
  lustre_location = source_tp.storage_backend.zone
  if not lustre_location:
    raise ValueError("Lustre zone is missing from storage backend")
  instance = f"lustre-{lustre_location}"
  return LustreToGcsClient(
      instance=instance,
      zone=lustre_location,
      project=project,
      service_account=service_account,
  )


@register_client(
    db_schema.BackendType.BACKEND_TYPE_GCS,
    db_schema.BackendType.BACKEND_TYPE_LUSTRE,
)
def _build_gcs_to_lustre(
    source_tp: db_schema.TierPath,
    target_tp: db_schema.TierPath,
    project: str | None,
    service_account: str | None,
) -> GcsToLustreClient:
  """Builds GcsToLustreClient from TierPaths."""
  del source_tp  # Unused
  lustre_location = target_tp.storage_backend.zone
  if not lustre_location:
    raise ValueError("Lustre zone is missing from storage backend")
  instance = f"lustre-{lustre_location}"
  return GcsToLustreClient(
      instance=instance,
      zone=lustre_location,
      project=project,
      service_account=service_account,
  )


def determine_client(
    source_tp: db_schema.TierPath,
    target_tp: db_schema.TierPath,
    project: str | None = None,
    service_account: str | None = None,
) -> GCPStorageClient:
  """Determines and returns the GCP Storage client based on backends."""
  source_type = source_tp.storage_backend.backend_type
  target_type = target_tp.storage_backend.backend_type
  key = (source_type, target_type)
  if key not in _CLIENT_BUILDERS:
    raise ValueError(
        f"Unsupported backend pair: {source_type} and {target_type}"
    )
  return _CLIENT_BUILDERS[key](source_tp, target_tp, project, service_account)


def get_client_from_status(
    status_dict: dict[str, Any],
    project: str | None = None,
    service_account: str | None = None,
) -> GCPStorageClient:
  """Resolves and instantiates the client based on status metadata."""
  client_type = status_dict.get("client_type")
  if not client_type:
    raise ValueError("Unknown or missing client type in status")

  if client_type == "GcsToGcsClient":
    return GcsToGcsClient(project=project, service_account=service_account)
  elif client_type == "LustreToGcsClient":
    zone = status_dict.get("zone")
    if not zone:
      raise ValueError("zone is missing from transfer_status")
    instance = f"lustre-{zone}"
    return LustreToGcsClient(
        instance=instance,
        zone=zone,
        project=project,
        service_account=service_account,
    )
  elif client_type == "GcsToLustreClient":
    zone = status_dict.get("zone")
    if not zone:
      raise ValueError("zone is missing from transfer_status")
    instance = f"lustre-{zone}"
    return GcsToLustreClient(
        instance=instance,
        zone=zone,
        project=project,
        service_account=service_account,
    )
  else:
    raise ValueError(f"Unknown or missing client type: {client_type}")
