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

"""Integration tests for GCP Storage Clients.

These tests run against GCP Managed Lustre or Parallelstore and GCS buckets.
To execute these tests, set the following environment variables:

  CTS_GCP_PROJECT: The GCP project ID.
  CTS_LUSTRE_LOCATION: The zone of the Lustre cluster (e.g. "us-east5-a").
  CTS_LUSTRE_INSTANCE: The Lustre instance ID.
  CTS_TEST_GCS_BUCKET1: The first GCS bucket (e.g. "gs://my-bucket1").
  CTS_TEST_GCS_BUCKET2: The second GCS bucket (e.g. "gs://my-bucket2").
  CTS_PARALLELSTORE_LOCATION: The zone of the Parallelstore instance (if
  different).
  CTS_PARALLELSTORE_INSTANCE: The Parallelstore instance ID (if different).

If these variables are not set, the tests are skipped automatically.
"""

import asyncio
import base64
import os
import time
import unittest
import uuid

from absl import logging
import google_crc32c
from orbax.checkpoint.experimental.tiering_service import gcp_storage_client
from orbax.checkpoint.experimental.tiering_service import job_worker


class RandomStream:
  """A file-like object generating random bytes up to a specified total size."""

  def __init__(self, total_size: int, chunk_size: int = 8 * 1024 * 1024):
    self.google_crc32c = google_crc32c
    self.total_size = total_size
    self.chunk_size = chunk_size
    self.bytes_read = 0
    self.checksum = self.google_crc32c.Checksum()

  def read(self, amt: int = -1) -> bytes:
    if self.bytes_read >= self.total_size:
      return b""
    if amt is None or amt == -1 or amt > self.chunk_size:
      amt = self.chunk_size
    remaining = self.total_size - self.bytes_read
    if amt > remaining:
      amt = remaining
    data = os.urandom(amt)
    self.bytes_read += len(data)
    self.checksum.update(data)
    return data

  def seek(self, offset, whence=0):
    if offset == 0 and whence == 0:
      self.bytes_read = 0
      self.checksum = self.google_crc32c.Checksum()
    else:
      raise NotImplementedError("Only seek(0, 0) is supported.")

  def tell(self):
    return self.bytes_read


class GcpLustreClientIntegrationTest(unittest.IsolatedAsyncioTestCase):

  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.project = os.environ.get("CTS_GCP_PROJECT")
    self.location = os.environ.get("CTS_LUSTRE_LOCATION")
    self.instance = os.environ.get("CTS_LUSTRE_INSTANCE")
    self.gcs_bucket = os.environ.get("CTS_TEST_GCS_BUCKET1")

    if not all([self.location, self.instance, self.gcs_bucket]):
      self.skipTest(
          "GcpLustreClientIntegrationTest skipped. Please set "
          "CTS_LUSTRE_LOCATION, CTS_LUSTRE_INSTANCE, and "
          "CTS_TEST_GCS_BUCKET1 environment variables to run integration tests."
      )

    assert self.location is not None
    assert self.instance is not None
    assert self.gcs_bucket is not None

    self.import_client = job_worker.GcsToLustreClient(
        project=self.project,
        location=self.location,
        instance=self.instance,
    )
    self.export_client = job_worker.LustreToGcsClient(
        project=self.project,
        location=self.location,
        instance=self.instance,
    )

  async def test_import_and_export_lifecycle(self):
    assert self.gcs_bucket is not None
    assert self.location is not None
    assert self.instance is not None
    test_id = str(uuid.uuid4())
    source_gcs_dir = (
        f"{self.gcs_bucket.rstrip('/')}/cts_integration_test_{test_id}/"
    )
    lustre_dir = f"/testfs/cts_integration_test_{test_id}/"
    destination_gcs_dir = (
        f"{self.gcs_bucket.rstrip('/')}/cts_integration_test_export_{test_id}/"
    )

    source_gcs_path = source_gcs_dir + "data.bin"
    destination_gcs_path = destination_gcs_dir + "data.bin"

    logging.info("Source GCS Path: %s", source_gcs_path)
    logging.info("Lustre Dir: %s", lustre_dir)
    logging.info("Destination GCS Path: %s", destination_gcs_path)

    # 1. Create a 1GB dummy file in GCS
    from google.cloud import storage  # pylint: disable=g-import-not-at-top

    storage_client = storage.Client(project=self.project)
    bucket_name = self.gcs_bucket.replace("gs://", "").split("/")[0]
    bucket = storage_client.bucket(bucket_name)

    blob_name = source_gcs_path.replace(f"gs://{bucket_name}/", "")
    blob = bucket.blob(blob_name)

    total_size = 1024 * 1024 * 1024  # 1GB
    total_size_mb = total_size / (1024 * 1024)
    stream = RandomStream(total_size)

    logging.info("Uploading 1GB file to GCS...")
    start_time = time.perf_counter()
    await asyncio.to_thread(blob.upload_from_file, stream, size=total_size)
    upload_duration = time.perf_counter() - start_time
    upload_bw = total_size_mb / upload_duration
    logging.info(
        "GCS Upload finished: %.2f seconds (%.2f MB/s)",
        upload_duration,
        upload_bw,
    )
    uploaded_crc = base64.b64encode(stream.checksum.digest()).decode("utf-8")

    export_blob_name = destination_gcs_path.replace(f"gs://{bucket_name}/", "")
    export_blob = bucket.blob(export_blob_name)

    try:
      # 2. Trigger import to Lustre
      logging.info("Triggering import to Lustre...")
      start_time = time.perf_counter()
      import_op = await self.import_client.trigger_copy(
          request_id=str(uuid.uuid4()),
          source_path=source_gcs_dir,
          destination_path=lustre_dir,
      )
      logging.info("Import operation triggered: %s", import_op)

      # 3. Poll import operation
      logging.info("Polling import...")
      while True:
        result = await self.import_client.poll_operation(import_op)
        logging.info("Import LRO %s status: %s", import_op, result.status)
        if result.status == gcp_storage_client.OperationStatus.SUCCESS:
          break
        elif result.status == gcp_storage_client.OperationStatus.FAILED:
          self.fail(
              f"Import operation failed: {result.detail_info.get('error')}"
          )
        await asyncio.sleep(5)
      import_duration = time.perf_counter() - start_time
      import_bw = total_size_mb / import_duration
      logging.info(
          "Lustre Import finished successfully: %.2f seconds (%.2f MB/s)",
          import_duration,
          import_bw,
      )

      # 4. Trigger export back to GCS
      logging.info("Triggering export to GCS...")
      start_time = time.perf_counter()
      export_op = await self.export_client.trigger_copy(
          request_id=str(uuid.uuid4()),
          source_path=lustre_dir,
          destination_path=destination_gcs_dir,
      )
      logging.info("Export operation triggered: %s", export_op)

      # 5. Poll export operation
      logging.info("Polling export...")
      while True:
        result = await self.export_client.poll_operation(export_op)
        logging.info("Export LRO %s status: %s", export_op, result.status)
        if result.status == gcp_storage_client.OperationStatus.SUCCESS:
          break
        elif result.status == gcp_storage_client.OperationStatus.FAILED:
          self.fail(
              f"Export operation failed: {result.detail_info.get('error')}"
          )
        await asyncio.sleep(5)
      export_duration = time.perf_counter() - start_time
      export_bw = total_size_mb / export_duration
      logging.info(
          "Lustre Export finished successfully: %.2f seconds (%.2f MB/s)",
          export_duration,
          export_bw,
      )

      # 6. Verify that the file was exported and matches the original
      logging.info("Verifying file integrity...")
      await asyncio.to_thread(export_blob.reload)
      exported_crc = export_blob.crc32c
      self.assertEqual(uploaded_crc, exported_crc)
      logging.info(
          "Verification succeeded! Exported file CRC32C matches original (%s)",
          exported_crc,
      )

    finally:
      # Cleanup GCS blobs
      try:
        await asyncio.to_thread(blob.delete)
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      try:
        await asyncio.to_thread(export_blob.delete)
      except Exception:  # pylint: disable=broad-exception-caught
        pass


class GcsToGcsClientIntegrationTest(unittest.IsolatedAsyncioTestCase):

  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.project = os.environ.get("CTS_GCP_PROJECT")
    self.gcs_bucket1 = os.environ.get("CTS_TEST_GCS_BUCKET1")
    self.gcs_bucket2 = os.environ.get("CTS_TEST_GCS_BUCKET2")

    if not all([self.project, self.gcs_bucket1, self.gcs_bucket2]):
      self.skipTest(
          "GcsToGcsClientIntegrationTest skipped. Please set "
          "CTS_GCP_PROJECT, "
          "CTS_TEST_GCS_BUCKET1, and CTS_TEST_GCS_BUCKET2 env vars."
      )

    assert self.gcs_bucket1 is not None
    assert self.gcs_bucket2 is not None

    self.client = gcp_storage_client.GcsToGcsClient(
        project=self.project,
    )

  async def test_gcs_to_gcs_lifecycle(self):
    assert self.gcs_bucket1 is not None
    assert self.gcs_bucket2 is not None
    test_id = str(uuid.uuid4())

    source_gcs_dir = (
        f"{self.gcs_bucket1.rstrip('/')}/cts_gcs_to_gcs_test_{test_id}/"
    )
    destination_gcs_dir = (
        f"{self.gcs_bucket2.rstrip('/')}/cts_gcs_to_gcs_test_export_{test_id}/"
    )

    source_gcs_path = source_gcs_dir + "data.bin"
    destination_gcs_path = destination_gcs_dir + "data.bin"

    logging.info("Source GCS Path: %s", source_gcs_path)
    logging.info("Destination GCS Path: %s", destination_gcs_path)

    # 1. Create a 100MB dummy file in GCS bucket 1
    from google.cloud import storage  # pylint: disable=g-import-not-at-top

    storage_client = storage.Client(project=self.project)
    bucket1_name = self.gcs_bucket1.replace("gs://", "").split("/")[0]
    bucket1 = storage_client.bucket(bucket1_name)
    blob_name = source_gcs_path.replace(f"gs://{bucket1_name}/", "")
    blob = bucket1.blob(blob_name)

    total_size = 100 * 1024 * 1024  # 100MB
    stream = RandomStream(total_size)

    logging.info("Uploading 100MB file to GCS Bucket 1...")
    await asyncio.to_thread(blob.upload_from_file, stream, size=total_size)
    uploaded_crc = base64.b64encode(stream.checksum.digest()).decode("utf-8")

    bucket2_name = self.gcs_bucket2.replace("gs://", "").split("/")[0]
    bucket2 = storage_client.bucket(bucket2_name)
    export_blob_name = destination_gcs_path.replace(f"gs://{bucket2_name}/", "")
    export_blob = bucket2.blob(export_blob_name)

    try:
      # 2. Trigger GCS-to-GCS transfer
      logging.info("Triggering GCS-to-GCS transfer...")
      start_time = time.perf_counter()
      job_request_id = str(uuid.uuid4())

      import_op = await self.client.trigger_copy(
          request_id=job_request_id,
          source_path=source_gcs_dir,
          destination_path=destination_gcs_dir,
      )
      logging.info("GCS-to-GCS transfer triggered, import LRO: %s", import_op)

      # 3. Poll LROs.
      req_id = import_op
      transfer_status = {"step": "import"}

      logging.info("Polling LROs...")
      while True:
        context = gcp_storage_client.TransferContext(
            job_request_id=job_request_id,
            source_path=source_gcs_dir,
            destination_path=destination_gcs_dir,
            transfer_status=transfer_status,
        )
        result = await self.client.poll_operation(req_id, context=context)
        logging.info(
            "Poll result: status=%s, detail_info=%s",
            result.status,
            result.detail_info,
        )

        if result.status == gcp_storage_client.OperationStatus.IN_PROGRESS:
          if "request_id" in result.detail_info:
            req_id = result.detail_info["request_id"]
          transfer_status = result.detail_info

        if result.status == gcp_storage_client.OperationStatus.SUCCESS:
          break
        elif result.status == gcp_storage_client.OperationStatus.FAILED:
          self.fail(f"Transfer failed: {result.detail_info.get('error')}")

        await asyncio.sleep(5)

      transfer_duration = time.perf_counter() - start_time
      logging.info(
          "GCS-to-GCS transfer finished successfully: %.2f seconds",
          transfer_duration,
      )

      # 4. Verify file in GCS bucket 2
      logging.info("Verifying file integrity in Bucket 2...")
      await asyncio.to_thread(export_blob.reload)
      exported_crc = export_blob.crc32c
      self.assertEqual(uploaded_crc, exported_crc)
      logging.info("Verification succeeded!")

    finally:
      # Cleanup GCS blobs
      try:
        await asyncio.to_thread(blob.delete)
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      try:
        await asyncio.to_thread(export_blob.delete)
      except Exception:  # pylint: disable=broad-exception-caught
        pass


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  logging.use_absl_handler()
  unittest.main()
