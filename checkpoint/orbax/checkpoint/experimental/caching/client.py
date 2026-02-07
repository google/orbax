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

"""Storage Service Client."""

import time
from absl import logging
import requests

SERVICE_URL = "http://service-dns/"


class StorageServiceClient:
  """Client for interacting with the Storage Tier Service."""

  def __init__(self, service_url: str | None = None):
    self._service_url = service_url or SERVICE_URL

  def resolve(self, execution_id: int, step: int) -> str:
    """Resolves an asset path from the service."""
    start = time.time()
    logging.info("Resolving ID-step: %s-%s.", execution_id, step)
    payload = {"execution_id": execution_id, "step": step}
    response = requests.post(f"{self._service_url}/resolve", json=payload)
    response.raise_for_status()
    result = response.json()["path"]
    end = time.time()
    logging.info("Resolved %s in %s seconds.", result, end - start)
    return result

  def finalize(self, execution_id: int, step: int) -> None:
    """Finalizes an asset in the service."""
    start = time.time()
    payload = {"execution_id": execution_id, "step": step}
    response = requests.post(f"{self._service_url}/finalize", json=payload)
    response.raise_for_status()
    end = time.time()
    logging.info(
        "Finalized %s %s in %s seconds.", execution_id, step, end - start
    )

  def prefetch(self, execution_id: int, step: int) -> None:
    """Prefetches an asset in the service."""
    start = time.time()
    payload = {"execution_id": execution_id, "step": step}
    response = requests.post(f"{self._service_url}/prefetch", json=payload)
    response.raise_for_status()
    end = time.time()
    logging.info(
        "Prefetched %s %s in %s seconds.", execution_id, step, end - start
    )

  def await_transfer(self, execution_id: int, step: int) -> None:
    """Waits for any ongoing transfer for the asset to complete."""
    start = time.time()
    payload = {"execution_id": execution_id, "step": step}
    response = requests.post(
        f"{self._service_url}/await_transfer", json=payload
    )
    response.raise_for_status()
    end = time.time()
    logging.info(
        "Awaited transfer %s %s in %s seconds.", execution_id, step, end - start
    )
