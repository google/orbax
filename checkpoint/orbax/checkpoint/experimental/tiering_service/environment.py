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

"""Checkpoint Tiering Service (CTS) client environment discovery utilities."""

import getpass
import os
from absl import logging
import httpx

_METADATA_URL = "http://metadata.google.internal/computeMetadata/v1"
_HEADERS = {"Metadata-Flavor": "Google"}


def get_current_user() -> str:
  """Returns the username of the current user."""
  try:
    return os.getlogin()
  except OSError:
    return getpass.getuser()


async def get_gcp_zone() -> str | None:
  """Gets the current zone from environment variable or GCP metadata server.

  Returns:
    The zone name (e.g., 'us-east5-a'), or None if not found/unavailable.
  """
  zone = os.environ.get("GCP_ZONE")
  if zone:
    return zone
  try:
    async with httpx.AsyncClient(timeout=2.0) as client:
      response = await client.get(
          f"{_METADATA_URL}/instance/zone", headers=_HEADERS
      )
      if response.status_code == 200:
        # The metadata server returns 'projects/<num>/zones/<zone>'
        zone_path = response.text.strip()
        return zone_path.split("/")[-1]
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning("Failed to query GCP metadata server for zone: %s", e)
  return None


async def get_gcp_region() -> str | None:
  """Gets the current region from environment variable or derived from zone.

  Returns:
    The region name (e.g., 'us-east5'), or None if not found.
  """
  region = os.environ.get("GCP_REGION")
  if region:
    return region
  zone = await get_gcp_zone()
  if zone:
    parts = zone.split("-")
    if len(parts) >= 2:
      return "-".join(parts[:-1])
  return None
