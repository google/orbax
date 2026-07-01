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

"""Checkpoint Tiering Service (CTS) client authentication utilities."""

import asyncio
from absl import logging
import google.auth
import google.auth.transport.requests

_CREDENTIALS = None


async def get_oauth_token() -> str | None:
  """Fetches the Google ADC OAuth 2.0 access token asynchronously with caching."""
  global _CREDENTIALS
  try:
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    if _CREDENTIALS is None:
      # Discover default credentials.
      _CREDENTIALS, _ = await asyncio.to_thread(
          google.auth.default, scopes=scopes
      )

    # If credentials are not valid (e.g. expired), refresh asynchronously.
    if not _CREDENTIALS.valid:
      request = google.auth.transport.requests.Request()
      await asyncio.to_thread(_CREDENTIALS.refresh, request)

    return _CREDENTIALS.token
  except Exception as e:  # pylint: disable=broad-exception-caught

    logging.warning("Failed to retrieve GCP OAuth token: %s", e)
  return None
