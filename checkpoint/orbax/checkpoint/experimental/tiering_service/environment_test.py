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

"""Tests for environment discovery utilities."""

import os
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import httpx
from orbax.checkpoint.experimental.tiering_service import environment


class EnvironmentTest(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    # Clear env variables to ensure tests are hermetic
    self.original_environ = dict(os.environ)
    if "GCP_ZONE" in os.environ:
      del os.environ["GCP_ZONE"]
    if "GCP_REGION" in os.environ:
      del os.environ["GCP_REGION"]

  def tearDown(self):
    os.environ.clear()
    os.environ.update(self.original_environ)
    super().tearDown()

  @mock.patch("os.getlogin")
  def test_get_current_user_os_login(self, mock_login):
    mock_login.return_value = "env-user"
    self.assertEqual(environment.get_current_user(), "env-user")

  @mock.patch("os.getlogin")
  @mock.patch("getpass.getuser")
  def test_get_current_user_fallback(self, mock_getuser, mock_login):
    mock_login.side_effect = OSError("No tty")
    mock_getuser.return_value = "fallback-user"
    self.assertEqual(environment.get_current_user(), "fallback-user")

  async def test_get_gcp_zone_from_env(self):
    os.environ["GCP_ZONE"] = "us-east5-a"
    zone = await environment.get_gcp_zone()
    self.assertEqual(zone, "us-east5-a")

  @mock.patch("httpx.AsyncClient.get")
  async def test_get_gcp_zone_from_metadata(self, mock_get):
    mock_response = mock.MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.text = "projects/12345/zones/us-central1-c"

    async def mock_get_fn(*unused_args, **unused_kwargs):
      return mock_response

    mock_get.side_effect = mock_get_fn

    zone = await environment.get_gcp_zone()
    self.assertEqual(zone, "us-central1-c")

  @mock.patch("httpx.AsyncClient.get")
  async def test_get_gcp_zone_metadata_server_unavailable(self, mock_get):
    async def mock_get_fn(*unused_args, **unused_kwargs):
      raise httpx.ConnectError("Network unreachable")

    mock_get.side_effect = mock_get_fn

    zone = await environment.get_gcp_zone()
    self.assertIsNone(zone)

  async def test_get_gcp_region_from_env(self):
    os.environ["GCP_REGION"] = "us-east5"
    region = await environment.get_gcp_region()
    self.assertEqual(region, "us-east5")

  @mock.patch("httpx.AsyncClient.get")
  async def test_get_gcp_region_derived_from_metadata(self, mock_get):
    mock_response = mock.MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.text = "projects/12345/zones/us-central1-c"

    async def mock_get_fn(*unused_args, **unused_kwargs):
      return mock_response

    mock_get.side_effect = mock_get_fn

    region = await environment.get_gcp_region()
    self.assertEqual(region, "us-central1")

  @mock.patch("httpx.AsyncClient.get")
  async def test_get_gcp_region_metadata_server_unavailable(self, mock_get):
    async def mock_get_fn(*unused_args, **unused_kwargs):
      raise httpx.ConnectError("Network unreachable")

    mock_get.side_effect = mock_get_fn

    region = await environment.get_gcp_region()
    self.assertIsNone(region)


if __name__ == "__main__":
  absltest.main()
