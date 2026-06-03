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

"""Authentication and permission verification for Tiering Service.

This module handles extracting OAuth tokens from gRPC metadata and verifying
user permissions on various storage backends (e.g. GCS, Lustre).
"""

from collections.abc import Collection
from absl import logging
import grpc
from orbax.checkpoint.experimental.tiering_service import db_schema

_BEARER_PREFIX = "Bearer "


async def get_oauth_token(context: grpc.aio.ServicerContext) -> str | None:
  """Extracts OAuth token from gRPC metadata.

  Looks for the 'authorization' header in the gRPC invocation metadata
  and extracts the bearer token if present.

  Args:
    context: The gRPC servicer context containing invocation metadata.

  Returns:
    The extracted OAuth token string, or None if not found or malformed.
  """
  logging.debug("Extracting OAuth token from metadata")
  metadata = dict(await context.invocation_metadata())
  # Standard header for OAuth tokens in gRPC is 'authorization'.
  auth_header = metadata.get("authorization")

  if auth_header is None:
    logging.warning("No authorization header found")
    return None

  if not auth_header.startswith(_BEARER_PREFIX):
    logging.warning("Malformed authorization header found.")
    return None

  logging.debug("Found authorization header.")
  return auth_header[len(_BEARER_PREFIX) :]


async def verify_gcs_permissions(
    token: str | None, path: str, permissions: Collection[str]
) -> bool:
  """Verifies if the caller has necessary permissions on GCS.

  Args:
    token: The OAuth token of the caller.
    path: The GCS path to check permissions for.
    permissions: A collection of permission strings to check.

  Returns:
    True if the permissions are verified, False otherwise.
  """
  logging.info("Verifying GCS permissions for path: %s", path)
  logging.debug("Requested permissions: %s", permissions)
  # TODO: b/503445654 - Implement actual IAM permission verification
  # For now, return True if a token is provided, and False otherwise,
  # to allow testing PERMISSION_DENIED errors.
  logging.debug("Permission check result: %s", token is not None)
  return token is not None


async def verify_lustre_permissions(token: str | None, path: str) -> bool:
  """Verifies if the caller has necessary permissions on Lustre.

  Args:
    token: The OAuth token of the caller.
    path: The Lustre path to check permissions for.

  Returns:
    True if the permissions are verified, False otherwise.
  """
  logging.info("Verifying Lustre permissions for path: %s", path)
  # TODO: b/503445654 - Implement actual Lustre permission verification
  # For now, return True if a token is provided, and False otherwise.
  return token is not None


async def has_read_permission(
    token: str | None,
    *,
    backend: db_schema.StorageBackend,
    path: str,
) -> bool:
  """Checks whether bearer token possesses permission scopes for storage read.

  Args:
    token: The OAuth token of the caller.
    backend: The StorageBackend target.
    path: The destination path on the backend.

  Returns:
    True if read permission is granted, False otherwise.
  """
  if backend.backend_type == db_schema.BackendType.BACKEND_TYPE_GCS:
    return await verify_gcs_permissions(token, path, ["storage.objects.get"])
  elif backend.backend_type == db_schema.BackendType.BACKEND_TYPE_LUSTRE:
    return await verify_lustre_permissions(token, path)
  else:
    logging.warning("Unknown backend type: %s", backend.backend_type)
    return False


async def has_write_permission(
    token: str | None,
    *,
    backend: db_schema.StorageBackend,
    path: str,
) -> bool:
  """Checks whether bearer token possesses permission scopes for storage write.

  Delegates the check to backend-specific verification functions based on the
  backend type.

  Args:
    token: The OAuth token of the caller.
    backend: The StorageBackend target.
    path: The destination path on the backend.

  Returns:
    True if write permission is granted, False otherwise.
  """
  if backend.backend_type == db_schema.BackendType.BACKEND_TYPE_GCS:
    return await verify_gcs_permissions(token, path, ["storage.objects.create"])
  elif backend.backend_type == db_schema.BackendType.BACKEND_TYPE_LUSTRE:
    return await verify_lustre_permissions(token, path)
  else:
    logging.warning("Unknown backend type: %s", backend.backend_type)
    return False
