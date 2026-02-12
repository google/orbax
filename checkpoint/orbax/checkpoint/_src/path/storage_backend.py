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

"""Checkpoint storage implementations."""

import abc
import dataclasses
import datetime
import enum
import os
from typing import Iterable

from absl import logging
import immutabledict
from orbax.checkpoint._src.path import atomicity
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint.google.path import cns2_atomicity
from orbax.checkpoint.google.path import cns2_utils
from orbax.checkpoint.google.path import tfhub_atomicity

from .learning.brain.contrib.hub.public.proto import metadata_pb2
from .learning.brain.contrib.hub.public.proto import options_pb2
from .learning.brain.contrib.hub.public.proto import realm_pb2
from .learning.brain.contrib.hub.public.python import client as client_lib
from .learning.brain.contrib.hub.public.python import handle as handle_lib
from .learning.deepmind.phoenix.v2 import fs_utils
from .pyglib import gfile
from .pyglib.contrib.gpathlib import gpath
from .util.task.python import error


Enum = enum.Enum

# Prefix for all step checkpoint tags with payload.
# Using ':' as a separator for consistency with existing tags.
_STEP_CHECKPOINT_TAG_PREFIX = 'step_checkpoint:'

_TFHUB_PATH_BY_REALM = immutabledict.immutabledict({
    realm_pb2.PROD_REALM: 'prod',
    realm_pb2.QUAL_REALM: 'qual',
    realm_pb2.ISOLATED_REALM: 'isolated',
})
_TFHUB_REALM_BY_PATH = immutabledict.immutabledict(
    {v: k for k, v in _TFHUB_PATH_BY_REALM.items()}
)


def get_tfhub_data_realm_from_path(path: gpath.GPath) -> realm_pb2.Realm:
  if path.parts[1] != 'tfhub':
    raise ValueError(f'Invalid TFHub path: {path!r}')
  realm_str = path.parts[2]
  realm = _TFHUB_REALM_BY_PATH.get(realm_str)
  if realm is None:
    raise ValueError(f'Unsupported TFHub realm in path: {realm_str!r}')
  return realm


@dataclasses.dataclass(frozen=True)
class CheckpointPathMetadata:
  """Internal representation of checkpoint path metadata.

  Attributes:
    path: The file system path of the checkpoint.
    status: The status of the checkpoint.
    version: The version of the checkpoint with an index and step number. (e.g.
      '1.step_1')
    tags: A list of tags associated with the checkpoint. Currently only
      supported for TFHub paths, for other paths this field will be `None`.
  """

  class Status(Enum):
    COMMITTED = 1
    UNCOMMITTED = 2

  path: str
  status: Status
  version: str | None
  tags: set[str] | None = None


@dataclasses.dataclass(frozen=True)
class CheckpointFilter:
  """Criteria for filtering checkpoints.

  TODO: b/466312058 This class will contain fields for filtering checkpoints by
  various criteria.
  """


@dataclasses.dataclass(frozen=True)
class CheckpointReadOptions:
  """Options for reading checkpoints.

  Attributes:
    filter: Optional filter criteria for selecting checkpoints.
    enable_strong_reads: If True, enables strong read consistency when querying
      checkpoints. This may have performance implications but ensures the most
      up-to-date results.
  """

  filter: CheckpointFilter | None = None
  enable_strong_reads: bool = False


def _gfile_delete_checkpoint(checkpoint_path: str | gpath.GPath):
  """Deletes a checkpoint using gfile.DeleteRecursively with error handling."""
  try:
    gfile.DeleteRecursively(checkpoint_path)
    logging.info('Removed old checkpoint (%s)', checkpoint_path)
  except gfile.GOSError:
    logging.exception('Failed to remove checkpoint (%s)', checkpoint_path)


# TODO(b/425293362): Implement this interface for Colossus, GCS and TFHub.
class StorageBackend(abc.ABC):
  """An abstract base class for a storage backend.

  This class defines a common interface for managing checkpoint paths in
  different file systems.
  """

  @abc.abstractmethod
  def list_checkpoints(self, base_path: str) -> list[CheckpointPathMetadata]:
    """Lists checkpoints for a given base path and version pattern."""
    raise NotImplementedError('Subclasses must provide implementation')

  @abc.abstractmethod
  def get_temporary_path_class(self) -> type[atomicity_types.TemporaryPath]:
    """Returns a TemporaryPath class for the storage backend."""
    raise NotImplementedError('Subclasses must provide implementation')

  @abc.abstractmethod
  def delete_checkpoint(self, checkpoint_path: str | gpath.GPath) -> None:
    """Deletes a checkpoint from the storage backend."""
    raise NotImplementedError('Subclasses must provide implementation')


class TFHubCheckpointTags:
  """Helper class to manage TF Hub checkpoint metadata tags."""

  class TagType(Enum):
    """Enum class for supported TF Hub checkpoint metadata tag types."""

    SERIES = 'series'

  _TAG_TYPES_BY_VALUE = immutabledict.immutabledict(
      {tag_type.value: tag_type for tag_type in TagType}
  )

  @classmethod
  def parse_tags(
      cls,
      tags: Iterable[str],
  ) -> set[tuple[TagType, str]]:
    """Parses TFHub tags into (TagType, value) tuples."""
    parsed_tags = set()
    for tag in tags:
      if tag.startswith(_STEP_CHECKPOINT_TAG_PREFIX):
        tag_suffix = tag[len(_STEP_CHECKPOINT_TAG_PREFIX) :]
        parts = tag_suffix.split('=', 1)
        if len(parts) == 2:
          tag_type_str, value = parts
          if tag_type_str in cls._TAG_TYPES_BY_VALUE:
            parsed_tags.add((cls._TAG_TYPES_BY_VALUE[tag_type_str], value))
    return parsed_tags

  def __init__(self):
    self._metadata: dict[self.TagType, list[str]] = {}

  def add(self, key: TagType, value: str) -> None:
    """Adds a key-value tag pair to the checkpoint metadata."""
    if key not in self._metadata:
      self._metadata[key] = []
    if value not in self._metadata[key]:
      self._metadata[key].append(value)
    else:
      logging.warning(
          'Value %s for key %s already exists in metadata. Skipping.',
          value,
          key,
      )

  def _format_tag_value(self, tag_type: TagType, value: str) -> str:
    """Formats a tag type and value into a string for TF Hub."""
    return f'{_STEP_CHECKPOINT_TAG_PREFIX}{tag_type.value}={value}'

  def as_tag_protos(self) -> list[metadata_pb2.TagProto]:
    """Returns a list of tags for TF Hub."""
    tags = []
    for tag_type, values in self._metadata.items():
      for value in values:
        tags.append(
            metadata_pb2.TagProto(value=self._format_tag_value(tag_type, value))
        )
    return tags


class TFHubStorageBackend(StorageBackend):
  """A StorageBackend implementation for TFHub checkpoints."""

  _TFHUB_VERSION_LABEL_PREFIX = 'v='
  _TFHUB_SNAPSHOT_DEFAULT_TTL = 5

  def __init__(self):
    super().__init__()
    self._realm: realm_pb2.Realm | None = None
    self._client: client_lib.Client | None = None

  def get_temporary_path_class(self) -> type[atomicity_types.TemporaryPath]:
    """Returns the TemporaryPath class for TFHub."""
    return tfhub_atomicity.TFHubTemporaryPath

  @classmethod
  def is_prod_tfhub_path(cls, path: str) -> bool:
    """Returns whether the given path is a prod TFHub path.

    TODO: b/425293362 - Consider moving this to fs_utils so that we do not have
    to expose the storage backend method to high level libraries such as
    Gemax Scale.

    Args:
      path: The TFHub path string (e.g., '/tfhub/prod/publisher/model').
    """
    if not fs_utils.is_tfhub_path(path):
      return False
    return path.split('/')[2] == _TFHUB_PATH_BY_REALM[realm_pb2.PROD_REALM]

  @classmethod
  def model_and_label_from_cvl_path(
      cls,
      path: str,
  ) -> tuple[str, str]:
    """Extracts the label from a path string that ends with 'v=<version>'.

    Example:
      path_string: '/tfhub/prod/publisher/model/v=1.step_1'
      returns: ('/tfhub/prod/publisher/model', '1.step_1')

    Args:
        path: The input string path.

    Returns:
        The extracted version string, or None if 'v=' is not found.

    Raises:
        ValueError: If the path does not contain a CVL (version label with '=').
    """
    base_path = os.path.dirname(path)
    label = os.path.basename(path)
    if not label.startswith(cls._TFHUB_VERSION_LABEL_PREFIX):
      raise ValueError(
          'Path does not contain a proper CVL (version label with'
          f' "{cls._TFHUB_VERSION_LABEL_PREFIX}"): {path}'
      )
    return base_path, label[len(cls._TFHUB_VERSION_LABEL_PREFIX) :]

  def _get_client(self, path: str) -> client_lib.Client:
    """Gets or creates a TFHub client for the given path's realm.

    The client is cached after the first creation. Subsequent calls must be
    for the same TFHub realm.

    Args:
      path: The TFHub path string (e.g., '/tfhub/prod/publisher/model').

    Returns:
      A `client_lib.Client` instance.

    Raises:
      ValueError: If the path belongs to a different realm than the one
        for which the client was already created.
    """
    realm = get_tfhub_data_realm_from_path(gpath.GPath(path))
    if self._client is None:
      logging.info('Creating TFHub client for realm: %s', realm)
      self._realm = realm
      self._client = client_lib.Client.create(realm)
    elif self._realm != realm:
      raise ValueError(
          'TFHub client is cached for realm'
          f' {realm_pb2.Realm.Name(self._realm)}, but got path in realm'
          f' {realm_pb2.Realm.Name(realm)}. Mixing realms is not supported.'
      )
    return self._client

  def _parse_handle_from_path(
      self,
      base_path: str,
  ) -> handle_lib.UnversionedHandle | None:
    """Attempts to parse a TFHub handle from the given `base_path`.

    This method tries to parse `base_path` as an `UnversionedHandle`.
    Non-unversioned paths are ignored and result in `None` being returned
    because `list_checkpoints` expects a base path representing a collection
    of versions, not a specific version. Returning `None` allows
    `list_checkpoints` to gracefully handle paths that don't conform to the
    expected unversioned handle format by returning an empty list.

    Args:
      base_path: The TFHub path string (e.g., '/tfhub/prod/publisher/model').

    Returns:
      A `handle_lib.UnversionedHandle` if parsing is successful, otherwise None.
    """
    try:
      return handle_lib.UnversionedHandle.from_gfile_filepath(base_path)
    except error.StatusNotOk:
      logging.warning(
          'Failed to parse base path as an unversioned handle: %s',
          base_path,
      )
      return None

  def _build_list_checkpoint_options(
      self,
      checkpoint_read_options: CheckpointReadOptions | None,
  ) -> options_pb2.ListAssetsOptionsProto:
    """Builds the ListAssetsOptionsProto for listing checkpoints.

    Args:
      checkpoint_read_options: The options to use for reading checkpoints.

    Returns:
      A configured ListAssetsOptionsProto.
    """
    options = options_pb2.ListAssetsOptionsProto()
    if checkpoint_read_options and checkpoint_read_options.enable_strong_reads:
      options.read_options.CopyFrom(
          options_pb2.ReadOptionsProto(
              max_staleness=datetime.timedelta(seconds=0)
          )
      )
    return options

  def _build_list_checkpoint_query(
      self,
      handle: handle_lib.UnversionedHandle,
  ) -> str:
    """Builds the query string for listing checkpoints.

    Args:
      handle: The unversioned TFHub handle to query.

    Returns:
      A query string for listing assets.
    """
    return f'handle:{handle} status:ACTIVE'

  def list_checkpoints(
      self,
      base_path: str,
      checkpoint_read_options: CheckpointReadOptions | None = None,
  ) -> list[CheckpointPathMetadata]:
    """Lists TFHub assets for a given publisher and handle.

    Note: This method only returns assets with ACTIVE status.

    Args:
      base_path: The base path for the TFHub assets, expected to start with
        '/tfhub/prod/'.
      checkpoint_read_options: The options to use for reading the checkpoints.

    Returns:
      A list of ModelMetadataProto objects matching the query.
    """
    handle = self._parse_handle_from_path(base_path)
    if handle is None:
      return []

    options = self._build_list_checkpoint_options(checkpoint_read_options)
    query = self._build_list_checkpoint_query(handle)
    client = self._get_client(base_path)
    realm = get_tfhub_data_realm_from_path(gpath.GPath(base_path))

    assets: list[CheckpointPathMetadata] = []
    while True:
      results = client.list_assets(options, query=query)
      options.continuation_token = results.continuation_token
      for model in results.models:
        if checkpoint_path_metadata := self._model_metadata_to_checkpoint_path(
            model, realm
        ):
          assets.append(checkpoint_path_metadata)
      if not results.continuation_token:
        break
    return assets

  def resolve_label_to_versioned_handle(
      self,
      path: str,
      label: str,
  ) -> handle_lib.Handle:
    """Resolves a version label to a versioned TFHub handle."""
    try:
      unversioned_handle = handle_lib.UnversionedHandle.from_gfile_filepath(
          path
      )
      return self._get_client(path).resolve_version_label(
          options_pb2.ResolveVersionLabelOptionsProto(),
          unversioned_handle,
          label,
      )
    except Exception as exc:
      raise ValueError(
          f'Failed to resolve version label: {label} for path: {path}'
      ) from exc

  def _hard_delete_asset(self, *, path: str, label: str) -> None:
    """Performs a hard delete of the given TFHub asset."""
    versioned_handle = self.resolve_label_to_versioned_handle(path, label)
    try:
      logging.info('Deleting asset with versioned handle: %s', versioned_handle)
      self._get_client(path).delete_asset(
          options_pb2.DeleteOptionsProto(
              deletion_strategy=options_pb2.DELETION_STRATEGY_HARD_DELETE
          ),
          versioned_handle,
      )
    except Exception as exc:
      raise ValueError(
          f'Failed to delete asset wit handle: {versioned_handle}'
      ) from exc

  def delete_checkpoint(self, checkpoint_path: str | gpath.GPath) -> None:
    """Deletes the TFHub asset corresponding to the given checkpoint path.

    The path is expected to contain a version label (e.g., 'v=<label>'). This
    method resolves the version label and then performs a hard delete of the
    specific asset version from TFHub.

    Args:
      checkpoint_path: The TFHub path of the checkpoint to delete.
    """
    if isinstance(checkpoint_path, gpath.GPath):
      checkpoint_path = str(checkpoint_path)
    path, label = self.model_and_label_from_cvl_path(checkpoint_path)
    self._hard_delete_asset(path=path, label=label)

  def create_snapshot(
      self,
      *,
      source_path: str,
      destination_path: str,
      ttl_days: int = _TFHUB_SNAPSHOT_DEFAULT_TTL,
  ) -> None:
    """Creates a TFHub snapshot from source to destination path.

    This method creates a new TFHub model at the destination path by copying
    the source model's metadata (alloc_config and permissions) and linking to
    the source's asset data. The snapshot has a configurable TTL.

    Args:
      source_path: Source TFHub path with CVL (e.g.,
        '/tfhub/prod/publisher/model/v=1.step_1').
      destination_path: Destination TFHub path with CVL (e.g.,
        '/tfhub/prod/publisher/snapshot/v=1.step_1').
      ttl_days: Time-to-live in days for the snapshot.

    Raises:
      ValueError: If source or destination paths do not contain a CVL.
    """
    if self._TFHUB_VERSION_LABEL_PREFIX not in gpath.GPath(source_path).name:
      raise ValueError(
          f'Source path must contain a CVL (version label): {source_path}'
      )
    if (
        self._TFHUB_VERSION_LABEL_PREFIX
        not in gpath.GPath(destination_path).name
    ):
      raise ValueError(
          'Destination path must contain a CVL (version label):'
          f' {destination_path}'
      )

    src_path, src_label = self.model_and_label_from_cvl_path(source_path)
    src_handle = self.resolve_label_to_versioned_handle(src_path, src_label)
    dst_path, dst_label = self.model_and_label_from_cvl_path(destination_path)

    tfhub_client = self._get_client(source_path)
    src_metadata = tfhub_client.get_model_metadata(
        options_pb2.OptionsProto(), src_handle
    )

    descriptor = metadata_pb2.ModelDescriptorProto(
        asset_descriptor=metadata_pb2.AssetDescriptorProto(
            labels=[
                metadata_pb2.LabelProto(
                    value=dst_label,
                    type=metadata_pb2.LabelType.LABEL_TYPE_CONSTANT,
                )
            ],
            documentation=metadata_pb2.AssetDocumentationProto(
                description=(
                    'Phoenix programmatically created TFHub snapshot of'
                    f' {source_path}.'
                )
            ),
            alloc_config=src_metadata.model_descriptor.asset_descriptor.alloc_config,
            permissions=src_metadata.model_descriptor.asset_descriptor.permissions,
            lifecycle=metadata_pb2.LifecycleProto(
                policy=metadata_pb2.LifecyclePolicyProto(
                    ttl_days=ttl_days,
                    # TODO(dicentra,mxberlot): Figure out if we might still need
                    # notifications for some snapshots.
                    notification_options=metadata_pb2.LifecycleNotificationOptionsProto(
                        notify_days_before=-1
                    ),
                )
            ),
        )
    )
    tfhub_client.create_model_and_assign_version(
        options_pb2.CreateOptionsProto(),
        handle_lib.UnversionedHandle.from_gfile_filepath(dst_path),
        descriptor,
        src_handle.to_gfile_filepath(self._realm),
    )

  def _model_metadata_to_checkpoint_path(
      self,
      model_metadata: metadata_pb2.ModelMetadataProto,
      realm: realm_pb2.Realm,
  ) -> CheckpointPathMetadata | None:
    """Creates a TFHubAsset from a ModelMetadataProto."""
    handle_proto = model_metadata.handle
    handle = (
        f'@{handle_proto.publisher}/{handle_proto.name}/{handle_proto.version}'
    )
    version_labels = (
        l.value
        for l in model_metadata.model_descriptor.asset_descriptor.labels
        if l.type == metadata_pb2.LabelType.LABEL_TYPE_CONSTANT
    )
    model_tags = [
        tag.value
        for tag in model_metadata.model_descriptor.asset_descriptor.tags
    ]
    version_label = next(version_labels, None)
    if not version_label:
      logging.warning(
          'Found an asset with no version label: %r returning None', handle
      )
      return None
    full_path = f'/tfhub/{_TFHUB_PATH_BY_REALM[realm]}/{handle_proto.publisher}/{handle_proto.name}'

    return CheckpointPathMetadata(
        path=full_path,
        status=CheckpointPathMetadata.Status.COMMITTED,
        version=version_label,
        tags=set(model_tags),
    )


class ColossusStorageBackend(StorageBackend):
  """A StorageBackend implementation for Colossus."""

  def get_temporary_path_class(self) -> type[atomicity_types.TemporaryPath]:
    return atomicity.AtomicRenameTemporaryPath

  def list_checkpoints(
      self,
      base_path: str,
  ) -> list[CheckpointPathMetadata]:
    """Lists checkpoints for a given base path and version pattern."""
    raise NotImplementedError(
        'list_checkpoints is not yet implemented for ColossusStorageBackend.'
    )

  def delete_checkpoint(self, checkpoint_path: str | gpath.GPath) -> None:
    """Deletes the checkpoint at the given path."""
    _gfile_delete_checkpoint(checkpoint_path)


# TODO(b/425293362): Refactor to have a single CnsAtomicRenameTemporaryPath that
# can subclass AtomicRenameTemporaryPath and encapsulate all CNS-specific
# logic.
class Colossus2StorageBackend(StorageBackend):
  """A StorageBackend implementation for Colossus 2."""

  def get_temporary_path_class(self) -> type[atomicity_types.TemporaryPath]:
    return cns2_atomicity.Cns2AtomicRenameTemporaryPath

  def list_checkpoints(self, base_path: str) -> list[CheckpointPathMetadata]:
    """Lists checkpoints for a given base path and version pattern."""
    raise NotImplementedError(
        'list_checkpoints is not yet implemented for Colossus2StorageBackend.'
    )

  def delete_checkpoint(self, checkpoint_path: str | gpath.GPath) -> None:
    """Deletes the checkpoint at the given path."""
    _gfile_delete_checkpoint(checkpoint_path)


class GCSStorageBackend(StorageBackend):
  """A StorageBackend implementation for GCS (Google Cloud Storage).

  # TODO(b/425293362): Implement this class.
  """

  def get_temporary_path_class(self) -> type[atomicity_types.TemporaryPath]:
    """Returns the final checkpoint path directly."""
    raise NotImplementedError(
        'get_temporary_path_class is not yet implemented for GCSStorageBackend.'
    )

  def list_checkpoints(self, base_path: str) -> list[CheckpointPathMetadata]:
    """Lists checkpoints for a given base path and version pattern."""
    raise NotImplementedError(
        'list_checkpoints is not yet implemented for GCSStorageBackend.'
    )

  def delete_checkpoint(self, checkpoint_path: str | gpath.GPath) -> None:
    """Deletes the checkpoint at the given path."""
    raise NotImplementedError(
        'delete_checkpoint is not yet implemented for GCSStorageBackend.'
    )


class LocalStorageBackend(StorageBackend):
  """A LocalStorageBackend implementation for local file systems.

  # TODO(b/425293362): Implement this class.
  """

  def get_temporary_path_class(self) -> type[atomicity_types.TemporaryPath]:
    """Returns the final checkpoint path directly."""
    raise NotImplementedError(
        'get_temporary_path_class is not yet implemented for'
        ' LocalStorageBackend.'
    )

  def list_checkpoints(self, base_path: str) -> list[CheckpointPathMetadata]:
    """Lists checkpoints for a given base path and version pattern."""
    raise NotImplementedError(
        'list_checkpoints is not yet implemented for LocalStorageBackend.'
    )

  def delete_checkpoint(self, checkpoint_path: str | gpath.GPath) -> None:
    """Deletes the checkpoint at the given path."""
    _gfile_delete_checkpoint(checkpoint_path)


_FILESYSTEM_TYPE_TO_STORAGE_BACKEND: immutabledict.immutabledict[
    fs_utils.FilesystemType, StorageBackend
] = immutabledict.immutabledict({
    fs_utils.FilesystemType.CNS: ColossusStorageBackend(),
    fs_utils.FilesystemType.GCS: GCSStorageBackend(),
    fs_utils.FilesystemType.TFHUB: TFHubStorageBackend(),
    fs_utils.FilesystemType.LOCAL: LocalStorageBackend(),
})


def resolve_storage_backend(
    path: str,
) -> StorageBackend:
  """Returns a StorageBackend object based on the given path.

  Args:
    path: The file system  specific path.

  Returns:
    A StorageBackend object.

  Raises:
    ValueError: If the path is not supported.
  """
  fs_type = fs_utils.filesystem_type(path)
  path = gpath.GPath(path)
  if fs_type == fs_utils.FilesystemType.CNS and cns2_utils.is_cns2_path(path):
    return Colossus2StorageBackend()
  else:
    storage_backend = _FILESYSTEM_TYPE_TO_STORAGE_BACKEND.get(fs_type)
  if not storage_backend:
    raise NotImplementedError(
        f'Storage backend not implemented for path: {path}'
    )
  logging.info('Resolved storage backend to: %s', type(storage_backend))
  return storage_backend
