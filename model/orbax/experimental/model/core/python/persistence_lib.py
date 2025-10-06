# Copyright 2025 The Orbax Authors.
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

"""Saving to disk.

This file contains facilities that can save a `Module` to disk.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import os
from typing import Optional

from absl import logging
from orbax.experimental.model.core.protos import manifest_pb2
from orbax.experimental.model.core.python import file_utils
from orbax.experimental.model.core.python import manifest_constants
from orbax.experimental.model.core.python import metadata
from orbax.experimental.model.core.python import unstructured_data
from orbax.experimental.model.core.python.device_assignment import DeviceAssignment
from orbax.experimental.model.core.python.manifest_util import build_manifest_proto
from orbax.experimental.model.core.python.saveable import Saveable
from orbax.experimental.model.core.python.unstructured_data import UnstructuredData


@dataclasses.dataclass
class GlobalSupplemental:
  """GlobalSupplemental info to be saved in the manifest.pb.

  Attributes:
    data: The global supplemental info to be saved.
    save_as: An optional relative file name. If set, `data` will be saved as a
      separate file using this file name (and in this case the supplemental info
      mustn't be a `file_location` already).
  """

  data: UnstructuredData

  save_as: str | None


@dataclasses.dataclass
class SaveOptions:
  """Options for saving to FSM SavedModel.

  This function may be used in the `options` argument in functions that
  save an FSM SavedModel.

  Attributes:
    function_aliases: A mapping from user-chosen function alias name to the
      function that runs on TPU.
    version: The serialization format version. With version >= 2 it generates
      manifest.pb whereas with version <= 1 it generates saved_model.pb.
    supplemental_info: Optional. An `UnstructuredData` (or a string-map of them)
      to be saved in the manifest.pb as the global supplemental info.
    visibility: Optional. A mapping from function name to its visibility (e.g.,
      `manifest_pb2.PUBLIC`, `manifest_pb2.PRIVATE`). If this parameter is not
      provided, all functions will be public. If only a subset of functions are
      provided in the mapping, the rest will be public by default.
      DeviceAssignmentByCoords
    device_assignment_by_coords: Optional. A sequence of DeviceAssignment to be
      saved in the manifest.pb.
  """

  version: int | None = None
  supplemental_info: Mapping[str, GlobalSupplemental] | None = None
  visibility: Mapping[str, manifest_pb2.Visibility] | None = None
  device_assignment_by_coords: Sequence[DeviceAssignment] | None = None


def _save_single_supplemental(
    supplemental_info: GlobalSupplemental,
    path: str,
) -> UnstructuredData:
  """Saves a single supplemental to disk."""
  data = supplemental_info.data
  if supplemental_info.save_as is not None:
    data = unstructured_data.write_inlined_data_to_file(
        data,
        path,
        supplemental_info.save_as,
    )
  return data


def _save_supplementals(
    supplemental_info: (
        GlobalSupplemental | Mapping[str, GlobalSupplemental] | None
    ),
    path: str,
) -> UnstructuredData | Mapping[str, UnstructuredData] | None:
  """Saves supplementals to disk."""
  if supplemental_info is None:
    return None
  if isinstance(supplemental_info, GlobalSupplemental):
    return _save_single_supplemental(supplemental_info, path)
  else:
    return {
        name: _save_single_supplemental(supp, path)
        for name, supp in supplemental_info.items()
    }


def save(
    m: dict[str, Saveable],
    path: str,
    options: Optional[SaveOptions] = None,
) -> None:
  """Saved the Module to disk."""
  assert options is not None
  assert options.version is not None
  assert options.version >= 2
  logging.info('Save version: %d', options.version)
  # Generate and export the Manifest proto.
  supplemental_info = _save_supplementals(options.supplemental_info, path)
  manifest_proto = build_manifest_proto(
      m,
      path,
      supplemental_info=supplemental_info,
      names_to_visibilities=options.visibility,
      device_assignment_by_coords=options.device_assignment_by_coords,
  )

  manifest_path = os.path.join(path, manifest_constants.MANIFEST_FILE_PATH)
  file_utils.mkdir_p(os.path.dirname(manifest_path))
  with file_utils.open_file(manifest_path, 'wb') as f:
    f.write(manifest_proto.SerializeToString())

  model_version = metadata.ModelVersion(
      version=manifest_constants.MANIFEST_VERSION,
      mime_type=manifest_constants.MANIFEST_MIME_TYPE,
      manifest_file_path=manifest_constants.MANIFEST_FILE_PATH,
  )
  # Write the main metadata to detect and parse an Orbax Model. The version file
  # should be THE LAST file to be written. It is used to validate the export and
  # identify an Orbax Model.
  model_version.save(
      os.path.join(path, manifest_constants.MODEL_VERSION_FILENAME)
  )


def load(saved_state_dir: str) -> manifest_pb2.Manifest:
  """Discovers and loads the manifest in the saved state directory."""
  model_version = metadata.ModelVersion.load(
      os.path.join(saved_state_dir, manifest_constants.MODEL_VERSION_FILENAME)
  )

  # TODO(b/447665358): Once there is more than one version of the model
  # we will need to check the list of supported versions and customize the
  # loading process accordingly.
  if model_version.version != manifest_constants.MANIFEST_VERSION:
    raise ValueError(f'Unsupported manifest version "{model_version.version}"')
  if model_version.mime_type != manifest_constants.MANIFEST_MIME_TYPE:
    raise ValueError(f'Unsupported manifest type "{model_version.mime_type}"')

  manifest_path = os.path.join(
      saved_state_dir, model_version.manifest_file_path
  )
  with file_utils.open_file(manifest_path, 'rb') as f:
    return manifest_pb2.Manifest.FromString(f.read())
