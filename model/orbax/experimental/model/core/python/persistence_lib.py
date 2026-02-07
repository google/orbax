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

"""Saving to disk.

This file contains facilities that can save an OBM module to disk or load a
persisted OBM module.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import os

from absl import logging
from orbax.experimental.model.core.protos import manifest_pb2
from orbax.experimental.model.core.python import device_assignment
from orbax.experimental.model.core.python import file_utils
from orbax.experimental.model.core.python import manifest_constants
from orbax.experimental.model.core.python import manifest_util
from orbax.experimental.model.core.python import metadata
from orbax.experimental.model.core.python import saveable
from orbax.experimental.model.core.python import unstructured_data


@dataclasses.dataclass
class GlobalSupplemental:
  """GlobalSupplemental info to be saved in the manifest.pb.

  Attributes:
    data: The global supplemental info to be saved.
    save_as: An optional relative file name. If set, `data` will be saved as a
      separate file using this file name (and in this case the supplemental info
      mustn't be a `file_location` already).
  """

  data: unstructured_data.UnstructuredData

  save_as: str | None = None


@dataclasses.dataclass
class SaveOptions:
  """Options for saving to FSM SavedModel.

  This function may be used in the `options` argument in functions that
  save an FSM SavedModel.

  Attributes:
    version: The serialization format version. With version >= 2 it generates
      manifest.pb whereas with version <= 1 it generates saved_model.pb.
    supplementals: Optional. A map of UnstructuredData to be saved in the
      manifest as the global supplemental info.
    visibility: Optional. A mapping from function name to its visibility (e.g.,
      `manifest_pb2.PUBLIC`, `manifest_pb2.PRIVATE`). If this parameter is not
      provided, all functions will be public. If only a subset of functions are
      provided in the mapping, the rest will be public by default.
    device_assignment_by_coords: Optional. A sequence of DeviceAssignment to be
      saved in the manifest.pb.
  """

  version: int | None = None
  supplementals: Mapping[str, GlobalSupplemental] | None = None
  visibility: Mapping[str, manifest_pb2.Visibility] | None = None
  device_assignment_by_coords: (
      Sequence[device_assignment.DeviceAssignment] | None
  ) = None


def save(
    module: dict[str, saveable.Saveable],
    target_dir: str,
    options: SaveOptions,
) -> None:
  """Saves module `module` in the target directory `target_dir`."""
  if options.version is None or options.version < 2:
    raise ValueError('Version must be >= 2')

  logging.info('Save version: %d', options.version)
  # Generate and export the manifest proto.
  supplemental_info = None
  if options.supplementals is not None:
    supplemental_info = {
        name: _save_supplemental(supp, target_dir)
        for name, supp in options.supplementals.items()
    }

  manifest_proto = manifest_util.build_manifest_proto(
      module,
      target_dir,
      supplementals=supplemental_info,
      visibilities=options.visibility,
      device_assignments=options.device_assignment_by_coords,
  )

  manifest_path = os.path.join(
      target_dir, manifest_constants.MANIFEST_FILE_PATH
  )
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
      os.path.join(target_dir, manifest_constants.MODEL_VERSION_FILENAME)
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


def _save_supplemental(
    supplemental: GlobalSupplemental,
    target_dir: str,
) -> unstructured_data.UnstructuredData:
  """Returns the supplemental data, either inlined or saved to disk."""
  if supplemental.save_as is None:
    return supplemental.data

  return unstructured_data.write_inlined_data_to_file(
      supplemental.data,
      target_dir,
      supplemental.save_as,
  )
