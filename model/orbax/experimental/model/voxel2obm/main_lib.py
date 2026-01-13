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

import collections
from collections.abc import Mapping
import os
import re
from typing import Any, Callable

from orbax.experimental.model import core as obm
from orbax.experimental.model.core.python import file_utils
from orbax.experimental.model.voxel2obm import voxel_asset_map_pb2


VOXEL_PROCESSOR_MIME_TYPE = 'application/protobuf; type=voxel.PlanProto'
VOXEL_PROCESSOR_VERSION = '0.0.1'
DEFAULT_VOXEL_MODULE_FOLDER = 'voxel_module'
VOXEL_ASSETS_FOLDER = 'assets'
VOXEL_ASSET_MAP_MIME_TYPE = (
    'application/protobuf; type=orbax_model_voxel_assets_map.VoxelAssetsMap'
)
VOXEL_ASSET_MAP_VERSION = '0.0.1'
VOXEL_ASSET_MAP_SUPPLEMENTAL_NAME = 'voxel_asset_map'


def voxel_plan_to_obm(
    # TODO(b/447200841): use the true type hint after voxel module is
    # implemented.
    voxel_module: Any,
    input_signature: obm.Tree[obm.ShloTensorSpec],
    output_signature: obm.Tree[obm.ShloTensorSpec],
    subfolder: str = DEFAULT_VOXEL_MODULE_FOLDER,
) -> obm.SerializableFunction:
  """Converts a Voxel plan to an `obm.SerializableFunction`.

  Args:
    voxel_module: The Voxel module to be converted.
    input_signature: The input signature of the Voxel module.
    output_signature: The output signature of the Voxel module.
    subfolder: The name of the subfolder for the converted module.

  Returns:
    An `obm.SerializableFunction` representing the Voxel module.
  """
  plan = voxel_module.export_plan()
  unstructured_data = obm.manifest_pb2.UnstructuredData(
      inlined_bytes=plan.SerializeToString(),
      mime_type=VOXEL_PROCESSOR_MIME_TYPE,
      version=VOXEL_PROCESSOR_VERSION,
  )

  obm_func = obm.SerializableFunction(
      body=obm.UnstructuredDataWithExtName(
          proto=unstructured_data,
          subfolder=subfolder,
          ext_name='pb',
      ),
      input_signature=input_signature,
      output_signature=output_signature,
  )
  return obm_func


def _normalize_file_name(file_name: str) -> str:
  """Strips numeric suffixes from a filename.

  For example, 'foo_1.txt' becomes 'foo.txt'. This helps group files
  for conflict resolution.

  Args:
    file_name: The base name of file.

  Returns:
    The normalized file name.
  """
  base, ext = os.path.splitext(file_name)
  # Strip numeric suffixes (_1, _123, _1_2, etc.) to normalize filenames
  # like 'foo_1.txt' and 'foo.txt' to the same base 'foo.txt' so that the
  # conflict can be resolved efficiently with the self._next_index_map.
  while re.search(r'_\d+$', base):
    base = re.sub(r'_\d+$', '', base)
  return f'{base}{ext}'


class _VoxelAssetMapBuilder:
  """Helper class to build VoxelAssetsMap efficiently."""

  def __init__(self):
    # Maps unique/sanitized filenames to their original source paths.
    # Used to detect filename collisions and ensure unique asset filenames.
    self._auxiliary_file_map: dict[str, str] = {}
    # Stores the next available index for a given base filename,
    # defaulting to 1 if the base filename hasn't been seen before.
    self._next_index_map: dict[str, int] = collections.defaultdict(lambda: 1)
    self._voxel_asset_map = voxel_asset_map_pb2.VoxelAssetsMap()

  @property
  def voxel_asset_map(self) -> voxel_asset_map_pb2.VoxelAssetsMap:
    return self._voxel_asset_map

  def add_asset(self, source_path: str, subfolder: str) -> None:
    """Adds an asset to the map, resolving name conflicts.

    If the base filename of the source_path is already used by a different
    source path in the map, a unique indexed filename (e.g., 'file_1.ext')
    is generated and used.

    Args:
      source_path: The original path of the asset file.
      subfolder: The name of the assets subfolder for the saved model.
    """
    # The asset has been added before, skip.
    if source_path in self._voxel_asset_map.assets:
      return

    file_name = os.path.basename(source_path)
    unique_file_name = _normalize_file_name(file_name)

    # If the file_name is already in use by a different source path,
    # or if we've previously generated indexed names for this base,
    # we need to find a unique indexed name.
    if unique_file_name in self._auxiliary_file_map:
      base, ext = os.path.splitext(unique_file_name)
      i = self._next_index_map[base]
      unique_file_name = f'{base}_{i}{ext}'
      self._next_index_map[base] += 1

    # Add the unique file name to the maps.
    self._auxiliary_file_map[unique_file_name] = source_path
    self._voxel_asset_map.assets[source_path] = os.path.join(
        subfolder, VOXEL_ASSETS_FOLDER, unique_file_name
    )


def _get_voxel_asset_map(
    asset_source_path: set[str], subfolder: str = DEFAULT_VOXEL_MODULE_FOLDER
) -> voxel_asset_map_pb2.VoxelAssetsMap:
  """Gets a VoxelAssetsMap proto for a given set of asset source paths.

  The VoxelAssetsMap proto contains a mapping from original asset paths to
  the new relative paths in the saved model directory.

  Args:
    asset_source_path: A set of source paths of the assets.
    subfolder: The name of the subfolder for the converted module.

  Returns:
    A VoxelAssetsMap proto.
  """
  builder = _VoxelAssetMapBuilder()
  for source_path in sorted(list(asset_source_path)):
    builder.add_asset(source_path, subfolder)
  return builder.voxel_asset_map


def _save_assets(
    voxel_asset_map: voxel_asset_map_pb2.VoxelAssetsMap, path: str
) -> None:
  """Saves asset files based on the provided VoxelAssetsMap.

  Iterates through the assets in voxel_asset_map and copies each asset from
  its source path to destination. The destination path is constructed by joining
  `path` with the asset's relative path, and destination directories are
  created as needed.

  Args:
    voxel_asset_map: A VoxelAssetsMap proto containing asset mappings.
    path: The base destination directory to save the assets.
  """
  for source_path, dest_relative_path in voxel_asset_map.assets.items():
    dest_path = os.path.join(path, dest_relative_path)
    file_utils.mkdir_p(os.path.dirname(dest_path))
    file_utils.copy(source_path, dest_path)
  return


def _asset_map_to_obm_supplemental(
    voxel_asset_map: voxel_asset_map_pb2.VoxelAssetsMap,
) -> obm.GlobalSupplemental:
  """Converts a VoxelAssetsMap proto to an obm.GlobalSupplemental object.

  Serializes the VoxelAssetsMap to bytes and wraps it in an
  obm.UnstructuredData object, returning it as part of an
  obm.GlobalSupplemental object.

  Args:
    voxel_asset_map: A VoxelAssetsMap proto to be converted.

  Returns:
    An obm.GlobalSupplemental object containing the serialized voxel asset map.
  """
  return obm.GlobalSupplemental(
      data=obm.UnstructuredData(
          inlined_bytes=voxel_asset_map.SerializeToString(),
          mime_type=VOXEL_ASSET_MAP_MIME_TYPE,
          version=VOXEL_ASSET_MAP_VERSION,
      ),
      save_as='voxel_asset_map.pb',
  )


def voxel_global_supplemental_closure(
    voxel_module: Any,
) -> Callable[[str], Mapping[str, obm.GlobalSupplemental]] | None:
  """Returns a closure for saving Voxel assets and creating supplemental data.

  This function first generates a VoxelAssetsMap based on asset_source_paths.
  It then returns a closure function. When called, the closure saves the
  assets to a specified destination and returns an obm.GlobalSupplemental object
  containing the asset map.

  Args:
    voxel_module: A Voxel module instance.

  Returns:
   A function that takes the asset destination path string, stores assets in it,
   and returns a dictionary of one entry, from the Voxel supplemental name to
   the obm.GlobalSupplemental object encoding the Voxel asset map.
  """
  asset_source_paths = voxel_module.export_assets()
  if not asset_source_paths:
    return None
  voxel_asset_map = _get_voxel_asset_map(asset_source_paths)

  def save_and_create_global_supplemental(
      path: str,
  ) -> Mapping[str, obm.GlobalSupplemental]:
    _save_assets(voxel_asset_map, path)
    return {
        VOXEL_ASSET_MAP_SUPPLEMENTAL_NAME: _asset_map_to_obm_supplemental(
            voxel_asset_map
        )
    }

  return save_and_create_global_supplemental
