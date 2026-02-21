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

"""Generates V1 checkpoints for compatibility testing."""

import json
import os

from absl import app
from absl import flags
from etils import epath
import jax.numpy as jnp
from orbax.checkpoint import test_utils
import orbax.checkpoint.experimental.v1 as ocp


FLAGS = flags.FLAGS


def _get_base_dir():
  if 'BUILD_WORKING_DIRECTORY' in os.environ:
    return os.path.join(
        os.environ['BUILD_WORKING_DIRECTORY'],
        'orbax/checkpoint/experimental/v1/_src/testing/compatibility/checkpoints',
    )
  return os.path.join(
      os.path.dirname(__file__),
      'checkpoints',
  )


flags.DEFINE_string(
    'base_dir',
    _get_base_dir(),
    'Base directory to save checkpoints.',
)


def create_pytree():
  return {'a': jnp.arange(8), 'b': {'c': jnp.array([1, 2, 3])}}


def create_json_object():
  return {'a': [1, 2, 3], 'b': 3, 'c': 'hello'}


def v1_metadata_present_composite_has_pytree(base_dir, checkpointables):
  """Checkpoint metadata present, composite, and has pytree checkpointable."""
  path = (
      base_dir
      / 'v1_checkpoints'
      / 'has_checkpoint_metadata'
      / 'composite_item_handler_checkpoint'
      / 'has_pytree_metadata'
  )
  if path.exists():
    return path

  ocp.save_checkpointables(path, checkpointables)
  (path / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v1_metadata_present_composite_missing_pytree(base_dir, checkpointables):
  """Checkpoint metadata present, composite, and no pytree checkpointable."""
  path = (
      base_dir
      / 'v1_checkpoints'
      / 'has_checkpoint_metadata'
      / 'composite_item_handler_checkpoint'
      / 'missing_pytree_metadata'
  )
  if path.exists():
    return path
  ocp.save_checkpointables(path, checkpointables)
  (path / 'pytree' / '_METADATA').unlink(missing_ok=True)  # del pytree metadata
  (path / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v1_metadata_present_corrupt_has_pytree(base_dir, checkpointables):
  """Checkpoint metadata present, corrupt, and has pytree checkpointable."""
  path = (
      base_dir
      / 'v1_checkpoints'
      / 'has_checkpoint_metadata'
      / 'corrupt_item_handler_checkpoint'
      / 'has_pytree_metadata'
  )
  if path.exists():
    return path
  ocp.save_checkpointables(path, checkpointables)
  # Manually corrupt the item_handlers field.
  checkpoint_metadata_path = path / '_CHECKPOINT_METADATA'
  if checkpoint_metadata_path.exists():
    with open(checkpoint_metadata_path, 'r') as f:
      data = json.load(f)
    if 'item_handlers' in data:
      data['item_handlers'] = {'pytree': 'invalid_type'}
    with open(checkpoint_metadata_path, 'w') as f:
      json.dump(data, f)
  else:
    raise FileNotFoundError(
        f'Metadata file not found: {checkpoint_metadata_path}'
    )
  (path / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v1_metadata_present_corrupt_missing_pytree(base_dir, checkpointables):
  """Checkpoint metadata present, corrupt, and missing pytree checkpointable."""
  path = (
      base_dir
      / 'v1_checkpoints'
      / 'has_checkpoint_metadata'
      / 'corrupt_item_handler_checkpoint'
      / 'missing_pytree_metadata'
  )
  if path.exists():
    return path
  ocp.save_checkpointables(path, checkpointables)
  (path / 'pytree' / '_METADATA').unlink(missing_ok=True)  # del pytree metadata
  checkpoint_metadata_path = path / '_CHECKPOINT_METADATA'
  # Manually corrupt the item_handlers field.
  if checkpoint_metadata_path.exists():
    with open(checkpoint_metadata_path, 'r') as f:
      data = json.load(f)
    if 'item_handlers' in data:
      data['item_handlers'] = {'pytree': 'invalid_type'}
    with open(checkpoint_metadata_path, 'w') as f:
      json.dump(data, f)
  else:
    raise FileNotFoundError(
        f'Metadata file not found: {checkpoint_metadata_path}'
    )
  (path / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v1_missing_metadata_composite_has_pytree(base_dir, checkpointables):
  """Checkpoint metadata missing, composite, and has pytree checkpointable."""
  path = (
      base_dir
      / 'v1_checkpoints'
      / 'missing_checkpoint_metadata'
      / 'composite_checkpoint'
      / 'has_pytree_metadata'
  )
  if path.exists():
    return path
  ocp.save_checkpointables(path, checkpointables)
  (path / '_CHECKPOINT_METADATA').unlink(missing_ok=True)
  (path / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v1_missing_metadata_composite_missing_pytree(base_dir, checkpointables):
  """Checkpoint metadata missing, composite, and no pytree checkpointable."""
  path = (
      base_dir
      / 'v1_checkpoints'
      / 'missing_checkpoint_metadata'
      / 'composite_checkpoint'
      / 'missing_pytree_metadata'
  )
  if path.exists():
    return path
  ocp.save_checkpointables(path, checkpointables)
  (path / '_CHECKPOINT_METADATA').unlink(missing_ok=True)
  (path / 'pytree' / '_METADATA').unlink(missing_ok=True)  # del pytree metadata
  (path / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v1_missing_field_in_checkpoint_metadata(
    base_dir, field_to_remove, checkpointables
):
  """Saves V1 checkpoint and removes a field from _CHECKPOINT_METADATA."""
  path = (
      base_dir
      / 'v1_checkpoints'
      / 'metadata_alterations'
      / f'missing_{field_to_remove}_metadata'
  )
  if path.exists():
    return path
  ocp.save_checkpointables(path, checkpointables)
  checkpoint_metadata_path = path / '_CHECKPOINT_METADATA'
  if checkpoint_metadata_path.exists():
    with open(checkpoint_metadata_path, 'r') as f:
      data = json.load(f)
    if field_to_remove in data:
      del data[field_to_remove]
    with open(checkpoint_metadata_path, 'w') as f:
      json.dump(data, f)
  else:
    raise FileNotFoundError(
        f'Metadata file not found: {checkpoint_metadata_path}'
    )
  (path / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v1_missing_pytree_data_files(
    base_dir, file_to_remove, checkpointables
):
  """Saves a checkpointables checkpoint and removes data files."""
  path = (
      base_dir
      / 'v1_checkpoints'
      / 'metadata_alterations'
      / f'missing_pytree_data_file_{file_to_remove}'
  )
  if path.exists():
    return path
  ocp.save_checkpointables(path, checkpointables)
  (path / 'pytree' / file_to_remove).unlink(missing_ok=True)
  (path / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v1_missing_pytree_data_dir(
    base_dir, dir_to_remove, checkpointables
):
  """Saves a checkpointables checkpoint and removes process directory."""
  path = (
      base_dir
      / 'v1_checkpoints'
      / 'metadata_alterations'
      / f'missing_pytree_data_dir_{dir_to_remove}'
  )
  if path.exists():
    return path
  ocp.save_checkpointables(path, checkpointables)
  (path / 'pytree' / dir_to_remove).rmtree()
  (path / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v1_dummy_checkpointable_present(base_dir, checkpointables):
  """Saves a checkpointables checkpoint and adds a dummy checkpointable."""
  path = (
      base_dir
      / 'v1_checkpoints'
      / 'general_alteration'
      / 'dummy_checkpointable_present'
  )
  if path.exists():
    return path
  ocp.save_checkpointables(path, checkpointables)
  (path / 'dummy').mkdir()
  (path / 'dummy' / '_METADATA').write_text('dummy')
  (path / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v1_delete_checkpointable(base_dir, checkpointables):
  """Saves a checkpointables checkpoint and adds a deleted checkpointable."""
  path = (
      base_dir
      / 'v1_checkpoints'
      / 'general_alteration'
      / 'deleted_checkpointable_present'
  )
  if path.exists():
    return path
  ocp.save_checkpointables(path, checkpointables)
  (path / 'state').rmtree()
  (path / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def main(argv):
  del argv
  base_dir = epath.Path(FLAGS.base_dir)
  base_dir.mkdir(parents=True, exist_ok=True)

  test_utils.set_tensorstore_driver_for_test()

  checkpointables = {
      'pytree': create_pytree(),
      'state': create_json_object(),
  }
  print('Generating V1 Checkpoints...')
  # Three categories we'll generate against:
  # 1. Is Checkpoint Metadata present?
  # - Yes, No
  # 2. Item_handler type
  # - dict(composite saved with save_checkpointables/pytree), invalid
  # 3. Is Pytree? (Pytree Metadata present in checkpointable dir?)
  # - Yes, No

  # The directory structure will follow the categories above:
  # base_dir /
  # 'v1_checkpoints' /
  # <Ckpt_Metadata_Present/Missing> /
  # <Item_Handler_Type> /
  # <Is_Pytree_Yes/No> /

  # Combinations
  # V1,Yes,dict,yes
  v1_metadata_present_composite_has_pytree(base_dir, checkpointables)
  # V1,Yes,dict,no
  v1_metadata_present_composite_missing_pytree(base_dir, checkpointables)
  # V1,Yes,invalid,yes
  v1_metadata_present_corrupt_has_pytree(base_dir, checkpointables)
  # V1,Yes,invalid,no
  v1_metadata_present_corrupt_missing_pytree(base_dir, checkpointables)
  # V1,No,dict(composite checkpoint),yes
  v1_missing_metadata_composite_has_pytree(base_dir, checkpointables)
  # V1,No,dict(composite checkpoint),no
  v1_missing_metadata_composite_missing_pytree(base_dir, checkpointables)
  # V1,No,invalid,yes
  # N/A
  # V1,No,invalid,no
  # N/A

  # Additional Corruptions
  # Will generate composite and direct pytree checkpoints for metadata
  # alterations.
  # The directory structure will be:
  # base_dir /
  # 'v1_checkpoints' /
  # 'metadata_alterations' /
  # <Alteration_Type> /

  # Missing Fields
  fields_to_remove = [
      'item_handlers',
      'metrics',
      'performance_metrics',
      'init_timestamp_nsecs',
      'commit_timestamp_nsecs',
      'custom_metadata',
  ]
  for field in fields_to_remove:
    v1_missing_field_in_checkpoint_metadata(base_dir, field, checkpointables)

  files_to_remove = ['_sharding', 'manifest.ocdbt']
  # Missing Data Files
  for file_to_remove in files_to_remove:
    v1_missing_pytree_data_files(base_dir, file_to_remove, checkpointables)

  dirs_to_remove = ['array_metadatas', 'd', 'ocdbt.process_0']
  # Missing Data Directories
  for dir_to_remove in dirs_to_remove:
    v1_missing_pytree_data_dir(base_dir, dir_to_remove, checkpointables)

  # Adding cases for dummy and deleted checkpointables within checkpoint, due to
  # user modification of checkpoint.
  # The directory structure will follow the categories above:
  # base_dir /
  # 'v1_checkpoints' /
  # 'general_alteration' /
  # <Alteration_Type> /
  v1_dummy_checkpointable_present(base_dir, checkpointables)
  v1_delete_checkpointable(base_dir, checkpointables)

  print(f'V1 Checkpoints generated at {base_dir}')


if __name__ == '__main__':
  app.run(main)
