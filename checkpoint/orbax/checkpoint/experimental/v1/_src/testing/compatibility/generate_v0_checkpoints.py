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

"""Generates V0 checkpoints for compatibility testing."""

import json
import os

from absl import app
from absl import flags
from etils import epath
import jax.numpy as jnp
from orbax.checkpoint import args
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.checkpointers import checkpointer as v0_checkpointer
from orbax.checkpoint._src.checkpointers import standard_checkpointer
from orbax.checkpoint._src.handlers import composite_checkpoint_handler

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


def _standard_checkpointer_save_pytree(path):
  """Saves a standard checkpoint using StandardCheckpointer."""
  if path.exists():
    return path
  pytree = create_pytree()
  with standard_checkpointer.StandardCheckpointer() as checkpointer:
    checkpointer.save(path, pytree)
  return path


def _checkpointer_save_composite_mixed(path):
  """Saves a standard checkpoint using Checkpointer + CompositeHandler."""
  if path.exists():
    return path
  json_object = create_json_object()
  pytree = create_pytree()
  checkpoint_args = args.Composite(**{
      'pytree': args.StandardSave(pytree),  # Will remove pytree metadata
      'state': args.JsonSave(json_object),
  })
  with v0_checkpointer.Checkpointer(
      composite_checkpoint_handler.CompositeCheckpointHandler()
  ) as checkpointer:
    checkpointer.save(path, checkpoint_args)
  return path


def v0_metadata_present_composite_has_pytree(base_dir):
  """Checkpoint metadata present, composite, and has pytree checkpointable."""
  path = (
      base_dir
      / 'v0_checkpoints'
      / 'has_checkpoint_metadata'
      / 'composite_item_handler_checkpoint'
      / 'has_pytree_metadata'
  )
  if path.exists():
    return path

  _checkpointer_save_composite_mixed(path)
  (path / 'pytree' / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v0_metadata_present_composite_missing_pytree(base_dir):
  """Checkpoint metadata present, composite, and no pytree checkpointable."""
  path = (
      base_dir
      / 'v0_checkpoints'
      / 'has_checkpoint_metadata'
      / 'composite_item_handler_checkpoint'
      / 'missing_pytree_metadata'
  )
  if path.exists():
    return path
  _checkpointer_save_composite_mixed(path)
  (path / 'pytree' / '_METADATA').unlink(missing_ok=True)  # del pytree metadata
  (path / 'pytree' / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v0_metadata_present_direct_has_pytree(base_dir):
  """Checkpoint metadata present, direct, and has pytree checkpointable."""
  path = (
      base_dir
      / 'v0_checkpoints'
      / 'has_checkpoint_metadata'
      / 'direct_item_handler_checkpoint'
      / 'has_pytree_metadata'
  )
  if path.exists():
    return path
  _standard_checkpointer_save_pytree(path)
  (path / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v0_metadata_present_direct_missing_pytree(base_dir):
  """Checkpoint metadata present, direct, and no pytree checkpointable."""
  path = (
      base_dir
      / 'v0_checkpoints'
      / 'has_checkpoint_metadata'
      / 'direct_item_handler_checkpoint'
      / 'missing_pytree_metadata'
  )
  if path.exists():
    return path
  _standard_checkpointer_save_pytree(path)
  (path / '_METADATA').unlink(missing_ok=True)  # del pytree metadata
  (path / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v0_metadata_present_corrupt_has_pytree(base_dir):
  """Checkpoint metadata present, corrupt, and has pytree checkpointable."""
  path = (
      base_dir
      / 'v0_checkpoints'
      / 'has_checkpoint_metadata'
      / 'corrupt_item_handler_checkpoint'
      / 'has_pytree_metadata'
  )
  if path.exists():
    return path
  _checkpointer_save_composite_mixed(path)
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
  (path / 'pytree' / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v0_metadata_present_corrupt_missing_pytree(base_dir):
  """Checkpoint metadata present, corrupt, and missing pytree checkpointable."""
  path = (
      base_dir
      / 'v0_checkpoints'
      / 'has_checkpoint_metadata'
      / 'corrupt_item_handler_checkpoint'
      / 'missing_pytree_metadata'
  )
  if path.exists():
    return path
  _checkpointer_save_composite_mixed(path)
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
  (path / 'pytree' / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v0_missing_metadata_composite_has_pytree(base_dir):
  """Checkpoint metadata missing, composite, and has pytree checkpointable."""
  path = (
      base_dir
      / 'v0_checkpoints'
      / 'missing_checkpoint_metadata'
      / 'composite_checkpoint'
      / 'has_pytree_metadata'
  )
  if path.exists():
    return path
  _checkpointer_save_composite_mixed(path)
  (path / '_CHECKPOINT_METADATA').unlink(missing_ok=True)
  (path / 'pytree' / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v0_missing_metadata_composite_missing_pytree(base_dir):
  """Checkpoint metadata missing, composite, and no pytree checkpointable."""
  path = (
      base_dir
      / 'v0_checkpoints'
      / 'missing_checkpoint_metadata'
      / 'composite_checkpoint'
      / 'missing_pytree_metadata'
  )
  if path.exists():
    return path
  _checkpointer_save_composite_mixed(path)
  (path / '_CHECKPOINT_METADATA').unlink(missing_ok=True)
  (path / 'pytree' / '_METADATA').unlink(missing_ok=True)  # del pytree metadata
  (path / 'pytree' / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v0_missing_metadata_direct_has_pytree(base_dir):
  """Checkpoint metadata missing, direct, and has pytree checkpointable."""
  path = (
      base_dir
      / 'v0_checkpoints'
      / 'missing_checkpoint_metadata'
      / 'direct_checkpoint'
      / 'has_pytree_metadata'
  )
  if path.exists():
    return path
  _standard_checkpointer_save_pytree(path)
  (path / '_CHECKPOINT_METADATA').unlink(missing_ok=True)
  (path / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v0_missing_metadata_direct_missing_pytree(base_dir):
  """Checkpoint metadata missing, direct, and no pytree checkpointable."""
  path = (
      base_dir
      / 'v0_checkpoints'
      / 'missing_checkpoint_metadata'
      / 'direct_checkpoint'
      / 'missing_pytree_metadata'
  )
  if path.exists():
    return path
  _standard_checkpointer_save_pytree(path)
  (path / '_CHECKPOINT_METADATA').unlink(missing_ok=True)
  (path / '_METADATA').unlink(missing_ok=True)  # del pytree metadata
  (path / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v0_composite_missing_field_in_checkpoint_metadata(
    base_dir, field_to_remove
):
  """Saves V0 checkpoint and removes a field from _CHECKPOINT_METADATA."""
  path = (
      base_dir
      / 'v0_checkpoints'
      / 'metadata_alterations'
      / 'composite_checkpoint'
      / f'missing_{field_to_remove}_metadata'
  )
  if path.exists():
    return path
  _checkpointer_save_composite_mixed(path)
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
  (path / 'pytree' / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v0_pytree_missing_field_in_checkpoint_metadata(base_dir, field_to_remove):
  """Saves V0 checkpoint and removes a field from _CHECKPOINT_METADATA."""
  path = (
      base_dir
      / 'v0_checkpoints'
      / 'metadata_alterations'
      / 'direct_checkpoint'
      / f'missing_{field_to_remove}_metadata'
  )
  if path.exists():
    return path
  _standard_checkpointer_save_pytree(path)
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


def v0_composite_missing_pytree_data_files(
    base_dir, file_to_remove
):
  """Saves a checkpointables checkpoint and removes data files."""
  path = (
      base_dir
      / 'v0_checkpoints'
      / 'metadata_alterations'
      / 'composite_checkpoint'
      / f'missing_pytree_data_file_{file_to_remove}'
  )
  if path.exists():
    return path
  _checkpointer_save_composite_mixed(path)
  (path / 'pytree' / file_to_remove).unlink(missing_ok=True)
  (path / 'pytree' / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v0_pytree_missing_pytree_data_files(
    base_dir, file_to_remove
):
  """Saves a pytree checkpoint and removes data files."""
  path = (
      base_dir
      / 'v0_checkpoints'
      / 'metadata_alterations'
      / 'direct_checkpoint'
      / f'missing_pytree_data_file_{file_to_remove}'
  )
  if path.exists():
    return path
  _standard_checkpointer_save_pytree(path)
  (path / file_to_remove).unlink(missing_ok=True)
  (path / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v0_composite_missing_pytree_data_dir(
    base_dir, dir_to_remove
):
  """Saves a checkpointables checkpoint and removes process directory."""
  path = (
      base_dir
      / 'v0_checkpoints'
      / 'metadata_alterations'
      / 'composite_checkpoint'
      / f'missing_pytree_data_dir_{dir_to_remove}'
  )
  if path.exists():
    return path
  _checkpointer_save_composite_mixed(path)
  (path / 'pytree' / dir_to_remove).rmtree()
  (path / 'pytree' / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def v0_pytree_missing_pytree_data_dir(
    base_dir, dir_to_remove
):
  """Saves a pytree checkpoint and removes process directory."""
  path = (
      base_dir
      / 'v0_checkpoints'
      / 'metadata_alterations'
      / 'direct_checkpoint'
      / f'missing_pytree_data_dir_{dir_to_remove}'
  )
  if path.exists():
    return path
  _standard_checkpointer_save_pytree(path)
  (path / dir_to_remove).rmtree()
  (path / 'descriptor').rmtree()  # GOOGLE_INTERNAL
  return path


def main(argv):
  del argv
  base_dir = epath.Path(FLAGS.base_dir)
  base_dir.mkdir(parents=True, exist_ok=True)

  test_utils.set_tensorstore_driver_for_test()

  print('Generating V0 Checkpoints...')
  # Three categories we'll generate against:
  # 1. Is Checkpoint Metadata present?
  # - Yes, No
  # 2. Item_handler type (save method)
  # - dict(composite), str(direct_pytree checkpoint), invalid
  # 3. Is Pytree? (Pytree Metadata present in checkpointable dir?)
  # - Yes, No

  # The directory structure will follow the categories above:
  # base_dir /
  # 'v0_checkpoints' /
  # <Ckpt_Metadata_Present/Missing> /
  # <Item_Handler_Type> /
  # <Is_Pytree_Yes/No> /
  # Optional_Data_Alteration

  # Combinations
  # V0,Yes,dict,yes
  v0_metadata_present_composite_has_pytree(base_dir)
  # V0,Yes,dict,no
  v0_metadata_present_composite_missing_pytree(base_dir)
  # V0,Yes,string,yes
  v0_metadata_present_direct_has_pytree(base_dir)
  # V0,Yes,string,no
  v0_metadata_present_direct_missing_pytree(base_dir)
  # V0,Yes,invalid,yes
  v0_metadata_present_corrupt_has_pytree(base_dir)
  # V0,Yes,invalid,no
  v0_metadata_present_corrupt_missing_pytree(base_dir)
  # V0,No,dict(composite checkpoint),yes
  v0_missing_metadata_composite_has_pytree(base_dir)
  # V0,No,dict(composite checkpoint),no
  v0_missing_metadata_composite_missing_pytree(base_dir)
  # V0,No,string(direct pytree checkpoint),yes
  v0_missing_metadata_direct_has_pytree(base_dir)
  # V0,No,string(direct pytree checkpoint),no
  v0_missing_metadata_direct_missing_pytree(base_dir)
  # V0,No,invalid,yes
  # N/A
  # V0,No,invalid,no
  # N/A

  # Additional Corruptions
  # Will generate composite and direct pytree checkpoints for metadata
  # alterations.
  # The directory structure will be:
  # base_dir /
  # 'v0_checkpoints' /
  # 'metadata_alterations' /
  # <Composite/Direct> /
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
    v0_composite_missing_field_in_checkpoint_metadata(base_dir, field)
    v0_pytree_missing_field_in_checkpoint_metadata(base_dir, field)

  files_to_remove = ['_sharding', 'manifest.ocdbt']
  # Missing Data Files
  for file_to_remove in files_to_remove:
    v0_composite_missing_pytree_data_files(base_dir, file_to_remove)
    v0_pytree_missing_pytree_data_files(base_dir, file_to_remove)

  dirs_to_remove = ['array_metadatas', 'd', 'ocdbt.process_0']
  # Missing Data Directories
  for dir_to_remove in dirs_to_remove:
    v0_composite_missing_pytree_data_dir(base_dir, dir_to_remove)
    v0_pytree_missing_pytree_data_dir(base_dir, dir_to_remove)

  print(f'V0 Checkpoints generated at {base_dir}')


if __name__ == '__main__':
  app.run(main)
