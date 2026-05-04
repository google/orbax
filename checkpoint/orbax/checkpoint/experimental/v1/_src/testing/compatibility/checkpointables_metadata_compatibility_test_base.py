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

"""Tests for V1 checkpointables_metadata API against V0/V1 Checkpoints."""
import os
from typing import Tuple, Type

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
from orbax.checkpoint import test_utils
import orbax.checkpoint.experimental.v1 as ocp
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout as checkpoint_layout_lib
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.testing.compatibility import test_utils as compatibility_test_utils


CheckpointLayoutEnum = options_lib.CheckpointLayout
InvalidLayoutError = checkpoint_layout_lib.InvalidLayoutError


_BASE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')


class CheckpointablesMetadataCompatibilityTestBase(parameterized.TestCase):
  """Tests for V1 checkpointables_metadata API against generated Checkpoints."""

  def setUp(self) -> None:
    super().setUp()
    self.base_dir = epath.Path(_BASE_DIR)
    self.expected_state = {
        'a': jnp.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=jnp.int32),
        'b': {'c': jnp.array([1, 2, 3], dtype=jnp.int32)},
    }
    self.expected_state_metadata = jax.tree.map(
        compatibility_test_utils.create_value_metadata, self.expected_state
    )
    self.expected_checkpointables_metadata = {
        'state': self.expected_state_metadata,
        'metadata': None,
    }

  def setup_registry(self) -> registration.CheckpointableHandlerRegistry:
    """Ensures we only have what we explicitly add."""
    registry = ocp.handlers.local_registry(include_global_registry=False)
    registry.add(ocp.handlers.PyTreeHandler, checkpointable_name='pytree')
    return registry

  def _determine_expected_outcome(
      self,
      version: str,
      metadata_present: bool,
      is_direct_checkpoint: bool,
      is_pytree: bool,
  ) -> Tuple[Type[Exception] | None, str | None]:
    """Determines failure cases. All other cases default to Success."""
    # If we indicate that this is a top level pytree checkpoint, but we do not
    # have pytree metadata, then our metadata resolution fails.
    if is_direct_checkpoint and metadata_present and not is_pytree:
      return FileNotFoundError, r'Metadata file .* does not exist at .*'

    # V1 strictly requires that a checkpoint has checkpoint metadata.
    if not is_direct_checkpoint and version == 'v1' and not metadata_present:
      return InvalidLayoutError, (
          r'Could not recognize the checkpoint at .* as a valid Orbax'
          r' checkpoint'
      )

    return None, None

  @parameterized.product(
      version=['v0', 'v1'],
      metadata_present=[True, False],
      is_direct_checkpoint=[True, False],
      is_pytree=[True, False],
  )
  def test_checkpointables_metadata_compatibility(
      self,
      version: str,
      metadata_present: bool,
      is_direct_checkpoint: bool,
      is_pytree: bool,
  ) -> None:
    """Tests checkpointables_metadata against various checkpoint formats.

    Args:
      version: v0 or v1.
      metadata_present: Whether the checkpoint has metadata files.
      is_direct_checkpoint: Whether the checkpoint is a direct checkpoint.
      is_pytree: Whether the checkpoint is a pytree checkpoint.
    """
    path = compatibility_test_utils.get_checkpoint_path(
        version, metadata_present, is_direct_checkpoint, is_pytree
    )
    if path is None or not path.exists():
      self.skipTest('Checkpoint for combination does not exist.')

    registry = self.setup_registry()

    error_type, expected_error_msg = (
        self._determine_expected_outcome(
            version,
            metadata_present,
            is_direct_checkpoint,
            is_pytree,
        )
    )

    with ocp.Context(
        checkpointables_options=ocp.options.CheckpointablesOptions(
            registry=registry
        )
    ):
      if error_type is None:
        loaded = ocp.checkpointables_metadata(path)
        # If the state checpointable is missing pytree metadata, then we expect
        # the metadata to be a dict of None values with keys corresponding to
        # the subdirectories of the checkpoint.
        if not is_pytree:
          subdirectories = [x for x in path.iterdir() if x.is_dir()]
          expected = {subdir.name: None for subdir in subdirectories}
        # If we find a direct checkpoint, we expect to load the top level
        # pytree metadata, this is due to both pytree_metadata and
        # checkpointables_metadata functions relying on the same
        # metadata resolution functionality.
        elif is_direct_checkpoint:
          expected = self.expected_state_metadata
        else:
          expected = self.expected_checkpointables_metadata

        actual = loaded.metadata
        test_utils.assert_tree_equal(self, expected, actual)
      else:
        with self.assertRaisesRegex(error_type, expected_error_msg):
          ocp.checkpointables_metadata(path)

  @parameterized.product(
      version=['v0', 'v1'],
      alteration=[
          'missing_item_handlers_metadata',
          'missing_metrics_metadata',
          'missing_performance_metrics_metadata',
          'missing_init_timestamp_nsecs_metadata',
          'missing_commit_timestamp_nsecs_metadata',
          'missing_custom_metadata_metadata',
          'missing_pytree_data_dir_array_metadatas',
      ],
  )
  def test_checkpointables_metadata_non_critical_corruptions(
      self, version: str, alteration: str
  ) -> None:
    """Tests checkpointables_metadata against non-critical corruptions.

    Args:
      version: The checkpoint version to test against.
      alteration: The alteration to apply to the checkpoint.
    """
    path = self.base_dir.joinpath(
        f'{version}_checkpoints',
        'composite_checkpoint',
        'non_critical_metadata_alterations',
        alteration,
    )
    # Missing sharding metadata results in a pytree identical to expected
    # values except sharding metadata is None.
    loaded = ocp.checkpointables_metadata(path)
    expected = self.expected_checkpointables_metadata
    actual = loaded.metadata
    test_utils.assert_tree_equal(self, expected, actual)

  @parameterized.product(
      version=['v0', 'v1'],
  )
  def test_checkpointables_metadata_missing_sharding_corruption(
      self, version: str
  ) -> None:
    """Tests checkpointables_metadata against missing sharding corruption.

    Args:
      version: The checkpoint version to test against.
    """
    path = self.base_dir.joinpath(
        f'{version}_checkpoints',
        'composite_checkpoint',
        'non_critical_metadata_alterations',
        'missing_pytree_data_file__sharding',
    )
    # Missing sharding metadata results in a pytree identical to expected
    # values except sharding metadata is None.
    loaded = ocp.checkpointables_metadata(path)
    self.assertIsNone(loaded.metadata['state']['a'].sharding_metadata)

  @parameterized.product(
      version=['v0', 'v1'],
      alteration=[
          'missing_pytree_data_file_manifest.ocdbt',
          'missing_pytree_data_dir_d',
      ],
  )
  def test_checkpointables_metadata_critical_corruptions(
      self, version: str, alteration: str
  ) -> None:
    """Tests checkpointables_metadata against critical corruptions.

    Args:
      version: The checkpoint version to test against.
      alteration: The alteration to apply to the checkpoint.
    """
    path = self.base_dir.joinpath(
        f'{version}_checkpoints',
        'composite_checkpoint',
        'critical_metadata_alterations',
        alteration,
    )
    # Doesnt fail as we are just accessing the metadata.
    loaded = ocp.checkpointables_metadata(path)
    self.assertIsNone(loaded.metadata.get('state'))


if __name__ == '__main__':
  absltest.main()
