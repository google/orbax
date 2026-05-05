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

"""Tests for V1 pytree_metadata API against generated V0 and V1 Checkpoints."""

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


class PytreeMetadataCompatibilityTestBase(parameterized.TestCase):
  """Tests for V1 pytree_metadata API against generated Checkpoints."""

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
    self.expected_metadata = None

  def setup_registry(
      self,
      name_registered: bool,
      handler_registered: bool,
  ) -> registration.CheckpointableHandlerRegistry:
    """Ensures we only have what we explicitly add."""
    registry = ocp.handlers.local_registry(include_global_registry=False)

    if handler_registered:
      # Register the handler without a specific name.
      # This allows resolution based on handler_typestr.
      # The recognized_handler_typestrs are those used to save the pytree
      # 'state' for V0 composite and direct checkpoints respectively.
      registry.add(
          ocp.handlers.PyTreeHandler,
          checkpointable_name=None,
          secondary_typestrs=[
              'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler',
              'orbax.checkpoint._src.handlers.standard_checkpoint_handler.StandardCheckpointHandler',
          ],
      )
      registry.add(
          ocp.handlers.JsonHandler,
          checkpointable_name=None,
          secondary_typestrs=[
              'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler',
          ],
      )

    if name_registered:
      # We register all checkpointables within the checkpoint as metadata
      # loading requires loading from all contents of the checkpoint.
      registry.add(ocp.handlers.PyTreeHandler, checkpointable_name='state')
      registry.add(ocp.handlers.JsonHandler, checkpointable_name='metadata')

    registry.add(ocp.handlers.PyTreeHandler, checkpointable_name='pytree')
    return registry

  def _determine_expected_outcome(
      self,
      version: str,
      checkpointable_name: str | None,
      metadata_present: bool,
      is_direct_checkpoint: bool,
      is_pytree: bool,
  ) -> Tuple[Type[Exception] | None, str | None]:
    """Encapsulates the complex boolean logic to determine load behavior."""
    # LAYOUT VALIDATION BEHAVIOR:
    if version == 'v1':
      # V1 strictly requires that checkpoint metadata is present.
      if not metadata_present:
        return (
            InvalidLayoutError,
            (
                r'Could not recognize the checkpoint at .* as a valid Orbax'
                r' checkpoint'
            ),
        )
      elif checkpointable_name is None:
        return (
            ValueError,
            r'Failed to interpret path .* as a .* Orbax PyTree',
        )
    elif version == 'v0':
      # V0 behavior specific to our generated checkpoints, as we know that
      # composite checkpoints cannot be loaded as pytrees and that direct
      # checkpoints can only be loaded as pytrees if checkpointable_name is
      # None.
      if not is_direct_checkpoint and checkpointable_name is None or (
          is_direct_checkpoint and checkpointable_name is not None
      ):
        return (
            InvalidLayoutError,
            r'Failed to interpret path .* as a .* Orbax PyTree',
        )

    # Load pytree enforces that pytree metadata is present.
    if not is_pytree:
      return (
          InvalidLayoutError,
          r'Failed to interpret path .* as a .* Orbax PyTree',
      )

    return None, None

  @parameterized.product(
      version=['v0', 'v1'],
      checkpointable_name=['state', None],
      name_registered=[True, False],
      metadata_present=[True, False],
      is_direct_checkpoint=[True, False],
      is_pytree=[True, False],
      handler_registered=[True, False],
  )
  def test_pytree_metadata_compatibility(
      self,
      version: str,
      checkpointable_name: str | None,
      name_registered: bool,
      metadata_present: bool,
      is_direct_checkpoint: bool,
      is_pytree: bool,
      handler_registered: bool,
  ) -> None:
    """Tests pytree_metadata compatibility across V0 and V1 checkpoints.

    Args:
      version: The checkpoint version to test against.
      checkpointable_name: The name of the checkpointable to load.
      name_registered: Whether the checkpointable name is registered.
      metadata_present: Whether the checkpoint metadata file is present.
      is_direct_checkpoint: Whether the checkpoint is a direct checkpoint.
      is_pytree: Whether the checkpointable is a pytree.
      handler_registered: Whether the handler is registered.
    """
    path = compatibility_test_utils.get_checkpoint_path(
        version, metadata_present, is_direct_checkpoint, is_pytree
    )
    if path is None or not path.exists():
      self.skipTest('Checkpoint for combination does not exist.')

    registry = self.setup_registry(
        name_registered,
        handler_registered,
    )

    error_type, error_msg = self._determine_expected_outcome(
        version,
        checkpointable_name,
        metadata_present,
        is_direct_checkpoint,
        is_pytree,
    )

    with ocp.Context(
        checkpointables_options=ocp.options.CheckpointablesOptions(
            registry=registry
        )
    ):
      if error_type is None:
        loaded = ocp.pytree_metadata(
            path,
            checkpointable_name=checkpointable_name,
        )
        expected = self.expected_state_metadata
        actual = loaded.metadata
        test_utils.assert_tree_equal(self, expected, actual)
      else:
        with self.assertRaisesRegex(error_type, error_msg):
          ocp.pytree_metadata(
              path,
              checkpointable_name=checkpointable_name,
          )

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
  def test_pytree_metadata_non_critical_corruptions(
      self, version: str, alteration: str
  ) -> None:
    """Tests pytree_metadata with non-critical corruptions.

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
    loaded = ocp.pytree_metadata(path, checkpointable_name='state')
    expected = self.expected_state_metadata
    actual = loaded.metadata
    test_utils.assert_tree_equal(self, expected, actual)

  @parameterized.product(
      version=['v0', 'v1'],
  )
  def test_pytree_metadata_missing_sharding_corruption(
      self, version: str
  ) -> None:
    """Tests pytree_metadata with missing sharding corruption.

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
    loaded = ocp.pytree_metadata(path, checkpointable_name='state')
    self.assertIsNone(loaded.metadata['a'].sharding_metadata)

  @parameterized.product(
      version=['v0', 'v1'],
      alteration=[
          'missing_pytree_data_file_manifest.ocdbt',
          'missing_pytree_data_dir_d',
      ],
  )
  def test_pytree_metadata_critical_corruptions(
      self, version: str, alteration: str
  ) -> None:
    """Tests pytree_metadata with critical corruptions.

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
    loaded = ocp.pytree_metadata(path, checkpointable_name='state')
    self.assertIsNone(loaded.metadata)


if __name__ == '__main__':
  absltest.main()
