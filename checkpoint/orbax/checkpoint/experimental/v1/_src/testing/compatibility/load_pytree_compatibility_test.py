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

"""Tests for V1 load_pytree API against generated V0 and V1 Checkpoints."""
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
from orbax.checkpoint.experimental.v1._src.testing.compatibility import test_utils as compatibility_test_utils
import orbax.checkpoint.utils


CheckpointLayoutEnum = options_lib.CheckpointLayout
InvalidLayoutError = checkpoint_layout_lib.InvalidLayoutError


_BASE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')


class LoadPytreeCompatibilityTest(parameterized.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.base_dir = epath.Path(_BASE_DIR)
    self.expected_state = {
        'a': jnp.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=jnp.int32),
        'b': {'c': jnp.array([1, 2, 3], dtype=jnp.int32)},
    }
    sharding = orbax.checkpoint.utils.make_single_device_sharding(
        jax.devices()[0]
    )
    self.abstract_state = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding),
        self.expected_state
    )

  def setup_registry(
      self,
      path: epath.Path,
      checkpointable_name: str | None,
      name_registered: bool,
      handler_registered: bool,
      pytree_registered: bool,
  ) -> registration.CheckpointableHandlerRegistry:
    """Ensures we only have what we explicitly add."""
    registry = ocp.handlers.local_registry(include_global_registry=False)

    # This reflects when a user has a checkpoint saved with unregistered
    # handler typestrs in metadata, if not handler_registered, handler
    # resolution will fail when trying to resolve based on metadata.
    if handler_registered:
      # The secondary_typestrs are those used to save the pytree
      # 'state' for V0 composite and direct checkpoints respectively.
      secondary_typestrs = [
          'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler',
          'orbax.checkpoint._src.handlers.standard_checkpoint_handler.StandardCheckpointHandler',
      ]
      registry.add(
          ocp.handlers.PyTreeHandler,
          checkpointable_name=None,
          secondary_typestrs=secondary_typestrs,
      )

    if name_registered:
      # Register the handler with a specific name, possibly the top-level
      # checkpointable name if provided, otherwise we use the path name assuming
      # we have a top-level pytree checkpoint.
      if checkpointable_name:
        registry.add(
            ocp.handlers.PyTreeHandler, checkpointable_name=checkpointable_name
        )
      else:
        registry.add(ocp.handlers.PyTreeHandler, checkpointable_name=path.name)

    if pytree_registered:
      # Register to scoped 'pytree' handler for fallback resolution.
      # Note this should standardly be present, though testing its presence to
      # ensure resolution works as expected without always relying on it.
      registry.add(ocp.handlers.PyTreeHandler, checkpointable_name='pytree')

    return registry

  def _determine_expected_outcome(
      self,
      version: str,
      checkpointable_name: str | None,
      abstract_pytree_provided: bool,
      name_registered: bool,
      metadata_present: bool,
      is_direct_checkpoint: bool,
      is_pytree: bool,
      handler_registered: bool,
      pytree_registered: bool,
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
      # V1 does not support loading a top-level pytree, redirects to V0.
      if checkpointable_name is None:
        return (
            ValueError,
            (
                r'Failed to interpret path .* as a .* Orbax PyTree'
            ),
        )

    # If checkpoint is not a pytree or if the layout logic fails to identify a
    # pytree checkpoint at the checkpointable_name path, then we expect an
    # InvalidLayoutError.
    if not is_pytree or (
        (is_direct_checkpoint and checkpointable_name)
        or (not is_direct_checkpoint and checkpointable_name is None)
    ):
      return (
          InvalidLayoutError,
          (
              r'Failed to interpret path .* as a .* Orbax PyTree'
          ),
      )

    # HANDLER RESOLUTION BEHAVIOR:
    can_resolve = (
        # If checkpointable name is explicitly registered to a handler, we can
        # use it for load.
        name_registered
        # If a handler is registered which corresponds to the handler typestr
        # derived from checkpoint metadata and abstract_pytree if either are
        # provided.
        or (
            handler_registered
            and (abstract_pytree_provided or metadata_present)
        )
        or pytree_registered
    )

    if not can_resolve:
      return (
          registration.NoEntryError,
          (
              r'Could not resolve a handler for .* and no \'pytree\' handler'
              r' found in .*'
          ),
      )

    return None, None

  @parameterized.product(
      version=['v0', 'v1'],
      checkpointable_name=['state', None],
      abstract_pytree_provided=[True, False],
      name_registered=[True, False],
      metadata_present=[True, False],
      is_direct_checkpoint=[True, False],
      is_pytree=[True, False],
      handler_registered=[True, False],
      pytree_registered=[True, False],
  )
  def test_load_pytree_compatibility(
      self,
      version: str,
      checkpointable_name: str | None,
      abstract_pytree_provided: bool,
      name_registered: bool,
      metadata_present: bool,
      is_direct_checkpoint: bool,
      is_pytree: bool,
      handler_registered: bool,
      pytree_registered: bool,
  ) -> None:
    path = compatibility_test_utils.get_checkpoint_path(
        version, metadata_present, is_direct_checkpoint, is_pytree
    )
    if path is None or not path.exists():
      self.skipTest('Checkpoint for combination does not exist.')

    registry = self.setup_registry(
        path,
        checkpointable_name,
        name_registered,
        handler_registered,
        pytree_registered,
    )

    error_type, expected_error_msg = (
        self._determine_expected_outcome(
            version,
            checkpointable_name,
            abstract_pytree_provided,
            name_registered,
            metadata_present,
            is_direct_checkpoint,
            is_pytree,
            handler_registered,
            pytree_registered,
        )
    )

    actual_abstract_pytree = (
        self.abstract_state if abstract_pytree_provided else None
    )

    with ocp.Context(
        checkpointables_options=ocp.options.CheckpointablesOptions(
            registry=registry
        )
    ):
      if error_type is None:
        loaded = ocp.load_pytree(
            path,
            checkpointable_name=checkpointable_name,
            abstract_pytree=actual_abstract_pytree,
        )
        test_utils.assert_tree_equal(self, loaded, self.expected_state)
      else:
        with self.assertRaisesRegex(error_type, expected_error_msg):
          ocp.load_pytree(
              path,
              checkpointable_name=checkpointable_name,
              abstract_pytree=actual_abstract_pytree,
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
          'missing_pytree_data_file__sharding',
      ],
  )
  def test_load_pytree_non_critical_corruptions(
      self, version: str, alteration: str
  ) -> None:

    path = self.base_dir.joinpath(
        f'{version}_checkpoints',
        'composite_checkpoint',
        'non_critical_metadata_alterations',
        alteration,
    )
    loaded = ocp.load_pytree(
        path, abstract_pytree=self.abstract_state, checkpointable_name='state'
    )
    test_utils.assert_tree_equal(self, loaded, self.expected_state)

  @parameterized.product(
      version=['v0', 'v1'],
      alteration=[
          'missing_pytree_data_file_manifest.ocdbt',
          'missing_pytree_data_dir_d',
      ],
  )
  def test_load_pytree_critical_corruptions(
      self, version: str, alteration: str
  ) -> None:
    path = self.base_dir.joinpath(
        f'{version}_checkpoints',
        'composite_checkpoint',
        'critical_metadata_alterations',
        alteration,
    )
    error_type = ValueError
    error_msg = r'Error opening .* driver:'
    with self.assertRaisesRegex(error_type, error_msg):
      ocp.load_pytree(
          path,
          checkpointable_name='state',
          abstract_pytree=self.abstract_state,
      )

  @parameterized.product(
      version=['v0', 'v1'],
  )
  def test_load_incorrect_path(self, version: str) -> None:
    checkpoint_path = (
        self.base_dir
        / f'{version}_checkpoints'
        / 'composite_checkpoint'
        / 'checkpoint_metadata_present'
        / 'pytree_checkpointable_has_metadata'
    )
    child_path = checkpoint_path / 'state'
    parent_path = checkpoint_path.parent
    with self.assertRaisesRegex(
        InvalidLayoutError,
        r'Could not recognize the checkpoint at .* as a valid Orbax checkpoint'
    ):
      ocp.load_pytree(child_path, checkpointable_name='state')
    with self.assertRaisesRegex(
        InvalidLayoutError,
        r'Could not recognize the checkpoint at .* as a valid Orbax checkpoint'
    ):
      ocp.load_pytree(parent_path, checkpointable_name='state')


if __name__ == '__main__':
  absltest.main()
