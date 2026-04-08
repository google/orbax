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

"""Tests for V1 load_checkpointables API against generated V0/V1 Checkpoints."""
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


class LoadCheckpointablesCompatibilityTest(parameterized.TestCase):

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
    self.expected_metadata = {'metadata': 'json_data'}
    self.abstract_metadata = None
    self.expected_checkpointables = {
        'state': self.expected_state,
        'metadata': self.expected_metadata,
    }
    self.abstract_checkpointables = {
        'state': self.abstract_state,
        'metadata': self.abstract_metadata,
    }

  def setup_registry(
      self,
      names_registered: bool,
      handler_registered: bool,
  ) -> registration.CheckpointableHandlerRegistry:
    """Ensures we only have what we explicitly add."""
    registry = ocp.handlers.local_registry(include_global_registry=False)

    if handler_registered:
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

    # Corresponds to all checkpointables in the checkpoint being explicitly
    # registered to a handler.
    if names_registered:
      registry.add(ocp.handlers.PyTreeHandler, checkpointable_name='state')
      registry.add(ocp.handlers.JsonHandler, checkpointable_name='metadata')

    registry.add(ocp.handlers.PyTreeHandler, checkpointable_name='pytree')

    return registry

  def _determine_expected_outcome(
      self,
      version: str,
      checkpointable_names_provided: bool,
      names_registered: bool,
      metadata_present: bool,
      is_direct_checkpoint: bool,
      has_pytree_metadata: bool,
      handler_registered: bool,
  ) -> Tuple[Type[Exception] | None, str | None]:
    """Encapsulates the complex boolean logic to determine load behavior."""
    # Direct checkpoints cannot be loaded with load_checkpointables.
    if version == 'v0' and is_direct_checkpoint:
      # Fails attempt to load explicit checkpointables.
      if checkpointable_names_provided:
        return KeyError, (
            r'Requested checkpointables: .* for loading were not found in the'
            r' checkpoint'
        )
      # In attempt to load everything, failure to load pytree contents as
      # checkpointables themselves.
      return KeyError, (
          r'Failed to load checkpointable: .* due to incompatible handler: .*'
      )

    # LAYOUT VALIDATION BEHAVIOR:
    # V1 strictly requires that checkpoint metadata is present. Additionally,
    # for the v0 composite checkpoint, we require metadata to resolve the
    # metadata checkpointable which is not handleable by PyTreeHandler.
    if version == 'v1' and not metadata_present:
      return InvalidLayoutError, (
          r'Could not recognize the checkpoint at .* as a valid Orbax'
          r' checkpoint'
      )
    # Since we are loading all checkpointables, we need pytree metadata to be
    # present for 'state' to properly resolve with PyTreeHandler for load.
    if not has_pytree_metadata:
      return registration.NoEntryError, (
          r'Failed to load checkpointable: .* due to incompatible'
          r' handler: .*'
      )

    # HANDLER RESOLUTION BEHAVIOR:
    # Given that names_registered corresponds to all checkpointables in our test
    # checkpoints being explicitly registered to a handler, we can
    # resolve the handlers for load without issue.
    if names_registered:
      return None, None
    # Otherwise, we require checkpoint handler metadata to be present and for
    # them to be registered to resolve the handlers for load.
    if (
        not handler_registered
        or not metadata_present
    ):
      return registration.NoEntryError, (
          r'Failed to load checkpointable: .* due to incompatible handler: .*'
      )

    return None, None

  @parameterized.product(
      version=['v0', 'v1'],
      checkpointable_names_provided=[True, False],
      abstract_checkpointables_provided=[True, False],
      names_registered=[True, False],
      metadata_present=[True, False],
      is_direct_checkpoint=[True, False],
      has_pytree_metadata=[True, False],
      handler_registered=[True, False],
  )
  def test_load_checkpointables_compatibility(
      self,
      version: str,
      checkpointable_names_provided: bool,
      abstract_checkpointables_provided: bool,
      names_registered: bool,
      metadata_present: bool,
      is_direct_checkpoint: bool,
      has_pytree_metadata: bool,
      handler_registered: bool,
  ) -> None:
    path = compatibility_test_utils.get_checkpoint_path(
        version, metadata_present, is_direct_checkpoint, has_pytree_metadata
    )
    if path is None or not path.exists():
      self.skipTest('Checkpoint for combination does not exist.')

    if not checkpointable_names_provided and abstract_checkpointables_provided:
      self.skipTest(
          'Cannot provide abstract_checkpointables without'
          ' checkpointable_names.'
      )

    registry = self.setup_registry(
        names_registered,
        handler_registered,
    )

    error_type, expected_error_msg = (
        self._determine_expected_outcome(
            version,
            checkpointable_names_provided,
            names_registered,
            metadata_present,
            is_direct_checkpoint,
            has_pytree_metadata,
            handler_registered,
        )
    )

    if checkpointable_names_provided:
      if abstract_checkpointables_provided:
        abstract_checkpointables = self.abstract_checkpointables
      else:
        abstract_checkpointables = {
            'state': None,
            'metadata': None,
        }
    else:
      abstract_checkpointables = None

    with ocp.Context(
        checkpointables_options=ocp.options.CheckpointablesOptions(
            registry=registry
        )
    ):
      if error_type is None:
        loaded = ocp.load_checkpointables(
            path,
            abstract_checkpointables=abstract_checkpointables,
        )
        test_utils.assert_tree_equal(
            self, loaded, self.expected_checkpointables
        )
      else:
        with self.assertRaisesRegex(error_type, expected_error_msg):
          ocp.load_checkpointables(
              path,
              abstract_checkpointables=abstract_checkpointables,
          )

  @parameterized.product(
      version=['v0', 'v1'],
      alteration=[
          'missing_metrics_metadata',
          'missing_performance_metrics_metadata',
          'missing_init_timestamp_nsecs_metadata',
          'missing_commit_timestamp_nsecs_metadata',
          'missing_custom_metadata_metadata',
          'missing_pytree_data_dir_array_metadatas',
          'missing_pytree_data_file__sharding',
      ],
  )
  def test_load_checkpointables_non_critical_corruptions(
      self, version: str, alteration: str
  ) -> None:
    path = self.base_dir.joinpath(
        f'{version}_checkpoints',
        'composite_checkpoint',
        'non_critical_metadata_alterations',
        alteration,
    )
    loaded = ocp.load_checkpointables(
        path, abstract_checkpointables=self.abstract_checkpointables
    )
    test_utils.assert_tree_equal(self, loaded, self.expected_checkpointables)

  @parameterized.product(
      version=['v0', 'v1'],
      alteration=[
          'missing_pytree_data_file_manifest.ocdbt',
          'missing_pytree_data_dir_d',
      ],
  )
  def test_load_checkpointables_critical_corruptions(
      self, version: str, alteration: str
  ) -> None:
    path = self.base_dir.joinpath(
        f'{version}_checkpoints',
        'composite_checkpoint',
        'critical_metadata_alterations',
        alteration,
    )
    error_type = registration.NoEntryError
    # Underlying error is due to Error opening driver, present in stack trace.
    error_msg = r'Failed to load checkpointable: .* due to incompatible handler'
    with self.assertRaisesRegex(error_type, error_msg):
      ocp.load_checkpointables(
          path, abstract_checkpointables=self.abstract_checkpointables
      )


if __name__ == '__main__':
  absltest.main()
