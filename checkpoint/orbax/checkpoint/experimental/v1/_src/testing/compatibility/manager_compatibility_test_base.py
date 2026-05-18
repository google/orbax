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

"""Tests for experimental Checkpointer compatibility against static CheckpointManager checkpoints.

The checkpoints verified by this test suite are checked into the repository
statically to ensure long-term backward compatibility against runtime changes.
"""

# TODO: b/513010203 - The `_test_base` py_library + py_tests pattern can be
# avoided with the `testonly = True` param to reduce the files/targets to
# only py_tests. This is cleaner, less files, and allows us to run specific
# tests in-file with the cider/jetski UIs. Clean up instances of this in the
# compatibility/ dir.

import json
import os
import shutil
from unittest import mock

from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.sharding_utils import make_single_device_sharding
import orbax.checkpoint.experimental.v1 as ocp
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.path import step
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.testing.compatibility import test_utils as compatibility_test_utils


Checkpointer = ocp.training.Checkpointer
RootMetadata = ocp.training.RootMetadata
CheckpointMetadata = ocp.training.CheckpointMetadata


_BASE_DIR = os.path.join(
    os.path.dirname(__file__), 'managed_checkpoints'
)


class ManagerCompatibilityTestBase(parameterized.TestCase):
  """Comprehensive static parameterization load testing of Checkpointer using V0 CheckpointManager checkpoints."""

  def setUp(self) -> None:
    super().setUp()
    self.base_dir = epath.Path(_BASE_DIR)

    # Expected checkpointables
    self.expected_state = {
        'a': jnp.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=jnp.int32),
        'b': {'c': jnp.array([1, 2, 3], dtype=jnp.int32)},
    }
    self.expected_metadata = {'metadata': 'json_data'}

    # Checkpointer/Manager related fields.
    self.expected_metrics = {'loss': 0.5}
    self.checkpointer_custom_metadata = {'foo': 'bar'}
    self.checkpoint_custom_metadata = {'custom': 'meta'}

    sharding = make_single_device_sharding(jax.devices()[0])
    if multihost.is_pathways_backend() or jax.process_count() > 1:
      self.expected_state = compatibility_test_utils.replicate_on_mesh(
          self.expected_state
      )
      mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('devices',))
      sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    self.expected_checkpointables = {
        'state': self.expected_state,
        'metadata': self.expected_metadata,
    }

    self.abstract_state = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding),
        self.expected_state
    )
    self.abstract_checkpointables = {
        'state': self.abstract_state,
        'metadata': None,
    }

    self.expected_state_metadata = jax.tree.map(
        compatibility_test_utils.create_value_metadata, self.expected_state
    )
    self.expected_checkpointables_metadata = {
        'state': self.expected_state_metadata,
        'metadata': None,
    }

    # Prevent CheckpointManager from attempting to call mkdir on read-only
    # runfiles directories.
    patcher1 = mock.patch.object(
        checkpoint_manager, '_create_root_directory', autospec=True
    )
    patcher2 = mock.patch.object(
        checkpoint_manager.CheckpointManager,
        '_maybe_save_root_metadata',
        autospec=True,
    )
    self.enter_context(patcher1)
    self.enter_context(patcher2)

  def setup_registry(self) -> registration.CheckpointableHandlerRegistry:
    """Sets up a registry for the test."""
    registry = ocp.handlers.local_registry()
    registry.add(
        ocp.handlers.PyTreeHandler,
        checkpointable_name='state',
        secondary_typestrs=[
            'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler',
        ],
    )

    return registry

  def _create_temporary_checkpoint(
      self,
      version: str,
      metrics_status: bool,
      root_metadata_status: bool,
      name_format: step.NameFormat | None = None,
  ) -> epath.Path:
    """Creates a temporary checkpoint for testing.

    Args:
      version: The checkpoint version to load.
      metrics_status: Whether metrics are present.
      root_metadata_status: Whether root metadata is present.
      name_format: The name format to use for the step directory.

    Returns:
      The path to the temporary checkpoint.
    """
    # Always use metadata_present=True to ensure a complete checkpoint exists.
    src_path = compatibility_test_utils.get_checkpoint_path(
        version,
        metadata_present=True,
        is_direct_checkpoint=False,
        is_pytree=True,
    )
    if src_path is None or not src_path.exists():
      raise ValueError(f'Source path does not exist: {src_path}')

    temp_dir = epath.Path(self.create_tempdir().full_path)
    step_folder = name_format.build_name(0) if name_format else '0'
    step_dir = temp_dir / step_folder
    step_dir.mkdir()
    shutil.copytree(str(src_path), str(step_dir), dirs_exist_ok=True)

    # Make step_dir writable so we can add files/directories inside it
    os.chmod(str(step_dir), 0o755)

    if metrics_status:
      metrics_dir = step_dir / 'metrics'
      metrics_dir.mkdir(exist_ok=True)
      metrics_file = metrics_dir / 'metrics'
      metrics_file.write_text('{"loss": 0.5}')

    # Handle metadata variation
    if root_metadata_status:
      # Write _ROOT_METADATA in temp_dir/metadata/
      metadata_dir = temp_dir / 'metadata'
      metadata_dir.mkdir(exist_ok=True)
      root_metadata_file = metadata_dir / '_ROOT_METADATA'
      root_metadata_file.write_text(
          json.dumps({'custom_metadata': {'foo': 'bar'}})
      )

    return temp_dir

  @parameterized.product(
      version=['v0', 'v1'],
      metrics_status=[True, False],
      root_metadata_status=[True, False],
      name_format=[
          None,
          step.standard_name_format(step_prefix='checkpoint'),
      ],
  )
  def test_properties(
      self,
      version: str,
      metrics_status: bool,
      root_metadata_status: bool,
      name_format: step.NameFormat | None = None,
  ) -> None:
    """Tests basic properties of Checkpointer initialized from a static path.

    Args:
      version: The checkpoint version to load.
      metrics_status: Whether metrics are present.
      root_metadata_status: Whether root metadata is present.
      name_format: The name format to use for the step directory.
    """
    path = self._create_temporary_checkpoint(
        version,
        metrics_status,
        root_metadata_status,
        name_format=name_format,
    )
    checkpointer = Checkpointer(path, step_name_format=name_format)
    self.enter_context(checkpointer)
    self.assertEqual(checkpointer.directory, path)
    self.assertIsNotNone(checkpointer.latest)
    assert checkpointer.latest is not None
    self.assertEqual(checkpointer.latest.step, 0)

  @parameterized.product(
      version=['v0', 'v1'],
      metrics_status=[True, False],
      root_metadata_status=[True, False],
      name_format=[
          None,
          step.standard_name_format(step_prefix='checkpoint'),
      ],
  )
  def test_root_metadata(
      self,
      version: str,
      metrics_status: bool,
      root_metadata_status: bool,
      name_format: step.NameFormat | None = None,
  ) -> None:
    """Verifies sequence-level root metadata parsing and existence.

    Args:
      version: The checkpoint version to load.
      metrics_status: Whether metrics are present.
      root_metadata_status: Whether root metadata is present.
      name_format: The name format to use for the step directory.
    """
    path = self._create_temporary_checkpoint(
        version,
        metrics_status,
        root_metadata_status,
        name_format=name_format,
    )
    checkpointer = Checkpointer(path, step_name_format=name_format)
    self.enter_context(checkpointer)
    root_metadata = checkpointer.root_metadata()
    self.assertIsInstance(root_metadata, RootMetadata)
    if root_metadata_status:
      self.assertDictEqual(
          root_metadata.custom_metadata, self.checkpointer_custom_metadata
      )
    else:
      self.assertIn(root_metadata.custom_metadata, (None, {}))

  @parameterized.product(
      version=['v0', 'v1'],
      metrics_status=[True, False],
      root_metadata_status=[True, False],
      name_format=[
          None,
          step.standard_name_format(step_prefix='checkpoint'),
      ],
  )
  def test_pytree_metadata(
      self,
      version: str,
      metrics_status: bool,
      root_metadata_status: bool,
      name_format: step.NameFormat | None = None,
  ) -> None:
    """Verifies structural array metadata layout resolution and validation.

    Args:
      version: The checkpoint version to load.
      metrics_status: Whether metrics are present.
      root_metadata_status: Whether root metadata is present.
      name_format: The name format to use for the step directory.
    """
    path = self._create_temporary_checkpoint(
        version,
        metrics_status,
        root_metadata_status,
        name_format=name_format,
    )
    registry = self.setup_registry()
    context = ocp.Context()
    context.checkpointables.registry = registry
    self.enter_context(context)
    checkpointer = Checkpointer(path, step_name_format=name_format)
    self.enter_context(checkpointer)

    metadata = checkpointer.pytree_metadata(0)
    self.assertIsInstance(metadata, CheckpointMetadata)
    actual = compatibility_test_utils.strip_sharding_metadata(
        metadata.metadata
    )
    expected = compatibility_test_utils.strip_sharding_metadata(
        self.expected_state_metadata
    )
    test_utils.assert_tree_equal(self, actual, expected)
    if metrics_status:
      self.assertDictEqual(metadata.metrics, self.expected_metrics)
    else:
      self.assertIsNone(metadata.metrics)

  @parameterized.product(
      version=['v0', 'v1'],
      metrics_status=[True, False],
      root_metadata_status=[True, False],
      name_format=[
          None,
          step.standard_name_format(step_prefix='checkpoint'),
      ],
  )
  def test_checkpointables_metadata(
      self,
      version: str,
      metrics_status: bool,
      root_metadata_status: bool,
      name_format: step.NameFormat | None = None,
  ) -> None:
    """Verifies multi-item checkpointables metadata retrieval.

    Args:
      version: The checkpoint version to load.
      metrics_status: Whether metrics are present.
      root_metadata_status: Whether root metadata is present.
      name_format: The name format to use for the step directory.
    """
    path = self._create_temporary_checkpoint(
        version,
        metrics_status,
        root_metadata_status,
        name_format=name_format,
    )
    registry = self.setup_registry()
    context = ocp.Context()
    context.checkpointables.registry = registry
    self.enter_context(context)
    checkpointer = Checkpointer(path, step_name_format=name_format)
    self.enter_context(checkpointer)

    metadata = checkpointer.checkpointables_metadata(0)
    self.assertIsInstance(metadata, CheckpointMetadata)
    self.assertDictEqual(
        metadata.custom_metadata, self.checkpoint_custom_metadata
    )
    if metrics_status:
      self.assertDictEqual(metadata.metrics, self.expected_metrics)
    else:
      self.assertIsNone(metadata.metrics)

    actual = metadata.metadata
    expected = self.expected_checkpointables_metadata
    if multihost.is_pathways_backend() or jax.process_count() > 1:
      expected = compatibility_test_utils.strip_sharding_metadata(expected)
      actual = compatibility_test_utils.strip_sharding_metadata(actual)
    test_utils.assert_tree_equal(self, actual, expected)

  @parameterized.product(
      version=['v0', 'v1'],
      metrics_status=[True, False],
      root_metadata_status=[True, False],
      name_format=[
          None,
          step.standard_name_format(step_prefix='checkpoint'),
      ],
  )
  def test_load_checkpointables(
      self,
      version: str,
      metrics_status: bool,
      root_metadata_status: bool,
      name_format: step.NameFormat | None = None,
  ) -> None:
    """Verifies explicit load dictionary alignment matching abstract shapes.

    Args:
      version: The checkpoint version to load.
      metrics_status: Whether metrics are present.
      root_metadata_status: Whether root metadata is present.
      name_format: The name format to use for the step directory.
    """
    path = self._create_temporary_checkpoint(
        version,
        metrics_status,
        root_metadata_status,
        name_format=name_format,
    )
    registry = self.setup_registry()
    context = ocp.Context()
    context.checkpointables.registry = registry
    self.enter_context(context)
    checkpointer = Checkpointer(path, step_name_format=name_format)
    self.enter_context(checkpointer)

    abstract_checkpointables = self.abstract_checkpointables
    loaded = checkpointer.load_checkpointables(0, abstract_checkpointables)
    test_utils.assert_tree_equal(self, self.expected_checkpointables, loaded)

  @parameterized.product(
      version=['v0', 'v1'],
      metrics_status=[True, False],
      root_metadata_status=[True, False],
      name_format=[
          None,
          step.standard_name_format(step_prefix='checkpoint'),
      ],
  )
  def test_load_pytree(
      self,
      version: str,
      metrics_status: bool,
      root_metadata_status: bool,
      name_format: step.NameFormat | None = None,
  ) -> None:
    """Verifies load_pytree API against generated Checkpoints.

    Args:
      version: The checkpoint version to load.
      metrics_status: Whether metrics are present.
      root_metadata_status: Whether root metadata is present.
      name_format: The step name format to use.
    """
    path = self._create_temporary_checkpoint(
        version,
        metrics_status,
        root_metadata_status,
        name_format=name_format,
    )
    registry = self.setup_registry()
    context = ocp.Context()
    context.checkpointables.registry = registry
    self.enter_context(context)
    checkpointer = Checkpointer(path, step_name_format=name_format)
    self.enter_context(checkpointer)

    loaded = checkpointer.load_pytree(
        0, abstract_pytree=self.abstract_state, checkpointable_name='state'
    )
    test_utils.assert_tree_equal(self, self.expected_state, loaded)
