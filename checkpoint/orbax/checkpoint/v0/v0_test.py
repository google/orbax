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

"""Tests for orbax.checkpoint.v0 public namespace and submodules."""

import inspect
from absl.testing import absltest
import orbax.checkpoint as ocp_top
from orbax.checkpoint import v0 as ocp_v0
import orbax.checkpoint.v0 as ocp_v0_direct  # pylint: disable=reimported


class V0NamespaceTest(absltest.TestCase):

  def test_namespace_identity(self):
    self.assertIs(ocp_v0, ocp_v0_direct)
    self.assertIs(ocp_top.v0, ocp_v0)

  def test_top_level_classes_and_functions_exposed_and_identical(self):
    expected_symbols = [
        'AbstractCheckpointManager',
        'AbstractCheckpointer',
        'ArrayCheckpointHandler',
        'ArrayRestoreArgs',
        'AsyncCheckpointHandler',
        'AsyncCheckpointer',
        'AsyncOptions',
        'BasePyTreeCheckpointHandler',
        'CheckpointHandler',
        'CheckpointHandlerRegistry',
        'CheckpointManager',
        'CheckpointManagerOptions',
        'Checkpointer',
        'CompositeCheckpointHandler',
        'CompositeOptions',
        'DefaultCheckpointHandlerRegistry',
        'Future',
        'JaxRandomKeyCheckpointHandler',
        'JsonCheckpointHandler',
        'NumpyRandomKeyCheckpointHandler',
        'PLACEHOLDER',
        'ProtoCheckpointHandler',
        'PyTreeCheckpointHandler',
        'PyTreeCheckpointer',
        'RestoreArgs',
        'RestoreTransform',
        'SaveArgs',
        'StandardCheckpointHandler',
        'StandardCheckpointer',
        'Transform',
        '__version__',
        'apply_transformations',
        'merge_trees',
    ]
    for symbol in expected_symbols:
      self.assertTrue(hasattr(ocp_v0, symbol), f'Missing {symbol} on v0')
      self.assertIs(
          getattr(ocp_v0, symbol),
          getattr(ocp_top, symbol),
          f'Mismatch for {symbol} between ocp.v0 and ocp',
      )

  def test_submodules_exposed_and_symbols_identical(self):
    expected_submodules = [
        'aggregate_handlers',
        'args',
        'arrays',
        'checkpoint_managers',
        'checkpoint_utils',
        'checkpointers',
        'handlers',
        'logging',
        'metadata',
        'msgpack_utils',
        'multihost',
        'options',
        'path',
        'pathways',
        'serialization',
        'step',
        'test_utils',
        'transform_utils',
        'tree',
        'type_handlers',
        'utils',
    ]
    for submod_name in expected_submodules:
      self.assertTrue(
          hasattr(ocp_v0, submod_name), f'Missing submodule {submod_name} on v0'
      )
      v0_submod = getattr(ocp_v0, submod_name)
      top_submod = getattr(ocp_top, submod_name)
      for attr_name in dir(top_submod):
        if attr_name.startswith('__'):
          continue
        top_attr = getattr(top_submod, attr_name)
        if inspect.ismodule(top_attr):
          continue
        if getattr(top_attr, '__module__', None) in (
            '__future__',
            'typing',
            'typing_extensions',
            'collections.abc',
        ) and attr_name not in ('PyTree',):
          continue
        self.assertTrue(
            hasattr(v0_submod, attr_name),
            f'Missing {attr_name} on ocp.v0.{submod_name}',
        )
        self.assertIs(
            getattr(v0_submod, attr_name),
            top_attr,
            f'Identity mismatch for ocp.v0.{submod_name}.{attr_name}',
        )

  def test_direct_submodule_imports(self):
    # pylint: disable=g-import-not-at-top
    import orbax.checkpoint.multihost as top_multihost
    from orbax.checkpoint.v0 import abstract_checkpoint_manager
    from orbax.checkpoint.v0 import aggregate_handlers
    from orbax.checkpoint.v0 import args
    from orbax.checkpoint.v0 import arrays
    from orbax.checkpoint.v0 import checkpoint_args
    from orbax.checkpoint.v0 import checkpoint_manager
    from orbax.checkpoint.v0 import checkpoint_managers
    from orbax.checkpoint.v0 import checkpoint_utils
    from orbax.checkpoint.v0 import checkpointers
    from orbax.checkpoint.v0 import future
    from orbax.checkpoint.v0 import handlers
    from orbax.checkpoint.v0 import logging
    from orbax.checkpoint.v0 import metadata
    from orbax.checkpoint.v0 import msgpack_utils
    from orbax.checkpoint.v0 import multihost
    from orbax.checkpoint.v0 import options
    from orbax.checkpoint.v0 import path
    from orbax.checkpoint.v0 import pathways
    from orbax.checkpoint.v0 import serialization
    from orbax.checkpoint.v0 import test_utils
    from orbax.checkpoint.v0 import testing
    from orbax.checkpoint.v0 import transform_utils
    from orbax.checkpoint.v0 import tree
    from orbax.checkpoint.v0 import type_handlers
    from orbax.checkpoint.v0 import utils
    from orbax.checkpoint.v0.logging import step_statistics
    from orbax.checkpoint.v0.metadata import tree as metadata_tree
    from orbax.checkpoint.v0.metadata import value as metadata_value
    from orbax.checkpoint.v0.multihost import multislice
    from orbax.checkpoint.v0.serialization import tensorstore_utils
    from orbax.checkpoint.v0.testing import local_path
    # pylint: enable=g-import-not-at-top

    self.assertIs(top_multihost.process_index, multihost.process_index)

    self.assertIs(
        abstract_checkpoint_manager.AbstractCheckpointManager,
        ocp_top.AbstractCheckpointManager,
    )
    self.assertIs(
        aggregate_handlers.MsgpackHandler,
        ocp_top.aggregate_handlers.MsgpackHandler,
    )
    self.assertIs(args.StandardSave, ocp_top.args.StandardSave)
    self.assertIs(args.CheckpointArgs, checkpoint_args.CheckpointArgs)
    self.assertIs(
        arrays.to_shape_dtype_struct, ocp_top.arrays.to_shape_dtype_struct
    )
    self.assertIs(
        checkpoint_manager.CheckpointManager, ocp_top.CheckpointManager
    )
    self.assertIs(
        checkpoint_managers.LatestN, ocp_top.checkpoint_managers.LatestN
    )
    self.assertIs(
        checkpoint_utils.checkpoints_iterator,
        ocp_top.checkpoint_utils.checkpoints_iterator,
    )
    self.assertIs(checkpointers.Checkpointer, ocp_top.Checkpointer)
    self.assertIs(future.Future, ocp_top.Future)
    self.assertIs(
        handlers.StandardCheckpointHandler, ocp_top.StandardCheckpointHandler
    )
    self.assertIs(logging.StandardLogger, ocp_top.logging.StandardLogger)
    self.assertIs(
        step_statistics.SaveStepStatistics,
        ocp_top.logging.step_statistics.SaveStepStatistics,
    )
    self.assertIs(metadata.StepMetadata, ocp_top.metadata.StepMetadata)
    self.assertIs(
        metadata_tree.ValueMetadataEntry,
        ocp_top.metadata.tree.ValueMetadataEntry,
    )
    self.assertIs(
        metadata_value.ArrayMetadata, ocp_top.metadata.value.ArrayMetadata
    )
    self.assertIs(
        msgpack_utils.msgpack_serialize,
        ocp_top.msgpack_utils.msgpack_serialize,
    )
    self.assertIs(multihost.process_index, ocp_top.multihost.process_index)
    self.assertIsNotNone(multislice)
    self.assertIs(options.FileOptions, ocp_top.options.FileOptions)
    self.assertIs(path.step, ocp_top.path.step)
    self.assertIs(
        pathways.CheckpointingImpl, ocp_top.pathways.CheckpointingImpl
    )
    self.assertIs(
        serialization.async_serialize, ocp_top.serialization.async_serialize
    )
    self.assertIs(
        tensorstore_utils.get_ts_context,
        ocp_top.serialization.ts_utils.get_ts_context,
    )
    self.assertIs(
        test_utils.assert_tree_equal, ocp_top.test_utils.assert_tree_equal
    )
    self.assertIsNotNone(testing.test_tree_utils)
    self.assertIsNotNone(local_path.LocalPath)
    self.assertIs(
        transform_utils.apply_transformations, ocp_top.apply_transformations
    )
    self.assertIs(tree.to_flat_dict, ocp_top.tree.to_flat_dict)
    self.assertIs(
        type_handlers.ArrayHandler, ocp_top.type_handlers.ArrayHandler
    )
    self.assertIs(utils.checkpoint_steps, ocp_top.utils.checkpoint_steps)

  def test_v0_src_and_top_src_identity(self):
    # pylint: disable=g-import-not-at-top
    from orbax.checkpoint._src.handlers import standard_checkpoint_handler as top_sch
    from orbax.checkpoint._src.serialization import type_handlers as top_th
    from orbax.checkpoint.v0._src.handlers import standard_checkpoint_handler as v0_sch
    from orbax.checkpoint.v0._src.serialization import type_handlers as v0_th
    # pylint: enable=g-import-not-at-top

    self.assertIs(v0_sch, top_sch)
    self.assertIs(v0_th, top_th)
    self.assertIs(ocp_v0._src.path.step, ocp_top._src.path.step)
    self.assertIs(
        ocp_v0._src.multihost.multihost,
        ocp_top._src.multihost.multihost,
    )

  def test_handler_typestr_compatibility(self):
    # pylint: disable=g-import-not-at-top
    from orbax.checkpoint.v0._src.handlers import handler_type_registry
    # pylint: enable=g-import-not-at-top

    self.assertEqual(
        ocp_v0.StandardCheckpointHandler.typestr(),
        'orbax.checkpoint.v0._src.handlers.standard_checkpoint_handler.'
        'StandardCheckpointHandler',
    )
    self.assertIs(
        handler_type_registry.get_handler_type(
            'orbax.checkpoint._src.handlers.standard_checkpoint_handler.'
            'StandardCheckpointHandler'
        ),
        ocp_v0.StandardCheckpointHandler,
    )
    self.assertIs(
        handler_type_registry.get_handler_type(
            'orbax.checkpoint.v0._src.handlers.standard_checkpoint_handler.'
            'StandardCheckpointHandler'
        ),
        ocp_v0.StandardCheckpointHandler,
    )


if __name__ == '__main__':
  absltest.main()
