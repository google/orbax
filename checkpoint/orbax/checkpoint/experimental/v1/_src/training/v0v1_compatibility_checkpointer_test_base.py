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

"""Base class for v0/v1 compatibility checkpointer tests."""

from typing import Any

from absl.testing import parameterized
from orbax.checkpoint import args as v0_args
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import handlers as v0_handlers
import orbax.checkpoint.experimental.v1 as ocp
from orbax.checkpoint.experimental.v1._src.testing import handler_utils
from orbax.checkpoint.experimental.v1._src.training import checkpointer_test_base
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


Checkpointer = ocp.training.Checkpointer
RootMetadata = ocp.training.RootMetadata
Foo = handler_utils.Foo
Bar = handler_utils.Bar
Baz = handler_utils.Baz


def build_handler_registry_and_checkpoint_args(
    checkpointables: dict[str, Any],
) -> tuple[v0_handlers.CheckpointHandlerRegistry, v0_args.Composite]:
  """Builds v0 handler registry and checkpoint args from checkpointables."""
  checkpoint_args = {}
  handler_registry = {}
  for name, checkpointable in checkpointables.items():
    if name == 'pytree':
      checkpoint_args['pytree'] = v0_args.PyTreeSave(checkpointable)
      handler_registry['pytree'] = v0_handlers.PyTreeCheckpointHandler()
    elif isinstance(checkpointable, (Foo, Bar, Baz)):
      checkpoint_args[name] = handler_utils.DataclassSaveArgs(checkpointable)
      handler_registry[name] = handler_utils.DataclassCheckpointHandler()
    else:
      raise NotImplementedError(
          f'Unsupported checkpointable: name={name}, value={checkpointable}'
      )
  return (
      v0_handlers.create_default_handler_registry(**handler_registry),
      v0_args.Composite(**checkpoint_args),
  )


class V0v1CompatibilityCheckpointerTestBase(
    checkpointer_test_base.CheckpointerTestBase.Test
):
  """Base class for v0/v1 compatibility checkpointer tests."""

  def create_checkpoint_manager(
      self,
      checkpointer: Checkpointer,
      handler_registry: v0_handlers.CheckpointHandlerRegistry,
  ) -> checkpoint_manager.CheckpointManager:
    return checkpoint_manager.CheckpointManager(
        self.directory,
        options=checkpoint_manager.CheckpointManagerOptions(
            create=False,
            enable_async_checkpointing=False,
            save_decision_policy=(
                checkpointer._manager._options.save_decision_policy  # pylint: disable=protected-access
            ),
            step_name_format=checkpointer._manager._options.step_name_format,  # pylint: disable=protected-access
            prevent_write_metrics=False,  # Write metrics on disk.
            best_fn=lambda metrics: metrics['loss'],  # Write metrics on disk.
        ),
        handler_registry=handler_registry,
    )

  def save_pytree(
      self,
      checkpointer: Checkpointer,
      step: int,
      pytree: tree_types.PyTreeOf[tree_types.LeafType],
      metrics: tree_types.JsonType | None = None,
      custom_metadata: tree_types.JsonType | None = None,
  ) -> bool:
    """Saves a pytree checkpoint with v0 CheckpointManager."""
    self.skipTest('b/422287659')
    manager = self.create_checkpoint_manager(
        checkpointer,
        handler_registry=v0_handlers.create_default_handler_registry(
            pytree=v0_handlers.PyTreeCheckpointHandler()
        ),
    )
    with manager:
      result = manager.save(
          step,
          args=v0_args.Composite(pytree=v0_args.PyTreeSave(pytree)),
          metrics=metrics,
          custom_metadata=custom_metadata,
      )
      manager.wait_until_finished()
      checkpointer.reload()  # Reload v1 checkpointer to sync with v0 ckpts.
      return result

  def save_checkpointables(
      self,
      checkpointer: Checkpointer,
      step: int,
      checkpointables: dict[str, Any],
      metrics: tree_types.JsonType | None = None,
      custom_metadata: tree_types.JsonType | None = None,
  ) -> bool:
    handler_registry, checkpoint_args = (
        build_handler_registry_and_checkpoint_args(checkpointables)
    )
    manager = self.create_checkpoint_manager(checkpointer, handler_registry)
    with manager:
      result = manager.save(
          step,
          args=checkpoint_args,
          metrics=metrics,
          custom_metadata=custom_metadata,
      )
      manager.wait_until_finished()
      checkpointer.reload()  # Reload v1 checkpointer to sync with v0 ckpts.
      return result

  def test_skips_when_ongoing_save(self):
    self.skipTest('Not relevant for v0/v1 compatibility.')

  def test_save_pytree_async(self):
    self.skipTest('Not relevant for v0/v1 compatibility.')

  def test_close(self):
    self.skipTest('Not relevant for v0/v1 compatibility.')

  def test_context_manager_close(self):
    self.skipTest('Not relevant for v0/v1 compatibility.')

  @parameterized.product(
      reinitialize_checkpointer=(True, False),
  )
  def test_root_metadata(self, reinitialize_checkpointer):
    if not reinitialize_checkpointer:
      self.skipTest('Not relevant for v0/v1 compatibility.')

    with checkpoint_manager.CheckpointManager(
        self.directory,
        options=checkpoint_manager.CheckpointManagerOptions(
            create=False,
            enable_async_checkpointing=False,
        ),
        handler_registry=v0_handlers.create_default_handler_registry(
            pytree=v0_handlers.PyTreeCheckpointHandler()
        ),
        metadata={'foo': 'bar'},  # saved during manager instance creation.
    ):
      pass
    # Does not overwrite custom metadata.
    checkpointer = Checkpointer(self.directory, custom_metadata={'baz': 2})
    self.enter_context(checkpointer)
    root_metadata = checkpointer.root_metadata()
    self.assertIsInstance(root_metadata, RootMetadata)
    self.assertDictEqual(root_metadata.custom_metadata, {'foo': 'bar'})

  def test_custom_checkpointables(self):
    # Use named handler to override v0 checkpoint_handlers.
    checkpointables_options = (
        ocp.options.CheckpointablesOptions.create_with_handlers(
            foo=handler_utils.FooHandler,
            bar=handler_utils.BarHandler,
        )
    )
    self.enter_context(
        ocp.Context(checkpointables_options=checkpointables_options)
    )
    super().test_custom_checkpointables()

  def test_load_with_switched_abstract_checkpointables(self):
    # Use named handler to override v0 checkpoint_handlers.
    checkpointables_options = (
        ocp.options.CheckpointablesOptions.create_with_handlers(
            bar=handler_utils.FooHandler,
            foo=handler_utils.BarHandler,
        )
    )
    self.enter_context(
        ocp.Context(checkpointables_options=checkpointables_options)
    )
    super().test_load_with_switched_abstract_checkpointables()

  def test_different_custom_checkpointables(self):
    # Use named handler to override v0 checkpoint_handlers.
    checkpointables_options = (
        ocp.options.CheckpointablesOptions.create_with_handlers(
            foo=handler_utils.FooHandler,
            bar=handler_utils.BarHandler,
        )
    )
    self.enter_context(
        ocp.Context(checkpointables_options=checkpointables_options)
    )
    super().test_different_custom_checkpointables()

  def test_custom_save_decision_policy(self):
    self.skipTest('b/422287659')

  def test_overwrites(self):
    self.skipTest('b/422287659')

  def test_steps(self):
    self.skipTest('b/422287659')

  def test_step_already_exists(self):
    self.skipTest('b/422287659')
