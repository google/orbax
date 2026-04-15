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

from typing import Any

from absl import flags
import jax
import orbax.checkpoint.experimental.v1 as ocp
from orbax.checkpoint.experimental.v1._src.testing import handler_utils
from orbax.checkpoint.experimental.v1._src.training import checkpointer_test_base
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types

from .pyglib.contrib.g3_multiprocessing import g3_multiprocessing
from absl.testing import absltest
from .testing.pybase import parameterized


FLAGS = flags.FLAGS

jax.config.update('jax_enable_x64', True)

Checkpointer = ocp.training.Checkpointer
save_decision_policies = ocp.training.save_decision_policies


class PathwaysCheckpointerTest(
    checkpointer_test_base.CheckpointerTestBase.Test,
    parameterized.TestCase,
):

  def save_pytree(
      self,
      checkpointer: Checkpointer,
      step: int,
      pytree: tree_types.PyTreeOf[tree_types.LeafType],
      metrics: tree_types.JsonType | None = None,
      custom_metadata: tree_types.JsonType | None = None,
  ) -> bool:
    """Sets up a checkpoint (to be overridden by subclasses)."""
    return checkpointer.save_pytree(
        step,
        pytree,
        metrics=metrics,
        custom_metadata=custom_metadata,
    )

  def save_checkpointables(
      self,
      checkpointer: Checkpointer,
      step: int,
      checkpointables: dict[str, Any],
      metrics: tree_types.JsonType | None = None,
      custom_metadata: tree_types.JsonType | None = None,
  ) -> bool:
    return checkpointer.save_checkpointables(
        step,
        checkpointables,
        metrics=metrics,
        custom_metadata=custom_metadata,
    )

  def test_custom_checkpointables(self):
    checkpointables_options = (
        ocp.options.CheckpointablesOptions.create_with_handlers(
            handler_utils.FooHandler,
            handler_utils.BarHandler,
        )
    )
    self.enter_context(
        ocp.Context(checkpointables_options=checkpointables_options)
    )
    super().test_custom_checkpointables()

  def test_load_with_switched_abstract_checkpointables(self):
    checkpointables_options = (
        ocp.options.CheckpointablesOptions.create_with_handlers(
            handler_utils.FooHandler,
            handler_utils.BarHandler,
        )
    )
    self.enter_context(
        ocp.Context(checkpointables_options=checkpointables_options)
    )
    super().test_load_with_switched_abstract_checkpointables()

  def test_different_custom_checkpointables(self):
    checkpointables_options = (
        ocp.options.CheckpointablesOptions.create_with_handlers(
            handler_utils.FooHandler,
            handler_utils.BarHandler,
        )
    )
    self.enter_context(
        ocp.Context(checkpointables_options=checkpointables_options)
    )
    super().test_different_custom_checkpointables()


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  g3_multiprocessing.handle_test_main(googletest.main)
