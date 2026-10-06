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

"""Tests for checkpoint file options."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax.numpy as jnp
from orbax.checkpoint import options as v0_options_lib
from orbax.checkpoint._src.serialization import limits
from orbax.checkpoint.experimental.v1 import options as public_v1_options
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as ocp_options
from orbax.checkpoint.experimental.v1._src.saving import saving
from orbax.checkpoint.experimental.v1._src.serialization import registration
from orbax.checkpoint.experimental.v1._src.serialization import types as serialization_types
from orbax.checkpoint.experimental.v1._src.training import checkpointer as training_checkpointer
from orbax.checkpoint.experimental.v1._src.training import save_decision_policies



class _MockMemoryProfiler(public_v1_options.MemoryProfiler):

  def __init__(self, total_memory_gib: float = 250.0):
    super().__init__()
    self._total_memory_gib = total_memory_gib
    self.start_count = 0
    self.end_count = 0

  def profiler_start(self) -> None:
    self.start_count += 1

  def profiler_end(self) -> None:
    self.end_count += 1

  @property
  def total_memory_gib(self) -> float:
    return self._total_memory_gib


class FileOptionsTest(parameterized.TestCase):

  def test_v0_conversion_with_none_options(self):
    opts = ocp_options.FileOptions()
    v0_opts = opts.v0()
    self.assertIsInstance(v0_opts, v0_options_lib.FileOptions)
    self.assertIsNone(v0_opts.path_permission_mode)
    self.assertFalse(v0_opts.skip_sync_file_validations)

  def test_v0_conversion_with_all_options(self):

    opts = ocp_options.FileOptions(
        path_permission_mode=0o777,
        skip_sync_file_validations=True,
    )
    v0_opts = opts.v0()
    self.assertIsInstance(v0_opts, v0_options_lib.FileOptions)
    self.assertEqual(v0_opts.path_permission_mode, 0o777)
    self.assertTrue(v0_opts.skip_sync_file_validations)


class AtomicityOptionsTest(parameterized.TestCase):

  def test_v0_conversion_maps_correctly(self):
    opts = ocp_options.AtomicityOptions(
        mode=ocp_options.AtomicityMode.COMMIT_FILE
    )

    v0_opts = opts.v0()

    self.assertIsInstance(v0_opts, v0_options_lib.AtomicityOptions)
    self.assertEqual(v0_opts.mode, v0_options_lib.AtomicityMode.COMMIT_FILE)


class MemoryOptionsTest(parameterized.TestCase):

  def test_memory_options_propagation(self):
    def is_prioritized_key_fn(path):
      del path
      return True

    ctx = context_lib.Context()
    ctx.memory.write_concurrent_bytes = 1024
    ctx.memory.read_concurrent_bytes = 2048
    ctx.memory.transfer_concurrent_bytes = 512
    ctx.memory.is_prioritized_key_fn = is_prioritized_key_fn  # pyrefly: ignore[bad-assignment]
    with ctx:
      with mock.patch(
          'orbax.checkpoint._src.handlers.base_pytree_checkpoint_handler.BasePyTreeCheckpointHandler',
          autospec=True,
      ) as mock_handler_class:
        # Mock async_save to return a future as expected by PyTreeHandler.save
        mock_handler_instance = mock_handler_class.return_value
        mock_handler_instance.async_save.return_value = (
            mock.MagicMock()
        )  # Awaitable

        pytree = {'a': jnp.ones((1,))}

        # save will eventually call BasePyTreeCheckpointHandler
        try:
          saving.save('/tmp/test', pytree)  # pyrefly: ignore[bad-argument-type]
        except Exception:  # pylint: disable=broad-except
          # We might get some errors because we mocked too much,
          # but we check if mock_handler_class was called.
          pass

        mock_handler_class.assert_called()
        # Find the call that has our expected arguments
        found = False
        for call in mock_handler_class.call_args_list:
          kwargs = call.kwargs
          if (
              kwargs.get('save_concurrent_bytes') == 1024
              and kwargs.get('restore_concurrent_bytes') == 2048
              and kwargs.get('save_device_host_concurrent_bytes') == 512
              and kwargs.get('is_prioritized_key_fn') == is_prioritized_key_fn  # pylint: disable=comparison-with-callable
          ):
            found = True
            break
        self.assertTrue(
            found,
            f'Expected call not found in {mock_handler_class.call_args_list}',
        )

  def test_memory_regulator_and_expected_surge_bytes_with_save(self):
    profiler = _MockMemoryProfiler(total_memory_gib=250.0)
    profiler._peak_usage_bytes = int(100 * 1024**3)
    regulator = public_v1_options.MemoryRegulator(
        max_memory_limit_gib=80.0, ki=0.0, kd=0.0, profiler=profiler
    )

    ctx = context_lib.Context()
    ctx.memory.memory_regulator = regulator
    # Verify regulator takes precedence over static transfer_concurrent_bytes.
    ctx.memory.transfer_concurrent_bytes = 512

    directory = epath.Path(self.create_tempdir().full_path)
    pytree = {'a': jnp.ones((2,))}

    with mock.patch.object(
        limits, 'get_byte_limiter', wraps=limits.get_byte_limiter
    ) as spy_limiter:
      with ctx:
        saving.save_checkpointables(
            directory / 'step0', {'model': pytree, 'opt': pytree}
        )
      spy_limiter.assert_any_call(50 * 1024**3)
      self.assertNotIn(mock.call(512), spy_limiter.call_args_list)

    # One regulate() call per checkpoint save even with two PyTrees.
    # error = 200 - 100 = 100 -> adjustment = 0.4 * 100 = 40 -> 10 + 40 = 50 GiB
    self.assertEqual(profiler.start_count, 1)
    self.assertEqual(profiler.end_count, 1)
    self.assertEqual(regulator.current_limit_bytes, 50 * 1024**3)

    # Simulate steady state at target (200 GiB) and an expected surge of 15 GiB.
    profiler._peak_usage_bytes = int(200 * 1024**3)
    surge_ctx = context_lib.Context(ctx)
    surge_ctx.memory.expected_surge_bytes = 15 * 1024**3
    self.assertIs(surge_ctx.memory.memory_regulator, regulator)

    with mock.patch.object(
        limits, 'get_byte_limiter', wraps=limits.get_byte_limiter
    ) as spy_limiter:
      with surge_ctx:
        saving.save_checkpointables(directory / 'step1', {'model': pytree})
      spy_limiter.assert_any_call(35 * 1024**3)

    self.assertEqual(profiler.start_count, 2)
    self.assertEqual(profiler.end_count, 2)
    self.assertEqual(regulator.current_limit_bytes, 35 * 1024**3)

    # Next save without expected_surge_bytes restores the 50 GiB limit.
    with mock.patch.object(
        limits, 'get_byte_limiter', wraps=limits.get_byte_limiter
    ) as spy_limiter:
      with ctx:
        saving.save_checkpointables(directory / 'step2', {'model': pytree})
      spy_limiter.assert_any_call(50 * 1024**3)

    self.assertEqual(profiler.start_count, 3)
    self.assertEqual(profiler.end_count, 3)
    self.assertEqual(regulator.current_limit_bytes, 50 * 1024**3)

    # If the regulator is removed, saving falls back to
    # transfer_concurrent_bytes (512).
    unregulated_ctx = context_lib.Context(ctx)
    unregulated_ctx.memory.memory_regulator = None
    with mock.patch.object(
        limits, 'get_byte_limiter', wraps=limits.get_byte_limiter
    ) as spy_limiter:
      with unregulated_ctx:
        saving.save_checkpointables(directory / 'step3', {'model': pytree})
      spy_limiter.assert_any_call(512)

    self.assertEqual(profiler.start_count, 3)
    self.assertEqual(profiler.end_count, 3)

  def test_memory_regulator_with_training_checkpointer(self):
    profiler = _MockMemoryProfiler(total_memory_gib=250.0)
    profiler._peak_usage_bytes = int(100 * 1024**3)
    regulator = public_v1_options.MemoryRegulator(
        max_memory_limit_gib=80.0, ki=0.0, kd=0.0, profiler=profiler
    )

    ctx = context_lib.Context()
    ctx.memory.memory_regulator = regulator

    directory = epath.Path(self.create_tempdir().full_path)
    pytree = {'a': jnp.ones((2,))}

    ckptr = training_checkpointer.Checkpointer(
        directory,
        save_decision_policy=save_decision_policies.FixedIntervalPolicy(2),  # pyrefly: ignore[bad-argument-type]
        context=ctx,
    )
    self.addCleanup(ckptr.close)

    # Step 0: saved -> regulated once.
    with mock.patch.object(
        limits, 'get_byte_limiter', wraps=limits.get_byte_limiter
    ) as spy_limiter:
      self.assertTrue(ckptr.save_checkpointables(0, {'model': pytree}))
      spy_limiter.assert_any_call(50 * 1024**3)
    self.assertEqual(profiler.start_count, 1)
    self.assertEqual(profiler.end_count, 1)
    self.assertEqual(regulator.current_limit_bytes, 50 * 1024**3)

    # Step 1: skipped by policy -> regulator must not step.
    self.assertFalse(ckptr.save_checkpointables(1, {'model': pytree}))
    self.assertEqual(profiler.start_count, 1)
    self.assertEqual(profiler.end_count, 1)
    self.assertEqual(regulator.current_limit_bytes, 50 * 1024**3)

    # Step 2: surge step via child Context.
    profiler._peak_usage_bytes = int(200 * 1024**3)
    surge_ctx = context_lib.Context(ctx)
    surge_ctx.memory.expected_surge_bytes = 15 * 1024**3
    with mock.patch.object(
        limits, 'get_byte_limiter', wraps=limits.get_byte_limiter
    ) as spy_limiter:
      with surge_ctx:
        self.assertTrue(ckptr.save_checkpointables(2, {'model': pytree}))
      spy_limiter.assert_any_call(35 * 1024**3)
    self.assertEqual(profiler.start_count, 2)
    self.assertEqual(profiler.end_count, 2)
    self.assertEqual(regulator.current_limit_bytes, 35 * 1024**3)

    # Step 4: normal save -> recovers from surge.
    with mock.patch.object(
        limits, 'get_byte_limiter', wraps=limits.get_byte_limiter
    ) as spy_limiter:
      self.assertTrue(ckptr.save_checkpointables(4, {'model': pytree}))
      spy_limiter.assert_any_call(50 * 1024**3)
    self.assertEqual(profiler.start_count, 3)
    self.assertEqual(profiler.end_count, 3)
    self.assertEqual(regulator.current_limit_bytes, 50 * 1024**3)

  def test_memory_options_callback_propagation(self):
    class DummyCallback(serialization_types.SerializationStatusCallback):

      def key_priority(
          self,
          keypath: serialization_types.tree_types.PyTreeKeyPath,
      ) -> serialization_types.TransferPriority:
        del keypath
        return serialization_types.TransferPriority.ASYNCHRONOUS_DEPRIORITIZED

      def on_transfer_start(
          self, keypath: serialization_types.tree_types.PyTreeKeyPath
      ) -> None:
        pass

      def on_transfer_end(
          self, keypath: serialization_types.tree_types.PyTreeKeyPath
      ) -> None:
        pass

      def on_write_start(
          self, keypath: serialization_types.tree_types.PyTreeKeyPath
      ) -> None:
        pass

      def on_write_end(
          self, keypath: serialization_types.tree_types.PyTreeKeyPath
      ) -> None:
        pass

    callback = DummyCallback()
    ctx = context_lib.Context()
    ctx.memory.serialization_status_callback = callback

    # Assert get_array_handler propagates it.
    handler = registration.get_array_handler(ctx)
    self.assertEqual(handler._callback, callback)


if __name__ == '__main__':
  absltest.main()
