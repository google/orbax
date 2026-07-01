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

"""Tests for checkpointing with Context."""

import asyncio
from concurrent import futures
import dataclasses
from absl.testing import absltest
import orbax.checkpoint.experimental.v1 as ocp
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as options_lib

ArrayOptions = options_lib.ArrayOptions
FileOptions = options_lib.FileOptions


@dataclasses.dataclass
class DummyChild(options_lib._ActiveContextGuard):
  value: int = 1


@dataclasses.dataclass
class DummyParent(options_lib._ActiveContextGuard):
  children: list[DummyChild] = dataclasses.field(default_factory=list)
  mapping: dict[str, DummyChild] = dataclasses.field(default_factory=dict)


def fake_checkpoint_operation() -> ocp.Context:
  return context_lib.get_context()


class ContextTest(absltest.TestCase):

  def test_default_context(self):
    ctx = fake_checkpoint_operation()
    self.assertEqual(ctx.array, ArrayOptions())

    with ocp.Context():
      ctx = fake_checkpoint_operation()
      self.assertEqual(ctx.array, ArrayOptions())

    context = ocp.Context()
    with context:
      ctx = fake_checkpoint_operation()
      self.assertEqual(ctx.array, ArrayOptions())

  def test_get_context_with_default(self):
    default_ctx = ocp.Context()
    default_ctx.array.saving.use_ocdbt = False

    custom_ctx = ocp.Context()
    custom_ctx.array.saving.use_zarr3 = False

    with self.subTest("no context set, no default provided"):
      ctx = context_lib.get_context()
      self.assertEqual(ctx.array, ArrayOptions())

    with self.subTest("no context set, default provided"):
      ctx = context_lib.get_context(default=default_ctx)
      self.assertIs(ctx, default_ctx)

    with self.subTest("context IS set, no default provided"):
      with custom_ctx:
        ctx = context_lib.get_context()
        self.assertIs(ctx, custom_ctx)

    with self.subTest("context IS set, default provided"):
      with custom_ctx:
        ctx = context_lib.get_context(default=default_ctx)
        self.assertIs(ctx, custom_ctx)
        self.assertIsNot(ctx, default_ctx)

    with self.subTest("no context set, default=None provided"):
      ctx = context_lib.get_context(default=None)
      self.assertEqual(ctx.array, ArrayOptions())

  def test_custom_context(self):
    ctx = ocp.Context()
    ctx.array.saving.use_zarr3 = False
    with ctx:
      ctx_2 = fake_checkpoint_operation()
      self.assertEqual(
          ctx_2.array,
          ArrayOptions(saving=ArrayOptions.Saving(use_zarr3=False)),
      )

  def test_custom_context_in_separate_thread_becomes_default(self):
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      ctx = ocp.Context()
      ctx.array.saving.use_zarr3 = False
      with ctx:
        future = executor.submit(fake_checkpoint_operation)
        active = future.result()
        self.assertEqual(active.array.saving.use_zarr3, True)

  def test_custom_context_in_same_thread_remains_custom(self):
    def test_fn():
      context = ocp.Context()
      context.array.saving.use_zarr3 = False
      with context:
        return fake_checkpoint_operation()

    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      future = executor.submit(test_fn)
      ctx = future.result()
      self.assertEqual(
          ctx.array,
          ArrayOptions(saving=ArrayOptions.Saving(use_zarr3=False)),
      )

  def test_nested_contexts(self):
    ctx1 = ocp.Context()
    ctx1.array.saving.use_zarr3 = False
    with ctx1:
      self.assertEqual(
          fake_checkpoint_operation().array.saving.use_zarr3, False
      )

      ctx2 = ocp.Context()  # absolute default slate by default
      ctx2.array.saving.use_ocdbt = False
      with ctx2:
        active = fake_checkpoint_operation()
        self.assertEqual(active.array.saving.use_zarr3, True)
        self.assertEqual(active.array.saving.use_ocdbt, False)

  def test_nested_contexts_with_inheritance(self):
    ctx1 = ocp.Context()
    ctx1.array.saving.use_zarr3 = False
    ctx1.file.path_permission_mode = 0o750
    with ctx1:
      active1 = fake_checkpoint_operation()
      self.assertEqual(active1.array.saving.use_zarr3, False)
      self.assertEqual(active1.file.path_permission_mode, 0o750)

      ctx2 = ocp.Context(active1)
      ctx2.array.saving.use_ocdbt = False
      with ctx2:
        active2 = fake_checkpoint_operation()
        self.assertEqual(active2.array.saving.use_zarr3, False)  # inherited
        self.assertEqual(active2.array.saving.use_ocdbt, False)  # mutated
        self.assertEqual(active2.file.path_permission_mode, 0o750)  # inherited

      active3 = fake_checkpoint_operation()
      self.assertEqual(active3.array.saving.use_zarr3, False)
      self.assertEqual(active3.array.saving.use_ocdbt, True)
      self.assertEqual(active3.file.path_permission_mode, 0o750)

  def test_concurrent_separate_instances_preserve_frozen_states(self):
    ctx1 = ocp.Context()
    ctx2 = ocp.Context()
    with ctx1:
      with self.assertRaises(RuntimeError):
        ctx1.array.saving.use_zarr3 = False
      # ctx2 option IDs are not in _FROZEN_IDS
      ctx2.array.saving.use_zarr3 = False

      with ctx2:
        with self.assertRaises(RuntimeError):
          ctx1.array.saving.use_zarr3 = False
        with self.assertRaises(RuntimeError):
          ctx2.array.saving.use_zarr3 = False

      # After exiting ctx2, ctx2 is no longer frozen
      ctx2.array.saving.use_zarr3 = True

  def test_exhaustive_frozen_context_mutability_checks(self):
    parent_ctx = ocp.Context()
    parent_ctx.array.saving.use_zarr3 = True

    with parent_ctx:
      # 1. Accessing properties on the outer parent_ctx raises RuntimeError
      # (it is frozen during the block)
      with self.assertRaises(RuntimeError):
        parent_ctx.array.saving.use_zarr3 = False
      with self.assertRaises(RuntimeError):
        parent_ctx.file.path_permission_mode = 0o750
      with self.assertRaises(RuntimeError):
        parent_ctx.checkpoint_layout = options_lib.CheckpointLayout.SAFETENSORS
      # Reading properties is fine.
      _ = parent_ctx.array

      # 2. get_context() returns the active parent_ctx directly. It is fully
      # frozen against mutation, but allows read access.
      active_parent = context_lib.get_context()
      self.assertTrue(active_parent.array.saving.use_zarr3)
      with self.assertRaises(RuntimeError):
        active_parent.array.saving.use_zarr3 = False

      # 3. Creating and mutating a child context template inside an enclosure
      # works perfectly.
      child_ctx = ocp.Context(parent_ctx)
      child_ctx.array.saving.use_zarr3 = False  #  Mutates fine before entry

      with child_ctx:
        # Inside inner block, both parent_ctx and child_ctx are frozen
        with self.assertRaises(RuntimeError):
          child_ctx.array.saving.use_zarr3 = True
        with self.assertRaises(RuntimeError):
          parent_ctx.array.saving.use_zarr3 = False

  def test_asyncio_tasks_inherit_frozen_context(self):
    ctx = ocp.Context()
    ctx.array.saving.use_zarr3 = True

    async def bg_operation():
      # Inside the asyncio task, the context and its frozen state are inherited
      active = context_lib.get_context()
      self.assertIs(active, ctx)
      self.assertTrue(active.array.saving.use_zarr3)
      # Attempting to mutate ctx from the running async task raises RuntimeError
      with self.assertRaises(RuntimeError):
        ctx.array.saving.use_zarr3 = False

    async def main():
      with ctx:
        # Spawn background task while context is active
        task = asyncio.create_task(bg_operation())
        await task

    asyncio.run(main())

  def test_asyncio_task_overlap(self):
    ctx = ocp.Context()
    ctx.array.saving.use_zarr3 = True

    resume_bg = asyncio.Event()
    bg_started = asyncio.Event()

    async def bg_save_operation():
      # 1. Verify active context and its frozen state before main() mutates it.
      active = context_lib.get_context()
      self.assertIs(active, ctx)
      self.assertTrue(active.array.saving.use_zarr3)
      with self.assertRaises(RuntimeError):
        ctx.array.saving.use_zarr3 = False

      bg_started.set()
      await resume_bg.wait()

      # 2. Verify that main() mutating the shared context object is visible.
      self.assertFalse(active.array.saving.use_zarr3)

      # 3. Even though main coroutine exited `with ctx:` and mutated use_zarr3,
      # this asyncio task still inherits the context snapshot where ctx frozen.
      with self.assertRaises(RuntimeError):
        ctx.array.saving.use_zarr3 = True

    async def main():
      with ctx:
        task = asyncio.create_task(bg_save_operation())
        await bg_started.wait()

      # Now main has exited `with ctx:`. In main's context var, ctx is unfrozen.
      ctx.array.saving.use_zarr3 = False  # Main can mutate it now

      # Resume bg task to verify it is still protected/frozen in its own context
      resume_bg.set()
      await task

    asyncio.run(main())

  def test_deeply_nested_options_frozen_check(self):
    ctx = ocp.Context()
    ctx.array.saving.storage_options.chunk_byte_size = 1024
    ctx.pytree.loading.partial_load = True
    with ctx:
      with self.assertRaises(RuntimeError):
        ctx.array.saving.storage_options.chunk_byte_size = 2048
      with self.assertRaises(RuntimeError):
        ctx.pytree.loading.partial_load = False
      # Reading is perfectly fine
      self.assertEqual(
          ctx.array.saving.storage_options.chunk_byte_size, 1024
      )
      self.assertTrue(ctx.pytree.loading.partial_load)

  def test_context_modification_after_exit(self):
    ctx = ocp.Context()
    ctx.array.saving.use_zarr3 = False
    with ctx:
      self.assertFalse(fake_checkpoint_operation().array.saving.use_zarr3)
      with self.assertRaises(RuntimeError):
        ctx.array.saving.use_zarr3 = True

    # Exiting the block unfreezes the context
    ctx.array.saving.use_zarr3 = True
    with ctx:
      self.assertTrue(fake_checkpoint_operation().array.saving.use_zarr3)
      with self.assertRaises(RuntimeError):
        ctx.array.saving.use_zarr3 = False

  def test_collect_ids_completeness(self):
    ctx = ocp.Context()
    collected_ids = context_lib._collect_ids(ctx)  # pylint: disable=protected-access

    # Verify top-level context and direct option attributes
    self.assertIn(id(ctx), collected_ids)
    self.assertIn(id(ctx.array), collected_ids)
    self.assertIn(id(ctx.pytree), collected_ids)
    self.assertIn(id(ctx.file), collected_ids)
    self.assertIn(id(ctx.memory), collected_ids)

    # Verify nested child dataclasses
    self.assertIn(id(ctx.array.saving), collected_ids)
    self.assertIn(id(ctx.array.saving.storage_options), collected_ids)
    self.assertIn(id(ctx.array.loading), collected_ids)
    self.assertIn(id(ctx.pytree.saving), collected_ids)
    self.assertIn(id(ctx.pytree.loading), collected_ids)

  def test_frozen_ids_lifecycle_and_nesting(self):
    ctx1 = ocp.Context()
    ctx2 = ocp.Context()

    initial_frozen = options_lib.FROZEN_IDS.get()
    self.assertNotIn(id(ctx1), initial_frozen)
    self.assertNotIn(id(ctx2), initial_frozen)

    with ctx1:
      frozen_outer = options_lib.FROZEN_IDS.get()
      self.assertIn(id(ctx1), frozen_outer)
      self.assertIn(id(ctx1.array.saving), frozen_outer)
      self.assertNotIn(id(ctx2), frozen_outer)

      with ctx2:
        frozen_inner = options_lib.FROZEN_IDS.get()
        self.assertIn(id(ctx1), frozen_inner)
        self.assertIn(id(ctx2), frozen_inner)
        self.assertIn(id(ctx1.array.saving), frozen_inner)
        self.assertIn(id(ctx2.array.saving), frozen_inner)

      # Exiting ctx2 pops ctx2 IDs but preserves ctx1 IDs
      frozen_after_inner = options_lib.FROZEN_IDS.get()
      self.assertIn(id(ctx1), frozen_after_inner)
      self.assertNotIn(id(ctx2), frozen_after_inner)

    # Exiting ctx1 restores exact initial state
    self.assertEqual(options_lib.FROZEN_IDS.get(), initial_frozen)

  def test_context_var_reset_on_exception(self):
    ctx = ocp.Context()
    initial_context = context_lib._CONTEXT.get()  # pylint: disable=protected-access
    initial_frozen = options_lib.FROZEN_IDS.get()

    try:
      with ctx:
        self.assertIs(context_lib._CONTEXT.get(), ctx)  # pylint: disable=protected-access
        self.assertIn(id(ctx), options_lib.FROZEN_IDS.get())
        raise ValueError("Intentional failure inside context enclosure")
    except ValueError:
      pass

    # Verify cleanup occurred despite exception
    self.assertIs(context_lib._CONTEXT.get(), initial_context)  # pylint: disable=protected-access
    self.assertEqual(options_lib.FROZEN_IDS.get(), initial_frozen)

  def test_deepcopy_preserves_non_field_attributes(self):
    ctx = ocp.Context()
    # Dynamically assign a non-field attribute to verify deepcopy preserves it
    ctx.array.saving._custom_field = "preserved_value"
    ctx_copy = ocp.Context(ctx)
    self.assertEqual(ctx_copy.array.saving._custom_field, "preserved_value")  # pyrefly: ignore[missing-attribute]

  def test_collect_ids_with_collections(self):
    child1 = DummyChild(10)
    child2 = DummyChild(20)
    child3 = DummyChild(30)
    parent = DummyParent(children=[child1, child2], mapping={"a": child3})

    ctx = ocp.Context()
    ctx._dummy_parent = parent  # pyrefly: ignore[missing-attribute]

    collected_ids = context_lib._collect_ids(ctx)  # pylint: disable=protected-access

    self.assertIn(id(parent), collected_ids)
    self.assertIn(id(child1), collected_ids)
    self.assertIn(id(child2), collected_ids)
    self.assertIn(id(child3), collected_ids)

    # Verify that entering the context freezes child1
    with ctx:
      with self.assertRaises(RuntimeError):
        child1.value = 15

  def test_active_context_freezes_non_fields_and_functions(self):
    ctx = ocp.Context()
    ctx.array.saving._custom_non_field = "initial_value"
    ctx.asynchronous.post_finalization_callback = lambda: None

    with ctx:
      with self.assertRaises(RuntimeError):
        ctx.array.saving._custom_non_field = "modified_value"
      with self.assertRaises(RuntimeError):
        ctx.asynchronous.post_finalization_callback = lambda: print("modified")


if __name__ == "__main__":
  absltest.main()
