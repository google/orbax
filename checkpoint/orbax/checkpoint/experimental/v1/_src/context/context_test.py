# Copyright 2024 The Orbax Authors.
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

from concurrent import futures
from absl.testing import absltest
import orbax.checkpoint.experimental.v1 as ocp
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as options_lib

ArrayOptions = options_lib.ArrayOptions


def fake_checkpoint_operation() -> ocp.Context:
  return context_lib.get_context()


class ContextTest(absltest.TestCase):

  def test_default_context(self):
    ctx = fake_checkpoint_operation()
    self.assertEqual(ctx.array_options, ArrayOptions())

    with ocp.Context():
      ctx = fake_checkpoint_operation()
      self.assertEqual(ctx.array_options, ArrayOptions())

    context = ocp.Context()
    with context:
      ctx = fake_checkpoint_operation()
      self.assertEqual(ctx.array_options, ArrayOptions())

  def test_custom_context(self):
    with ocp.Context(
        array_options=ArrayOptions(saving=ArrayOptions.Saving(use_zarr3=False))
    ):
      ctx = fake_checkpoint_operation()
      self.assertEqual(
          ctx.array_options,
          ArrayOptions(saving=ArrayOptions.Saving(use_zarr3=False)),
      )

    context = ocp.Context(
        array_options=ArrayOptions(saving=ArrayOptions.Saving(use_zarr3=False))
    )
    with context:
      ctx = fake_checkpoint_operation()
      self.assertEqual(
          ctx.array_options,
          ArrayOptions(saving=ArrayOptions.Saving(use_zarr3=False)),
      )

  def test_custom_context_in_separate_thread_becomes_default(self):
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      with ocp.Context(
          array_options=ArrayOptions(
              saving=ArrayOptions.Saving(use_zarr3=False)
          )
      ):
        future = executor.submit(fake_checkpoint_operation)
        ctx = future.result()
        self.assertEqual(ctx.array_options, ArrayOptions())

    with ocp.Context(
        array_options=ArrayOptions(saving=ArrayOptions.Saving(use_zarr3=False))
    ):
      with futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fake_checkpoint_operation)
        ctx = future.result()
        self.assertEqual(ctx.array_options, ArrayOptions())

  def test_custom_context_in_same_thread_remains_custom(self):
    def test_fn():
      with ocp.Context(
          array_options=ArrayOptions(
              saving=ArrayOptions.Saving(use_zarr3=False)
          )
      ):
        ctx = fake_checkpoint_operation()
        return ctx

    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      future = executor.submit(test_fn)
      ctx = future.result()
      self.assertEqual(
          ctx.array_options,
          ArrayOptions(saving=ArrayOptions.Saving(use_zarr3=False)),
      )

  def test_nested_contexts(self):
    with ocp.Context(
        array_options=ArrayOptions(saving=ArrayOptions.Saving(use_zarr3=False))
    ):
      ctx = fake_checkpoint_operation()
      self.assertEqual(
          ctx.array_options,
          ArrayOptions(saving=ArrayOptions.Saving(use_zarr3=False)),
      )

      with ocp.Context(
          array_options=ArrayOptions(
              saving=ArrayOptions.Saving(use_ocdbt=False)
          )
      ):
        ctx = fake_checkpoint_operation()
        self.assertEqual(
            ctx.array_options,
            ArrayOptions(
                saving=ArrayOptions.Saving(use_zarr3=True, use_ocdbt=False)
            ),
        )

      ctx = fake_checkpoint_operation()
      self.assertEqual(
          ctx.array_options,
          ArrayOptions(
              saving=ArrayOptions.Saving(use_zarr3=False, use_ocdbt=True)
          ),
      )

  def test_nested_contexts_with_inheritance(self):
    with ocp.Context(
        array_options=ArrayOptions(saving=ArrayOptions.Saving(use_zarr3=False))
    ):
      ctx = fake_checkpoint_operation()
      self.assertEqual(
          ctx.array_options,
          ArrayOptions(saving=ArrayOptions.Saving(use_zarr3=False)),
      )
      with ocp.Context(
          ctx,
          array_options=ArrayOptions(
              saving=ArrayOptions.Saving(use_ocdbt=False, use_zarr3=True)
          ),
      ):
        ctx = fake_checkpoint_operation()
        self.assertEqual(
            ctx.array_options,
            ArrayOptions(
                saving=ArrayOptions.Saving(use_ocdbt=False, use_zarr3=True)
            ),
        )
      ctx = fake_checkpoint_operation()
      self.assertEqual(
          ctx.array_options,
          ArrayOptions(saving=ArrayOptions.Saving(use_zarr3=False)),
      )


if __name__ == "__main__":
  absltest.main()
