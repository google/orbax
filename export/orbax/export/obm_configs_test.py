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

"""Tests for obm_configs."""

from absl.testing import absltest
from orbax.export import obm_configs


class ObmConfigsTest(absltest.TestCase):

  def test_batch_options_disable_batch(self):
    batch_options = obm_configs.BatchOptions(
        batch_component=obm_configs.BatchComponent.NO_BATCHING,
    )
    self.assertIsNone(batch_options.max_batch_size)

  def test_batch_options_raise_error_without_max_batch_size_and_allowed_batch_sizes(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError,
        "max_batch_size must be provided when allowed_batch_sizes is empty."
        " Got: max_batch_size: None; allowed_batch_sizes: None.",
    ):
      obm_configs.BatchOptions(
          batch_component=obm_configs.BatchComponent.MODEL_FUNCTION,
      )

  def test_batch_options_with_allowed_batch_sizes_only(self):
    batch_options = obm_configs.BatchOptions(
        batch_component=obm_configs.BatchComponent.MODEL_FUNCTION,
        allowed_batch_sizes=[2, 4, 8],
    )
    self.assertEqual(batch_options.max_batch_size, 8)

  def test_batch_options_with_max_batch_size_only(self):
    batch_options = obm_configs.BatchOptions(
        batch_component=obm_configs.BatchComponent.MODEL_FUNCTION,
        max_batch_size=8,
    )
    self.assertEqual(batch_options.max_batch_size, 8)

  def test_batch_options_raise_error_with_non_positive_allowed_batch_sizes(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError,
        r"`allowed_batch_sizes` must be positive. Got: \[0, 4, 16\]",
    ):
      obm_configs.BatchOptions(
          batch_component=obm_configs.BatchComponent.MODEL_FUNCTION,
          allowed_batch_sizes=[0, 4, 16],
      )

  def test_batch_options_raise_error_with_unsorted_allowed_batch_sizes(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError,
        r"`allowed_batch_sizes` must be sorted in ascending order. Got:"
        r" \[4, 16, 2\]",
    ):
      obm_configs.BatchOptions(
          batch_component=obm_configs.BatchComponent.MODEL_FUNCTION,
          allowed_batch_sizes=[4, 16, 2],
      )

  def test_batch_options_raise_error_with_inconsistent_max_batch_size_and_allowed_batch_sizes(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError,
        r"`max_batch_size` must be equal to the largest one in"
        r" `allowed_batch_sizes` when large batch splitting is disabled.",
    ):
      obm_configs.BatchOptions(
          batch_component=obm_configs.BatchComponent.MODEL_FUNCTION,
          max_batch_size=8,
          allowed_batch_sizes=[2, 4, 16],
          disable_large_batch_splitting=True,
      )

  def test_batch_options_raise_error_with_smaller_max_batch_size(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError,
        r"`max_batch_size` must be larger than or equal to the largest one in"
        r" `allowed_batch_sizes`.",
    ):
      obm_configs.BatchOptions(
          batch_component=obm_configs.BatchComponent.MODEL_FUNCTION,
          max_batch_size=8,
          allowed_batch_sizes=[2, 4, 16],
          disable_large_batch_splitting=False,
      )

  def test_batch_options_success_with_larger_max_batch_size(
      self,
  ):
    obm_configs.BatchOptions(
        batch_component=obm_configs.BatchComponent.MODEL_FUNCTION,
        max_batch_size=32,
        allowed_batch_sizes=[2, 4, 16],
        disable_large_batch_splitting=False,
    )

  def test_batch_options_raise_error_with_negative_batch_timeout_micros(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError,
        r"`batch_timeout_micros` must be non-negative. Got: -1",
    ):
      obm_configs.BatchOptions(
          batch_component=obm_configs.BatchComponent.MODEL_FUNCTION,
          max_batch_size=8,
          batch_timeout_micros=-1,
      )

  def test_batch_options_raise_error_with_non_positive_num_batch_threads(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError,
        r"`num_batch_threads` must be at least 1. Got: 0",
    ):
      obm_configs.BatchOptions(
          batch_component=obm_configs.BatchComponent.MODEL_FUNCTION,
          max_batch_size=8,
          num_batch_threads=0,
      )

  def test_batch_options_raise_error_with_non_positive_max_enqueued_batches(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError,
        r"`max_enqueued_batches` must be at least 1. Got: 0",
    ):
      obm_configs.BatchOptions(
          batch_component=obm_configs.BatchComponent.MODEL_FUNCTION,
          max_batch_size=8,
          max_enqueued_batches=0,
      )


if __name__ == "__main__":
  absltest.main()
