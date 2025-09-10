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

"""Configuration classes for Orbax Model Export."""
from collections.abc import Sequence
import dataclasses
import enum
import itertools


# LINT.IfChange
@enum.unique
class BatchComponent(enum.Enum):
  """The computation component to batch."""

  NO_BATCHING = "no_batching"
  MODEL_FUNCTION = "model_function"
  WHOLE_PIPELINE = "whole_pipeline"
  PRE_PROCESSOR_AND_MODEL_FUNCTION = "pre_processor_and_model_function"
  MODEL_FUNCTION_AND_POST_PROCESSOR = "model_function_and_post_processor"


@enum.unique
class BatchPaddingPolicy(enum.Enum):
  """The batch padding policy for the batch scheduler.

  Options:
    PAD_UP: Pad up to the next allowed batch size.
    BATCH_DOWN: Batch down to a smaller allowed batch size.

  See the documentation of BatchOptions.BatchPaddingPolicy for details.
  """
  PAD_UP = "pad_up"
  BATCH_DOWN = "batch_down"
  # TODO: b/443993280 - Add MINIMIZE_TPU_COST_PER_REQUEST once it's supported in
  # JSV.


# LINT.ThenChange(//depot//orbax/export/obm_export.py)


@dataclasses.dataclass(kw_only=True)
class BatchOptions:
  """Batch options for the Orbax model.

  Attributes:
    batch_component: The component to batch.
    max_batch_size: The maximum allowed batch size for any input. If
      `allowed_batch_sizes` is provided, this can be None and infered from the
      largest value in the list. Otherwise, this must be provided.
    batch_timeout_micros: Maximum number of microseconds to wait before
      outputting an incomplete batch.
    num_batch_threads: Number of scheduling threads for processing batches of
      work. Determines the number of batches processed in parallel. This should
      be roughly in line with the number of TPU cores available.
    max_enqueued_batches: Maximum number of batches enqueued for processing
      before requests are failed fast. Default is 250.
    allowed_batch_sizes: Optional list of allowed batch sizes. If left empty,
      all batch sizes no larger than `max_batch_size` are allowed. Otherwise,
      supplies a list of batch sizes. The entries must increase monotonically.
    disable_large_batch_splitting: Whether to disable large batch splitting.
    batch_padding_policy: The batch padding policy for the batch scheduler.
      Default is PAD_UP.
  """
  batch_component: BatchComponent
  max_batch_size: int | None = None
  batch_timeout_micros: int = 0
  allowed_batch_sizes: Sequence[int] | None = None
  num_batch_threads: int = 1
  max_enqueued_batches: int = 250
  disable_large_batch_splitting: bool = False
  batch_padding_policy: BatchPaddingPolicy = BatchPaddingPolicy.PAD_UP

  def __post_init__(self):
    """Validates the batch options."""

    if self.allowed_batch_sizes:
      if self.allowed_batch_sizes[0] < 1:
        raise ValueError(
            "`allowed_batch_sizes` must be positive. Got:"
            f" {self.allowed_batch_sizes}"
        )

      # `allowed_batch_sizes` should be sorted in ascending order.
      for a, b in itertools.pairwise(self.allowed_batch_sizes):
        if a >= b:
          raise ValueError(
              "`allowed_batch_sizes` must be sorted in ascending order. Got:"
              f" {self.allowed_batch_sizes}"
          )

      if self.max_batch_size is None:
        self.max_batch_size = self.allowed_batch_sizes[-1]
      else:
        # When `allowed_batch_sizes` is provided, `max_batch_size` must be
        # larger than or equal to the largest one in the list.
        if (
            self.disable_large_batch_splitting
            and self.max_batch_size != self.allowed_batch_sizes[-1]
        ):
          raise ValueError(
              "`max_batch_size` must be equal to the largest one in"
              " `allowed_batch_sizes` when large batch splitting is disabled."
              " Got: {self.max_batch_size} vs"
              f" {self.allowed_batch_sizes[-1]}. Set `max_batch_size` to None"
              " to automatically infer it from `allowed_batch_sizes`."
          )
        if (
            not self.disable_large_batch_splitting
            and self.max_batch_size < self.allowed_batch_sizes[-1]
        ):
          raise ValueError(
              "`max_batch_size` must be larger than or equal to the largest"
              f" one in `allowed_batch_sizes`. Got: {self.max_batch_size} vs"
              f" {self.allowed_batch_sizes[-1]}."
          )
        if (
            self.max_batch_size > self.allowed_batch_sizes[-1]
            and self.disable_large_batch_splitting
        ):
          raise ValueError(
              "`max_batch_size` must be equal to the largest one in"
              " `allowed_batch_sizes` when large batch splitting is disabled."
              " Got: {self.max_batch_size} vs"
              f" {self.allowed_batch_sizes[-1]}."
          )

    # When `allowed_batch_sizes` is not provided, `max_batch_size` must be
    # provided.
    if self.max_batch_size is None:
      raise ValueError(
          "`max_batch_size` must be provided when `allowed_batch_sizes` is"
          " empty."
      )

    if self.batch_timeout_micros < 0:
      raise ValueError(
          "`batch_timeout_micros` must be non-negative. Got:"
          f" {self.batch_timeout_micros}"
      )

    if self.num_batch_threads <= 0:
      raise ValueError(
          "`num_batch_threads` must be at least 1. Got:"
          f" {self.num_batch_threads}"
      )

    if self.max_enqueued_batches <= 0:
      raise ValueError(
          "`max_enqueued_batches` must be at least 1. Got:"
          f" {self.max_enqueued_batches}"
      )
