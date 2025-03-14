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
# LINT.ThenChange(//depot//orbax/export/obm_export.py)


@dataclasses.dataclass(kw_only=True)
class BatchOptions:
  """Batch options for the Orbax model.

  Attributes:
    batch_component: The component to batch.
    max_batch_size: The maximum allowed batch size for any input. If
      `allowed_batch_sizes` is provided, this can be None or must be the largest
      one in the list. Otherwise, this must be provided.
    batch_timeout_micros: Maximum number of microseconds to wait before
      outputting an incomplete batch.
    allowed_batch_sizes: Optional list of allowed batch sizes. If left empty,
      all batch sizes no larger than `max_batch_size` are allowed. Otherwise,
      supplies a list of batch sizes. The entries must increase monotonically.
    disable_large_batch_splitting: Whether to disable large batch splitting.
  """
  batch_component: BatchComponent
  max_batch_size: int | None = None
  batch_timeout_micros: int
  allowed_batch_sizes: Sequence[int]
  disable_large_batch_splitting: bool = False

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
        # When `allowed_batch_sizes` is provided, `max_batch_size` must be the
        # largest one in the list.
        if self.max_batch_size != self.allowed_batch_sizes[-1]:
          raise ValueError(
              "`max_batch_size` must be the largest one in"
              f" `allowed_batch_sizes`. Got: {self.max_batch_size} vs"
              f" {self.allowed_batch_sizes[-1]}. Set `max_batch_size` to None"
              " to automatically infer it."
          )

    # When `allowed_batch_sizes` is not provided, `max_batch_size` must be
    # provided.
    if self.max_batch_size is None:
      raise ValueError(
          "`max_batch_size` must be provided when `allowed_batch_sizes` is"
          " empty."
      )

    if self.batch_timeout_micros <= 0:
      raise ValueError(
          "`batch_timeout_micros` must be non-negative. Got:"
          f" {self.batch_timeout_micros}"
      )
