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
import enum


# LINT.IfChange
@enum.unique
class BatchComponent(enum.Enum):
  """The commputation component to batch."""

  NO_BATCHING = "no_batching"
  MODEL_FUNCTION = "model_function"
# LINT.ThenChange(//depot//orbax/export/obm_export.py)


class BatchOptions:
  """Batch options for the Orbax model."""

  def __init__(
      self,
      *,
      batch_component: BatchComponent,
      max_batch_size: int | None = None,
      batch_timeout_micros: int,
      allowed_batch_sizes: Sequence[int],
      disable_large_batch_splitting: bool = False,
  ):
    """Initializes the batch options.

    Args:
      batch_component: The component to batch.
      max_batch_size: The maximum batch size.
      batch_timeout_micros: The batch timeout in microseconds.
      allowed_batch_sizes: The allowed batch sizes. Must be sorted in ascending
        order.
      disable_large_batch_splitting: Whether to disable large batch splitting.
    """

    # `allowed_batch_sizes` should be sorted in ascending order.
    if not all(
        allowed_batch_size > 0 for allowed_batch_size in allowed_batch_sizes
    ):
      raise ValueError(
          f"`allowed_batch_sizes` must be positive. Got: {allowed_batch_sizes}"
      )
    for i in range(len(allowed_batch_sizes) - 1):
      if allowed_batch_sizes[i] >= allowed_batch_sizes[i + 1]:
        raise ValueError(
            "`allowed_batch_sizes` must be sorted in ascending order. Got:"
            f" {allowed_batch_sizes}"
        )

    # When `allowed_batch_sizes` is provided, `max_batch_size` must be the
    # largest one in the list.
    if allowed_batch_sizes:
      if max_batch_size is None:
        max_batch_size = allowed_batch_sizes[-1]
      elif max_batch_size != allowed_batch_sizes[-1]:
        raise ValueError(
            "`max_batch_size` must be the largest one in"
            f" `allowed_batch_sizes`. Got: {max_batch_size} vs"
            f" {allowed_batch_sizes[-1]}"
        )

    # When `allowed_batch_sizes` is not provided, `max_batch_size` must be
    # provided.
    if not allowed_batch_sizes and max_batch_size is None:
      raise ValueError(
          "`max_batch_size` must be provided when `allowed_batch_sizes` is"
          " empty."
      )

    if batch_timeout_micros <= 0:
      raise ValueError(
          "`batch_timeout_micros` must be non-negative. Got:"
          f" {batch_timeout_micros}"
      )

    self.batch_component = batch_component
    self.max_batch_size = max_batch_size
    self.batch_timeout_micros = batch_timeout_micros
    self.allowed_batch_sizes = allowed_batch_sizes
    self.disable_large_batch_splitting = disable_large_batch_splitting
