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

"""Configuration classes for Orbax Model Export."""

from collections.abc import Mapping, Sequence
import dataclasses
import enum
import itertools
import logging
import jax


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
    MINIMIZE_TPU_COST_PER_REQUEST: Chooses to either PAD_UP or BATCH_DOWN so as
    to minimize the TPU costs per real request.

  See the documentation of BatchOptions.BatchPaddingPolicy for details.
  """

  PAD_UP = "pad_up"
  BATCH_DOWN = "batch_down"
  MINIMIZE_TPU_COST_PER_REQUEST = "minimize_tpu_cost_per_request"


@enum.unique
class MixedPriorityBatchingPolicy(enum.Enum):
  """The mixed priority batch policy for the batch scheduler.

  Options:
    LOW_PRIORITY_PADDING_WITH_MAX_BATCH_SIZE: Pad low priority inputs up to the
    max_batch_size.
    LOW_PRIORITY_PADDING_WITH_NEXT_ALLOWED_BATCH_SIZE: Pad low priority inputs
    up to the next allowed batch size.
    PRIORITY_ISOLATION: Keep low and high priority inputs in separate batches.
    PRIORITY_MERGE: Merge low priority inputs into the high priority batch.
  """
  LOW_PRIORITY_PADDING_WITH_MAX_BATCH_SIZE = "low_priority_padding_with_max_batch_size"
  LOW_PRIORITY_PADDING_WITH_NEXT_ALLOWED_BATCH_SIZE = "low_priority_padding_with_next_allowed_batch_size"
  PRIORITY_ISOLATION = "priority_isolation"
  PRIORITY_MERGE = "priority_merge"


# LINT.ThenChange(//depot//orbax/export/obm_export.py)


@dataclasses.dataclass(kw_only=True)
class LowPriorityBatchOptions:
  """Low priority batch options for the Orbax model.

  This is used to configure the batching behavior for low priority inputs. If
  not provided, the priority batching is disabled. See the documentation of
  BatchOptions.LowPriorityBatchOptions for details.

  Attributes:
    max_batch_size: The maximum allowed batch size for low priority inputs. If
      low priority `allowed_batch_sizes` is provided, this can be None and
      infered from the largest value in the list. Otherwise, this must be
      provided.
    batch_timeout_micros: Maximum number of microseconds to wait before
      outputting an incomplete low priority batch.
    allowed_batch_sizes: Optional list of allowed batch sizes. If left empty,
      all batch sizes no larger than `max_batch_size` are allowed. Otherwise,
      supplies a list of batch sizes. The entries must increase monotonically.
    max_enqueued_batches: Maximum number of batches enqueued for processing
      before requests are failed fast. Default is 250.
  """

  max_batch_size: int | None = None
  batch_timeout_micros: int = 0
  allowed_batch_sizes: Sequence[int] | None = None
  max_enqueued_batches: int = 250


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
    low_priority_batch_options: The batch options for low priority inputs.
    mixed_priority_batching_policy: The mixed priority batching policy for the
      batch scheduler. Default is LOW_PRIORITY_PADDING_WITH_MAX_BATCH_SIZE.
  """

  batch_component: BatchComponent
  max_batch_size: int | None = None
  batch_timeout_micros: int = 0
  allowed_batch_sizes: Sequence[int] | None = None
  num_batch_threads: int = 1
  max_enqueued_batches: int = 250
  disable_large_batch_splitting: bool = False
  batch_padding_policy: BatchPaddingPolicy = BatchPaddingPolicy.PAD_UP
  low_priority_batch_options: LowPriorityBatchOptions | None = None
  mixed_priority_batching_policy: MixedPriorityBatchingPolicy = (
      MixedPriorityBatchingPolicy.LOW_PRIORITY_PADDING_WITH_MAX_BATCH_SIZE
  )

  def _validate_batch_options(
      self,
      max_batch_size: int,
      allowed_batch_sizes: Sequence[int] | None,
      batch_timeout_micros: int,
      max_enqueued_batches: int,
      is_low_priority_batch_options: bool = False,
  ) -> None:
    """Validates the batch options.

    Args:
      max_batch_size: The max batch size.
      allowed_batch_sizes: The allowed batch sizes.
      batch_timeout_micros: The batch timeout in microseconds.
      max_enqueued_batches: The max number of enqueued batches.
      is_low_priority_batch_options: Whether the batch options are for low
        priority batches.

    Raises:
      ValueError: If the batch options are invalid.
    """
    low_priority_prefix = (
        "low_priority_batch_options." if is_low_priority_batch_options else ""
    )
    if max_batch_size < 1:
      raise ValueError(
          f"`{low_priority_prefix}max_batch_size` must be positive. Got:"
          f" {max_batch_size}"
      )
    if allowed_batch_sizes:
      # The allowed batch sizes must be positive and sorted in ascending order.
      for a, b in itertools.pairwise(allowed_batch_sizes):
        if a >= b:
          raise ValueError(
              f"`{low_priority_prefix}allowed_batch_sizes` must be sorted in"
              f" ascending order. Got: {allowed_batch_sizes}"
          )
      if allowed_batch_sizes[0] < 1:
        raise ValueError(
            f"`{low_priority_prefix}allowed_batch_sizes` must be positive. Got:"
            f" {allowed_batch_sizes}"
        )

      # When `allowed_batch_sizes` and `max_batch_size` are provided,
      # `max_batch_size` must be larger than or equal to the largest allowed
      # batch size.
      if (
          self.disable_large_batch_splitting
          and max_batch_size != allowed_batch_sizes[-1]
      ):
        raise ValueError(
            f"`{low_priority_prefix}max_batch_size` must be equal to the"
            f" largest one in `{low_priority_prefix}allowed_batch_sizes` when"
            f" large batch splitting is disabled. Got: {max_batch_size} vs"
            f" {allowed_batch_sizes[-1]}. Set"
            f" `{low_priority_prefix}max_batch_size` to None to automatically"
            f" infer it from `{low_priority_prefix}allowed_batch_sizes`."
        )
      if (
          not self.disable_large_batch_splitting
          and max_batch_size < allowed_batch_sizes[-1]
      ):
        raise ValueError(
            f"`{low_priority_prefix}max_batch_size` must be larger than or"
            " equal to the largest one in"
            f" `{low_priority_prefix}allowed_batch_sizes`. Got:"
            f" {max_batch_size} vs {allowed_batch_sizes[-1]}."
        )

    if batch_timeout_micros < 0:
      raise ValueError(
          f"`{low_priority_prefix}batch_timeout_micros` must be non-negative."
          f" Got: {batch_timeout_micros}"
      )

    if max_enqueued_batches <= 0:
      raise ValueError(
          f"`{low_priority_prefix}max_enqueued_batches` must be at least 1."
          f" Got: {max_enqueued_batches}"
      )

  def __post_init__(self):
    """Validates the batch options."""
    if self.batch_component == BatchComponent.NO_BATCHING:
      return

    if self.max_batch_size is None:
      if self.allowed_batch_sizes:
        self.max_batch_size = self.allowed_batch_sizes[-1]
      else:
        raise ValueError(
            "max_batch_size must be provided when allowed_batch_sizes is empty."
            f" Got: max_batch_size: {self.max_batch_size}; allowed_batch_sizes:"
            f" {self.allowed_batch_sizes}."
        )

    self._validate_batch_options(
        self.max_batch_size,
        self.allowed_batch_sizes,
        self.batch_timeout_micros,
        self.max_enqueued_batches,
    )

    if self.num_batch_threads < 0:
      raise ValueError(
          "`num_batch_threads` must be at least 1. Got:"
          f" {self.num_batch_threads}"
      )
    elif self.num_batch_threads == 0:
      # Set num_batch_threads to 1 if it is 0.
      logging.warning(
          "num_batch_threads is %d. Setting it to 1.",
          self.num_batch_threads,
      )
      self.num_batch_threads = 1

    if self.low_priority_batch_options is not None:
      if self.low_priority_batch_options.max_batch_size is None:
        if self.low_priority_batch_options.allowed_batch_sizes:
          self.low_priority_batch_options.max_batch_size = (
              self.low_priority_batch_options.allowed_batch_sizes[-1]
          )
        else:
          raise ValueError(
              "Low priority max_batch_size must be provided when"
              " allowed_batch_sizes is empty. Got:"
              " low_priority_batch_options.max_batch_size:"
              f" {self.low_priority_batch_options.max_batch_size};"
              " low_priority_batch_options.allowed_batch_sizes:"
              f" {self.low_priority_batch_options.allowed_batch_sizes}."
          )
      self._validate_batch_options(
          self.low_priority_batch_options.max_batch_size,
          self.low_priority_batch_options.allowed_batch_sizes,
          self.low_priority_batch_options.batch_timeout_micros,
          self.low_priority_batch_options.max_enqueued_batches,
          is_low_priority_batch_options=True,
      )


@dataclasses.dataclass(kw_only=True)
class Jax2ObmOptions:
  """Options for jax2obm conversion.

  Attributes:
    native_serialization_platforms: The native serialization platforms to use
      when converting the jax function to an obm function (i.e., 'cpu', 'cuda',
      'rocm', 'tpu'). The values are case-insensitive.
    checkpoint_path: Path for checkpoint weights relative to the output model
      directory. (e.g., if 'ckpt' is provided, weights will be saved in
      '<output_model_dir>/ckpt').
    weights_name: This name serves as a key in `OrbaxModule.checkpoint` to
      identify a specific set of weights that will be used by the exported JAX
      function.
    polymorphic_constraints: Input polymorphic constraints.
    load_all_checkpoint_weights: If set to True, all weights from the checkpoint
      will be loaded, including those not used by the exported function(s).
      Defaults to False, which only loads necessary weights to save memory
      during serving.
    xla_flags_per_platform: XLA flags per platform for the model.
    jax_mesh: Mesh for the model.
    persist_xla_flags: Whether to persist XLA flags in the exported model. If
      true, it will allow the model to be served with these XLA flags by
      overriding default XLA flags used by the serving runtime.
    enable_bf16_optimization: If set to True, float32 weights are converted to
      bfloat16 to save memory.
    enable_auto_layout: If set to True, automatically generate the optimal
      layout for model parameters to improve serving performance.
  """

  # TODO: b/448900820 - Consider constraint the type to the proto enums.
  native_serialization_platforms: Sequence[str] | str | None = None
  checkpoint_path: str | None = None
  weights_name: str | None = None
  polymorphic_constraints: (
      Mapping[str, Sequence[str]] | Sequence[str] | None
  ) = None
  # TODO: b/448900820 - Remove this variable, we should always load necessary
  # weights only.
  load_all_checkpoint_weights: bool = False
  xla_flags_per_platform: Mapping[str, Sequence[str]] | None = None
  jax_mesh: jax.sharding.Mesh | None = None
  persist_xla_flags: bool = True
  enable_bf16_optimization: bool = False
  enable_auto_layout: bool = False


@dataclasses.dataclass(kw_only=True)
class ObmExportOptions:
  """Options for Orbax Model Export.

  Attributes:
    batch_options: Configures if and how incoming inference requests are batched
      together for efficiency (a technique called online batching). See
      `BatchOptions` above for configuration details. If set to None, batching
      is disabled, and requests are processed individually. #
  """

  batch_options: BatchOptions | None = None
