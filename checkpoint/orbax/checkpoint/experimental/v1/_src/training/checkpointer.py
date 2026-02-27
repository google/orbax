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

"""Defines a class for managing a sequence of checkpoints in a training loop."""

from __future__ import annotations

import typing
from typing import Any, Iterable, Sequence

from absl import logging
from etils import epy
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
import orbax.checkpoint.experimental.v1._src.handlers.global_registration  # pylint: disable=unused-import
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.loading import loading
from orbax.checkpoint.experimental.v1._src.metadata import loading as metadata_loading
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import step as path_step_lib
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.saving import saving
from orbax.checkpoint.experimental.v1._src.synchronization import types as async_types
from orbax.checkpoint.experimental.v1._src.training import errors
from orbax.checkpoint.experimental.v1._src.training import preservation_policies
from orbax.checkpoint.experimental.v1._src.training import save_decision_policies
from orbax.checkpoint.experimental.v1._src.training.metadata import types as training_metadata_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


CheckpointMetadata = training_metadata_types.CheckpointMetadata
RootMetadata = training_metadata_types.RootMetadata


PYTREE_CHECKPOINTABLE_KEY = checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY


class _AsyncSaveResponse(async_types.AsyncResponse[bool]):

  def __init__(
      self, manager: checkpoint_manager.CheckpointManager, saved: bool
  ):
    self._manager = manager
    self._saved = saved

  def result(self, timeout: float | None = None) -> bool:
    del timeout  # Ignored.
    self._manager.wait_until_finished()
    return self._saved


def _resolve_integer_step(
    step: int | CheckpointMetadata,
) -> int:
  if isinstance(step, int):
    return step
  return step.step


@typing.final
class Checkpointer(epy.ContextManager):
  """An object that manages a sequence of checkpoints in a training loop."""

  def __init__(
      self,
      directory: path_types.PathLike,
      *,
      save_decision_policy: (
          save_decision_policies.SaveDecisionPolicy | None
      ) = None,
      preservation_policy: (
          preservation_policies.PreservationPolicy | None
      ) = None,
      step_name_format: (
          path_step_lib.NameFormat[path_step_lib.Metadata] | None
      ) = None,
      custom_metadata: tree_types.JsonType | None = None,
  ):
    """Initializes a Checkpointer.

    IMPORTANT: This class is not thread safe. All APIs should be called across
    all available processes, from the main thread.

    The Checkpointer is intended for use in a training loop, where a sequence
    of checkpoints are saved at regular intervals. Example usage::

      # Configure the frequency at which checkpoints are saved.
      save_decision = ocp.training.save_decision
      save_decision_policy = save_decision.AnySavePolicy([
          save_decision.FixedIntervalPolicy(1000),
          save_decision.PreemptionPolicy(),
      ])

      # Configure the checkpoints to preserve (avoid garbage collection).
      preservation = ocp.training.preservation
      preservation_policy = preservation.AnyPreservationPolicy([
          preservation.LatestN(10),
          preservation.EveryNSteps(10000),
      ])

      with ocp.training.Checkpointer(
          directory,
          save_decision_policy=save_decision_policy,
          preservation_policy=preservation_policy,
      ) as ckptr:
        if ckptr.latest is None:
          model_state = init_from_scratch(rng)
        else:
          model_state = ckptr.load_pytree()  # Loads latest checkpoint.
          # Note: prefer to specify the abstract tree if available.
          model_state = ckptr.load_pytree(
              ckptr.latest, abstract_pytree=abstract_model_state)
        start_step = ckptr.latest.step if ckptr.latest else 0
        for step in range(start_step, num_steps):
          model_state = train_step(model_state)
          # Saves a checkpoint if needed (according to `save_decision_policy`).
          ckptr.save_pytree(step, model_state)

    Prefer to use the context manager style as shown above, which ensures that
    the Checkpointer is closed properly and any outstanding async operations are
    completed.

    Args:
      directory: The root directory where checkpoints are stored. The directory
        will be created if it does not exist.
      save_decision_policy: A policy used to determine when a checkpoint should
        be saved. If not provided, the `Checkpointer` saves as often as possible
        by default (assuming no checkpoint is currently being saved), and saves
        when a preemption is detected by the JAX distributed system.
      preservation_policy: A policy used to determine when a checkpoint should
        be preserved. Any checkpoints not preserved are garbage collected. If
        not provided,
      step_name_format: An object used to specify the format for step paths. By
        default, steps are rendered as simple integers, like
        `/root/directory/<step>`.
      custom_metadata: A JSON dictionary representing user-specified custom
        metadata. This should be information that is relevant to the entire
        sequence of checkpoints, rather than to any single checkpoint.
    """

    context = context_lib.get_context()

    default_save_decision_policy = save_decision_policies.AnySavePolicy([
        save_decision_policies.InitialSavePolicy(),
        save_decision_policies.ContinuousCheckpointingPolicy(),
        save_decision_policies.PreemptionCheckpointingPolicy(),
    ])
    save_decision_policy = save_decision_policy or default_save_decision_policy
    default_preservation_policy = preservation_policies.PreserveAll()
    preservation_policy = preservation_policy or default_preservation_policy

    self._step_name_format = (
        step_name_format or path_step_lib.standard_name_format()
    )
    options = checkpoint_manager.CheckpointManagerOptions(
        save_decision_policy=save_decision_policy,
        preservation_policy=preservation_policy,
        step_name_format=step_name_format,
        max_to_keep=None,  # Unlimited.
        # TODO(b/401541834) Configure todelete_subdir.
        # TODO(b/401541834) Enable background deletion.
        async_options=context.async_options.v0(),
        file_options=context.file_options.v0(),
        multiprocessing_options=context.multiprocessing_options.v0(),
        temporary_path_class=context.file_options.temporary_path_class,
        # Prevent the checkpoint manager from writing metrics on its own. This
        # class will take responsibility for writing metrics.
        prevent_write_metrics=True,
    )
    self._manager = checkpoint_manager.CheckpointManager(
        directory,
        options=options,
        metadata=custom_metadata,
    )

  @property
  def directory(self) -> path_types.Path:
    """The root directory where checkpoint steps are located."""
    return self._manager.directory

  @property
  def latest(self) -> CheckpointMetadata[None] | None:
    """Returns the latest :py:class:`.CheckpointMetadata`, or None if no checkpoints exist.

    See `checkpoints` documentation below.

    Returns:
      The latest checkpoint, or None if no checkpoints exist.
    """
    if not self.checkpoints:
      return None
    return self.checkpoints[-1]

  @property
  def checkpoints(self) -> Sequence[CheckpointMetadata[None]]:
    """Returns a list of :py:class:`.CheckpointMetadata`, sorted ascending by step.

    The method returns a list of :py:class:`.CheckpointMetadata` objects, which
    contain selected properties describing the checkpoint. Contrast this with
    the methods :py:func:`.pytree_metadata` and
    :py:func:`.checkpointables_metadata`, which may perform a more expensive
    disk read to retrieve additional information. This method only returns
    cheap cacheable properties like step and timestamp. The return value is
    annotated as :py:class:`.CheckpointMetadata[None]` because the core
    `metadata` property is not retrieved, and is therefore `None`.

    The property is cached to avoid repeated disk reads. This is not a problem
    unless checkpoints are manually deleted, or deleted by some other job or
    class that `Checkpointer` is unaware of. Note that doing this is
    discouraged.

    Returns:
      A list of checkpoints, sorted ascending by step.
    """

    infos = sorted(self._manager._checkpoints, key=lambda info: info.step)  # pylint: disable=protected-access
    return [
        CheckpointMetadata[None](
            info.step,
            metadata=None,
            metrics=info.metrics,
            commit_timestamp_nsecs=int(info.time.timestamp() * 1e9),
        )
        for info in infos
    ]

  def _resolve_existing_checkpoint(
      self, step: int | CheckpointMetadata | None
  ) -> CheckpointMetadata[None]:
    if step is None:
      latest = self.latest
      if latest is None:
        raise errors.StepNotFoundError(
            'Specified `step=None`, but no checkpoints were found.'
        )
      return latest
    step = _resolve_integer_step(step)
    for checkpoint in self.checkpoints:
      if checkpoint.step == step:
        return checkpoint
    raise errors.StepNotFoundError(f'No checkpoint found at step {step}.')

  def should_save(self, step: int) -> bool:
    """Returns whether a checkpoint should be saved at the given step."""
    step = _resolve_integer_step(step)
    return self._manager.should_save(step)

  def save_pytree(
      self,
      step: int,
      pytree: tree_types.PyTreeOf[tree_types.LeafType],
      *,
      force: bool = False,
      overwrite: bool = False,
      metrics: tree_types.JsonType | None = None,
      custom_metadata: tree_types.JsonType | None = None,
  ) -> bool:
    """Saves a checkpoint, if dictated by :py:class:`.SaveDecisionPolicy`.

    This function behaves similarly to :py:func:`.save_pytree` (see
    documentation), but performs additional tasks related to managing a sequence
    of checkpoint steps.

    It consists roughly of the following steps:
      - Check whether a checkpoint should be saved at the given step.
      - Check whether a save is already in progress. If so, wait for it to
        finish.
      - Save to a directory given by `root_directory / <step_format>`.
      - Perform garbage collection if necessary.
      - Return whether a checkpoint was saved or not.

    It is important to note that the `Checkpointer` never allows saving more
    than one checkpoint at a time. Depending on the
    :py:class:`.SaveDecisionPolicy`, a checkpoint may be saved or skipped at a
    given step, but if a save is initiated, as dictated by the policy, then it
    will proceed as normal as long as no other save is currently in progress. If
    a save is already in progress, the function will block until the previous
    save has finished.

    Args:
      step: The step number to save.
      pytree: The PyTree to save.
      force: If True, ignores all :py:class:`.SaveDecisionPolicy` checks, and
        always decides to save a checkpoint.
      overwrite: If True, deletes any existing checkpoint at the given step
        before saving. Otherwise, raises an error if the checkpoint already
        exists.
      metrics: A PyTree of metrics to be saved with the checkpoint.
      custom_metadata: A JSON dictionary representing user-specified custom
        metadata. This should be information that is relevant to the checkpoint
        at the given step, rather than to the entire sequence of checkpoints.

    Returns:
      Whether a checkpoint was saved or not.
    """
    return self.save_pytree_async(
        step,
        pytree,
        force=force,
        overwrite=overwrite,
        metrics=metrics,
        custom_metadata=custom_metadata,
    ).result()

  def save_checkpointables(
      self,
      step: int,
      checkpointables: dict[str, Any],
      *,
      force: bool = False,
      overwrite: bool = False,
      metrics: tree_types.JsonType | None = None,
      custom_metadata: tree_types.JsonType | None = None,
  ) -> bool:
    """Saves a set of checkpointables at the given step."""
    return self.save_checkpointables_async(
        step,
        checkpointables,
        force=force,
        overwrite=overwrite,
        metrics=metrics,
        custom_metadata=custom_metadata,
    ).result()

  def save_pytree_async(
      self,
      step: int,
      pytree: tree_types.PyTreeOf[tree_types.LeafType],
      *,
      force: bool = False,
      overwrite: bool = False,
      metrics: tree_types.JsonType | None = None,
      custom_metadata: tree_types.JsonType | None = None,
  ) -> async_types.AsyncResponse[bool]:
    """Saves a checkpoint asynchronously, if dictated by :py:class:`.SaveDecisionPolicy`.

    See documentation for :py:func:`.save_pytree` for full details. This
    function is essentially the same, except that it executes mostly in the
    background, and blocks for as little time as possible (primarily to
    transfer weights from device to host).

    Args:
      step: The step number to save.
      pytree: The PyTree to save.
      force: If True, ignores all :py:class:`.SaveDecisionPolicy` checks, and
        always decides to save a checkpoint.
      overwrite: If True, deletes any existing checkpoint at the given step
        before saving. Otherwise, raises an error if the checkpoint already
        exists.
      metrics: A PyTree of metrics to be saved with the checkpoint.
      custom_metadata: A JSON dictionary representing user-specified custom
        metadata. This should be information that is relevant to the checkpoint
        at the given step, rather than to the entire sequence of checkpoints.

    Returns:
      An AsyncResponse, which can be awaited via `result()`, which returns a
      bool indicating whether a checkpoint was saved or not.
    """
    return self.save_checkpointables_async(
        step,
        {PYTREE_CHECKPOINTABLE_KEY: pytree},
        force=force,
        overwrite=overwrite,
        metrics=metrics,
        custom_metadata=custom_metadata,
    )

  def save_checkpointables_async(
      self,
      step: int,
      checkpointables: dict[str, Any],
      *,
      force: bool = False,
      overwrite: bool = False,
      metrics: tree_types.JsonType | None = None,
      custom_metadata: tree_types.JsonType | None = None,
  ) -> async_types.AsyncResponse[bool]:
    """Saves a set of checkpointables asynchronously at the given step."""
    if overwrite:
      logging.info(
          'Specified `overwrite`: deleting existing checkpoint %d if it'
          ' exists.',
          step,
      )
      try:
        self._manager.delete(step)
      except FileNotFoundError:
        pass
    elif step in [c.step for c in self.checkpoints]:
      raise errors.StepAlreadyExistsError(f'Step {step} already exists.')

    checkpointer, args = saving.get_v0_checkpointer_and_args(
        checkpointables, metrics=metrics, context=context_lib.get_context()
    )
    self._manager._checkpointer = checkpointer  # pylint: disable=protected-access
    saved = self._manager.save(
        step,
        args=args,
        metrics=metrics,
        force=force,
        custom_metadata=custom_metadata,
    )
    return _AsyncSaveResponse(self._manager, saved)

  def load_pytree(
      self,
      step: int | CheckpointMetadata | None = None,
      abstract_pytree: (
          tree_types.PyTreeOf[tree_types.AbstractLeafType] | None
      ) = None,
  ) -> tree_types.PyTreeOf[tree_types.LeafType]:
    """Loads a PyTree checkpoint at the given step.

    This function behaves similarly to :py:func:`.load_pytree` (see
    documentation).

    Args:
      step: The step number or :py:class:`.CheckpointMetadata` to load.
      abstract_pytree: The abstract PyTree to load.

    Returns:
      The loaded PyTree.
    """
    return self.load_checkpointables(
        step, {PYTREE_CHECKPOINTABLE_KEY: abstract_pytree}
    )[PYTREE_CHECKPOINTABLE_KEY]

  def load_checkpointables(
      self,
      step: int | CheckpointMetadata | None = None,
      abstract_checkpointables: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    """Loads a set of checkpointables at the given step."""
    step = self._resolve_existing_checkpoint(step).step
    return  loading.load_checkpointables(
        self.directory / self._step_name_format.build_name(step),
        abstract_checkpointables,
    )

  def load_pytree_async(
      self,
      step: int | CheckpointMetadata | None = None,
      abstract_pytree: (
          tree_types.PyTreeOf[tree_types.AbstractLeafType] | None
      ) = None,
  ) -> async_types.AsyncResponse[tree_types.PyTreeOf[tree_types.LeafType]]:
    """Not yet supported."""
    raise NotImplementedError()

  def load_checkpointables_async(
      self,
      step: int | CheckpointMetadata | None = None,
      abstract_checkpointables: dict[str, Any] | None = None,
  ) -> async_types.AsyncResponse[dict[str, Any]]:
    """Loads a set of checkpointables asynchronously at the given step."""
    raise NotImplementedError()

  def pytree_metadata(
      self, step: int | CheckpointMetadata | None = None
  ) -> training_metadata_types.CheckpointMetadata[
      metadata_types.PyTreeMetadata
  ]:
    """Returns checkpoint metadata for the given step.

    Retrieves metadata describing the structure of the PyTree stored at the
    given step. If no step is provided, the method resolves to the latest
    available checkpoint.

    Args:
      step: The step number to retrieve metadata for. If `None`, the latest step
        is used. Can also be a :py:class:`.CheckpointMetadata` object, from
        which the step is extracted.

    Returns:
      A :py:class:`.CheckpointMetadata` object containing
      :py:class:`.PyTreeMetadata`, along with checkpoint timestamp and metrics
      information.
    """
    checkpoint = self._resolve_existing_checkpoint(step)
    del step
    checkpoint_metadata = metadata_loading.pytree_metadata(
        self._manager.directory
        / self._step_name_format.build_name(checkpoint.step)
    )
    return training_metadata_types.CheckpointMetadata[
        metadata_types.PyTreeMetadata
    ](
        step=checkpoint.step,
        metadata=checkpoint_metadata.metadata,
        init_timestamp_nsecs=checkpoint_metadata.init_timestamp_nsecs,
        commit_timestamp_nsecs=checkpoint_metadata.commit_timestamp_nsecs,
        custom_metadata=checkpoint_metadata.custom_metadata,
        metrics=checkpoint.metrics,
    )

  def checkpointables_metadata(
      self, step: int | CheckpointMetadata | None = None
  ) -> training_metadata_types.CheckpointMetadata[dict[str, Any]]:
    """Returns checkpoint metadata for the given step.

    Retrieves metadata describing the structure of the checkpointables stored at
    the given step. If no step is provided, the method resolves to the latest
    available checkpoint.

    Args:
      step: The step number to retrieve metadata for. If `None`, the latest step
        is used. Can also be a :py:class:`.CheckpointMetadata` object, from
        which the step is extracted.

    Returns:
      A :py:class:`.CheckpointMetadata` object containing a `dict[str, Any]`
      describing the checkpointables, along with checkpoint timestamp and
      metrics information.
    """
    checkpoint = self._resolve_existing_checkpoint(step)
    del step
    checkpoint_metadata = metadata_loading.checkpointables_metadata(
        self._manager.directory
        / self._step_name_format.build_name(checkpoint.step)
    )
    return training_metadata_types.CheckpointMetadata[dict[str, Any]](
        step=checkpoint.step,
        metadata=checkpoint_metadata.metadata,
        init_timestamp_nsecs=checkpoint_metadata.init_timestamp_nsecs,
        commit_timestamp_nsecs=checkpoint_metadata.commit_timestamp_nsecs,
        custom_metadata=checkpoint_metadata.custom_metadata,
        metrics=checkpoint.metrics,
    )

  def root_metadata(
      self,
  ) -> (
      training_metadata_types.CheckpointMetadata[metadata_types.PyTreeMetadata]
      | training_metadata_types.RootMetadata
  ):
    metadata = self._manager.metadata(None)
    return RootMetadata(
        directory=self.directory, custom_metadata=metadata.custom_metadata
    )

  def reload(self):
    """Reloads internal properties from the root directory.

    Updates the list of available checkpoints by rescanning the storage
    location. Use this method to sync the checkpointer with the file system
    if checkpoints have been added or removed externally.
    """
    self._manager.reload()

  def is_saving_in_progress(self) -> bool:
    """Returns whether a checkpoint save operation is currently in progress.

    Checks if there are any background persistence operations
    currently active.

    Returns:
      `True` if a save operation is in progress, `False` otherwise.
    """
    return self._manager.is_saving_in_progress()

  def wait(self):
    """Waits for any outstanding async operations to complete.

    This method blocks until all background tasks, such as asynchronous saves,
    have finished. Use this method to ensure that all operations are finalized
    before proceeding with dependent actions.
    """
    self._manager.wait_until_finished()

  def close(self):
    """Waits for pending async operations to complete and releases resources.

    This method blocks until all background tasks, such as asynchronous saves,
    have finished. It also performs necessary cleanup, such as closing
    file handles.
    """
    self._manager.close()

  def __contextmanager__(
      self,
  ) -> Iterable[Checkpointer]:
    try:
      yield self
    finally:
      self.close()
