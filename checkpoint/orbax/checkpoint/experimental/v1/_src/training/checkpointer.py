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

"""Defines a class for managing a sequence of checkpoints in a training loop."""

from __future__ import annotations

from typing import Iterable, Literal, Sequence, overload

from absl import logging
from etils import epy
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import pytree_handler
from orbax.checkpoint.experimental.v1._src.metadata import loading as metadata_loading
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import step as path_step_lib
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import types as async_types
from orbax.checkpoint.experimental.v1._src.training import errors
from orbax.checkpoint.experimental.v1._src.training import save_decision_policies
from orbax.checkpoint.experimental.v1._src.training.metadata import types as training_metadata_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types
from orbax.checkpoint.handlers import handler_registration


CheckpointMetadata = training_metadata_types.CheckpointMetadata
RootMetadata = training_metadata_types.RootMetadata


_PYTREE_CHECKPOINTABLE_KEY = 'pytree'


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


class Checkpointer(epy.ContextManager):
  """An object that manages a sequence of checkpoints in a training loop."""

  def __init__(
      self,
      directory: path_types.PathLike,
      *,
      save_decision_policy: (
          save_decision_policies.SaveDecisionPolicy | None
      ) = None,
      step_name_format: (
          path_step_lib.NameFormat[path_step_lib.Metadata] | None
      ) = None,
      custom_metadata: tree_types.JsonType | None = None,
      # TODO(b/371005679): Add PreservationPolicy.
  ):
    """Initializes a Checkpointer.

    The Checkpointer is intended for use in a training loop, where a sequence
    of checkpoints are saved at regular intervals. Example usage::

      save_decision_policies = ocp.training.save_decision_policies
      # Save every 1000 steps, or when a preemption is detected.
      save_decision_policy = save_decision_policies.AnySavePolicy(
          save_decision_policies.FixedIntervalPolicy(1000),
          save_decision_policies.PreemptionPolicy(),
      )
      with ocp.training.Checkpointer(
          directory,
          save_decision_policy,
          # TODO(b/371005679): Add example for PreservationPolicy.
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
      step_name_format: An object used to specify the format for step paths. By
        default, steps are rendered as simple integers, like
        `/root/directory/<step>`.
      custom_metadata: A JSON dictionary representing user-specified custom
        metadata. This should be information that is relevant to the entire
        sequence of checkpoints, rather than to any single checkpoint.
    """
    default_save_decision_policy = save_decision_policies.AnySavePolicy([
        save_decision_policies.InitialSavePolicy(),
        save_decision_policies.ContinuousCheckpointingPolicy(),
        save_decision_policies.PreemptionCheckpointingPolicy(),
    ])
    save_decision_policy = save_decision_policy or default_save_decision_policy
    context = context_lib.get_context()
    options = checkpoint_manager.CheckpointManagerOptions(
        save_decision_policy=save_decision_policy,
        step_name_format=step_name_format,
        max_to_keep=None,  # Unlimited.
        # TODO(b/401541834) Configure todelete_subdir.
        # TODO(b/401541834) Enable background deletion.
        async_options=context.async_options.v0(),
        file_options=context.file_options.v0(),
        multiprocessing_options=context.multiprocessing_options.v0(),
        temporary_path_class=context.file_options.temporary_path_class,
    )
    self._step_name_format = (
        step_name_format or path_step_lib.standard_name_format()
    )
    v0_handler = pytree_handler.create_v0_handler(context_lib.get_context())
    registry = handler_registration.create_default_handler_registry(
        **{_PYTREE_CHECKPOINTABLE_KEY: v0_handler}
    )
    self._manager = checkpoint_manager.CheckpointManager(
        directory,
        options=options,
        handler_registry=registry,
        metadata=custom_metadata,
    )

  @property
  def directory(self) -> path_types.Path:
    """The root directory where checkpoint steps are located."""
    return self._manager.directory

  @property
  def latest(self) -> CheckpointMetadata[None] | None:
    """Returns the latest `CheckpointMetadata`, or None if no checkpoints exist.

    See `checkpoints` documentation below.

    Returns:
      The latest `CheckpointMetadata`, or None if no checkpoints exist.
    """
    if not self.checkpoints:
      return None
    return self.checkpoints[-1]

  @property
  def checkpoints(self) -> Sequence[CheckpointMetadata[None]]:
    """Returns a list of `CheckpointMetadata`, sorted ascending by step.

    The method returns a list of `CheckpointMetadata` objects, which contain
    selected properties describing the checkpoint. Contrast this with the
    methods `pytree_metadata` and `checkpointables_metadata`, which may perform
    a more expensive disk read to retrieve additional information. This method
    only returns cheap cacheable properties like step and timestamp. The return
    value is annotated as `CheckpointMetadata[None]` because the core `metadata`
    property is not retrieved, and is therefore `None`.

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
        )
        for info in infos
    ]

  def _select_checkpoint(
      self, step: int | CheckpointMetadata
  ) -> CheckpointMetadata[None]:
    """Returns the checkpoint metadata at the given step."""
    step = self._resolve_step(step)
    for checkpoint in self.checkpoints:
      if checkpoint.step == step:
        return checkpoint
    raise ValueError(f'No checkpoint found at step {step}.')

  def should_save(self, step: int) -> bool:
    """Returns whether a checkpoint should be saved at the given step."""
    step = self._resolve_step(step)
    return self._manager.should_save(step)

  @overload
  def _resolve_step(self, step: int | CheckpointMetadata) -> int:
    ...

  @overload
  def _resolve_step(self, step: Literal[None]) -> None:
    ...

  def _resolve_step(self, step: int | CheckpointMetadata | None) -> int | None:
    if isinstance(step, CheckpointMetadata):
      return step.step
    return step

  def save_pytree(
      self,
      step: int,
      pytree: tree_types.PyTreeOf[tree_types.LeafType],
      *,
      force: bool = False,
      metrics: tree_types.JsonType | None = None,
      custom_metadata: tree_types.JsonType | None = None,
  ) -> bool:
    """Saves a PyTree checkpoint at the given step.

    This function behaves similarly to `ocp.save_pytree` (see documentation),
    but performs additional tasks related to managing a sequence of checkpoint
    steps.

    It consists roughly of the following steps:
      - Check whether a checkpoint should be saved at the given step.
      - Save to a directory given by `root_directory / <step_format>`.
      - Perform garbage collection if necessary.
      - Return whether a checkpoint was saved or not.

    Args:
      step: The step number to save.
      pytree: The PyTree to save.
      force: If True, deletes any existing checkpoint at the given step before
        saving.
      metrics: A PyTree of metrics to be saved with the checkpoint.
      custom_metadata: A JSON dictionary representing user-specified custom
        metadata. This should be information that is relevant to the checkpoint
        at the given step, rather than to the entire sequence of checkpoints.

    Returns:
      Whether a checkpoint was saved or not.
    """
    response = self.save_pytree_async(
        step,
        pytree,
        force=force,
        metrics=metrics,
        custom_metadata=custom_metadata,
    )
    return response.result()

  def save_pytree_async(
      self,
      step: int,
      pytree: tree_types.PyTreeOf[tree_types.LeafType],
      *,
      force: bool = False,
      metrics: tree_types.JsonType | None = None,
      custom_metadata: tree_types.JsonType | None = None,
  ) -> async_types.AsyncResponse[bool]:
    """Saves a PyTree checkpoint asynchronously at the given step.

    See documentation for `save_pytree` for more details. This function executes
    in the background, and blocks for as little time as possible.

    Args:
      step: The step number to save.
      pytree: The PyTree to save.
      force: If True, deletes any existing checkpoint at the given step before
        saving.
      metrics: A PyTree of metrics to be saved with the checkpoint.
      custom_metadata: A JSON dictionary representing user-specified custom
        metadata. This should be information that is relevant to the checkpoint
        at the given step, rather than to the entire sequence of checkpoints.

    Returns:
      An AsyncResponse, which can be awaited via `result()`, which returns a
      bool indicating whether a checkpoint was saved or not.

    Raises:
      StepAlreadyExistsError: If a checkpoint already exists at the given step.
    """
    args = pytree_handler.create_v0_save_args(context_lib.get_context(), pytree)
    if force:
      logging.info(
          'Specified `force`: deleting existing checkpoint %d if it exists.',
          step,
      )
      try:
        self._manager.delete(step)
      except FileNotFoundError:
        pass
    elif step in [c.step for c in self.checkpoints]:
      raise errors.StepAlreadyExistsError(f'Step {step} already exists.')
    saved = self._manager.save(
        step,
        args=args_lib.Composite(**{_PYTREE_CHECKPOINTABLE_KEY: args}),
        force=force,
        metrics=metrics,
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

    This function behaves similarly to `ocp.load_pytree` (see documentation).

    Args:
      step: The step number or `CheckpointMetadata` to load.
      abstract_pytree: The abstract PyTree to load.

    Returns:
      The loaded PyTree.
    """
    step = self._resolve_step(step)
    args = pytree_handler.create_v0_restore_args(
        context_lib.get_context(), abstract_pytree
    )
    restored = self._manager.restore(
        step,
        args=args_lib.Composite(**{_PYTREE_CHECKPOINTABLE_KEY: args}),
    )
    return restored[_PYTREE_CHECKPOINTABLE_KEY]

  def load_pytree_async(
      self,
      step: int | CheckpointMetadata | None = None,
      abstract_pytree: (
          tree_types.PyTreeOf[tree_types.AbstractLeafType] | None
      ) = None,
  ) -> async_types.AsyncResponse[tree_types.PyTreeOf[tree_types.LeafType]]:
    """Not yet supported."""
    raise NotImplementedError()

  def pytree_metadata(
      self, step: int
  ) -> training_metadata_types.CheckpointMetadata[
      metadata_types.PyTreeMetadata
  ]:
    """Returns checkpoint metadata for the given step."""
    checkpoint_metadata = metadata_loading.pytree_metadata(
        self._manager.directory / self._step_name_format.build_name(step)
    )
    return training_metadata_types.CheckpointMetadata[
        metadata_types.PyTreeMetadata
    ](
        step=step,
        metadata=checkpoint_metadata.metadata,
        init_timestamp_nsecs=checkpoint_metadata.init_timestamp_nsecs,
        commit_timestamp_nsecs=checkpoint_metadata.commit_timestamp_nsecs,
        custom_metadata=checkpoint_metadata.custom_metadata,
        descriptor=checkpoint_metadata.descriptor,
        metrics=self._select_checkpoint(step).metrics,
    )

  def root_metadata(
      self,
  ) -> (
      training_metadata_types.CheckpointMetadata[metadata_types.PyTreeMetadata]
      | training_metadata_types.RootMetadata
  ):
    metadata = self._manager.metadata(None)
    return RootMetadata(custom_metadata=metadata.custom_metadata)

  def reload(self):
    """Reloads internal properties from the root directory."""
    self._manager.reload()

  def close(self):
    """Ensures any outstanding async operations are completed before closing."""
    self._manager.close()

  def __contextmanager__(
      self,
  ) -> Iterable[Checkpointer]:
    try:
      yield self
    finally:
      self.close()
