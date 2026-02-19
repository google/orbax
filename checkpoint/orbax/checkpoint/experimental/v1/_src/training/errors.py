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

"""Errors encountered during training."""


class StepAlreadyExistsError(FileExistsError):
  """Raised when a step intended for saving already exists.

  This error is raised when a training step attempts to save at a step number
  that already contains a checkpoint. This is intended to prevent accidental
  overwriting of existing checkpoints without explicit intention.

  Example Usage:
    This error is typically handled during the saving process::

      try:
        checkpoint_manager.save(step=100)
      except StepAlreadyExistsError:
        print("Checkpoint already exists at step 100.")

  Attributes:
      step (int): The step number that already exists.
      path (str): The path to the directory where the step already exists.
  """


class StepNotFoundError(FileNotFoundError):
  """Raised when a requested step is not found.

  This error is raised when a restoration operation requests a specific
  training step that does not exist in the checkpoint directory. This serves
  as a signal that the requested history is missing or has been deleted.

  Example Usage:
    This error is typically handled during the restoration process::

      try:
        checkpoint_manager.restore(step=200)
      except StepNotFoundError:
        print("Step 200 not found. Restoring latest available step.")
        checkpoint_manager.restore(step=checkpoint_manager.latest_step())

  Attributes:
    step (int): The step number that was requested but not found.
    path (str): The path where the step was expected to be found.
  """
