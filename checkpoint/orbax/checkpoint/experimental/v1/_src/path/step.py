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

"""Utilities for working with paths constructed from steps."""
import abc
import dataclasses
from typing import Generic, Iterator, Protocol, Sequence, TypeVar

from absl import logging
from etils import epath
from orbax.checkpoint._src.path import step as step_lib
from orbax.checkpoint.experimental.v1._src.training.metadata import types as training_metadata_types

CheckpointMetadata = training_metadata_types.CheckpointMetadata

MetadataT = TypeVar('MetadataT', bound='CheckpointMetadata')


class NameFormat(Protocol, Generic[MetadataT]):
  """Responsible for naming and querying steps."""

  def build_name(self, step: int) -> str:
    """Returns `step` name.

    *Implementation hint:* Implement it to build a name for the given step using
    the class's custom formatting attributes. Since it is mainly meant for
    building names to save checkpoints, it can raise error if this NameFormat is
    just meant for finding already existing step paths.

    Args:
      step: Step number.
    """
    ...

  @abc.abstractmethod
  def find_all(self, base_path: epath.PathLike) -> Iterator[MetadataT]:
    """Returns metadata of all steps.

    NOTE: Ignores uncommitted checkpoints.

    *Implementation hint:* Implement it to find all step folders under
    `base_path` performing IO operations if needed. Use
    `build_step_metadatas(...)` helper function to build all the `MetadataT`
    using the found step paths.

    Args:
      base_path: *root* Path under which Step folders are placed.
    """
    ...

  @abc.abstractmethod
  def find_step(self, base_path: epath.PathLike, step: int) -> MetadataT:
    """Returns the metadata for `step` or raises ValueError.

    NOTE: Ignores uncommitted checkpoints.

    *Implementation hint:* Implement it to find the step folder under
    `base_path` performing IO operations if needed.

    Args:
      base_path: *root* Path under which Step folders are placed.
      step: Step number.

    Raises:
      ValueError if no committed paths for the requested step is found.
    """
    ...


class _StandardNameFormat(NameFormat[CheckpointMetadata[None]]):
  """NameFormat for 'standard' steps for common Orbax use cases."""

  def __init__(
      self,
      step_prefix: str | None = None,
      step_format_fixed_length: int | None = None,
      single_host_load_and_broadcast: bool = False,
  ):
    self._delegate = step_lib.standard_name_format(
        step_prefix=step_prefix,
        step_format_fixed_length=step_format_fixed_length,
        single_host_load_and_broadcast=single_host_load_and_broadcast,
    )

  def build_name(self, step: int) -> str:
    return self._delegate.build_name(step)

  def find_all(
      self, base_path: epath.PathLike
  ) -> Iterator[CheckpointMetadata[None]]:
    result = self._delegate.find_all(base_path)
    for metadata in result:
      yield CheckpointMetadata(
          step=metadata.step,
          path=metadata.path,
          metadata=None,
          init_timestamp_nsecs=metadata.init_timestamp_nsecs,
          commit_timestamp_nsecs=metadata.commit_timestamp_nsecs,
      )

  def find_step(
      self, base_path: epath.PathLike, step: int
  ) -> CheckpointMetadata[None]:
    result = self._delegate.find_step(base_path, step)
    return CheckpointMetadata(
        step=result.step,
        path=result.path,
        metadata=None,
        init_timestamp_nsecs=result.init_timestamp_nsecs,
        commit_timestamp_nsecs=result.commit_timestamp_nsecs,
    )


def standard_name_format(
    *,
    step_prefix: str | None = None,
    step_format_fixed_length: int | None = None,
    single_host_load_and_broadcast: bool = False,
) -> NameFormat[CheckpointMetadata[None]]:
  """Returns NameFormat for 'standard' steps for common Orbax use cases.

  NOTE: Ignores uncommitted checkpoints.

  Naming examples:
   * step_prefix=None    step_format_fixed_length=None  ->  23
   * step_prefix=None    step_format_fixed_length=4     ->  0023
   * step_prefix=step    step_format_fixed_length=None  ->  step_23
   * step_prefix=step    step_format_fixed_length=4     ->  step_0023

  Args:
    step_prefix: Optional fixed string prefixed to step. Note an *underscore* is
      appended before applying it.
    step_format_fixed_length: Optional length of the zero padded step. e.g. 6
      for 000123.
    single_host_load_and_broadcast: If True, the jax process=0 will list all
      steps and broadcast them to all other processes. NOTE: Ignored if jax
      backend is not multi controller.
  """
  return _StandardNameFormat(
      step_prefix=step_prefix,
      step_format_fixed_length=step_format_fixed_length,
      single_host_load_and_broadcast=single_host_load_and_broadcast,
  )


@dataclasses.dataclass(frozen=True)
class _CompositeNameFormat(NameFormat[CheckpointMetadata[None]]):
  """A NameFormat that supports reading multiple step namings, but just one format to write.

  Attributes:
    write_name_format: NameFormat used to build step names meant for writing
      checkpoints. Must be present in `read_name_formats` at a preferred
      priority position.
    read_name_formats: Sequence (ordered) of NameFormats used to find steps for
      reading checkpoints. It acts like an *or*, where the first one to match is
      returned.
  """

  write_name_format: NameFormat[CheckpointMetadata[None]]
  read_name_formats: Sequence[NameFormat[CheckpointMetadata[None]]]

  def __post_init__(self):
    if self.write_name_format not in self.read_name_formats:
      read_name_formats = ','.join(str(f) for f in self.read_name_formats)
      raise ValueError(
          f'write_name_format: {self.write_name_format} must be present in'
          f' read_name_formats: [{read_name_formats}].'
      )

  def __str__(self):
    read_name_formats = ','.join(str(f) for f in self.read_name_formats)
    return f'Composite([{read_name_formats}])'

  def build_name(self, step: int) -> str:
    """Returns `step` name using `write_name_format`."""
    return self.write_name_format.build_name(step)

  def find_all(
      self, base_path: epath.PathLike
  ) -> Iterator[CheckpointMetadata[None]]:
    """Returns metadata of all steps."""
    found_paths = set()
    for read_name_format in self.read_name_formats:
      for step_metadata in read_name_format.find_all(base_path):
        if step_metadata.path not in found_paths:
          found_paths.add(step_metadata.path)
          yield step_metadata

  def find_step(
      self, base_path: epath.PathLike, step: int
  ) -> CheckpointMetadata[None]:
    """Returns the metadata for `step` or raises ValueError."""
    errors = []  # Used to raise the final collated error if needed.
    for read_name_format in self.read_name_formats:
      try:
        return read_name_format.find_step(base_path, step)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.info(
            'Failed to find step=%s with NameFormat=%s under %s. Error: %s',
            step,
            read_name_format,
            base_path,
            e,
        )
        errors.append(e)

    # Raise the concatenated errors.
    messages = [f'{e}' for e in errors]
    raise ValueError('\n'.join(messages))


def composite_name_format(
    write_name_format: NameFormat[CheckpointMetadata[None]],
    read_name_formats: Sequence[NameFormat[CheckpointMetadata[None]]],
) -> NameFormat[CheckpointMetadata[None]]:
  """Returns *composite* NameFormat supporting multiple read/single write formats.

  Args:
    write_name_format: NameFormat used to build step names meant for writing
      checkpoints. Must be present in `read_name_formats` at a preferred
      priority position.
    read_name_formats: Sequence (ordered) of NameFormats used to find steps for
      reading checkpoints. Please note that to resolve conflicts (and avoid
      raising errors) in case of multiple NameFormats matching a given step, the
      sequence should be provided in highest to lowest priority order:
      NameFormat appearing earlier in the sequence is preferred.
  """
  return _CompositeNameFormat(write_name_format, read_name_formats)
