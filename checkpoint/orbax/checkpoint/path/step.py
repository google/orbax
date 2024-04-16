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

"""Orbax step storage entities."""

import abc
from collections.abc import Callable
import dataclasses
import re
from typing import Iterator, Optional, Protocol, Sequence, TypeVar
from absl import logging
from etils import epath

# This file mode gives full permissions to OWNER, GROUP and OTHER.
WORLD_READABLE_MODE = 0o777

MetadataT = TypeVar('MetadataT', bound='Metadata')


@dataclasses.dataclass(frozen=True)
class Metadata:
  """Metadata of a step."""

  step: int
  path: epath.Path


class NameFormat(Protocol):
  """Protocol responsible for naming and querying steps."""

  @abc.abstractmethod
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
  def build_metadata(
      self, step_path: epath.Path, *args, **kwargs
  ) -> Optional[MetadataT]:
    """Returns metadata for given `step_path` if it is valid or None.

    *Implementation hint:* Implement it to build the `MetadataT` instance using
    the given `step_path`. It should be called from `find_metadata(...)`,
    `find_step(...)` and `find_all(...)`. This method may not perform IO
    operations.

    Args:
      step_path: Path to folder containing step data.
      *args: Overridable args meant to provide custom params.
      **kwargs: Overridable kwargs meant to provide custom params.
    """
    ...

  @abc.abstractmethod
  def find_metadata(
      self, base_path: epath.PathLike, step: int, *args, **kwargs
  ) -> Optional[MetadataT]:
    """Returns metadata for given `base_path` and `step` or None.

    *Implementation hint:* Implement it to find the step folder under
    `base_path` performing IO operations if needed. Use `build_metadata(...)` to
    build the `MetadataT` using the found step path.

    Args:
      base_path: *root* Path under which Step folders are placed.
      step: Step number.
      *args: Overridable args meant to provide custom params.
      **kwargs: Overridable kwargs meant to provide custom params.
    """
    ...

  def find_all(self, base_path: epath.PathLike) -> Iterator[MetadataT]:
    """Returns metadata of all steps.

    *Implementation hint:* Implement it to find all step folders under
    `base_path` performing IO operations if needed. Use `build_metadata(...)`
    and `build_step_metadatas(...)` to build all the `MetadataT` using the found
    step paths.

    Args:
      base_path: *root* Path under which Step folders are placed.
    """
    ...

  def find_step(self, base_path: epath.PathLike, step: int) -> MetadataT:
    """Returns the metadata for `step` or raises ValueError.

    *Implementation hint:* Implement it to find the step folder under
    `base_path` performing IO operations if needed. Use `build_metadata(...)` to
    build the `MetadataT` using the found step path.

    Args:
      base_path: *root* Path under which Step folders are placed.
      step: Step number.
    """
    metadata = self.find_metadata(base_path, step)
    if metadata is not None:
      return metadata

    # Raise detailed error message.
    try:
      name = self.build_name(step)
    except Exception:  # pylint: disable=broad-exception-caught
      name = f'*{step}'  # build_name may raise error.
    raise ValueError(
        f'No step path found with name={name}, NameFormat={self} for'
        f' step={step} under {base_path}.'
    )


def build_step_path(
    base_path: epath.PathLike, name_format: NameFormat, step: int
) -> epath.Path:
  """Returns `step` path under `base_path` for step `name_format`."""
  return epath.Path(base_path) / name_format.build_name(step)


def build_step_metadatas(
    step_paths: Iterator[epath.Path],
    build_metadata: Callable[[epath.Path], Optional[MetadataT]],
) -> Iterator[MetadataT]:
  """Yields filtered metadata mapped with `step_paths`.

  Args:
    step_paths: Iterator of step paths.
    build_metadata: Callable to match and build step metadata from `step_paths`
      elements. If a `step_paths` element doesn't match then it returns None.

  Yields:
    Step metadata.
  """
  for step_path in step_paths:
    if step_metadata := build_metadata(step_path):
      yield step_metadata


def step_prefix_with_underscore(step_prefix: Optional[str]) -> str:
  """Returns `step_prefix` appended with `underscore` or <empty> if None."""
  return '' if step_prefix is None else f'{step_prefix}_'


@dataclasses.dataclass(frozen=True)
class _StandardNameFormat(NameFormat):
  """NameFormat for 'standard' steps for common Orbax use cases.

  Naming examples:
   * step_prefix=None    step_format_fixed_length=None  ->  23
   * step_prefix=None    step_format_fixed_length=4     ->  0023
   * step_prefix=step    step_format_fixed_length=None  ->  step_23
   * step_prefix=step    step_format_fixed_length=4     ->  step_0023

  Attributes:
    step_prefix: Optional fixed string prefixed to step. Note an *underscore* is
      appended before applying it.
    step_format_fixed_length: Optional length of the zero padded step. e.g. 6
      for 000123.
  """

  step_prefix: Optional[str] = None
  step_format_fixed_length: Optional[int] = None

  def build_name(self, step: int) -> str:
    """Returns `(prefix_)?(zero padding)?step` name."""
    if self.step_format_fixed_length is not None:
      step_str = f'{step:0{self.step_format_fixed_length}d}'
    else:
      step_str = f'{step}'

    # [prefix]step
    return f'{step_prefix_with_underscore(self.step_prefix)}{step_str}'

  def build_metadata(
      self, step_path: epath.Path, step: Optional[int] = None
  ) -> Optional[Metadata]:
    """Returns metadata for given `step_path` if it is valid or None."""
    if not step_path.is_dir():
      return None

    if step is not None:
      # step already known, just check exists.
      if step_path.exists():
        return Metadata(step=step, path=step_path)

    # Regex: [prefix]*(step)
    if self.step_format_fixed_length and self.step_format_fixed_length > 0:
      zero_present = rf'0\d{{{self.step_format_fixed_length-1}}}'
      zero_not_present = rf'[1-9]\d{{{self.step_format_fixed_length-1}}}\d*'
      zero_padded_step_group = rf'({zero_present}|{zero_not_present})'
    else:
      zero_padded_step_group = r'(0|[1-9]\d*)'
    name_regex = f'^{step_prefix_with_underscore(self.step_prefix)}{zero_padded_step_group}$'

    match = re.search(name_regex, step_path.name)
    if match is None:
      return None
    (step_,) = match.groups()
    step_ = int(step_)

    return Metadata(step=step_, path=step_path)

  def find_metadata(
      self, base_path: epath.PathLike, step: int
  ) -> Optional[MetadataT]:
    """Returns metadata for given `base_path` and `step` or None."""
    step_path = build_step_path(base_path, self, step)
    return self.build_metadata(step_path, step=step)

  def find_all(self, base_path: epath.PathLike) -> Iterator[Metadata]:
    """Returns metadata of all steps matching with name_format attributes."""
    # <step_prefix>_?<0 padding>?*
    step_paths = epath.Path(base_path).glob(
        f'{step_prefix_with_underscore(self.step_prefix)}*'
    )
    return build_step_metadatas(step_paths, self.build_metadata)


def standard_name_format(
    step_prefix: Optional[str] = None,
    step_format_fixed_length: Optional[int] = None,
) -> NameFormat:
  """Returns NameFormat for 'standard' steps for common Orbax use cases.

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
  """
  return _StandardNameFormat(
      step_prefix=step_prefix, step_format_fixed_length=step_format_fixed_length
  )


@dataclasses.dataclass(frozen=True)
class _CompositeNameFormat(NameFormat):
  """Supports reading multiple step namings, but just one format to write.

  Attributes:
    write_name_format: NameFormat used to build step names meant for writing
      checkpoints. Must be present in `read_name_formats` at a preferred
      priority position.
    read_name_formats: Sequence (ordered) of NameFormats used to find steps for
      reading checkpoints. It acts like an *or*, where the first one to match is
      returned.
  """

  write_name_format: NameFormat
  read_name_formats: Sequence[NameFormat]

  def __post_init__(self):
    if self.write_name_format not in self.read_name_formats:
      raise ValueError(
          f'write_name_format: {self.write_name_format} must be present in'
          f' read_name_formats: {self.read_name_formats}.'
      )

  def build_name(self, step: int) -> str:
    """Returns `step` name using `write_name_format`."""
    return self.write_name_format.build_name(step)

  def build_metadata(
      self, step_path: epath.Path, step: Optional[int] = None
  ) -> Optional[Metadata]:
    """Returns metadata for given `step_path` if it is valid or None."""
    for read_name_format in self.read_name_formats:
      metadata = read_name_format.build_metadata(step_path, step)
      if metadata is not None:
        return metadata
    return None

  def find_metadata(
      self, base_path: epath.PathLike, step: int
  ) -> Optional[Metadata]:
    """Returns metadata for given `base_path` and `step` or None."""
    try:
      return self.find_step(base_path, step)
    except ValueError:
      return None

  def find_all(self, base_path: epath.PathLike) -> Iterator[Metadata]:
    """Returns metadata of all steps."""
    step_paths = epath.Path(base_path).iterdir()
    return build_step_metadatas(step_paths, self.build_metadata)

  def find_step(self, base_path: epath.PathLike, step: int) -> Metadata:
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
    write_name_format: NameFormat,
    read_name_formats: Sequence[NameFormat],
) -> NameFormat:
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
