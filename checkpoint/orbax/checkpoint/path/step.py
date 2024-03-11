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

from collections.abc import Callable
import dataclasses
import re
from typing import Iterator, Optional, Protocol, TypeVar
from etils import epath

MetadataT = TypeVar('MetadataT', bound='Metadata')


@dataclasses.dataclass(frozen=True)
class Metadata:
  """Metadata of a step."""

  step: int
  path: epath.Path


class NameFormat(Protocol):
  """Protocol responsible for naming and querying steps."""

  def build_name(self, step: int) -> str:
    """Returns `step` name."""
    ...

  def find_all(self, base_path: epath.PathLike) -> Iterator[MetadataT]:
    """Returns metadata of all steps."""
    ...

  def find_step(self, base_path: epath.PathLike, step: int) -> MetadataT:
    """Returns the metadata for `step` or raises ValueError."""
    ...


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


def step_prefix_with_underscore(step_prefix: Optional[str]) -> Optional[str]:
  """Returns `step_prefix` appended with `underscore`."""
  return step_prefix if step_prefix is None else f'{step_prefix}_'


def select_the_only_metadata(metadatas: Iterator[MetadataT]) -> MetadataT:
  """Returns the only metadata expected in `metadatas` or raises ValueError."""
  selected = None
  for metadata in metadatas:
    if selected is not None:
      raise ValueError(f'Multiple matches found: {selected}, {metadata} ...')
    selected = metadata
  if selected is None:
    raise ValueError('No matches found.')
  return selected


@dataclasses.dataclass(frozen=True)
class StandardNameFormat(NameFormat):
  """NameFormat for 'standard' steps sufficient for most of the Orbax needs.

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
    step_prefix_ = step_prefix_with_underscore(self.step_prefix) or ''

    if self.step_format_fixed_length is not None:
      step_str = f'{step:0{self.step_format_fixed_length}d}'
    else:
      step_str = f'{step}'

    # [prefix]step
    return f'{step_prefix_}{step_str}'

  def find_all(self, base_path: epath.PathLike) -> Iterator[Metadata]:
    """Returns metadata of all steps matching with name_format attributes."""
    step_prefix_ = step_prefix_with_underscore(self.step_prefix) or ''

    def _build_metadata(step_path: epath.Path) -> Optional[Metadata]:
      if not step_path.is_dir():
        return None
      # Regex: [prefix]*(step)
      if self.step_format_fixed_length and self.step_format_fixed_length > 0:
        zero_present = rf'0\d{{{self.step_format_fixed_length-1}}}'
        zero_not_present = rf'[1-9]\d{{{self.step_format_fixed_length-1}}}\d*'
        zero_padded_step_group = rf'({zero_present}|{zero_not_present})'
      else:
        zero_padded_step_group = r'(0|[1-9]\d*)'
      name_regex = f'^{step_prefix_}{zero_padded_step_group}$'

      match = re.search(name_regex, step_path.name)
      if match is None:
        return None
      (step_,) = match.groups()
      step_ = int(step_)

      return Metadata(step=step_, path=step_path)

    # <step_prefix>_?<0 padding>?*
    step_paths = epath.Path(base_path).glob(f'{step_prefix_}*')

    return build_step_metadatas(step_paths, _build_metadata)

  def find_step(self, base_path: epath.PathLike, step: int) -> Metadata:
    """Returns the metadata for `step` matching with name_format attributes."""

    def _build_metadata(step_path: epath.Path) -> Optional[Metadata]:
      if not step_path.is_dir():
        return None
      return Metadata(step=step, path=step_path)

    step_paths = epath.Path(base_path).glob(self.build_name(step))

    metadatas = build_step_metadatas(step_paths, _build_metadata)
    return select_the_only_metadata(metadatas)
