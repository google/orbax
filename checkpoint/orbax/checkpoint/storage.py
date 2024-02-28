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

"""Orbax storage entities."""

import dataclasses
from typing import Iterator, Optional, Protocol, Type

from absl import logging
from etils import epath


class Root(Protocol):
  """Root dir context containing step subdirs."""

  def __init__(self, *args, **kwargs):
    ...

  def lookup(self, step: int, step_lookup: 'StepLookup') -> 'StepMetadata':
    """Returns a `StepMetadata` for `step` with `step_lookup`."""
    ...

  def define(
      self, step: int, step_def: Optional['StepDef'] = None
  ) -> 'StepMetadata':
    """Returns a `StepMetadata` defined with `step` and `StepDef`."""
    ...


class _Root(Root):
  """Default internal implementation of `Root`."""

  def __init__(self, path: epath.PathLike):
    self._path = epath.Path(path)

  def lookup(self, step: int, step_lookup: 'StepLookup') -> 'StepMetadata':
    """Returns a `StepMetadata` for `step` with `step_lookup`."""

    def _step_metadatas(
        handler: 'StepLookupHandler',
    ) -> Iterator[StepMetadata]:
      for step_path in handler.get_step_paths(self._path):
        if step_metadata := handler.build_step_metadata(step_path):
          yield step_metadata

    step_lookup_handler = step_lookup.step_lookup_handler(step)
    step_metadatas = _step_metadatas(step_lookup_handler)
    return step_lookup_handler.select_step_metadata(step_metadatas)

  def define(
      self, step: int, step_def: Optional['StepDef'] = None
  ) -> 'StepMetadata':
    """Returns a `StepMetadata` defined with `step` and `StepDef`."""
    return StepMetadata(
        step=step,
        path=self._path / default_step_name(step=step, step_def=step_def),
    )


def root(path: epath.PathLike, root_factory: Type[Root] = _Root) -> Root:
  """Returns a `Root` instance containing all steps.

  Args:
    path: Path containing step subdirs.
    root_factory: Factory to return custom implementation on `Root`.
  """
  logging.info('Creating %s(%s)', root_factory, path)
  return root_factory(path)


@dataclasses.dataclass(frozen=True)
class StepDef:
  """Specifications of a step used to define StepMetadata.

  For example:
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

  def step_prefix_underscore(self) -> Optional[str]:
    """Returns `step_prefix` with an *underscore* appended."""
    return f'{self.step_prefix}_' if self.step_prefix else None


@dataclasses.dataclass(frozen=True)
class StepMetadata:
  """Metadata about a step."""

  step: int
  path: epath.Path


class StepLookup(Protocol):
  """Step lookup request."""

  def step_lookup_handler(self, step: int) -> 'StepLookupHandler':
    """Returns a `StepLookupHandler` to process this `StepLookup` for `step`."""
    ...


class StepLookupHandler(Protocol):
  """Protocol to process `StepLookup` request."""

  def __init__(self, *args, **kwargs):
    ...

  def get_step_paths(self, base_path: epath.PathLike) -> Iterator[epath.Path]:
    """Returns step paths under `base_path`."""
    ...

  def build_step_metadata(
      self, step_path: epath.Path
  ) -> Optional[StepMetadata]:
    """Returns step metadata from `step_path` or None."""
    ...

  def select_step_metadata(
      self, step_metadatas: Iterator[StepMetadata]
  ) -> StepMetadata:
    """Returns a step metadata from `step_metadatas`."""
    ...


class DefaultStepLookupHandler(StepLookupHandler):
  """Default impl of `StepLookupHandler` to process `StepLookup` request."""

  def __init__(self, *, step: int, step_lookup: StepLookup, glob_pattern: str):
    self._step = step
    self._step_lookup = step_lookup
    self._glob_pattern = glob_pattern

  def get_step_paths(self, base_path: epath.PathLike) -> Iterator[epath.Path]:
    """Returns step paths under `base_path`."""
    logging.info(
        'StepLookupHandler: %s, listing step subdirs from: %s', self, base_path
    )
    return epath.Path(base_path).glob(self._glob_pattern)

  def build_step_metadata(
      self, step_path: epath.Path
  ) -> Optional[StepMetadata]:
    """Returns step metadata from `step_path` or None."""
    if not step_path.is_dir():
      return None
    return StepMetadata(step=self._step, path=step_path)

  def select_step_metadata(
      self, step_metadatas: Iterator[StepMetadata]
  ) -> StepMetadata:
    """Returns one step metadata from `step_metadatas`.

    Args:
      step_metadatas: Iterator[StepMetadata] used for extracting an element.
        Expected to contain just one element.

    Returns:
      A StepMetadata.
    Raises:
      ValueError: If `step_metadatas` contains more than one elements.
      ValueError: If `step_metadatas` is empty.
    """

    selected = None
    for step_metadata in step_metadatas:
      if selected is not None:
        raise ValueError(
            f'Multiple matches found for {self._step_lookup}: {selected},'
            f' {step_metadata} ...'
        )
      selected = step_metadata
    if selected is None:
      raise ValueError(f'No matches found for {self._step_lookup}.')
    return selected


def default_step_name(*, step: int, step_def: Optional[StepDef] = None) -> str:
  """Returns `(prefix_)?(zero padding)?step` name."""
  fixed_prefix = (step_def and step_def.step_prefix_underscore()) or ''

  if step_def and step_def.step_format_fixed_length is not None:
    step_str = f'{step:0{step_def.step_format_fixed_length}d}'
  else:
    step_str = f'{step}'

  # [prefix]step
  return f'{fixed_prefix}{step_str}'


@dataclasses.dataclass(frozen=True)
class DefaultStepLookup(StepLookup):
  """Matches `*<step>` pattern.

  `*<step>` could further match `(fixed prefix)?(zero padding)?<step>` pattern.

  For example:
   * step_prefix=None    step_format_fixed_length=None  ->  23
   * step_prefix=None    step_format_fixed_length=4     ->  0023
   * step_prefix=step    step_format_fixed_length=None  ->  step_23
   * step_prefix=step    step_format_fixed_length=4     ->  step_0023

  Attributes:
    step_prefix: Optional fixed string prefixed to step. Note an *underscore* is
      appended before applying it.
    step_format_fixed_length: Optional length of padded step number. # e.g. 6
      for 000123.
    step_lookup_handler_factory: StepLookupHandler factory. Default:
      `DefaultStepLookupHandler`.
  """

  step_prefix: Optional[str] = None
  step_format_fixed_length: Optional[int] = None
  step_lookup_handler_factory: Type[StepLookupHandler] = (
      DefaultStepLookupHandler
  )

  def step_lookup_handler(self, step: int) -> 'StepLookupHandler':
    """Returns a `StepLookupHandler` to process this `DefaultStepLookup` for `step`."""
    return self.step_lookup_handler_factory(
        step=step,
        step_lookup=self,
        glob_pattern=default_step_name(
            step=step,
            step_def=StepDef(
                step_prefix=self.step_prefix,
                step_format_fixed_length=self.step_format_fixed_length,
            ),
        ),
    )
