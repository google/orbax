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

"""Definitions and helpers for constructing OCDBT process spec strings.

These are used by Orbax OCDBT integration to name/identify per-process
subdirectories where OCDBT checkpoints are written.

When writing an OCDBT TensorStore checkpoint, per-process subdirectories are
used to isolate processes from each other. The names of these subdirectories
must conform to a specific format, defined in this module as OcdbtProcessSpec.
"""

import dataclasses
import re


# Standard OCDBT prefix for per-process subdirectories.
PROCESS_PREFIX = 'ocdbt.process_'
# Suffix for per-replica subdirectories, used when replica-separate folders are
# enabled.
REPLICA_SUFFIX = 'replica_'

# Process ID must be a non-empty string of alphanumeric characters.
_PROCESS_ID_RE = r'[A-Za-z0-9]+'
_PROCESS_PREFIX_RE = r'ocdbt\.process_'

_OCDBT_SPEC_PATTERN = re.compile(
    rf'^{_PROCESS_PREFIX_RE}({REPLICA_SUFFIX})?({_PROCESS_ID_RE})$'
)


@dataclasses.dataclass(frozen=True)
class OcdbtProcessSpec:
  """Orbax OCDBT process spec.

  Allows to construct and parse per-process subdirectory names for OCDBT
  TensorStore checkpoints. To get a subdirectory name, construct an instance of
  this class and use the `__str__` method.

  Attributes:
    process_id: The process ID, must be a non-empty string of alphanumeric
      characters.
    use_replica_suffix: Whether to use the replica suffix in the process spec.
      If False, the process spec will be of the form
      `ocdbt.process_{process_id}`. If True, the process spec will be of the
      form `ocdbt.process_replica_{process_id}`.
  """

  process_id: str
  use_replica_suffix: bool = False

  def __post_init__(self):
    if not re.fullmatch(rf'^{_PROCESS_ID_RE}$', self.process_id):
      raise ValueError(
          f'Invalid OCDBT process id: {self.process_id}. Must conform to'
          f' {_PROCESS_ID_RE} pattern.'
      )

  def __str__(self) -> str:
    """Returns the string representation of the OcdbtProcessSpec.

    This should be used as per-process subdirectory name for OCDBT TensorStore
    checkpoints.
    """
    replica_suffix = REPLICA_SUFFIX if self.use_replica_suffix else ''
    return f'{PROCESS_PREFIX}{replica_suffix}{self.process_id}'

  @classmethod
  def parse(cls, s: str) -> 'OcdbtProcessSpec':
    """Parses the OcdbtProcessSpec from a string."""
    m = _OCDBT_SPEC_PATTERN.fullmatch(s)
    if not m:
      raise ValueError(f"Bad string '{s}' for Orbax/OCDBT process spec")
    return OcdbtProcessSpec(
        process_id=m.group(2), use_replica_suffix=(m.group(1) is not None)
    )
