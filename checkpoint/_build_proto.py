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

"""PEP 517 build backend wrapping flit_core to compile Orbax protos.

Referenced by `build-backend` in pyproject.toml; generates
checkpoint_operation_pb2.py before flit_core builds an sdist or wheel.
"""

from __future__ import annotations

from collections.abc import Callable
import importlib.util
import pathlib
import subprocess
import sys
from typing import ParamSpec

from flit_core import buildapi

_ROOT = pathlib.Path(__file__).resolve().parent
_PROTO_REL = 'orbax/checkpoint/proto/checkpoint_operation.proto'
_PB2_REL = 'orbax/checkpoint/proto/checkpoint_operation_pb2.py'
_P = ParamSpec('_P')

get_requires_for_build_wheel = buildapi.get_requires_for_build_wheel
get_requires_for_build_sdist = buildapi.get_requires_for_build_sdist
get_requires_for_build_editable = buildapi.get_requires_for_build_editable
prepare_metadata_for_build_wheel = buildapi.prepare_metadata_for_build_wheel
prepare_metadata_for_build_editable = (
    buildapi.prepare_metadata_for_build_editable
)


def _compile_protos() -> None:
  """Compiles checkpoint_operation.proto into checkpoint_operation_pb2.py."""
  if importlib.util.find_spec('grpc_tools') is None:
    # Sdists ship the generated module, so they build without grpcio-tools.
    if (_ROOT / _PB2_REL).is_file():
      return
    raise RuntimeError(
        f'grpcio-tools is required to compile {_PROTO_REL}; install '
        'grpcio-tools>=1.81.0 or build with PEP 517 isolation enabled.'
    )
  subprocess.run(
      [
          sys.executable,
          '-m',
          'grpc_tools.protoc',
          '-I=.',
          '--python_out=.',
          _PROTO_REL,
      ],
      cwd=_ROOT,
      check=True,
  )


def _with_protos(hook: Callable[_P, str]) -> Callable[_P, str]:
  """Wraps a flit_core build hook to compile protos first.

  Args:
    hook: flit_core PEP 517 hook that builds an sdist or wheel.

  Returns:
    Hook with the same arguments that runs protoc before calling `hook`.
  """

  def build(*args: _P.args, **kwargs: _P.kwargs) -> str:
    _compile_protos()
    return hook(*args, **kwargs)

  return build


build_wheel = _with_protos(buildapi.build_wheel)
build_editable = _with_protos(buildapi.build_editable)
build_sdist = _with_protos(buildapi.build_sdist)
