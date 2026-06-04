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

"""Run manifest — environment snapshot written once per suite.

Captures git SHA + dirty flag, JAX / orbax / tensorstore versions, host
identity, jax topology counts, and the XLA_FLAGS / LIBTPU_INIT_ARGS
strings that govern the run. The point is reproducibility: every TB
dashboard should be able to answer "what code + what env produced this?"
without grepping the launcher log.
"""

from __future__ import annotations

import dataclasses
import datetime
import importlib
import os
import socket
import subprocess


def _safe_check_output(cmd: list[str]) -> str:
  """Runs cmd and returns its stripped stdout, or an empty string on error."""
  try:
    return (
        subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=5)
        .decode("utf-8")
        .strip()
    )
  except (
      subprocess.CalledProcessError,
      FileNotFoundError,
      OSError,
      subprocess.TimeoutExpired,
  ):
    return ""


def _safe_module_version(import_path: str) -> str:
  try:
    mod = importlib.import_module(import_path)
    return getattr(mod, "__version__", "unknown")
  except ImportError:
    return "missing"


def _capture_topology() -> tuple[int, int, int, str]:
  """Best-effort jax topology snapshot."""
  try:
    import jax  # pylint: disable=g-import-not-at-top

    devices = jax.devices()
    kind = devices[0].device_kind if devices else "unknown"
    return (
        jax.process_count(),
        jax.process_index(),
        jax.device_count(),
        kind,
    )
  except Exception:  # pylint: disable=broad-exception-caught
    return (1, 0, 0, "unknown")


@dataclasses.dataclass(frozen=True)
class RunManifest:
  """Environment + code snapshot captured once per benchmark suite."""

  captured_at: str
  hostname: str
  git_sha: str
  git_dirty: bool
  jax_version: str
  orbax_version: str
  tensorstore_version: str
  jax_process_count: int
  jax_process_index: int
  jax_device_count: int
  jax_device_kind: str
  xla_flags: str
  libtpu_init_args: str

  def as_markdown(self) -> str:
    """Renders the run manifest as grouped markdown tables."""
    code_table = [
        "| field | value |",
        "|---|---|",
        f"| `git_sha` | `{self.git_sha}` |",
        f"| `git_dirty` | `{self.git_dirty}` |",
        f"| `jax_version` | `{self.jax_version}` |",
        f"| `orbax_version` | `{self.orbax_version}` |",
        f"| `tensorstore_version` | `{self.tensorstore_version}` |",
    ]
    xla_flags = self.xla_flags or "(unset)"
    libtpu_init_args = self.libtpu_init_args or "(unset)"
    env_table = [
        "| field | value |",
        "|---|---|",
        f"| `hostname` | `{self.hostname}` |",
        f"| `captured_at` | `{self.captured_at}` |",
        f"| `XLA_FLAGS` | `{xla_flags}` |",
        f"| `LIBTPU_INIT_ARGS` | `{libtpu_init_args}` |",
    ]
    topo_table = [
        "| field | value |",
        "|---|---|",
        f"| `jax_process_count` | `{self.jax_process_count}` |",
        f"| `jax_process_index` | `{self.jax_process_index}` |",
        f"| `jax_device_count` | `{self.jax_device_count}` |",
        f"| `jax_device_kind` | `{self.jax_device_kind}` |",
    ]
    return "\n".join([
        "## Run manifest",
        "",
        "### Code",
        "",
        *code_table,
        "",
        "### Environment",
        "",
        *env_table,
        "",
        "### Topology",
        "",
        *topo_table,
        "",
    ])


def capture_run_manifest() -> RunManifest:
  """Captures every reproducibility-relevant fact about the current process."""
  git_sha = _safe_check_output(["git", "rev-parse", "HEAD"]) or "unknown"
  git_status = _safe_check_output(["git", "status", "--porcelain"])
  git_dirty = bool(git_status.strip())

  pc, pi, dc, kind = _capture_topology()

  return RunManifest(
      captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      hostname=socket.gethostname(),
      git_sha=git_sha,
      git_dirty=git_dirty,
      jax_version=_safe_module_version("jax"),
      orbax_version=_safe_module_version("orbax.checkpoint"),
      tensorstore_version=_safe_module_version("tensorstore"),
      jax_process_count=pc,
      jax_process_index=pi,
      jax_device_count=dc,
      jax_device_kind=kind,
      xla_flags=os.environ.get("XLA_FLAGS", ""),
      libtpu_init_args=os.environ.get("LIBTPU_INIT_ARGS", ""),
  )
