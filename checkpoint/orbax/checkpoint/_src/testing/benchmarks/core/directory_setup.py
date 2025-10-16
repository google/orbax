# Copyright 2025 The Orbax Authors.
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

"""Functions for test directory setup in Orbax benchmark tests."""

from absl import logging
from etils import epath
import jax


def setup_test_directory(name: str, base_path: str | None = None) -> epath.Path:
  """Creates a unique, clean test directory for a benchmark run.

  It supports both local filesystems and cloud storage (like GCS) via etils.

  Args:
      name: The name of the test, used to create the directory.
      base_path: The parent directory. Defaults to /tmp/orbax_benchmarks/.

  Returns:
      A path pointing to the created directory.
  """
  base_path = "/tmp/orbax_benchmarks" if base_path is None else base_path
  path = epath.Path(base_path) / name
  logging.info("Setting up test directory at: %s", path)
  if jax.process_index() == 0:
    if path.exists():
      logging.warning("Test directory %s already exists. Deleting it.", path)
      path.rmtree()
    path.mkdir(parents=True)
  return path
