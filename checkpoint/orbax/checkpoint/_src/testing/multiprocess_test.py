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

"""Helper for running multi-process tests."""

import os

import jax
from orbax.checkpoint._src.multihost import multihost

from .learning.brain.research.jax.tests.multiprocess import multiprocess_test

main = multiprocess_test.main


class MultiProcessTest(multiprocess_test.MultiProcessTest):
  # TODO(b/378138653) Support TPUless MultiProcessTest.

  def setUp(self):
    """Start distributed service."""
    if multihost.is_pathways_backend():
      assert (
          jax.process_count() == 1
      ), "Expected 1 process for Pathways backend."
    else:
      super().setUp()

  def tearDown(self):
    """Stop distributed service."""
    if multihost.is_pathways_backend():
      return
    super().tearDown()

  def multiprocess_create_tempdir(
      self, name: str | None = None
  ) -> str:
    """Creates a temporary directory for the test."""
    directory = self._get_tempdir_path_test()
    if name is not None:
      directory = os.path.join(directory, name)
    if jax.process_index() == 0:
      os.makedirs(directory, exist_ok=False)
    jax.experimental.multihost_utils.sync_global_devices(
        "multiprocess_create_tempdir"
    )
    return directory
