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

"""Tests for Pathways deleter.

Note: It is important not to pass `self` to the dispatched function.
"""

from absl import flags
import jax
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import pathways as multihost_pathways
from orbax.checkpoint._src.path import step as step_lib
from orbax.checkpoint.experimental.v1._src.emergency import deleter as deleter_lib
from orbax.checkpoint.testing import local_path as local_path_test_lib

from .pyglib.contrib.g3_multiprocessing import g3_multiprocessing
from absl.testing import absltest
from .testing.pybase import parameterized


FLAGS = flags.FLAGS

jax.config.update('jax_enable_x64', True)


class DeleterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._directory = local_path_test_lib.create_local_path_base(self)
    self.assertTrue(self._directory.exists())
    self.assertTrue(multihost.is_pathways_backend())
    self._dispatcher = dispatchers.RemotePythonDispatcher()

  def test_dispatch_function(self):
    local_path = local_path_test_lib.LocalPath(self._directory)
    jax.block_until_ready(self._dispatcher.dispatch(local_path.mkdir))

    for i in range(multihost_pathways.worker_count(None)):
      path = self._directory / f'local_{i}'
      self.assertTrue(path.exists(), f'Path {path} does not exist.')

  def test_delete_local_step(self):
    name_format = step_lib.standard_name_format()
    local_path = local_path_test_lib.LocalPath(self._directory)
    deleter = deleter_lib.create_checkpoint_deleter(
        local_path, name_format=name_format  # pytype: disable=wrong-arg-types
    )

    def _make_step_path():
      path = local_path / name_format.build_name(1)
      path.mkdir(parents=True, exist_ok=False)

    jax.block_until_ready(self._dispatcher.dispatch(_make_step_path))

    worker_count = multihost_pathways.worker_count(None)
    self.assertEqual(worker_count, 4)
    for worker_id in range(worker_count):
      path = self._directory / f'local_{worker_id}' / name_format.build_name(1)
      self.assertTrue(path.exists())

    deleter.delete(1)
    for worker_id in range(worker_count):
      path = self._directory / f'local_{worker_id}' / name_format.build_name(1)
      self.assertFalse(
          path.exists(), f'Path {path} still exists on worker {worker_id}'
      )


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  g3_multiprocessing.handle_test_main(googletest.main)
