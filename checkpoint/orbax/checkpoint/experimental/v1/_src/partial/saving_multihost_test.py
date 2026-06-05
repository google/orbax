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

from __future__ import annotations

import asyncio
from unittest import mock

from absl.testing import parameterized
from etils import epath
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.path.snapshot import snapshot
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.partial import path as partial_path_lib
from orbax.checkpoint.experimental.v1._src.partial import saving
from orbax.checkpoint.experimental.v1._src.partial import saving_test
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils


create_numpy_pytree = array_test_utils.create_numpy_pytree
create_sharded_pytree = array_test_utils.create_sharded_pytree


class PartialSavingMultihostTest(
    saving_test.PartialSavingTest, multiprocess_test.MultiProcessTest
):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.multiprocess_create_tempdir(name='partial_saving_multihost_test')
    )
    self.pytree, self.abstract_pytree = create_sharded_pytree()
    self.numpy_pytree, self.abstract_numpy_pytree = create_numpy_pytree()

    self.context = context_lib.Context()
    self.context.multiprocessing.primary_host = 0
    self.context.multiprocessing.barrier_sync_key_prefix = 'PartialSavingTest'

    test_utils.set_tensorstore_driver_for_test()
    test_utils.sync_global_processes('PartialSavingTest:setup_complete')

  def tearDown(self):
    super().tearDown()
    test_utils.sync_global_processes('PartialSavingTest:teardown_complete')
    self.loop.close()

  @property
  def primary_host(self):
    return self.context.multiprocessing_options.primary_host

  @parameterized.parameters(True, False)
  def test_finalize_conforming(self, finalize_partial_path: bool):
    final_path = self.directory / 'test_finalize_conforming'
    partial_path = partial_path_lib.add_partial_save_suffix(final_path)
    path_to_finalize = partial_path if finalize_partial_path else final_path

    if multihost.is_primary_host(self.primary_host):
      partial_path.mkdir(parents=True)
    test_utils.sync_global_processes(
        'test_finalize_conforming:make_partial_path'
    )
    self.assertTrue(partial_path.exists())
    self.assertFalse(final_path.exists())

    saving.finalize(path_to_finalize)

    self.assertFalse(partial_path.exists())
    self.assertTrue(final_path.exists())

  @parameterized.parameters(True, False)
  def test_finalize_final_exists(self, finalize_partial_path: bool):
    final_path = self.directory / 'test_finalize_final_exists'
    partial_path = partial_path_lib.add_partial_save_suffix(final_path)
    path_to_finalize = partial_path if finalize_partial_path else final_path

    if multihost.is_primary_host(self.primary_host):
      partial_path.mkdir(parents=True)
      final_path.mkdir(parents=True)
    test_utils.sync_global_processes('test_finalize_final_exists:1')
    self.assertTrue(partial_path.exists())
    self.assertTrue(final_path.exists())

    with self.assertRaises(FileExistsError):
      saving.finalize(path_to_finalize)

  def test_finalize_rename_os_error(self):
    final_path = self.directory / 'test_finalize_rename_os_error'
    partial_path = partial_path_lib.add_partial_save_suffix(final_path)
    path_to_finalize = final_path

    if multihost.is_primary_host(self.primary_host):
      partial_path.mkdir(parents=True)
    test_utils.sync_global_processes('test_finalize_rename_os_error:1')
    self.assertTrue(partial_path.exists())
    self.assertFalse(final_path.exists())

    async def mock_rename(src, dst):
      del src, dst
      raise OSError('Test error.')

    with mock.patch(
        'orbax.checkpoint.experimental.v1._src.partial.saving.async_path.rename',
        new=mock_rename,
    ):
      with self.assertRaises(OSError):
        saving.finalize(path_to_finalize)

    self.assertTrue(partial_path.exists())
    self.assertFalse(final_path.exists())

  def test_finalize_file_collision(self):
    final_path = self.directory / 'test_finalize_file_collision'
    partial_path = partial_path_lib.add_partial_save_suffix(final_path)

    saving.save(final_path, {'a': 1})
    saving.save(final_path, {'b': 2})

    pending_dirs = asyncio.run(snapshot.list_pending_dirs(partial_path))
    self.assertLen(pending_dirs, 2)

    if multihost.is_primary_host(self.primary_host):
      for p_dir in pending_dirs:
        (p_dir / 'colliding_file.txt').write_text('collision')

    test_utils.sync_global_processes('test_finalize_file_collision:write')

    if multihost.is_primary_host(self.primary_host):
      with self.assertRaisesRegex(
          FileExistsError,
          'File collision on colliding_file.txt during finalize. Overwriting '
          'destination file is not allowed.',
      ):
        saving.finalize(final_path)
    else:
      with self.assertRaisesRegex(
          OSError, 'Partial checkpoint finalization failed.'
      ):
        saving.finalize(final_path)


if __name__ == '__main__':
  multiprocess_test.main()
