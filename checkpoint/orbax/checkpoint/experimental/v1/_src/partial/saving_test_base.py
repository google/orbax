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

"""Base class for partial saving tests."""

# pylint: disable=missing-class-docstring,protected-access,missing-function-docstring

from __future__ import annotations

import asyncio

from absl.testing import parameterized
from etils import epath
from orbax.checkpoint import test_utils
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.partial import saving
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils


create_numpy_pytree = array_test_utils.create_numpy_pytree
create_sharded_pytree = array_test_utils.create_sharded_pytree


class PartialSavingTestBase:

  class Test(parameterized.TestCase):

    def setUp(self):
      super().setUp()

      self.directory = (
          epath.Path(self.create_tempdir(name='partial_saving_test').full_path)
          / 'ckpt'
      )
      self.pytree, self.abstract_pytree = create_sharded_pytree()
      self.numpy_pytree, self.abstract_numpy_pytree = create_numpy_pytree()

      self.context = context_lib.Context(
          multiprocessing_options=options_lib.MultiprocessingOptions(
              primary_host=0,
              barrier_sync_key_prefix='PartialSavingTest',
          )
      )

      test_utils.set_tensorstore_driver_for_test()
      test_utils.sync_global_processes('PartialSavingTest:setup_complete')

    def tearDown(self):
      super().tearDown()
      test_utils.sync_global_processes('PartialSavingTest:teardown_complete')

    @parameterized.parameters(True, False)
    def test_finalize(self, finalize_partial_path: bool):
      final_path = self.directory
      partial_path = saving._add_partial_save_suffix(final_path)

      path_to_finalize = partial_path if finalize_partial_path else final_path

      with self.subTest('conforming'):
        if multihost.is_primary_host(
            self.context.multiprocessing_options.primary_host
        ):
          partial_path.mkdir(parents=True)
        multihost.sync_global_processes(
            'PartialSavingTest:test_finalize:conforming_sync_1'
        )
        asyncio.run(
            multihost.sync_global_processes(
                multihost.unique_barrier_key(
                    'PartialSavingTest:test_finalize:conforming_sync_1',
                    prefix=self.context.multiprocessing_options.barrier_sync_key_prefix,
                ),
                processes=self.context.multiprocessing_options.active_processes,
            )
        )
        self.assertTrue(partial_path.exists())
        self.assertFalse(final_path.exists())

        saving.finalize(path_to_finalize)

        self.assertFalse(partial_path.exists())
        self.assertTrue(final_path.exists())

        if multihost.is_primary_host(
            self.context.multiprocessing_options.primary_host
        ):
          final_path.rmtree()
        asyncio.run(
            multihost.sync_global_processes(
                multihost.unique_barrier_key(
                    'PartialSavingTest:test_finalize:conforming_sync_2',
                    prefix=self.context.multiprocessing_options.barrier_sync_key_prefix,
                ),
                processes=self.context.multiprocessing_options.active_processes,
            )
        )

      with self.subTest('partial_missing'):
        self.assertFalse(partial_path.exists())
        self.assertFalse(final_path.exists())

        with self.assertRaises(FileNotFoundError):
          saving.finalize(path_to_finalize)

      with self.subTest('final_exists'):
        if multihost.is_primary_host(
            self.context.multiprocessing_options.primary_host
        ):
          partial_path.mkdir(parents=True)
          final_path.mkdir(parents=True)
        asyncio.run(
            multihost.sync_global_processes(
                multihost.unique_barrier_key(
                    'PartialSavingTest:test_finalize:final_exists_sync_1',
                    prefix=self.context.multiprocessing_options.barrier_sync_key_prefix,
                ),
                processes=self.context.multiprocessing_options.active_processes,
            )
        )
        self.assertTrue(partial_path.exists())
        self.assertTrue(final_path.exists())

        with self.assertRaises(FileExistsError):
          saving.finalize(path_to_finalize)

        if multihost.is_primary_host(
            self.context.multiprocessing_options.primary_host
        ):
          partial_path.rmtree()
          final_path.rmtree()
        asyncio.run(
            multihost.sync_global_processes(
                multihost.unique_barrier_key(
                    'PartialSavingTest:test_finalize:final_exists_sync_2',
                    prefix=self.context.multiprocessing_options.barrier_sync_key_prefix,
                ),
                processes=self.context.multiprocessing_options.active_processes,
            )
        )
