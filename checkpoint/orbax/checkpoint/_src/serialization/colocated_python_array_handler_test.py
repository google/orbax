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

import unittest

from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.serialization import cloud_pathways_type_handlers
from orbax.checkpoint._src.serialization import type_handlers

from .learning.brain.research.jax.tests.multiprocess import multiprocess_test


class ColocatedPythonArrayHandlerTest(
    unittest.IsolatedAsyncioTestCase, parameterized.TestCase
):
  """Test class."""

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='colocated_test').full_path
    )
    # TODO(b/364139319): Support more devices and processes.
    assert jax.device_count() == 8
    assert jax.process_count() == 1

    test_utils.set_tensorstore_driver_for_test()

    test_utils.sync_global_processes(
        'ColocatedPythonArrayHandlerTest:setup_complete'
    )

  def tearDown(self):
    test_utils.sync_global_processes(
        'ColocatedPythonArrayHandlerTest:tests_complete'
    )
    super().tearDown()

  @parameterized.product(
      use_ocdbt=(True, False),
      use_zarr3=(True, False),
  )
  async def test_serialize_deserialize(self, use_ocdbt, use_zarr3):
    handler = cloud_pathways_type_handlers.ColocatedPythonArrayHandler()
    sharding = jax.sharding.NamedSharding(
        mesh=jax.sharding.Mesh(
            devices=np.asarray(jax.devices()).reshape(2, 4),
            axis_names=('x', 'y'),
        ),
        spec=jax.sharding.PartitionSpec('y'),
    )
    arr = jax.device_put(np.arange(32), sharding)
    info = test_utils.get_param_info('a', self.directory, is_ocdbt=use_ocdbt)
    info.use_zarr3 = use_zarr3

    futures = await handler.serialize([arr], [info])
    for f in futures:
      f.result()
    test_utils.sync_global_processes(
        'ColocatedPythonArrayHandlerTest:serialized'
    )

    restored = await handler.deserialize(
        [info], [type_handlers.ArrayRestoreArgs(sharding=sharding)]
    )
    test_utils.assert_array_equal(self, arr, restored[0])

  async def test_serialize_no_save_args(self):
    handler = cloud_pathways_type_handlers.ColocatedPythonArrayHandler()
    sharding = jax.sharding.NamedSharding(
        mesh=jax.sharding.Mesh(
            devices=np.asarray(jax.devices()).reshape(2, 4),
            axis_names=('x', 'y'),
        ),
        spec=jax.sharding.PartitionSpec('y'),
    )
    arr = jax.device_put(np.arange(32), sharding)
    info = test_utils.get_param_info('a', self.directory)

    with self.assertRaises(ValueError):
      await handler.serialize([arr], [info], args=None)

  async def test_serialize_deserialize_random_key(self):
    handler = cloud_pathways_type_handlers.ColocatedPythonArrayHandler()
    sharding = jax.sharding.NamedSharding(
        mesh=jax.sharding.Mesh(
            devices=np.asarray(jax.devices()).reshape(2, 4),
            axis_names=('x', 'y'),
        ),
        spec=jax.sharding.PartitionSpec('y'),
    )
    key = jax.random.key(0)
    info = test_utils.get_param_info('a', self.directory)

    futures = await handler.serialize([key], [info])
    for f in futures:
      f.result()
    test_utils.sync_global_processes(
        'ColocatedPythonArrayHandlerTest:serialized'
    )

    restored = await handler.deserialize(
        [info], [type_handlers.ArrayRestoreArgs(sharding=sharding)]
    )
    test_utils.assert_array_equal(self, key, restored[0])


if __name__ == '__main__':
  multiprocess_test.main()
