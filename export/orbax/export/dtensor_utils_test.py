# Copyright 2023 The Orbax Authors.
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

from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax.experimental import pjit
from jax.sharding import PartitionSpec as P
import numpy as np
from orbax.export import dtensor_utils
from tensorflow.experimental import dtensor


PartitionSpec = Optional[P]


def _create_sharded_jax_array(
    global_arr: np.ndarray, pspec: PartitionSpec, mesh: jax.sharding.Mesh
) -> jax.Array:
  with mesh:
    return pjit.pjit(lambda x: x, out_axis_resources=pspec)(global_arr)


def _get_dtensor_full_value(
    dt_arr: dtensor_utils.DTensor,
    dmesh: dtensor.Mesh,
) -> np.ndarray:
  return dtensor.relayout(
      dt_arr, dtensor.Layout([dtensor.UNSHARDED for _ in dt_arr.shape], dmesh)
  ).numpy()


class DtensorUtilsTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    dtensor_utils.initialize_dtensor()

  def setUp(self):
    super().setUp()
    self.assertTrue(dtensor_utils.dtensor_initialized())
    self._mesh_shape = (4, 2)
    self._mesh = self._create_mesh(self._mesh_shape, ('x', 'y'))

  def _create_mesh(
      self, shape: tuple[int, ...], axis_names: tuple[str, ...]
  ) -> jax.sharding.Mesh:
    devices = np.asarray(jax.devices()).reshape(*shape)
    return jax.sharding.Mesh(devices, axis_names)

  def test_jax_mesh_to_dtensor_mesh(self):
    dmesh = dtensor_utils.jax_mesh_to_dtensor_mesh(self._mesh)
    self.assertSequenceEqual(dmesh.shape(), self._mesh_shape)
    self.assertEqual(dmesh.dim_size('x'), 4)
    self.assertEqual(dmesh.dim_size('y'), 2)

  @parameterized.named_parameters(
      ('_replicated', (8, 4), None),
      ('_fully_sharded', (8, 4), P('y', 'x')),
      ('_dim0_sharded', (8, 4), P('x')),
      ('_dim0_sharded_tuple', (8, 4), P(('x',))),
      ('_dim1_sharded', (8, 4), P(None, 'y')),
      ('_dim1_and_dim2_sharded', (6, 4, 8), P(None, 'x', 'y')),
  )
  def test_jax_array_to_dtensor(
      self, shape: tuple[int, ...], pspec: PartitionSpec
  ):
    global_arr = np.arange(np.prod(shape)).reshape(shape)
    jax_arr = _create_sharded_jax_array(global_arr, pspec, self._mesh)
    dmesh = dtensor_utils.jax_mesh_to_dtensor_mesh(self._mesh)
    dt_arr = dtensor_utils.jax_array_to_dtensor(jax_arr, pspec, dmesh)
    np.testing.assert_equal(_get_dtensor_full_value(dt_arr, dmesh), global_arr)

  def test_sharding_across_multi_mesh_axes_unsupported(self):
    global_arr = np.arange(16).reshape((8, 2))
    pspec = P(('x', 'y'))
    jax_arr = _create_sharded_jax_array(global_arr, pspec, self._mesh)
    dmesh = dtensor_utils.jax_mesh_to_dtensor_mesh(self._mesh)
    with self.assertRaisesRegex(
        ValueError,
        (
            'not support partitioning of an array dimension across multiple'
            ' mesh axes'
        ),
    ):
      dtensor_utils.jax_array_to_dtensor(jax_arr, pspec, dmesh)

  @parameterized.parameters(
      ((1, 8), ('x', 'y'), P(None, ('x', 'y'))),
      ((8, 1), ('x', 'y'), P(('x', 'y'), None)),
      ((1, 2, 4), ('x', 'y', 'z'), P(None, 'y', ('x', 'z'))),
      ((1, 2, 4), ('x', 'y', 'z'), P(None, 'y', ('z', 'x'))),
      ((1, 2, 4), ('x', 'y', 'z'), P(None, None, ('y', 'x'))),
      ((1, 1, 8), ('x', 'y', 'z'), P(None, None, ('x', 'y', 'z'))),
      ((1, 8, 1), ('x', 'y', 'z'), P(None, ('x', 'y', 'z'), None)),
      ((8, 1, 1), ('x', 'y', 'z'), P(('x', 'y', 'z'), None, None)),
      ((1, 1, 8), ('x', 'y', 'z'), P(None, 'y', ('x', 'z'))),
      ((1, 1, 8), ('x', 'y', 'z'), P('x', None, ('y', 'z'))),
      ((1, 1, 8), ('x', 'y', 'z'), P('x', None, ('z', 'y'))),
      ((1, 1, 8), ('x', 'y', 'z'), P(('x', 'y'), None, 'z')),
  )
  def test_sharding_across_multi_mesh_axes(
      self,
      mesh_shape: tuple[int, ...],
      axis_names: tuple[str, ...],
      pspec: P,
  ):
    arr_shape = np.asarray(mesh_shape) * 2
    global_arr = np.arange(np.prod(arr_shape)).reshape(arr_shape)
    mesh = self._create_mesh(mesh_shape, axis_names)
    jax_arr = _create_sharded_jax_array(global_arr, pspec, mesh)
    dmesh = dtensor_utils.jax_mesh_to_dtensor_mesh(mesh)
    dt_arr = dtensor_utils.jax_array_to_dtensor(jax_arr, pspec, dmesh)
    np.testing.assert_equal(_get_dtensor_full_value(dt_arr, dmesh), global_arr)

  def test_jax_dtensor_mesh_mismatch(self):
    global_arr = np.arange(16).reshape((8, 2))
    pspec = P('x', 'y')
    jax_arr = _create_sharded_jax_array(global_arr, pspec, self._mesh)
    dmesh = dtensor.create_mesh([('x', 2), ('y', 4)])
    with self.assertRaisesRegex(
        ValueError, 'must be a multiple of the size of mesh axis "y"'
    ):
      dtensor_utils.jax_array_to_dtensor(jax_arr, pspec, dmesh)

  def test_get_current_dtensor_mesh(self):
    self.assertIsNone(dtensor_utils.get_current_dtensor_mesh())
    with dtensor_utils.maybe_enable_dtensor_export_on(self._mesh):
      self.assertIsInstance(
          dtensor_utils.get_current_dtensor_mesh(), dtensor.Mesh
      )
    self.assertIsNone(dtensor_utils.get_current_dtensor_mesh())

    with dtensor_utils.maybe_enable_dtensor_export_on(None):
      self.assertIsNone(dtensor_utils.get_current_dtensor_mesh())


if __name__ == '__main__':
  absltest.main()
