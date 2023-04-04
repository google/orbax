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

"""Test JaxModule/ExportManager in DTensor context."""

from absl.testing import absltest

import jax
from jax.experimental import pjit
from jax.sharding import PartitionSpec as P
import numpy as np
from orbax.export import dtensor_utils
from orbax.export.export_manager import ExportManager
from orbax.export.jax_module import JaxModule
from orbax.export.serving_config import ServingConfig
import tensorflow as tf


def _create_sharded_jax_array(
    global_arr: np.ndarray, pspec: P, mesh: jax.sharding.Mesh
) -> jax.Array:
  with mesh:
    return pjit.pjit(lambda x: x, out_axis_resources=pspec)(global_arr)


class DtensorExportTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    dtensor_utils.initialize_dtensor()

  def setUp(self):
    super().setUp()
    self.assertTrue(dtensor_utils.dtensor_initialized())
    self._mesh_shape = (4, 2)
    devices = np.asarray(jax.devices()).reshape(*self._mesh_shape)
    self._mesh = jax.sharding.Mesh(devices, ('x', 'y'))
    self._export_dir = self.create_tempdir().full_path

  def test_jax_module_export(self):
    if jax.config.jax2tf_default_native_serialization:
      self.skipTest(
          'TODO(b/274311054): Could not legalize op: tf.XlaCallModule')
    w = np.random.rand(16, 8).astype(np.float32)
    x = np.random.rand(2, 16).astype(np.float32)
    with self._mesh:
      pspec = P('x', 'y')
      params = {'w': _create_sharded_jax_array(w, pspec, self._mesh)}
      pspecs = {'w': pspec}

      pjit_model_fn = pjit.pjit(
          lambda p, x: x @ p['w'],
          in_shardings=(pspecs, None),
          out_shardings=None,
      )

      with dtensor_utils.maybe_enable_dtensor_export_on(self._mesh):
        jm = JaxModule(params, pjit_model_fn, pspecs=pspecs)
        em = ExportManager(
            jm,
            [
                ServingConfig(
                    'serving_default',
                    [tf.TensorSpec((2, 16), tf.float32, name='x')],
                    tf_postprocessor=lambda y: {'y': y},
                )
            ],
        )
        em.save(self._export_dir)


  def test_dtensor_export_error(self):
    pspec = P('x', 'y')
    w = np.random.rand(16, 8).astype(np.float32)
    sharded_w = _create_sharded_jax_array(
        np.random.rand(16, 8).astype(np.float32), pspec, self._mesh
    )
    with self.assertRaisesRegex(ValueError, '`pspecs` is not specified'):
      with dtensor_utils.maybe_enable_dtensor_export_on(self._mesh):
        JaxModule({'w': sharded_w}, lambda x: x)

    with self.assertRaisesRegex(
        ValueError, 'JaxModule is not created within a DTensor export context'
    ):
      JaxModule({'w': w}, lambda x: x, pspecs={'w': pspec})


if __name__ == '__main__':
  absltest.main()
