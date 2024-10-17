# Copyright 2024 The Orbax Authors.
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

from absl.testing import absltest
import jax
import jax.numpy as jnp
from orbax.export import constants
from orbax.export import serving_config as osc
from orbax.export.modules import obm_module


class ObmModuleTest(absltest.TestCase):

  def test_obm_module_missing_serving_config_input_signature(self):
    def apply_fn(params, inputs):
      return params + inputs

    params_shape = (2, 5)
    params_dtype = jnp.dtype(jnp.float32)
    params = jnp.array(jnp.ones(params_shape, dtype=params_dtype))
    function_name = 'simple_add'
    with self.assertRaises(ValueError):
      obm_module.ObmModule(
          params=params,
          apply_fn={function_name: apply_fn},
          export_version=constants.ExportModelType.TF_SAVEDMODEL,
          serving_configs=[osc.ServingConfig(signature_key=function_name)],
      )

  def test_obm_module_missing_serving_config_signature_key(self):
    def apply_fn(params, inputs):
      return params + inputs

    params_shape = (2, 5)
    params_dtype = jnp.dtype(jnp.float32)
    params = jnp.array(jnp.ones(params_shape, dtype=params_dtype))
    function_name = 'simple_add'
    with self.assertRaises(ValueError):
      obm_module.ObmModule(
          params=params,
          apply_fn={function_name: apply_fn},
          export_version=constants.ExportModelType.TF_SAVEDMODEL,
          serving_configs=[
              osc.ServingConfig(
                  signature_key=function_name,
              )
          ],
      )

  def test_obm_module_multiple_apply_fns(self):
    def apply_fn(params, inputs):
      return params + inputs

    def apply_fn_2(params, inputs):
      return params - inputs

    params_shape = (2, 5)
    params_dtype = jnp.dtype(jnp.float32)
    params = jnp.array(jnp.ones(params_shape, dtype=params_dtype))
    function_name = 'simple_add'
    function_name_2 = 'simple_subtract'
    with self.assertRaises(NotImplementedError):
      obm_module.ObmModule(
          params=params,
          apply_fn={function_name: apply_fn, function_name_2: apply_fn_2},
          export_version=constants.ExportModelType.TF_SAVEDMODEL,
          serving_configs=[
              osc.ServingConfig(
                  signature_key=function_name,
                  input_signature=jax.ShapeDtypeStruct((2, 5), jnp.float32),
              ),
          ],
      )

  def test_obm_module_unsupported_native_serialization_platform(self):
    def apply_fn(params, inputs):
      return params + inputs

    params_shape = (2, 5)
    params_dtype = jnp.dtype(jnp.float32)
    params = jnp.array(jnp.ones(params_shape, dtype=params_dtype))
    function_name = 'simple_add'

    with self.assertRaises(ValueError):
      obm_module.ObmModule(
          params=params,
          apply_fn={function_name: apply_fn},
          export_version=constants.ExportModelType.TF_SAVEDMODEL,
          jax2obm_kwargs={
              constants.NATIVE_SERIALIZATION_PLATFORM: (
                  'bad_serialization_platform'
              )
          },
          serving_configs=[
              osc.ServingConfig(
                  signature_key=function_name,
                  input_signature=jax.ShapeDtypeStruct((2, 5), jnp.float32),
              )
          ],
      )

  def test_obm_module_missing_serving_configs(self):
    def apply_fn(params, inputs):
      return params + inputs

    params_shape = (2, 5)
    params_dtype = jnp.dtype(jnp.float32)
    params = jnp.array(jnp.ones(params_shape, dtype=params_dtype))
    function_name = 'simple_add'
    with self.assertRaises(ValueError):
      obm_module.ObmModule(
          params=params,
          apply_fn={function_name: apply_fn},
          export_version=constants.ExportModelType.TF_SAVEDMODEL,
          serving_configs={},
      )


if __name__ == '__main__':
  absltest.main()
