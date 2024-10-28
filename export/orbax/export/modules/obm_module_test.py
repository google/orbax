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
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from orbax.export import constants
from orbax.export import serving_config as osc
from orbax.export.modules import obm_module


class ObmModuleTest(parameterized.TestCase):

  def assertHasAttr(self, obj, attr_name):
    self.assertTrue(
        hasattr(obj, attr_name),
        msg=f'Object (of type {type(obj)}) does not have attribute {attr_name}',
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
    with self.assertRaisesRegex(
        NotImplementedError, r'only supports a single method'
    ):
      obm_module.ObmModule(
          params=params,
          apply_fn={function_name: apply_fn, function_name_2: apply_fn_2},
      )


if __name__ == '__main__':
  absltest.main()
