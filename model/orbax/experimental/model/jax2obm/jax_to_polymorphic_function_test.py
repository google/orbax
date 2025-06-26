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

import os

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from orbax.experimental.model import core as obm
from orbax.experimental.model.jax2obm import constants
from orbax.experimental.model.jax2obm import jax_to_polymorphic_function as jax_to_poly_fn
import tensorflow as tf


def read_checkpoint_values(
    prefix: str,
) -> dict[str, tuple[np.ndarray, obm.DType]]:
  loaded = tf.train.load_checkpoint(prefix)
  contents = {}
  for key in loaded.get_variable_to_dtype_map().keys():
    contents[key] = loaded.get_tensor(key)
  return contents


class JaxToPolymorphicFunctionTest(parameterized.TestCase):

  @parameterized.parameters(
      (f'_{native_serialization_platform}', native_serialization_platform)
      for native_serialization_platform in (
          [constants.OrbaxNativeSerializationType.CPU],
          None
      )
  )
  def test_simple_e2e(self, native_serialization_platforms):
    # TODO(b/332756633): Also test with `@jax.jit`.
    def jax_fn(inputs, params):
      return params + inputs, params - inputs

    params_shape = (5, 10)
    params_dtype = np.dtype(np.float32)
    jax_params = jnp.array(np.ones(params_shape, dtype=params_dtype))
    em_fn_ = jax_to_poly_fn.convert(
        jax.jit(jax_fn),
        native_serialization_platforms=native_serialization_platforms,
    )
    em_params = jax.tree.map(jax_to_poly_fn.convert_to_variable, jax_params)
    input_spec = (
        obm.TensorSpec(params_shape, obm.np_dtype_to_dtype(params_dtype)),
    )
    em_fn = obm.function(
        lambda inputs: em_fn_(inputs, em_params), input_signature=input_spec
    )
    obm_module = dict()
    # TODO(b/328686660): Test saving polymorphic function `forward_fn`.
    # obm_module.forward_fn = em_fn
    obm_module['concrete_forward_fn'] = em_fn.get_concrete_function(input_spec)

    save_path = os.path.join(self.create_tempdir())
    obm.save(obm_module, save_path)

    # Validate exported model using tf.saved_model.load.
    loaded = tf.saved_model.load(save_path)

    inp = np.random.rand(*params_shape).astype(params_dtype)
    self.assertEqual(loaded.signatures.keys(), {'concrete_forward_fn'})
    loaded_fn = loaded.signatures['concrete_forward_fn']
    out = loaded_fn(tf.constant(inp))
    expected = jax_fn(inp, jax_params)

    np.testing.assert_array_equal(out['output_0'], expected[0])
    np.testing.assert_array_equal(out['output_1'], expected[1])

    ckpt = read_checkpoint_values(
        os.path.join(save_path, 'variables', 'variables')
    )
    self.assertEqual(ckpt.keys(), {'concrete_forward_fn/capture_0'})
    np.testing.assert_array_equal(
        ckpt['concrete_forward_fn/capture_0'], jax_params
    )
