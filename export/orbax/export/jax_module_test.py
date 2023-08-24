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

"""Tests for jax_module."""

import collections

from absl.testing import parameterized
import jax
from jax import sharding
import jax.numpy as jnp
import numpy as np
from orbax.export import jax_module
import tensorflow as tf


def _register_custom_dict_to_jax(dict_cls):
  def _flatten_with_keys(xs):
    sorted_keys = sorted(xs.keys())
    return tuple(
        [(jax.tree_util.DictKey(k), xs[k]) for k in sorted_keys]
    ), tuple(sorted_keys)

  jax.tree_util.register_pytree_with_keys(
      dict_cls,
      flatten_with_keys=_flatten_with_keys,
      unflatten_func=lambda keys, xs: dict_cls(zip(keys, xs)),
  )
  return dict_cls


@_register_custom_dict_to_jax
class YetAnotherDict(dict):
  pass


class JaxModuleTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(dict, YetAnotherDict)
  def test_variable_names(self, top_level_dict_cls=dict):
    MyTuple = collections.namedtuple('MyTuple', 'x y')

    @_register_custom_dict_to_jax
    class MyDict(dict):
      pass

    params = top_level_dict_cls(
        a=jnp.array(1),
        b=jnp.array([3, 4]),
        c=MyTuple(jnp.array(-1), jnp.array(-2)),
        d=MyDict(
            e=jnp.array(1),
            f=[jnp.array([5, 6]), jnp.array([7, 8])],
        ),
        empty_nodes=[dict(), tuple(), list(), MyDict(), YetAnotherDict()],
    )
    variable_names_to_vals = {
        v.name: v
        for v in jax_module.JaxModule(params, lambda params, x: x).variables
    }
    self.assertEqual(
        {
            'a:0': np.array(1),
            'b:0': np.array([3, 4]),
            'c.x:0': np.array(-1),
            'c.y:0': np.array(-2),
            'd.e:0': np.array(1),
            'd.f.0:0': np.array([5, 6]),
            'd.f.1:0': np.array([7, 8]),
        },
        variable_names_to_vals,
    )

  def test_variable_names_contains_tilde(self):
    """Test that variable containing ~ are escaped.

    https://github.com/google/orbax/issues/420
    """
    params = {
        'model/~/linear': {
            'w': jnp.array(1),
            'b': jnp.array(2),
        }
    }
    variable_names_to_vals = {
        v.name: v
        for v in jax_module.JaxModule(params, lambda params, x: x).variables
    }
    self.assertEqual(
        {
            'model/_/linear.w:0': np.array(1),
            'model/_/linear.b:0': np.array(2),
        },
        variable_names_to_vals,
    )

  def test_variable_names_custom_node_not_registered(self):
    @_register_custom_dict_to_jax
    class MyDict(dict):
      pass

    params = MyDict(
        a=jnp.array(1),
        b=[jnp.array([5, 6]), jnp.array([7, 8])],
    )
    variables = jax_module.JaxModule(params, lambda params, x: x).variables
    names = {v.name for v in variables}
    self.assertLen(names, len(variables))

  def test_trainable(self):
    params = {'x': jnp.array(1), 'y': jnp.array(2)}
    trainable = {'x': True, 'y': False}
    jm = jax_module.JaxModule(params, lambda params, x: x, trainable=trainable)
    self.assertLen(jm.trainable_variables, 1)
    self.assertEqual(jm.trainable_variables[0].name, 'x:0')
    self.assertEqual(jm.trainable_variables[0], jnp.array(1))
    self.assertTrue(jm.with_gradient)

    jm = jax_module.JaxModule(params, lambda params, x: x)
    self.assertEmpty(jm.trainable_variables)
    self.assertFalse(jm.with_gradient)

    jm = jax_module.JaxModule(params, lambda params, x: x, trainable=True)
    self.assertLen(jm.trainable_variables, 2)
    self.assertTrue(jm.with_gradient)

    jm = jax_module.JaxModule(params, lambda params, x: x, trainable=False)
    self.assertEmpty(jm.trainable_variables)
    self.assertFalse(jm.with_gradient)

  def test_jax_array(self):
    global_mesh = sharding.Mesh(np.array(jax.devices('cpu')), 'x')
    mesh_axes = sharding.PartitionSpec('x')
    global_input_shape = (jax.device_count('cpu'), 2)
    global_input_data = np.arange(
        np.prod(global_input_shape)).reshape(global_input_shape)

    arr = jax.make_array_from_callback(
        global_input_shape, jax.sharding.NamedSharding(global_mesh, mesh_axes),
        lambda idx: global_input_data[idx])
    self.assertIsInstance(arr, jax.Array)
    variables = jax_module.JaxModule(
        {'arr': arr}, lambda params, x: x
    ).variables
    self.assertLen(variables, 1)
    self.assertEqual(variables[0].name, 'arr:0')
    self.assertAllEqual(variables[0], global_input_data)

  @parameterized.parameters(True, False)
  def test_call_jax_module_methods(self, jit_compile):

    def linear(params, x):
      return params['w'] @ x + params['b']

    key_w, key_b, key_x = jax.random.split(jax.random.PRNGKey(1234), 3)
    params = {
        'w': jax.random.normal(key_w, shape=(8, 8)),
        'b': jax.random.normal(key_b, shape=(8, 1)),
    }
    x = jax.random.normal(key_x, shape=(8, 1))

    jm = jax_module.JaxModule(params, linear, jit_compile=jit_compile)
    self.assertAllClose(
        jm.methods[jax_module.DEFAULT_METHOD_KEY](x),
        jm.jax_methods[jax_module.DEFAULT_METHOD_KEY](x),
    )

  @parameterized.parameters(True, False)
  def test_polymorphic_shapes(self, jit_compile):

    def linear(params, batch):
      return params['w'] @ batch + params['b']

    key_w, key_b, key_x = jax.random.split(jax.random.PRNGKey(1234), 3)
    params = {
        'w': jax.random.normal(key_w, shape=(8, 8)),
        'b': jax.random.normal(key_b, shape=(8, 1)),
    }

    with self.assertRaisesRegex(ValueError,
                                'Do not use `polymorphic_shapes`'):
      jax_module.JaxModule(
          params, linear, jax2tf_kwargs={'polymorphic_shapes': [None, 'b, ...']}
      )

    jm = jax_module.JaxModule(
        params,
        linear,
        jit_compile=jit_compile,
        input_polymorphic_shape='b, ...',
    )

    @tf.function(
        autograph=False,
        jit_compile=False,
        input_signature=[tf.TensorSpec([None, 8, 1], tf.float32)])
    def traced(x):
      return jm.methods[jax_module.DEFAULT_METHOD_KEY](x)

    key_x1, key_x2 = jax.random.split(key_x, 2)
    x1 = jax.random.normal(key_x1, shape=(8, 8, 1))  # batch size is 8
    self.assertAllClose(traced(x1), linear(params, x1))

    x2 = jax.random.normal(key_x2, shape=(16, 8, 1))  # batch size is 16
    self.assertAllClose(traced(x2), linear(params, x2))

  def test_multi_functions(self):
    jm = jax_module.JaxModule(
        params={'delta': jnp.ones((), jnp.int32)},
        apply_fn={
            'add': lambda params, x: x + params['delta'],
            'sub': lambda params, x: x - params['delta'],
        },
        input_polymorphic_shape={
            'add': None,
            'sub': 'b, ...',  # Make `sub` batch polymorphic.
        },
    )

    # `add` cannot accept polymorphic shapes.
    with self.assertRaisesRegex(ValueError, 'polymorphic shape'):
      jm.methods['add'].get_concrete_function(tf.TensorSpec([None], tf.int32))

    # `add` can accept fixed shapes.
    jm.methods['add'].get_concrete_function(tf.TensorSpec([1], tf.int32))
    # `sub` can accept polymorphic shapes.
    jm.methods['sub'].get_concrete_function(tf.TensorSpec([None], tf.int32))

  def test_init_invalid_argument(self):
    params = {'delta': jnp.ones((), jnp.int32)},
    apply_fns = {
        'add': lambda params, x: x + params['delta'],
        'sub': lambda params, x: x - params['delta'],
    }

    with self.assertRaisesRegex(ValueError, '`input_polymorphic_shape` must'):
      jax_module.JaxModule(params, apply_fns)

    with self.assertRaisesRegex(ValueError, '`input_polymorphic_shape` must'):
      jax_module.JaxModule(
          params,
          apply_fns,
          input_polymorphic_shape={
              'add': None,
          },
      )

    with self.assertRaisesRegex(ValueError, '`jax2tf_kwargs` must'):
      jax_module.JaxModule(
          params,
          apply_fns,
          input_polymorphic_shape=jax.tree_util.tree_map(
              lambda x: None, apply_fns
          ),
          jax2tf_kwargs={'enable_xla': False},
      )

    with self.assertRaisesRegex(ValueError, '`jit_compile` must'):
      jax_module.JaxModule(
          params,
          apply_fns,
          input_polymorphic_shape=jax.tree_util.tree_map(
              lambda x: None, apply_fns
          ),
          jit_compile={'add': False},
      )

    with self.assertRaisesRegex(ValueError, 'contains trainable'):
      jax_module.JaxModule(
          params,
          lambda p, x: x,
          trainable=True,
          jax2tf_kwargs={'with_gradient': False},
      )

    with self.assertRaisesRegex(ValueError, 'does not contain trainable'):
      jax_module.JaxModule(
          params,
          lambda p, x: x,
          trainable=False,
          jax2tf_kwargs={'with_gradient': True},
      )

  def test_jax2tf_native_serialization_platforms(self):
    params = {}
    my_module = jax_module.JaxModule(params, lambda params, x: x)
    self.assertEqual(my_module.native_serialization_platforms, ['cpu'])

    my_module = jax_module.JaxModule(
        params,
        lambda params, x: x,
        jax2tf_kwargs=dict(native_serialization_platforms=['cuda']),
    )
    self.assertEqual(my_module.native_serialization_platforms, ['cuda'])

    with self.assertRaisesRegex(
        NotImplementedError,
        'native_serialization_platforms is not yet implemented for multiple'
        ' platforms',
    ):
      _ = jax_module.JaxModule(
          params,
          lambda params, x: x,
          jax2tf_kwargs=dict(
              native_serialization_platforms=['cuda', 'tpu', 'cpu']
          ),
      )


if __name__ == '__main__':
  tf.test.main()
