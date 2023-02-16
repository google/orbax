# Copyright 2022 The Orbax Authors.
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
from flax import serialization
from flax.core import FrozenDict
import jax
from jax.experimental.global_device_array import GlobalDeviceArray as GDA
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec
import numpy as np
from orbax.export.jax_module import JaxModule
import tensorflow as tf


def _unzip2(xys):
  return [x for x, y in xys], [y for x, y in xys]


def _register_custom_dict_to_jax(dict_cls):
  jax.tree_util.register_pytree_node(
      dict_cls,
      lambda xs: _unzip2(sorted(xs.items()))[::-1],
      lambda keys, xs: dict_cls(zip(keys, xs)),
  )
  return dict_cls


class JaxModuleTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(dict, FrozenDict)
  def test_variable_names(self, top_level_dict_cls=dict):
    MyTuple = collections.namedtuple('MyTuple', 'x y')

    @_register_custom_dict_to_jax
    class MyDict(dict):
      pass

    serialization.register_serialization_state(
        MyDict,
        lambda xs: serialization.to_state_dict(dict(xs)),
        lambda xs, sd: MyDict(serialization.from_state_dict(dict(xs), sd)),
    )

    params = top_level_dict_cls(
        a=jnp.array(1),
        b=jnp.array([3, 4]),
        c=MyTuple(jnp.array(-1), jnp.array(-2)),
        d=MyDict(
            e=jnp.array(1),
            f=[jnp.array([5, 6]), jnp.array([7, 8])],
        ),
        empty_nodes=[dict(), tuple(), list(), MyDict(), FrozenDict()],
    )
    variable_names_to_vals = {
        v.name: v for v in JaxModule(params, lambda params, x: x).variables
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

  def test_variable_names_custom_node_not_registered(self):
    @_register_custom_dict_to_jax
    class MyDict(dict):
      pass

    params = MyDict(
        a=jnp.array(1),
        b=[jnp.array([5, 6]), jnp.array([7, 8])],
    )
    variables = JaxModule(params, lambda params, x: x).variables
    names = {v.name for v in variables}
    self.assertLen(names, len(variables))

  def test_trainable(self):
    params = {'x': jnp.array(1), 'y': jnp.array(2)}
    trainable = {'x': True, 'y': False}
    jm = JaxModule(params, lambda params, x: x, trainable=trainable)
    self.assertLen(jm.trainable_variables, 1)
    self.assertEqual(jm.trainable_variables[0].name, 'x:0')
    self.assertEqual(jm.trainable_variables[0], jnp.array(1))
    self.assertTrue(jm.with_gradient)

    jm = JaxModule(params, lambda params, x: x)
    self.assertEmpty(jm.trainable_variables)
    self.assertFalse(jm.with_gradient)

    jm = JaxModule(params, lambda params, x: x, trainable=True)
    self.assertLen(jm.trainable_variables, 2)
    self.assertTrue(jm.with_gradient)

    jm = JaxModule(params, lambda params, x: x, trainable=False)
    self.assertEmpty(jm.trainable_variables)
    self.assertFalse(jm.with_gradient)

  def test_gda(self):
    global_mesh = Mesh(np.array(jax.devices('cpu')), 'x')
    mesh_axes = PartitionSpec('x')
    global_input_shape = (jax.device_count('cpu'), 2)
    global_input_data = np.arange(
        np.prod(global_input_shape)).reshape(global_input_shape)

    gda = GDA.from_callback(global_input_shape, global_mesh, mesh_axes,
                            lambda idx: global_input_data[idx])
    variables = JaxModule({'gda': gda}, lambda params, x: x).variables
    self.assertLen(variables, 1)
    self.assertEqual(variables[0].name, 'gda:0')
    self.assertAllEqual(variables[0], global_input_data)

  def test_jax_array(self):
    global_mesh = Mesh(np.array(jax.devices('cpu')), 'x')
    mesh_axes = PartitionSpec('x')
    global_input_shape = (jax.device_count('cpu'), 2)
    global_input_data = np.arange(
        np.prod(global_input_shape)).reshape(global_input_shape)

    arr = jax.make_array_from_callback(
        global_input_shape, jax.sharding.NamedSharding(global_mesh, mesh_axes),
        lambda idx: global_input_data[idx])
    self.assertIsInstance(arr, jax.Array)
    self.assertNotIsInstance(arr, GDA)
    variables = JaxModule({'arr': arr}, lambda params, x: x).variables
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

    jax_module = JaxModule(params, linear, jit_compile=jit_compile)
    self.assertAllClose(jax_module.methods[JaxModule.DEFAULT_METHOD_KEY](x),
                        jax_module.jax_methods[JaxModule.DEFAULT_METHOD_KEY](x))

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
      JaxModule(
          params,
          linear,
          jax2tf_kwargs={'polymorphic_shapes': [None, 'b, ...']})

    jax_module = JaxModule(
        params,
        linear,
        jit_compile=jit_compile,
        input_polymorphic_shape='b, ...')

    @tf.function(
        autograph=False,
        jit_compile=False,
        input_signature=[tf.TensorSpec([None, 8, 1], tf.float32)])
    def traced(x):
      return jax_module.methods[JaxModule.DEFAULT_METHOD_KEY](x)

    key_x1, key_x2 = jax.random.split(key_x, 2)
    x1 = jax.random.normal(key_x1, shape=(8, 8, 1))  # batch size is 8
    self.assertAllClose(traced(x1), linear(params, x1))

    x2 = jax.random.normal(key_x2, shape=(16, 8, 1))  # batch size is 16
    self.assertAllClose(traced(x2), linear(params, x2))

  def test_multi_functions(self):
    jax_module = JaxModule(
        params={'delta': jnp.ones((), jnp.int32)},
        apply_fn={
            'add': lambda params, x: x + params['delta'],
            'sub': lambda params, x: x - params['delta'],
        },
        input_polymorphic_shape={
            'add': None,
            'sub': 'b, ...',  # Make `sub` batch polymorphic.
        })

    # `add` cannot accept polymorphic shapes.
    with self.assertRaisesRegex(ValueError, 'polymorphic shape'):
      jax_module.methods['add'].get_concrete_function(
          tf.TensorSpec([None], tf.int32))

    # `add` can accept fixed shapes.
    jax_module.methods['add'].get_concrete_function(
        tf.TensorSpec([1], tf.int32))
    # `sub` can accept polymorphic shapes.
    jax_module.methods['sub'].get_concrete_function(
        tf.TensorSpec([None], tf.int32))

  def test_init_invalid_argument(self):
    params = {'delta': jnp.ones((), jnp.int32)},
    apply_fns = {
        'add': lambda params, x: x + params['delta'],
        'sub': lambda params, x: x - params['delta'],
    }

    with self.assertRaisesRegex(ValueError, '`input_polymorphic_shape` must'):
      JaxModule(params, apply_fns)

    with self.assertRaisesRegex(ValueError, '`input_polymorphic_shape` must'):
      JaxModule(
          params, apply_fns, input_polymorphic_shape={
              'add': None,
          })

    with self.assertRaisesRegex(ValueError, '`jax2tf_kwargs` must'):
      JaxModule(
          params,
          apply_fns,
          input_polymorphic_shape=jax.tree_util.tree_map(
              lambda x: None, apply_fns),
          jax2tf_kwargs={'enable_xla': False})

    with self.assertRaisesRegex(ValueError, '`jit_compile` must'):
      JaxModule(
          params,
          apply_fns,
          input_polymorphic_shape=jax.tree_util.tree_map(
              lambda x: None, apply_fns),
          jit_compile={'add': False})

    with self.assertRaisesRegex(ValueError, 'contains trainable'):
      JaxModule(
          params,
          lambda p, x: x,
          trainable=True,
          jax2tf_kwargs={'with_gradient': False},
      )

    with self.assertRaisesRegex(ValueError, 'does not contain trainable'):
      JaxModule(
          params,
          lambda p, x: x,
          trainable=False,
          jax2tf_kwargs={'with_gradient': True},
      )


if __name__ == '__main__':
  tf.test.main()
