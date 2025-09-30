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

"""Tests for reconstructing JAX function from OBM serialization protos."""

import os
from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
import numpy as np
import orbax.checkpoint as ocp
from orbax.experimental.model import core as obm
from orbax.experimental.model.jax2obm import jax_supplemental_pb2
from orbax.experimental.model.jax2obm import main_lib
from orbax.experimental.model.jax2obm import obm_to_jax
from orbax.experimental.model.test_utils import simple_orchestration
from orbax.experimental.model.test_utils import simple_orchestration_pb2

F0 = jax_supplemental_pb2.DTypeRefinement.f0
DType = obm.ShloDType


def _make_shape_refm(
    refinements: Sequence[str] | None,
) -> jax_supplemental_pb2.ShapeRefinement | None:
  if refinements is None:
    return None
  return jax_supplemental_pb2.ShapeRefinement(
      dimension_sizes=(
          jax_supplemental_pb2.DimensionSizeRefinement(size=r)
          for r in refinements
      )
  )


class ObmToJaxTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (f'_{idx}', shlo_dim_size, refinement, expected_result)
      for idx, (shlo_dim_size, refinement, expected_result) in enumerate((
          (None, None, None),
          (None, '3', 3),
          (None, 'a', 'a'),
          (None, 'None', 'None'),
          (3, None, 3),
          (3, '3', 3),
      ))
  )
  def test_refine_dim_size_ok(self, shlo_dim_size, refinement, expected_result):
    self.assertEqual(
        obm_to_jax._refine_dim_size(shlo_dim_size, refinement),
        expected_result,
    )

  @parameterized.named_parameters(
      (f'_{shlo_dim_size}_{refinement}', shlo_dim_size, refinement)
      for shlo_dim_size, refinement in (
          (3, 'a'),
          (3, 'None'),
          (3, '4'),
      )
  )
  def test_refine_dim_size_error(self, shlo_dim_size, refinement):
    with self.assertRaisesRegex(
        ValueError, r'is not a valid refinement of the ShloDimSize'
    ):
      obm_to_jax._refine_dim_size(shlo_dim_size, refinement)

  @parameterized.named_parameters(
      (f'_{idx}', shlo_shape, refinement, err_msg)
      for idx, (shlo_shape, refinement, err_msg) in enumerate((
          (None, [], r'Can not convert an unknown-rank shape to JAX'),
          ([], [None], None),
          ([2, 3], [], None),
          ([2, 3], [None], None),
          ([2, 3], [None] * 3, None),
      ))
  )
  def test_refine_shape_error(self, shlo_shape, refinement, err_msg):
    refinement = jax_supplemental_pb2.ShapeRefinement(
        dimension_sizes=[
            jax_supplemental_pb2.DimensionSizeRefinement(size=size)
            for size in refinement
        ]
    )
    if err_msg is None:
      err_msg = r'rank.* is not equal to.* rank'
    with self.assertRaisesRegex(ValueError, err_msg):
      obm_to_jax._refine_shape(shlo_shape, refinement)

  @parameterized.named_parameters(
      (
          f'_{shlo_dtype}_{refinement}_{expected_result}',
          shlo_dtype,
          refinement,
          expected_result,
      )
      for shlo_dtype, refinement, expected_result in (
          (DType.f32, None, np.float32),
          (DType.bool, None, np.bool_),
          (DType.bool, F0, jax.dtypes.float0),
      )
  )
  def test_refine_dtype_ok(self, shlo_dtype, refinement, expected_result):
    self.assertEqual(
        obm_to_jax._refine_dtype(shlo_dtype, refinement),
        expected_result,
    )

  def test_refine_dtype_error(self):
    with self.assertRaisesRegex(
        ValueError, r'is not a valid refinement of the ShloDType'
    ):
      obm_to_jax._refine_dtype(DType.f32, F0)

  @parameterized.named_parameters(
      (  # pylint: disable=g-complex-comprehension
          f'_{idx}_{use_map}',
          shlo_tensor_specs,
          refinements,
          expected_result,
          use_map,
      )
      for use_map in (False, True)
      for idx, (shlo_tensor_specs, refinements, expected_result) in enumerate((
          ([], [], []),
          (
              [((None, 3), DType.f32), ((), DType.f32), ((), DType.bool)],
              [(('b', '3'), None), None, ((), F0)],
              [
                  (('b', 3), np.float32),
                  ((), np.float32),
                  ((), jax.dtypes.float0),
              ],
          ),
      ))
  )
  def test_refine_tensor_specs_ok(
      self, shlo_tensor_specs, refinements, expected_result, use_map
  ):
    if use_map:
      refinements_proto = jax_supplemental_pb2.ShapeDTypeRefinements(
          map=jax_supplemental_pb2.ShapeDTypeRefinementMap(
              idx_to_refinement={
                  idx: jax_supplemental_pb2.ShapeDTypeRefinement(
                      shape=_make_shape_refm(pair[0]), dtype=pair[1]
                  )
                  for idx, pair in enumerate(refinements)
                  if pair is not None
              }
          )
      )
    else:
      refinements_proto = jax_supplemental_pb2.ShapeDTypeRefinements(
          list=jax_supplemental_pb2.ShapeDTypeRefinementList(
              refinements=(
                  jax_supplemental_pb2.ShapeDTypeRefinement(
                      shape=_make_shape_refm(shape), dtype=dtype
                  )
                  for shape, dtype in (
                      (None, None) if r is None else r for r in refinements
                  )
              )
          )
      )
    self.assertEqual(
        obm_to_jax._refine_tensor_specs(
            [
                obm.ShloTensorSpec(shape=shape, dtype=dtype)
                for shape, dtype in shlo_tensor_specs
            ],
            refinements_proto,
        ),
        expected_result,
    )

  @parameterized.named_parameters(
      (f'_{shlo_tensor_specs}_{refinements}', shlo_tensor_specs, refinements)
      for shlo_tensor_specs, refinements in (
          ([((), DType.f32)] * 2, [(None, None)]),
          ([((), DType.f32)] * 2, [(None, None)] * 3),
      )
  )
  def test_refine_tensor_specs_error(self, shlo_tensor_specs, refinements):
    with self.assertRaisesRegex(
        ValueError,
        r'The number of refinements .*is not equal to the number of tensor'
        r' specs',
    ):
      obm_to_jax._refine_tensor_specs(
          shlo_tensor_specs,
          jax_supplemental_pb2.ShapeDTypeRefinements(
              list=jax_supplemental_pb2.ShapeDTypeRefinementList(
                  refinements=(
                      jax_supplemental_pb2.ShapeDTypeRefinement(
                          shape=shape, dtype=dtype
                      )
                      for shape, dtype in refinements
                  )
              )
          ),
      )

  @parameterized.named_parameters(
      (f'_{idx}', shape)
      for idx, shape in enumerate((
          (None,),
          (2, None, 3),
          ('a', None),
          (None, 'b'),
          ('a', None, 3),
      ))
  )
  def test_to_jax_shape_error(self, shape):
    with self.assertRaisesRegex(
        ValueError, r'jax.core.ShapedArray does not allow `None` dimensions'
    ):
      obm_to_jax._to_jax_shape(shape, jax.export.SymbolicScope())

  @parameterized.named_parameters(
      (f'_{idx}', shape_dtype)
      for idx, shape_dtype in enumerate((
          ((), np.float32),
          ((2,), np.float32),
          ((2, 3), np.float32),
          (('a',), np.float32),
          (('a', 'b'), np.float32),
          (('a', 'a'), np.float32),
          ((2, 'a', 3, 'b', 'a', 4), np.float32),
      ))
  )
  def test_to_jax_shaped_array_ok(self, shape_dtype):
    shape, dtype = shape_dtype
    result = obm_to_jax._to_jax_shaped_array(
        shape_dtype, jax.export.SymbolicScope()
    )
    self.assertIsInstance(result, jax.core.ShapedArray)
    self.assertEqual(result.dtype, dtype)
    self.assertLen(result.shape, len(shape))
    for i, (dim, expected_dim) in enumerate(zip(result.shape, shape)):
      self.assertEqual(
          str(dim), str(expected_dim), f'The {i}-th dimension size is wrong.'
      )

  def test_obm_functions_to_jax_function_simple_jax_e2e(self):
    def jax_model_fn(params, x):
      return 2 * params * x

    jax_params = jnp.array(2.5)
    params_args_spec = main_lib.get_shape_dtype_struct(jax_params)
    input_args_spec = params_args_spec

    # Export the jax function to OBM.
    obm_shlo_fn = main_lib.convert(
        jax.jit(jax_model_fn),
        (params_args_spec, input_args_spec),
        {},
    )

    obm_module = dict()
    model_function_name = 'my_model_fn'
    obm_module[model_function_name] = obm_shlo_fn

    weights_name = 'my_weights'
    checkpoint_path = 'my_checkpoint/'
    obm_module[weights_name] = main_lib.convert_path_to_value(
        checkpoint_path,
        mime_type='orbax_checkpoint',
    )

    save_dir_path = os.path.join(self.create_tempdir())
    supplemental_filename = 'my_orchestration.pb'

    obm.save(
        obm_module,
        save_dir_path,
        obm.SaveOptions(
            version=2,
            supplemental_info={
                simple_orchestration.TEST_ORCHESTRATION_SUPPLEMENTAL_NAME: (
                    obm.GlobalSupplemental(
                        simple_orchestration.create(
                            model_function_name=model_function_name,
                            weights_name=weights_name,
                        ),
                        supplemental_filename,
                    )
                )
            },
        ),
    )

    # All of those information will be provided by the manifest at load time.
    del model_function_name
    del weights_name
    del checkpoint_path

    # Loading now.
    manifest_proto = obm.load(save_dir_path)

    # Loads the orchestration.
    orch_filename = manifest_proto.supplemental_info[
        simple_orchestration.TEST_ORCHESTRATION_SUPPLEMENTAL_NAME
    ].file_system_location.string_path
    pipeline_proto = simple_orchestration_pb2.Pipeline()
    with open(os.path.join(save_dir_path, orch_filename), 'rb') as f:
      pipeline_proto.ParseFromString(f.read())

    # Loads the model function
    loaded_model_function_name = pipeline_proto.model_function_name
    loaded_obm_function = manifest_proto.objects[
        loaded_model_function_name
    ].function

    # and its supplemental.
    jax_supplemental_filename = (
        loaded_obm_function.body.stable_hlo_body.supplemental_info[
            obm.JAX_SPECIFIC_INFO
        ].file_system_location.string_path
    )
    jax_supplemental_proto = jax_supplemental_pb2.Function()
    with open(
        os.path.join(save_dir_path, jax_supplemental_filename), 'rb'
    ) as f:
      jax_supplemental_proto.ParseFromString(f.read())

    # Deserialize the jax function and compare results from the 2 jax functions.
    deserialized_jax_exported = obm_to_jax.obm_functions_to_jax_function(
        loaded_obm_function,
        jax_supplemental_proto,
    )

    test_input = jnp.array(7.0)

    result_from_deserialized_jax_call = deserialized_jax_exported.call(
        jax_params, test_input
    )
    result_from_original_jax_call = jax_model_fn(jax_params, test_input)

    self.assertEqual(
        result_from_deserialized_jax_call, result_from_original_jax_call
    )
    self.assertEqual(result_from_deserialized_jax_call, 35.0)

  def test_obm_functions_to_jax_function_mnist(self):
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

    class JaxMnist(nn.Module):
      """Flax MNIST model."""

      @nn.compact
      def __call__(self, x):
        """See base class."""
        x = nn.Conv(features=32, kernel_size=(4, 4))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(4, 4))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x

    model = JaxMnist()

    input_args_spec = jax.ShapeDtypeStruct((4, 28, 28, 1), jnp.float64)
    params = model.init(
        jax.random.PRNGKey(666),
        jnp.ones(shape=input_args_spec.shape, dtype=input_args_spec.dtype),
    )

    def get_mesh():
      devices = mesh_utils.create_device_mesh((2, 2, 2))
      return jax.sharding.Mesh(devices, ('b', 'x', 'y'))

    mesh = get_mesh()

    params_sharding_spec = jax.tree_util.tree_map(
        lambda _: NamedSharding(mesh, jax.sharding.PartitionSpec('y')), params
    )
    input_sharding_spec = NamedSharding(
        mesh, PartitionSpec('b', 'x', None, None)
    )

    model_apply_fn = jax.jit(
        model.apply,
        in_shardings=(
            params_sharding_spec,
            input_sharding_spec,
        ),
        out_shardings=NamedSharding(mesh, PartitionSpec('b', 'y')),
    )

    params_args_spec = main_lib.get_shape_dtype_struct(params)

    # Export the jax function to OBM.
    obm_shlo_fn = main_lib.convert(
        model_apply_fn,
        (params_args_spec, input_args_spec),
        {},
    )

    obm_module = dict()

    model_function_name = 'mnist_forward_fn'
    obm_module[model_function_name] = obm_shlo_fn
    save_dir_path = os.path.join(self.create_tempdir())

    # Save the params to orbax checkpoint, which will be loaded later.
    checkpoint_path = 'my_checkpoint/'
    checkpoint_abs_path = os.path.join(save_dir_path, checkpoint_path)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(checkpoint_abs_path, params)
    weights_name = 'my_weights'
    obm_module[weights_name] = main_lib.convert_path_to_value(
        checkpoint_path,
        mime_type='orbax_checkpoint',
    )

    obm.save(
        obm_module,
        save_dir_path,
        obm.SaveOptions(
            version=2,
            supplemental_info={
                simple_orchestration.TEST_ORCHESTRATION_SUPPLEMENTAL_NAME: (
                    obm.GlobalSupplemental(
                        simple_orchestration.create(
                            model_function_name=model_function_name,
                            weights_name=weights_name,
                        ),
                        'my_orchestration.pb',
                    )
                )
            },
        ),
    )

    # All of those information will be provided by the manifest at load time.
    del model_function_name
    del weights_name
    del checkpoint_path
    del checkpoint_abs_path

    # Loading now.
    manifest_proto = obm.load(save_dir_path)

    # Loads the orchestration.
    orch_filename = manifest_proto.supplemental_info[
        simple_orchestration.TEST_ORCHESTRATION_SUPPLEMENTAL_NAME
    ].file_system_location.string_path
    pipeline_proto = simple_orchestration_pb2.Pipeline()
    with open(os.path.join(save_dir_path, orch_filename), 'rb') as f:
      pipeline_proto.ParseFromString(f.read())

    # Loads the model function
    loaded_model_function_name = pipeline_proto.model_function_name
    loaded_obm_function = manifest_proto.objects[
        loaded_model_function_name
    ].function

    # and its supplemental.
    jax_supplemental_filename = (
        loaded_obm_function.body.stable_hlo_body.supplemental_info[
            obm.JAX_SPECIFIC_INFO
        ].file_system_location.string_path
    )
    jax_supplemental_proto = jax_supplemental_pb2.Function()
    with open(
        os.path.join(save_dir_path, jax_supplemental_filename), 'rb'
    ) as f:
      jax_supplemental_proto.ParseFromString(f.read())

    # Deserialize the jax function and compare results from the 2 jax functions.
    deserialized_jax_exported = obm_to_jax.obm_functions_to_jax_function(
        loaded_obm_function,
        jax_supplemental_proto,
    )

    # Restore/load the params from the saved orbax checkpoint,
    # this will be fed into the deserialized jax function only.
    loaded_weights_name = pipeline_proto.weights_name
    loaded_checkpoint_path = manifest_proto.objects[
        loaded_weights_name
    ].value.external.data.file_system_location.string_path
    restored_params = checkpointer.restore(
        os.path.join(save_dir_path, loaded_checkpoint_path)
    )

    test_input_data = jax.device_put(
        jax.random.uniform(
            jax.random.PRNGKey(999), (4, 28, 28, 1), dtype=jnp.float64
        ),
        input_sharding_spec,
    )

    result_from_original_jax_call = model_apply_fn(params, test_input_data)
    result_from_deserialized_jax_call = deserialized_jax_exported.call(
        jax.device_put(restored_params, params_sharding_spec),
        test_input_data,
    )

    self.assertTrue(
        jnp.array_equal(
            result_from_deserialized_jax_call, result_from_original_jax_call
        )
    )


if __name__ == '__main__':
  absltest.main()
