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

"""MLEM saved model builder test."""

from collections.abc import Sequence

import jax
from jax import export as jax_export
import jax.numpy as jnp
import numpy as np
from orbax.experimental.model.core.python import concrete_function
from orbax.experimental.model.core.python import module
from orbax.experimental.model.core.python import save_lib
from orbax.experimental.model.core.python import shlo_function
from orbax.experimental.model.core.python import signature
from orbax.experimental.model.core.python.function import np_dtype_to_shlo_dtype
from orbax.experimental.model.core.python.function import ShloTensorSpec
from orbax.experimental.model.core.python.saved_model_proto import saved_model_builder

from absl.testing import absltest

save = save_lib.save
Tensor = concrete_function.Tensor
ConcreteFunction = concrete_function.ConcreteFunction
Variable = concrete_function.Variable
TensorSpec = signature.TensorSpec


def jax_spec_from_aval(x: jax.core.AbstractValue) -> jax.ShapeDtypeStruct:
  assert isinstance(x, jax.core.ShapedArray)
  return jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)


def jax_spec_to_shlo_spec(
    jax_spec: Sequence[jax.ShapeDtypeStruct],
) -> tuple[ShloTensorSpec, ...]:
  return tuple(
      ShloTensorSpec(shape=x.shape, dtype=np_dtype_to_shlo_dtype(x.dtype))
      for x in jax_spec
  )


def jax_spec_to_tensor_spec(x: jax.ShapeDtypeStruct) -> TensorSpec:
  return TensorSpec(
      shape=x.shape, dtype=concrete_function.dtype_from_np_dtype(x.dtype)
  )


class SavedModelBuilderTest(absltest.TestCase):

  def test_build_with_function_aliases(self):
    @jax.jit
    def f(xy, captured):
      return captured + xy, captured - xy

    myvar = jnp.array(np.ones((5, 10), dtype=np.float32))
    jax_in_spec = (
        jax.ShapeDtypeStruct(shape=[], dtype=myvar.dtype),
        jax.ShapeDtypeStruct(shape=myvar.shape, dtype=myvar.dtype),
    )
    exported = jax_export.export(f)(*jax_in_spec)
    jax_out_spec = tuple(jax_spec_from_aval(x) for x in exported.out_avals)

    input_signature = (jax_spec_to_tensor_spec(jax_in_spec[0]),)
    output_signature = tuple(jax_spec_to_tensor_spec(x) for x in jax_out_spec)
    func = ConcreteFunction(
        input_signature=input_signature,
        output_signature=output_signature,
        base_fn=shlo_function.ShloFunction(
            input_signature=jax_spec_to_shlo_spec(jax_in_spec),
            output_signature=jax_spec_to_shlo_spec(jax_out_spec),
            mlir_module_serialized=exported.mlir_module_serialized,
            calling_convention_version=exported.calling_convention_version,
            module_kept_var_idx=exported.module_kept_var_idx,
            lowering_platforms=exported.platforms,
            supplemental_info=None,
            physical_in_dtypes=(None,) * len(jax_in_spec),
            physical_out_dtypes=(None,) * len(jax_out_spec),
        ),
        captured_vars=(Variable(Tensor(np.asarray(myvar))),),
    )

    m = module.Module()
    m.add_concrete_function("add", func)
    builder = saved_model_builder.SavedModelProtoBuilder()
    named_vars, named_functions = save_lib.named_variables_and_functions(m)
    for variable, name in named_vars.items():
      builder.add_variable(variable, name)
    for fn, name in named_functions.items():
      builder.add_function(fn, name)
    builder.add_function_alias({"tpu_func": m.add}, named_functions)
    proto = builder.build()

    self.assertEqual(
        proto.meta_graphs[0].meta_info_def.function_aliases,
        {"add": "tpu_func"},
    )


if __name__ == "__main__":
  absltest.main()
