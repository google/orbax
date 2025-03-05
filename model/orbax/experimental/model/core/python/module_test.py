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

# pylint: disable=g-importing-member
from typing import Tuple

from absl.testing import parameterized
import numpy as np
from orbax.experimental.model.core.python import polymorphic_function
from orbax.experimental.model.core.python import test_utils
from orbax.experimental.model.core.python import tracing
from orbax.experimental.model.core.python.concrete_function import ConcreteFunction
from orbax.experimental.model.core.python.concrete_function import dtype_to_np_dtype
from orbax.experimental.model.core.python.concrete_function import spec_from_ndarray
from orbax.experimental.model.core.python.concrete_function import Tensor
from orbax.experimental.model.core.python.concrete_function import TreeOfVars
from orbax.experimental.model.core.python.concrete_function import Variable
from orbax.experimental.model.core.python.module import Module
from orbax.experimental.model.core.python.polymorphic_function import PolymorphicFunction
from orbax.experimental.model.core.python.shlo_function import ShloFunction
from orbax.experimental.model.core.python.signature import float64
from orbax.experimental.model.core.python.signature import Signature
from orbax.experimental.model.core.python.signature import TensorSpec
from orbax.experimental.model.core.python.signature import TreeOfTensorSpecs
from orbax.experimental.model.core.python.tree_util import tree_map

from absl.testing import absltest


# TODO(b/329305005): Support nested structures.
TreeOfNdarrays = np.ndarray


def spec_from_np_arrays(tree: TreeOfNdarrays) -> TreeOfTensorSpecs:
  assert isinstance(tree, np.ndarray)
  arr = tree
  return spec_from_ndarray(arr)


def signature_from_np_arrays(tple: Tuple[TreeOfNdarrays, ...]) -> Signature:
  return tuple(map(spec_from_np_arrays, tple))


def zeros_from_spec(spec: TensorSpec) -> Tensor:
  return Tensor(np.zeros(spec.shape, dtype_to_np_dtype(spec.dtype)))


class ModuleTest(test_utils.ObmTestCase):

  @parameterized.named_parameters(
      (  # pylint: disable=g-complex-comprehension
          f"_{use_input_signature}_{use_abstract_eval}",
          use_input_signature,
          use_abstract_eval,
      )
      for use_abstract_eval in (False, True)
      for use_input_signature in (False, True)
  )
  def test_simple_case(self, use_input_signature, use_abstract_eval):
    if use_input_signature and use_abstract_eval:
      self.skipTest(
          "use_abstract_eval has effect only when use_input_signature=False."
      )
    expected_np_params = np.array([[1.1, 2.2]])
    expected_input_spec = TensorSpec(shape=(2, 3), dtype=float64)
    expected_output_spec = TensorSpec(shape=(1, 3), dtype=float64)
    expected_shlo_module = b"My MatMul SHLO Module"

    @tracing.add_variable_support
    def fn_(
        x: tracing.SymbolicTensor, y: tracing.SymbolicTensor
    ) -> Tuple[tracing.TreeOfGraphEdgeTensors, ...]:
      self.assertTrue(tracing.is_tracing())
      input_sig = tree_map(tracing.spec_from_symbolic_tensors, (x, y))
      # TODO(wangpeng): Remove this annotation once `Signature` is
      #   based on `tree_util.Tree`.
      input_sig: Signature
      fn = ConcreteFunction(
          input_signature=input_sig,
          output_signature=(expected_output_spec,),
          base_fn=ShloFunction(
              input_signature=(),
              output_signature=(),
              mlir_module_serialized=expected_shlo_module,
              calling_convention_version=0,
              module_kept_var_idx=(),
              lowering_platforms=(),
              supplemental_info=None,
              physical_in_dtypes=(),
              physical_out_dtypes=(),
          ),
      )
      output = tracing.add_node(input_=(x, y), op=fn)
      return output

    variables = tree_map(lambda x: Variable(Tensor(x)), expected_np_params)
    # TODO(wangpeng): Remove this annotation once `TreeOfVars` is
    #   based on `tree_util.Tree`.
    variables: TreeOfVars
    kwargs = {}
    if use_input_signature:
      kwargs["input_signature"] = (expected_input_spec,)

    def fn_2(inputs):
      # Somehow the typechecker can't see the annotation `variables: TreeOfVars`
      # above, so we need to re-annotate.
      # TODO(wangpeng): Remove this annotation and change `fn_2` back to a
      #   lambda once `TreeOfVars` is based on `tree_util.Tree`.
      vs = variables  # can't annotate captured variable, so need a new alias
      vs: TreeOfVars
      return fn_(inputs, vs)

    fn = polymorphic_function.function(fn_2, **kwargs)
    if not use_input_signature:
      # If `input_signature` wasn't provided to `function`, we need to do a
      # tracing to add a new concrete function, via either `fn.__call__` or
      # `fn.abstract_eval`.
      if use_abstract_eval:
        output_spec = fn.abstract_eval((expected_input_spec,))
        self.assertTreeEquiv(output_spec, (expected_output_spec,))
      else:
        fn(tree_map(zeros_from_spec, expected_input_spec))
    m = Module()
    m.forward_fn = fn

    self.assertSetEqual(set(m.get_dict().keys()), set(["forward_fn"]))
    retrieved_fn = m.forward_fn
    self.assertIsInstance(retrieved_fn, PolymorphicFunction)
    self.assertLen(retrieved_fn.concrete_fns, 1)
    retrieved_cfn = retrieved_fn.concrete_fns[0]
    self.assertIsInstance(retrieved_cfn, ConcreteFunction)
    retrieved_cfn: ConcreteFunction
    self.assertTreeEquiv(retrieved_cfn.input_signature, (expected_input_spec,))
    self.assertTreeEquiv(
        retrieved_cfn.output_signature, (expected_output_spec,)
    )
    retrieved_captures = retrieved_cfn.captured_vars
    # Checks that the Python object identities of the captures are preserved.
    self.assertTreeEquiv(retrieved_captures, (variables,), lambda a, b: a is b)
    retrieved_base_fn = retrieved_cfn.base_fn
    self.assertIsInstance(retrieved_base_fn, ShloFunction)
    retrieved_base_fn: ShloFunction
    self.assertEqual(
        retrieved_base_fn.mlir_module_serialized, expected_shlo_module
    )


if __name__ == "__main__":
  absltest.main()
