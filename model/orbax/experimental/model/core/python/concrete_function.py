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

"""Function (aka concrete function)."""

# pylint: disable=g-importing-member
from dataclasses import dataclass
from typing import Any, Tuple
import numpy as np
from orbax.experimental.model.core.python.shlo_function import ShloFunction
from orbax.experimental.model.core.python.signature import assert_sub_specs
from orbax.experimental.model.core.python.signature import DType
from orbax.experimental.model.core.python.signature import Shape
from orbax.experimental.model.core.python.signature import Signature
from orbax.experimental.model.core.python.signature import TensorSpec
from orbax.experimental.model.core.python.signature import TreeOfTensorSpecs
from orbax.experimental.model.core.python.util import dtypes


def np_shape_to_shape(shape: Tuple[int, ...]) -> Shape:
  return shape


def np_dtype_to_dtype(dtype: np.dtype) -> DType:
  return dtypes.numpy_to_tf_dtype(dtype)


def dtype_to_np_dtype(dtype: DType) -> np.dtype:
  return dtypes.tf_to_numpy_dtype(dtype)


def ndarray_to_spec(arr: np.ndarray) -> TensorSpec:
  return TensorSpec(
      shape=np_shape_to_shape(arr.shape),
      dtype=dtypes.numpy_to_tf_dtype(arr.dtype),
  )


@dataclass
class Tensor:

  np_array: np.ndarray
  spec: TensorSpec

  def __init__(self, np_array: np.ndarray):
    self.np_array = np_array
    self.spec = ndarray_to_spec(np_array)


@dataclass
class Variable:

  value: Tensor


# A PyTree where all leaf nodes are variables
# TODO(b/329305005): Support nested structures.
TreeOfVars = Variable


def is_tree_of_vars(x: Any) -> bool:
  return isinstance(x, TreeOfVars)


@dataclass
class ConcreteFunction:
  """A function with an input and output signature.

  It corresponds to concrete functions in TF.

  A `Function` is usually built by starting from a free function (a
  function that doesn't have captures, so that it receives all input
  information through arguments) and then binding some arguments of that free
  function to some fixed values (called "captures").

  Attributes:
    input_signature: the input signature.
    output_signature: the output signature.
    base_fn: a free function from which this function was built.
    captured_vars: a tuple of `TreeOfVars`. The tuple components are bound to
      the front-most (positional) arguments of `base_fn`.
  """

  # An input signature has to be a tuple (or list), otherwise
  # "partial application" is meaningless.
  input_signature: Signature
  # TODO(b/329306885): Support non-tuple non-list output signature.
  output_signature: Signature
  # The base (i.e. capture-free) function
  base_fn: ShloFunction

  # Captured variables to be *appended* (not *prepended*) to the argument list
  # when calling `base_fn`.
  # TODO(b/329308295): Support binding to arbitrary positional or keyword
  #   arguments, not just the back-most ones.
  # TODO(b/329308274): Support negative indices for positional arguments.
  captured_vars: Tuple[TreeOfVars, ...] = ()


def assert_nonempty_and_get_head(
    t: Tuple[Any, ...]
) -> Tuple[Any, Tuple[Any, ...]]:
  assert t
  return t[0], t[1:]


def assert_nonempty_and_get_last(
    t: Tuple[Any, ...]
) -> Tuple[Tuple[Any, ...], Any]:
  assert t
  return t[0:-1], t[-1]


def vars_to_spec(tree: TreeOfVars) -> TreeOfTensorSpecs:
  assert isinstance(tree, Variable)
  var: Variable = tree
  return var.value.spec


def _partially_apply_to_variables(
    f: ConcreteFunction, arg: TreeOfVars
) -> ConcreteFunction:
  other_args_spec, expected_arg_spec = assert_nonempty_and_get_last(
      f.input_signature
  )
  arg_spec = vars_to_spec(arg)
  assert_sub_specs(arg_spec, expected_arg_spec)
  return ConcreteFunction(
      input_signature=other_args_spec,
      output_signature=f.output_signature,
      base_fn=f.base_fn,
      captured_vars=(arg,) + f.captured_vars,
  )


def partial(
    f: ConcreteFunction, arg: Any, *, bind_last_arg: bool = False
) -> ConcreteFunction:
  """Partially applies a function `f` to an argument `arg`.

  `arg` will be bound to the first (positional) argument of `f`. The returned
  function
  will have the same signature as `f` sans the first (positional) argument.

  Args:
    f: the `Function` to be partially applied.
    arg: the value to be bound to the target argument.
    bind_last_arg: a boolean. If `True`, we will bind the last (positional)
      argument instead of the first.

  Returns:
    A `Function`.
  """
  if not is_tree_of_vars(arg):
    raise NotImplementedError(
        f"We only support `arg` of type `TreeOfVars`, not {type(arg)}"
    )
  if not bind_last_arg:
    raise NotImplementedError("Only bind_last_arg=True is supported.")
  return _partially_apply_to_variables(f, arg)
