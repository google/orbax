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

"""Polymorphic functions (i.e. functions with multiple signatures)."""

# pylint: disable=g-importing-member
from dataclasses import dataclass
from dataclasses import field
from typing import List, Optional

from orbax.experimental.model.core.python import tracing
from orbax.experimental.model.core.python.concrete_function import ConcreteFunction
from orbax.experimental.model.core.python.concrete_function import Tensor
from orbax.experimental.model.core.python.signature import assert_sub_signature
from orbax.experimental.model.core.python.signature import Signature


def spec_from_values(*args, **kwargs) -> Signature:
  if kwargs:
    raise NotImplementedError("Keyword arguments are not supporeted.")
  specs = []
  for arg in args:
    if not isinstance(arg, Tensor):
      raise NotImplementedError("Only `Tensor`s are supporeted.")
    specs.append(arg.spec)
  return tuple(specs)


@dataclass
class PolymorphicFunction:
  """A function that has multiple input and output signatures.

  It corresponds to `tf.function`s (aka polymorphic functions) in TF.

  Each input signature is backed by a `ConcreteFunction`. If a new input
  signature is used to call this
  `PolymorphicFunction`, it will use a Python function `python_fn` (if
  available) create a new `ConcreteFunction`.

  Caution: Currently only supports at most one concrete function.
  """

  python_fn: Optional[tracing.Tracable] = None
  concrete_fns: List[ConcreteFunction] = field(default_factory=list)

  def __call__(self, *args, **kwargs) -> None:  # pylint: disable=g-doc-args
    """Looks up or creates (by tracing) a concrete function matching the input.

    Returns nothing.
    """
    spec = spec_from_values(*args, **kwargs)
    _ = self.abstract_eval(spec)
    # We can't run the concrete function and get actual output values,
    # so we don't return anything.

  # TODO(b/329304515): Support polymorphism (i.e. more than one concrete
  #   functions).
  def abstract_eval(self, spec: Signature) -> Signature:
    """Calls or creates (by tracing) a concrete function with the given spec.

    Returns the output spec of that concrete function.

    Since only spec is returned, the concrete function is not really "called",
    but just asked about its `output_signature`. Hence this is an
    "abstract evaluation" or "symbolic call".

    Args:
      spec: an input signature to be used to do the abstract evaluation.

    Returns:
      The output signature resulted from this abstract evaluation.
    """
    return self.get_concrete_function(spec).output_signature

  def get_concrete_function(self, spec: Signature) -> ConcreteFunction:  # pylint: disable=missing-function-docstring
    if not self.concrete_fns:
      if self.python_fn is None:
        raise ValueError(
            "We need to do a new tracing, but `python_fn` is None."
        )
      cfn = tracing.trace(self.python_fn, spec)
      self.concrete_fns.append(cfn)
    elif len(self.concrete_fns) == 1:
      cfn = self.concrete_fns[0]
      assert_sub_signature(spec, cfn.input_signature)
    else:
      raise NotImplementedError(
          "Expected zero or one concrete function, got"
          f" {len(self.concrete_fns)}."
      )
    return cfn


# TODO(b/329735824): Consider renaming `em.function` to `em.export` or `em.jit`.
def function(
    f: tracing.Tracable, *, input_signature: Optional[Signature] = None
) -> PolymorphicFunction:
  """Creates a `PolymorphicFunction` from a `Tracable`.

  Args:
    f: a `Tracable`.
    input_signature: an optional input signature. If given, the
      returned `PolymorphicFunction` will contain a concrete function
      matching the `input_signature`. Otherwise, the
      `PolymorphicFunction` will contain no concrete functions.

  Returns:
    A `PolymorphicFunction`.
  """
  fn = PolymorphicFunction(python_fn=f)
  if input_signature is not None:
    _ = fn.abstract_eval(input_signature)
  return fn
