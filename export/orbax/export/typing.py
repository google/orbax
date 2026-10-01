# Copyright 2026 The Orbax Authors.
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

"""Common typing for export."""

from collections.abc import Callable, Mapping, Sequence, Set
import dataclasses
from typing import Any, TypeVar, Union
import jaxtyping
import tensorflow as tf


T = TypeVar('T')
Nested = Union[T, tuple[Any, ...], Sequence[Any], Mapping[str, Any]]
WarmupExample = Union[list[Mapping[str, Any]], Mapping[str, Any]]
NestedTfTrackable = Nested[tf.saved_model.experimental.TrackableResource]


PyTree = jaxtyping.PyTree

# ApplyFn take two arguments, the first one is the model_params, the second one
# is the model_inputs.
ApplyFn = Callable[[PyTree, PyTree], PyTree]


@dataclasses.dataclass
class ApplyFnInfo:
  """Information about an apply function.

  Attributes:
    apply_fn: The apply function, which takes `model_params` and `model_inputs`
      as arguments. `model_inputs` must be a dictionary with keys matching
      `input_keys`. The function must return a dictionary with keys matching
      `output_keys`.
    input_keys: The keys of the input dict that the `apply_fn` expects. These
      keys are also used to determine the topological ordering of the `apply_fn`
      and other `DataProcessor`s in the pipeline.
    output_keys: The keys of the output dict that the `apply_fn` produces. These
      keys are also used to determine the topological ordering of the `apply_fn`
      and other `DataProcessor`s in the pipeline.
    use_input_aliasing: If True, the exported function takes one extra input
      argument per output, aliased to that output, so that an execution writes
      its results into the buffers passed for those inputs (donated at serving
      time) instead of allocating them. The extra arguments are hidden from the
      serving signature, and the runtime fills them in by output name. Only
      supported by Orbax Model export. It makes every pipeline that serves
      this function graph-based (keyword-arguments based), so processors (if
      any) must be given as `data_processors`, the input signature must be a
      mapping keyed by input name, and every output must have a unique name.
      `Jax2ObmOptions.enable_auto_layout` must be off. Donation is only served
      from AOT executables, so input shapes must be concrete: with
      `input_polymorphic_shape`, `input_polymorphic_shape_symbol_values` is
      required.
  """

  apply_fn: ApplyFn
  input_keys: Set[str]
  output_keys: Set[str]
  use_input_aliasing: bool = False
