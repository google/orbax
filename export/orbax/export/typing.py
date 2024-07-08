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

"""Common typing for export."""

from typing import Any, Callable, Mapping, Sequence, TypeVar, Union
import jaxtyping
from orbax.export import utils as orbax_export_utils
import tensorflow as tf


T = TypeVar('T')
Nested = Union[T, tuple[Any, ...], Sequence[Any], Mapping[str, Any]]
WarmupExample = Union[list[Mapping[str, Any]], Mapping[str, Any]]
NestedTfTrackable = Nested[tf.saved_model.experimental.TrackableResource]
NestedTfTensorSpec = Nested[
    Union[tf.TensorSpec, orbax_export_utils.TensorSpecWithDefault]
]

PyTree = jaxtyping.PyTree

# ApplyFn take two arguments, the first one is the model_params, the second one
# is the model_inputs.
ApplyFn = Callable[[PyTree, PyTree], PyTree]
