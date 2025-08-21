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

"""The data processor class for tensroflow-based pre/post-processors."""

from collections.abc import Callable, Mapping, Sequence
import functools
from typing import Any, Tuple

import jax
import jaxtyping
from orbax.export.data_processors import data_processor_base
import tensorflow as tf

ConcreteFunction = tf.types.experimental.ConcreteFunction
PyTree = jaxtyping.PyTree


class TfDataProcessor(data_processor_base.DataProcessor):
  """DataProcessor Class for TensorFlow-based processors.

  Attributes:
    input_signature: The input signature for the TensorFlow model.
    output_signature: The output signature for the TensorFlow model.
  """

  def __init__(self, **kwargs):
    """Initializes the TfDataProcessor.

    Args:
      **kwargs: Additional keyword arguments.
    """
    super().__init__(**kwargs)
    self._tf_callable = kwargs['tf_callable']
    self._concrete_function: ConcreteFunction | None = None
