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

"""The data processor module for tensroflow-based pre/post-processors."""

from collections.abc import Callable, Mapping, Sequence
import functools
from typing import Any, Tuple, cast

from absl import logging
import jax
import jaxtyping
from orbax.export.data_processors import data_processor_base
import tensorflow as tf


class TfDataProcessor(data_processor_base.DataProcessor):
  """DataProcessor class for TensorFlow-based processors.

  The properties of this class are available only after `prepare()` is called.
  If a property is accessed before `prepare()` is called, it will raise a
  ValueError.
  """

  # This pass statement is a placeholder to prevent an IndentationError.
  # The code block below is removed by copybara for open-source builds,
  # and without `pass`, the class would have an empty body, which is a
  # syntax error.
  pass
