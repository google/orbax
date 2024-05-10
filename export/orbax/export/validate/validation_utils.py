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

"""The collection of validation utility functions."""
import enum
import json
from typing import Any, Tuple

import jax
import numpy as np
import tensorflow as tf

Status = enum.Enum('Status', ['Pass', 'Fail'])


def split_tf_floating_and_discrete_groups(
    inputs: list[Any],
) -> Tuple[np.ndarray, list[Any]]:
  """Flatten `inputs` and split them into floating and non-floating groups."""

  def is_float(x):
    if isinstance(x, tf.Tensor):
      return x.dtype.is_floating
    else:
      return np.issubdtype(np.asarray(x).dtype, np.floating)

  flattened = jax.tree_util.tree_leaves(inputs)
  # Convert float_vals into 1D numpy array for easy comparison.
  float_vals = [np.asarray(x).flatten() for x in flattened if is_float(x)]
  if float_vals:
    float_vals = np.concatenate(float_vals, axis=0)
  else:
    float_vals = np.array([])

  discrete_vals = [x for x in flattened if not is_float(x)]
  return float_vals, discrete_vals


class EnhancedJSONEncoder(json.JSONEncoder):
  """internal class to support json output."""

  def default(self, o):
    if isinstance(o, enum.Enum):
      return o.name
    elif hasattr(o, 'numpy'):
      return o.numpy()
    elif hasattr(o, 'tolist'):
      return o.tolist()
    elif isinstance(o, bytes):
      return o.decode('utf-8', errors='backslashreplace')
    return json.JSONEncoder.default(self, o)


def get_latency_stat(latencies: list[float]):
  """Internal helper function to generate latency stats table."""
  num_batches = len(latencies)
  avg_in_ms = float(np.average(latencies) * 1000)
  p90_in_ms = float(np.percentile(latencies, 90) * 1000)
  p99_in_ms = float(np.percentile(latencies, 99) * 1000)
  return num_batches, avg_in_ms, p90_in_ms, p99_in_ms
