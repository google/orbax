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

"""Define ValidationJob class here."""

from collections.abc import Mapping, Sequence
import dataclasses
import time
from typing import Any, Callable, TypeVar

import dataclasses_json


INPUT = TypeVar('INPUT')


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class ValidationSingleJobResult:
  """Create the data structure to contain a single validate Job Result."""
  latencies: Sequence[float]
  outputs: Sequence[Any]
  xprof_url: str
  metadata: Mapping[str, Any]

  def __post_init__(self):
    if not isinstance(self.outputs, Sequence):
      raise ValueError('The output must be list since we take batch inputs.')

  def maybe_convert_result_to_dict(self):
    """Converts outputs to a dict that matches the default SavedModel."""

    def _convert(results):
      if not isinstance(results, Mapping):
        output = {}
        for i, x in enumerate(results):
          output[f'output_{i}'] = x
        return output
      return results

    self.outputs = list(map(_convert, self.outputs))


def _run_inference(infer_step_fn, batched_examples, latencies, outputs):
  for inputs in batched_examples:
    t0 = time.time()
    result = infer_step_fn(inputs)
    latency = time.time() - t0
    latencies.append(latency)
    outputs.append(result)


class ValidationJob:
  """Run inference job and output the ValidationSingleJobResult object."""

  def __init__(
      self,
      baseline_inference_fn: Callable[[INPUT], Any],
      candidate_inference_fn: Callable[[INPUT], Any],
      batch_input: Sequence[INPUT],
      with_xprof: bool = False,
  ):
    """Output the ValidationSingleJobResult.

    Feed batched input data into `inference_fn` and get latencies and results.
    The batch_input is batched dataset.

    Args:
      baseline_inference_fn:  baseline function.
      candidate_inference_fn: loaded model candidate function.
      batch_input:  Batch of inputs.
      with_xprof: whether to run xprof. Unused.
      # Unless GOOGLE-INTERNAL.
    """
    self._baseline_inference_fn = baseline_inference_fn
    self._candidate_inference_fn = candidate_inference_fn
    self._batch_input = batch_input
    self._with_xprof = with_xprof

  def _calc_result(
      self,
      infer_step: Callable[[INPUT], Any],
      batched_examples: Sequence[INPUT],
  ):
    """Feed batch_input into `apply_fn` and run."""
    # Warm up in case they are jit functions.
    _ = infer_step(batched_examples[0])

    # Generate baseline model baseline result.
    latencies = list()
    outputs = list()
    xprof_url = 'N/A'
    _run_inference(
        infer_step,
        batched_examples,
        latencies,
        outputs,
    )

    # TODO(johnqiangzhang): Add metadata info based on the user feedback.
    metadata = {}
    result = ValidationSingleJobResult(
        latencies,
        outputs,
        xprof_url,
        metadata)
    return result

  def calc_baseline_result(self) -> ValidationSingleJobResult:
    """Feed batch_input into baseline `inference_fn` and run."""
    return self._calc_result(self._baseline_inference_fn, self._batch_input)

  def calc_candidate_result(self) -> ValidationSingleJobResult:
    """Feed batch_input into candidate `inference_fn` and run."""
    return self._calc_result(self._candidate_inference_fn, self._batch_input)
