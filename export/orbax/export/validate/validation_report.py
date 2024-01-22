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

"""Define ValidationReport class here."""
import dataclasses
import pathlib
from typing import Any, Dict, Optional, Union

from absl import logging
import dataclasses_json
import jax
import numpy as np
from orbax.export.validate.validation_job import ValidationSingleJobResult
from orbax.export.validate.validation_utils import get_latency_stat
from orbax.export.validate.validation_utils import split_tf_floating_and_discrete_groups
from orbax.export.validate.validation_utils import Status

XprofURL = str
MetaData = Any


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class ValidationReportOption:
  """Option for ValidationReport class."""
  floating_atol: float = 1e-7
  floating_rtol: float = 1e-7
  max_non_floating_mismatch_ratio: float = 1e-2
  output_report_path: Optional[Union[str, pathlib.Path]] = None
  print_debug_info: bool = False

  def __post_init__(self):
    """check if option value is legal.

    Raises:
      OverflowError: raise if floating_atol < 0 or floating_rtol < 0.
    """
    if self.floating_atol < 0:
      raise OverflowError('floating_atol should be larger than zero.')
    if self.floating_rtol < 0:
      raise OverflowError('floating_rtol should be larger than zero.')


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class LatencyStat:
  """The latency indicator of ML job."""
  num_batches: int
  avg_in_ms: float
  p90_in_ms: float
  p99_in_ms: float


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class FloatingPointDiffReport:
  total: int
  max_diff: float
  max_rel_diff: float
  all_close: bool
  all_close_absolute_tolerance: float
  all_close_relative_tolerance: float


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class NonFloatingPointDiffReport:
  total_flattened_tensors: int
  mismatches: int
  mismatch_ratio: float
  max_non_floating_mismatch_ratio: float


@dataclasses_json.dataclass_json
@dataclasses.dataclass(init=False, eq=False)
class ValidationReport:
  """Generate validation report based on ValidationSingleJobResult.
  """
  outputs: Dict[str, Union[FloatingPointDiffReport, NonFloatingPointDiffReport]]
  latency: Dict[str, LatencyStat]
  xprof_url: Dict[str, XprofURL]
  metadata: Dict[str, MetaData]
  status: Status

  def __init__(self,
               baseline: ValidationSingleJobResult,
               candidate: ValidationSingleJobResult,
               option: Optional[ValidationReportOption] = None):
    """Generate validation result report with users config options.

    Args:
      baseline:  The baseline ValidationSingleJobResult.
      candidate:  The candidate ValidationSingleJobResult. The comparing
        criterions will be apply on candidate.
      option: ValidationReport options.
    """
    if not option:
      self._option = ValidationReportOption()
    else:
      self._option = option

    floating_atol = self._option.floating_atol
    floating_rtol = self._option.floating_rtol
    max_non_floating_mismatch_ratio = (
        self._option.max_non_floating_mismatch_ratio)

    baseline_latencies = baseline.latencies
    baseline_outputs = baseline.outputs
    baseline_url = baseline.xprof_url

    self.status = Status.Pass

    candidate_latencies = candidate.latencies
    candidate_outputs = candidate.outputs
    # TODO(b/251969924): check baseline and candidate have same structure.
    candidate_url = candidate.xprof_url

    num_batches, avg_in_ms, p90_in_ms, p99_in_ms = get_latency_stat(
        baseline_latencies)
    baseline_latency_stat = LatencyStat(num_batches, avg_in_ms, p90_in_ms,
                                        p99_in_ms)
    num_batches, avg_in_ms, p90_in_ms, p99_in_ms = get_latency_stat(
        candidate_latencies)
    candidate_latency_stat = LatencyStat(num_batches, avg_in_ms, p90_in_ms,
                                         p99_in_ms)

    baseline_outputs_tree_def = jax.tree_util.tree_structure(baseline_outputs)
    candidate_outputs_tree_def = jax.tree_util.tree_structure(candidate_outputs)
    if baseline_outputs_tree_def != candidate_outputs_tree_def:
      raise ValueError(
          'baseline and candidate result have diff tree_def.'
          f'baseline tree_def = {baseline_outputs_tree_def}'
          f'candidate tree_def = {candidate_outputs_tree_def}'
      )

    baseline_floatings, baseline_non_floatings = (
        split_tf_floating_and_discrete_groups(baseline_outputs)
    )
    candidate_floatings, candidate_non_floatings = (
        split_tf_floating_and_discrete_groups(candidate_outputs)
    )

    if baseline_floatings.size != candidate_floatings.size:
      raise ValueError(
          'baseline and candidate floating result have different length. '
          f'baseline = {baseline_floatings}, candidate = {candidate_floatings}'
      )
    if len(baseline_non_floatings) != len(candidate_non_floatings):
      raise ValueError(
          'baseline and candidate non-floating result have different length. '
          f'baseline = {baseline_floatings}, candidate = {candidate_floatings}'
      )

    self.outputs = {}
    if baseline_floatings.size == 0:
      logging.info('No floating-point outputs.')
    else:
      max_diff = np.abs(candidate_floatings - baseline_floatings).max()
      max_rel_diff = (np.abs(candidate_floatings - baseline_floatings) /
                      np.maximum(np.abs(baseline_floatings), 1e-6)).max()
      all_close = np.allclose(candidate_floatings, baseline_floatings,
                              floating_atol, floating_rtol)
      if all_close:
        logging.info(
            'Baseline and candidate floating-point results are all close '
            '(atol=%f, rtol=%f). max_diff=%f, max_rel_diff=%f', floating_atol,
            floating_rtol, max_diff, max_rel_diff)
      else:
        logging.warning(
            'Baseline and candidate floating-point results are not all close. '
            'max_diff=%f, max_rel_diff=%f.', max_diff, max_rel_diff)
        if self._option.print_debug_info:
          logging.warning('baseline_floatings = %s', baseline_floatings)
          logging.warning('candidate_floatings = %s', candidate_floatings)
        self.status = Status.Fail

      self.outputs['FloatingPointDiffReport'] = FloatingPointDiffReport(
          total=int(baseline_floatings.size),
          max_diff=float(max_diff),
          max_rel_diff=float(max_rel_diff),
          all_close=all_close,
          all_close_absolute_tolerance=floating_atol,
          all_close_relative_tolerance=floating_rtol,
      )

    mismatches = sum(
        np.all(j != t)
        for j, t in zip(baseline_non_floatings, candidate_non_floatings))
    total_non_floatings = len(baseline_non_floatings)
    mismatch_ratio = .0
    if total_non_floatings > 0:
      mismatch_ratio = mismatches / total_non_floatings
    if mismatch_ratio <= max_non_floating_mismatch_ratio:
      logging.info(
          '%d Baseline/Candidate mismatches over %d non-floating-point results.'
          'Mismatch ratio is %f (<= %f threshold).', mismatches,
          total_non_floatings, mismatch_ratio, max_non_floating_mismatch_ratio)
    else:
      logging.warning(
          (
              '%d Baseline/Candidate mismatches over %d non-floating-point'
              ' results.Mismatch ratio is %f (> %f threshold).'
          ),
          mismatches,
          total_non_floatings,
          mismatch_ratio,
          max_non_floating_mismatch_ratio,
      )
      if self._option.print_debug_info:
        logging.warning('baseline_non_floatings = %s', baseline_non_floatings)
        logging.warning('candidate_non_floatings = %s', candidate_non_floatings)
      self.status = Status.Fail

    self.outputs['NonFloatingPointDiffReport'] = NonFloatingPointDiffReport(
        total_flattened_tensors=int(total_non_floatings),
        mismatches=int(mismatches),
        mismatch_ratio=float(mismatch_ratio),
        max_non_floating_mismatch_ratio=float(max_non_floating_mismatch_ratio))

    # Create the result python dict.
    self.latency = {
        'baseline': baseline_latency_stat,
        'candidate': candidate_latency_stat
    }

    self.xprof_url = {'baseline': baseline_url, 'candidate': candidate_url}

    self.metadata = {
        'baseline': baseline.metadata,
        'candidate': candidate.metadata
    }
