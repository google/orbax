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

"""Tests for validate module."""
import json

from absl.testing import absltest
from orbax.export.validate.validation_job import ValidationSingleJobResult
from orbax.export.validate.validation_report import ValidationReport
from orbax.export.validate.validation_report import ValidationReportOption
from orbax.export.validate.validation_utils import Status


class ValidateReportTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.baseline_result = ValidationSingleJobResult(
        outputs=[{
            '1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 2, 3, 4, 5, 6]
        }],
        latencies=[88.0, 88.0, 90.0, 91.0, 88.5, 88.7, 87.0],
        xprof_url='N/A',
        metadata={})

    self.candidate_result = ValidationSingleJobResult(
        outputs=[{
            '1': [0.2, 0.4, 0.6, 0.8, 1.0, 0.6, 1, 2, 3, 4, 5, 6]
        }],
        latencies=[88.0, 88.0, 90.0, 91.0, 88.5, 88.7, 87.0],
        xprof_url='N/A',
        metadata={})

    self.candidate_result_2 = ValidationSingleJobResult(
        outputs=[{
            '1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 2, 4, 6, 8, 5, 6]
        }],
        latencies=[88.0, 88.0, 90.0, 91.0, 88.5, 88.7, 87.0],
        xprof_url='N/A',
        metadata={})

    self.candidate_result_3 = ValidationSingleJobResult(
        outputs=[{
            '1': [0.2, 0.4, 0.6, 0.8, 1.0, 0.6, 2, 4, 6, 8, 5, 6]
        }],
        latencies=[88.0, 88.0, 90.0, 91.0, 88.5, 88.7, 87.0],
        xprof_url='N/A',
        metadata={})

  def test_validation_report_option(self):
    with self.assertRaises(OverflowError):
      ValidationReportOption(floating_atol=-1)
    with self.assertRaises(OverflowError):
      ValidationReportOption(floating_rtol=-1)

  def test_status_pass(self):
    validation_report = ValidationReport(self.baseline_result,
                                         self.baseline_result)
    self.assertEqual(validation_report.status, Status.Pass)

  def test_status_fail(self):
    validation_report = ValidationReport(self.baseline_result,
                                         self.candidate_result)
    self.assertEqual(validation_report.status, Status.Fail)

    # Test validate func: float data pass but but non-float fail.
    validation_report = ValidationReport(self.baseline_result,
                                         self.candidate_result_2)
    self.assertEqual(validation_report.status, Status.Fail)

    # Test both float and non-float result fail on validation check.
    validation_report = ValidationReport(self.baseline_result,
                                         self.candidate_result_3)
    self.assertEqual(validation_report.status, Status.Fail)

  def test_to_dict_and_to_json(self):
    validation_report = ValidationReport(self.baseline_result,
                                         self.candidate_result)

    # test to_dict.
    result_dict = validation_report.to_dict()
    expect_dict = {
        'outputs': {
            'FloatingPointDiffReport': {
                'total': 6,
                'max_diff': 0.5,
                'max_rel_diff': 1.0,
                'all_close': False,
                'all_close_absolute_tolerance': 1e-07,
                'all_close_relative_tolerance': 1e-07
            },
            'NonFloatingPointDiffReport': {
                'total_flattened_tensors': 6,
                'mismatches': 0,
                'mismatch_ratio': 0.0,
                'max_non_floating_mismatch_ratio': 0.01
            }
        }
    }
    self.assertDictEqual(result_dict['outputs'], expect_dict['outputs'])

    # test to_json
    result_json = validation_report.to_json()
    new_result_dict = json.loads(result_json)
    self.assertDictEqual(new_result_dict['outputs'], expect_dict['outputs'])

  def test_diff_tree_def(self):
    baseline_result = ValidationSingleJobResult(
        outputs=[{'1': [1, 2, 3, 4, 5, 6]}],  # 6 non-floats
        latencies=[90.0],
        xprof_url='N/A',
        metadata={},
    )

    candidate_result = ValidationSingleJobResult(
        outputs=[{'1': [1, 2, 3, 4, 5]}],  # 5 non-floats
        latencies=[88.0],
        xprof_url='N/A',
        metadata={},
    )
    with self.assertRaisesRegex(
        ValueError,
        'baseline and candidate result have diff tree_def.',
    ):
      _ = ValidationReport(baseline_result, candidate_result)

  def test_unequal_data_points(self):
    baseline_result = ValidationSingleJobResult(
        outputs=[{'1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}],  # 6 floats
        latencies=[90.0],
        xprof_url='N/A',
        metadata={},
    )

    candidate_result = ValidationSingleJobResult(
        outputs=[{'1': [1.0, 2.0, 3.0, 4.0, 5.0, 6]}],  # 5 floats + 1 int
        latencies=[88.0],
        xprof_url='N/A',
        metadata={},
    )
    with self.assertRaisesRegex(
        ValueError,
        'baseline and candidate floating result have different length.',
    ):
      _ = ValidationReport(baseline_result, candidate_result)


if __name__ == '__main__':
  absltest.main()
