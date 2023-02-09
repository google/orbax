# Copyright 2022 The Orbax Authors.
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
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from jax.experimental import jax2tf
import jax.numpy as jnp
import numpy as np
from orbax.export.validate.validation_job import ValidationJob
from orbax.export.validate.validation_job import ValidationSingleJobResult
from orbax.export.validate.validation_report import ValidationReport
from orbax.export.validate.validation_utils import Status
import tensorflow as tf


class ValidationJobTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(with_xprof=False, testcase_name="without_xprof"),
      dict(with_xprof=True, testcase_name="with_xprof"),
  )
  def test_validation_job_report(self, with_xprof):
    """Test validation job to get the output."""
    func_jax = lambda x: jnp.sin(jnp.cos(x))
    func_tf = tf.function(jax2tf.convert(func_jax), autograph=False)
    batch_input = list(np.arange(128).reshape((16, 8)).astype(np.float32))
    validation_job = ValidationJob(
        func_jax, func_tf, batch_input, with_xprof=with_xprof)
    jax_result = validation_job.calc_baseline_result()
    tf_result = validation_job.calc_candidate_result()
    validation_report = ValidationReport(jax_result, tf_result)
    self.assertEqual(validation_report.status, Status.Pass)
    logging.info(validation_report)

    # test discrete value
    func_jax = lambda x: x * x
    func_tf = tf.function(jax2tf.convert(func_jax), autograph=False)
    batch_input = list(np.arange(128).reshape((16, 8)).astype(int))
    validation_job = ValidationJob(
        func_jax, func_tf, batch_input, with_xprof=with_xprof)
    jax_result = validation_job.calc_baseline_result()
    tf_result = validation_job.calc_candidate_result()
    validation_report = ValidationReport(jax_result, tf_result)
    self.assertEqual(validation_report.status, Status.Pass)
    logging.info(validation_report)


class ValidationSingleJobResultTest(parameterized.TestCase):

  def test_maybe_convert_result_to_dict(self):
    model_output = [tf.constant(1) for _ in range(12)]
    batch_outputs = [model_output] * 30
    single_job_result = ValidationSingleJobResult(
        outputs=batch_outputs,
        latencies=[88.0] * 30,
        xprof_url="N/A",
        metadata={},
    )
    single_job_result.maybe_convert_result_to_dict()
    new_output = single_job_result.outputs[0]
    self.assertIsInstance(
        new_output,
        dict,
        f"new_output suppose to be python dict but got {type(new_output)}",
    )
    self.assertIn(
        "output_11",
        new_output,
        (
            "output_11 should in new_output but not."
            f"new_output keys is {new_output.keys()}"
        ),
    )


if __name__ == "__main__":
  absltest.main()
