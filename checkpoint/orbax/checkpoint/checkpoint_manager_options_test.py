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

"""CheckpointManagerOptions tests."""

import dataclasses
import datetime
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import orbax.checkpoint as ocp

MOCK_STEP_NAME_FORMAT = mock.create_autospec(ocp.step.NameFormat)


class CheckpointManagerOptionsTest(parameterized.TestCase):

  @parameterized.parameters(
      (
          {
              'save_interval_steps': 1,
          },
      ),
      (
          {
              'save_interval_steps': 0,
              'max_to_keep': 1,
          },
      ),
      (
          {
              'save_interval_steps': 0,
              'keep_time_interval': datetime.timedelta(seconds=1),
          },
      ),
      (
          {
              'save_interval_steps': 0,
              'keep_period': 1,
          },
      ),
      (
          {
              'save_interval_steps': 0,
              'create': True,
          },
      ),
      (
          {
              'save_interval_steps': 0,
              'create': False,
              'cleanup_tmp_directories': True,
          },
      ),
      (
          {
              'save_interval_steps': 0,
              'create': False,
              'cleanup_tmp_directories': False,
              'save_on_steps': [1],
          },
      ),
      (
          {
              'save_interval_steps': 0,
              'create': False,
              'cleanup_tmp_directories': False,
              'save_on_steps': None,
              'todelete_subdir': 'ttl=1h',
          },
      ),
      (
          {
              'save_interval_steps': 1,
              'save_on_steps': [1],
              'should_save_fn': lambda step: True,
          },
      ),
  )
  def test_side_effect_options_update_for_read_only(self, kwargs):
    kwargs.update({'read_only': True})
    options = ocp.CheckpointManagerOptions(**kwargs)
    self.assertEqual(options.save_interval_steps, 0)
    self.assertIsNone(options.max_to_keep)
    self.assertIsNone(options.keep_time_interval)
    self.assertIsNone(options.keep_period)
    self.assertFalse(options.create)
    self.assertFalse(options.cleanup_tmp_directories)
    self.assertEmpty(options.save_on_steps)
    self.assertIsNone(options.todelete_subdir)
    self.assertIsNone(options.should_save_fn)
    self.assertIsNone(options.should_keep_fn)

  def test_replace_for_read_only(self):
    options = ocp.CheckpointManagerOptions(
        read_only=True, create=False, save_interval_steps=0
    )
    self.assertEmpty(options.save_on_steps)
    updated_options = dataclasses.replace(options, step_prefix='prefix')
    self.assertEmpty(updated_options.save_on_steps)

  def test_replace_for_should_keep_fn(self):
    options = ocp.CheckpointManagerOptions(
        keep_period=1,
        should_keep_fn=lambda step: True,
    )
    self.assertIsNone(options.keep_period)
    self.assertIsNotNone(options.should_keep_fn)

  @parameterized.named_parameters(
      # dict(
      #     testcase_name='error_step_name_format_false',
      #     single_host_load_and_broadcast=True,
      #     step_name_format=ocp.step.standard_name_format(
      #         single_host_load_and_broadcast=False
      #     ),
      #     expected_single_host_load_and_broadcast=None, # Both None for Error.
      #     expected_step_name_format=None,  # Both None for Error.
      # ),
      # dict(
      #     testcase_name='error_step_name_format_true',
      #     single_host_load_and_broadcast=False,
      #     step_name_format=ocp.step.standard_name_format(
      #         single_host_load_and_broadcast=True
      #     ),
      #     expected_single_host_load_and_broadcast=None, # Both None for Error.
      #     expected_step_name_format=None,  # Both None for Error.
      # ),
      # dict(
      #     testcase_name='error_enabled_with_non_supporting_step_name_format',
      #     single_host_load_and_broadcast=True,
      #     step_name_format=MOCK_STEP_NAME_FORMAT,
      #     expected_single_host_load_and_broadcast=None, # Both None for Error.
      #     expected_step_name_format=None,  # Both None for Error.
      # ),
      dict(
          testcase_name='both_true',
          single_host_load_and_broadcast=True,
          step_name_format=ocp.step.standard_name_format(
              single_host_load_and_broadcast=True
          ),
          expected_single_host_load_and_broadcast=True,
          expected_step_name_format=ocp.step.standard_name_format(
              single_host_load_and_broadcast=True
          ),
      ),
      dict(
          testcase_name='both_false',
          single_host_load_and_broadcast=False,
          step_name_format=ocp.step.standard_name_format(
              single_host_load_and_broadcast=False
          ),
          expected_single_host_load_and_broadcast=False,
          expected_step_name_format=ocp.step.standard_name_format(
              single_host_load_and_broadcast=False
          ),
      ),
      dict(
          testcase_name='only_single_host_load_and_broadcast_true',
          single_host_load_and_broadcast=True,
          step_name_format=None,
          expected_single_host_load_and_broadcast=True,
          expected_step_name_format=None,
      ),
      dict(
          testcase_name='only_single_host_load_and_broadcast_false',
          single_host_load_and_broadcast=False,
          step_name_format=None,
          expected_single_host_load_and_broadcast=False,
          expected_step_name_format=None,
      ),
      dict(
          testcase_name='disabled_with_non_supporting_step_name_format',
          single_host_load_and_broadcast=False,
          step_name_format=MOCK_STEP_NAME_FORMAT,
          expected_single_host_load_and_broadcast=False,
          expected_step_name_format=MOCK_STEP_NAME_FORMAT,
      ),
  )
  def test_single_host_load_and_broadcast(
      self,
      single_host_load_and_broadcast: bool,
      step_name_format: ocp.step.NameFormat | None,
      expected_single_host_load_and_broadcast: bool | None,
      expected_step_name_format: ocp.step.NameFormat | None,
  ):
    if (
        expected_single_host_load_and_broadcast is None
        and expected_step_name_format is None
    ):
      with self.assertRaises(ValueError):
        ocp.CheckpointManagerOptions(
            single_host_load_and_broadcast=single_host_load_and_broadcast,
            step_name_format=step_name_format,
        )
    else:
      options = ocp.CheckpointManagerOptions(
          single_host_load_and_broadcast=single_host_load_and_broadcast,
          step_name_format=step_name_format,
      )
      self.assertEqual(
          options.single_host_load_and_broadcast,
          expected_single_host_load_and_broadcast,
      )
      self.assertEqual(options.step_name_format, expected_step_name_format)


if __name__ == '__main__':
  absltest.main()
