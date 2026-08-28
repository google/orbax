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

from absl.testing import absltest
from absl.testing import parameterized

from orbax.checkpoint._src.serialization import ocdbt_process_spec


class OcdbtProcessSpecTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='no_replica_suffix_numeric_id',
          process_id='123',
          use_replica_suffix=False,
          expected_spec='ocdbt.process_123',
      ),
      dict(
          testcase_name='with_replica_suffix_numeric_id',
          process_id='123',
          use_replica_suffix=True,
          expected_spec='ocdbt.process_replica_123',
      ),
      dict(
          testcase_name='no_replica_suffix_alphanumeric_id',
          process_id='h0',
          use_replica_suffix=False,
          expected_spec='ocdbt.process_h0',
      ),
      dict(
          testcase_name='with_replica_suffix_alphanumeric_id',
          process_id='w13',
          use_replica_suffix=True,
          expected_spec='ocdbt.process_replica_w13',
      ),
      dict(
          testcase_name='no_replica_suffix_alphabetic_id',
          process_id='abc',
          use_replica_suffix=False,
          expected_spec='ocdbt.process_abc',
      ),
      dict(
          testcase_name='with_replica_suffix_alphabetic_id',
          process_id='abc',
          use_replica_suffix=True,
          expected_spec='ocdbt.process_replica_abc',
      ),
  )
  def test_create(
      self,
      process_id: str,
      use_replica_suffix: bool,
      expected_spec: str,
  ):
    spec = ocdbt_process_spec.OcdbtProcessSpec(
        process_id=process_id, use_replica_suffix=use_replica_suffix
    )
    self.assertEqual(str(spec), expected_spec)

  @parameterized.named_parameters(
      dict(
          testcase_name='empty_process_id',
          process_id='',
      ),
      dict(
          testcase_name='non_alphanumeric_process_id',
          process_id='123!',
      ),
  )
  def test_create_invalid_process_id(self, process_id: str):
    with self.assertRaisesRegex(
        ValueError, rf'Invalid OCDBT process id: {process_id}'
    ):
      ocdbt_process_spec.OcdbtProcessSpec(
          process_id=process_id, use_replica_suffix=False
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='no_replica_suffix_numeric_id',
          spec='ocdbt.process_123',
          expected_process_id='123',
          expected_use_replica_suffix=False,
      ),
      dict(
          testcase_name='with_replica_suffix_numeric_id',
          spec='ocdbt.process_replica_123',
          expected_process_id='123',
          expected_use_replica_suffix=True,
      ),
      dict(
          testcase_name='no_replica_suffix_alphanumeric_id',
          spec='ocdbt.process_h0',
          expected_process_id='h0',
          expected_use_replica_suffix=False,
      ),
      dict(
          testcase_name='with_replica_suffix_alphanumeric_id',
          spec='ocdbt.process_replica_w13',
          expected_process_id='w13',
          expected_use_replica_suffix=True,
      ),
      dict(
          testcase_name='no_replica_suffix_alphabetic_id',
          spec='ocdbt.process_abc',
          expected_process_id='abc',
          expected_use_replica_suffix=False,
      ),
      dict(
          testcase_name='with_replica_suffix_alphabetic_id',
          spec='ocdbt.process_replica_abc',
          expected_process_id='abc',
          expected_use_replica_suffix=True,
      ),
  )
  def test_parse(
      self,
      spec: str,
      expected_process_id: str,
      expected_use_replica_suffix: bool,
  ):
    parsed_spec = ocdbt_process_spec.OcdbtProcessSpec.parse(spec)
    self.assertEqual(parsed_spec.process_id, expected_process_id)
    self.assertEqual(
        parsed_spec.use_replica_suffix, expected_use_replica_suffix
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='empty_string',
          spec='',
      ),
      dict(
          testcase_name='no_process_prefix',
          spec='replica_123',
      ),
      dict(
          testcase_name='empty_process_id',
          spec='ocdbt.process_',
      ),
      dict(
          testcase_name='empty_process_id_with_replica_suffix',
          spec='ocdbt.process_replica_',
      ),
      dict(
          testcase_name='empty_replica_suffix',
          spec='ocdbt.process__123',
      ),
      dict(
          testcase_name='non_alphanumeric_process_id',
          spec='ocdbt.process_123!',
      ),
      dict(
          testcase_name='extra_characters_at_start',
          spec='abc.ocdbt.process_123',
      ),
      dict(
          testcase_name='extra_characters_at_end',
          spec='ocdbt.process_123.abc',
      ),
  )
  def test_parse_invalid_spec(self, spec: str):
    with self.assertRaisesRegex(
        ValueError, r'Bad string .* for Orbax/OCDBT process spec'
    ):
      ocdbt_process_spec.OcdbtProcessSpec.parse(spec)


if __name__ == '__main__':
  absltest.main()
