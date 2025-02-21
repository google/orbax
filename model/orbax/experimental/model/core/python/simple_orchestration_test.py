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

from absl.testing import parameterized
from orbax.experimental.model.core.python import simple_orchestration
from absl.testing import absltest


class SimpleOrchestrationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="_ok_1",
          model_function_signature=(
              ((1, 2), {}),
              100,
          ),
          expected_signature_or_error=(((2,), {}), 100),
      ),
      dict(
          testcase_name="_ok_2",
          model_function_signature=(
              (1, 2),
              100,
          ),
          expected_signature_or_error=(((2,), {}), 100),
      ),
      dict(
          testcase_name="_ok_3",
          pre_processor_signature=(((1,), {}), 2),
          model_function_signature=(
              2,
              3,
          ),
          post_processor_signature=(3, 4),
          expected_signature_or_error=(((1,), {}), 4),
      ),
      dict(
          testcase_name="_ok_4",
          model_function_signature=(
              (1, 2, 3, 4, 5),
              100,
          ),
          num_of_weights=3,
          expected_signature_or_error=((4, 5), (100,)),
      ),
      dict(
          testcase_name="_ok_5",
          model_function_signature=(
              (1, 2, 3),
              100,
          ),
          num_of_weights=3,
          expected_signature_or_error=((), (100,)),
      ),
      dict(
          testcase_name="_error_1",
          model_function_signature=(
              1,
              100,
          ),
          expected_signature_or_error=(
              ValueError,
              (
                  r"The positional-arguments part of model_function_signature"
                  r" must be a sequence.*Got:"
              ),
          ),
      ),
      dict(
          testcase_name="_error_2",
          model_function_signature=(
              (1, 2, 3),
              100,
          ),
          expected_signature_or_error=(
              ValueError,
              (
                  r"The positional part of model_function_signature must have"
                  r" exactly two elements.*Got:"
              ),
          ),
      ),
      dict(
          testcase_name="_error_3",
          model_function_signature=(
              (1, 2, 3),
              100,
          ),
          num_of_weights=-1,
          expected_signature_or_error=(
              ValueError,
              r"num_of_weights must be non-negative.*Got:",
          ),
      ),
      dict(
          testcase_name="_error_4",
          model_function_signature=(
              (1, 2, 3),
              100,
          ),
          num_of_weights=4,
          expected_signature_or_error=(
              ValueError,
              (
                  r"The positional part of model_function_signature must have"
                  r" at least 4 elements.*Got:"
              ),
          ),
      ),
  )
  def test_calculate_signature(
      self,
      *,
      model_function_signature,
      pre_processor_signature=None,
      post_processor_signature=None,
      num_of_weights=None,
      expected_signature_or_error,
  ):
    def f():
      return simple_orchestration.calculate_signature(
          model_function_signature=model_function_signature,
          pre_processor_signature=pre_processor_signature,
          post_processor_signature=post_processor_signature,
          num_of_weights=num_of_weights,
      )

    if isinstance(expected_signature_or_error[0], type):
      with self.assertRaisesRegex(*expected_signature_or_error):
        f()
    else:
      signature = f()
      self.assertEqual(signature, expected_signature_or_error)


if __name__ == "__main__":
  absltest.main()
