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

from absl.testing import absltest
from orbax.experimental.model.core.protos import manifest_pb2
from orbax.experimental.model.test_utils import simple_orchestration
from orbax.experimental.model.test_utils import simple_orchestration_pb2
from absl.testing import absltest


class SimpleOrchestrationTest(googletest.TestCase):

  def test_create(self):
    expected_proto = manifest_pb2.UnstructuredData(
        inlined_bytes=simple_orchestration_pb2.Pipeline(
            model_function_name="model_function_name",
            weights_name="weights_name",
            pre_processor_name="pre_processor_name",
            post_processor_name="post_processor_name",
        ).SerializeToString(),
        mime_type=simple_orchestration.TEST_ORCHESTRATION_MIME_TYPE,
        version=simple_orchestration.TEST_ORCHESTRATION_VERSION,
    )
    self.assertEqual(
        simple_orchestration.create(
            model_function_name="model_function_name",
            weights_name="weights_name",
            pre_processor_name="pre_processor_name",
            post_processor_name="post_processor_name",
        ),
        expected_proto,
    )


if __name__ == "__main__":
  absltest.main()
