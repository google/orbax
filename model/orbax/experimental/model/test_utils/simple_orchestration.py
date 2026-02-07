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

"""Pipeline: pre-processor + model-function + post-processor."""

from orbax.experimental.model.core.protos import manifest_pb2
from orbax.experimental.model.test_utils import simple_orchestration_pb2

TEST_ORCHESTRATION_MIME_TYPE: str = "simple_orchestration"
TEST_ORCHESTRATION_VERSION: str = "0.0.1"
TEST_ORCHESTRATION_SUPPLEMENTAL_NAME = "orchestration"


def create(
    *,
    model_function_name: str | None = None,
    weights_name: str | None = None,
    pre_processor_name: str | None = None,
    post_processor_name: str | None = None,
) -> manifest_pb2.UnstructuredData:
  """Returns an `UnstructuredData` containing a `Pipeline` proto.

  The `Pipeline` proto will be returned as `inlined_bytes`.

  Args:
    model_function_name: The name of the model function.
    weights_name: The name of the weights.
    pre_processor_name: The name of the pre-processor function.
    post_processor_name: The name of the post-processor function.

  Returns:
    An `UnstructuredData` containing in its `inlined_bytes` field a
    `Pipeline` proto.
  """
  orch = simple_orchestration_pb2.Pipeline()
  if model_function_name is not None:
    orch.model_function_name = model_function_name
  if weights_name is not None:
    orch.weights_name = weights_name
  if pre_processor_name is not None:
    orch.pre_processor_name = pre_processor_name
  if post_processor_name is not None:
    orch.post_processor_name = post_processor_name

  return manifest_pb2.UnstructuredData(
      inlined_bytes=orch.SerializeToString(),
      mime_type=TEST_ORCHESTRATION_MIME_TYPE,
      version=TEST_ORCHESTRATION_VERSION,
  )
