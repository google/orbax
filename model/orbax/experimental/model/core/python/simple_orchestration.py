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

"""Simple orchestration: pre-processor + model-function + post-processor."""

from orbax.experimental.model.core.protos import simple_orchestration_pb2
from orbax.experimental.model.core.python import unstructured_data


CURRENT_SIMPLE_ORCHESTRATION_MIME_TYPE: str = "simple_orchestration"
CURRENT_SIMPLE_ORCHESTRATION_VERSION: str = "0.0.1"


def serialize_simple_orchestration_proto(
    proto: simple_orchestration_pb2.SimpleOrchestration,
) -> unstructured_data.UnstructuredData:
  return unstructured_data.UnstructuredData(
      inlined_bytes=proto.SerializeToString(),
      mime_type=CURRENT_SIMPLE_ORCHESTRATION_MIME_TYPE,
      version=CURRENT_SIMPLE_ORCHESTRATION_VERSION,
  )


def simple_orchestration(
    model_function_name: str,
    weights_name: str,
    pre_processor_name: str | None = None,
    post_processor_name: str | None = None,
) -> unstructured_data.UnstructuredData:
  """Returns an `UnstructuredData` containing a `SimpleOrchestration` proto.

  The `SimpleOrchestration` proto will be returned as `inlined_bytes`.

  Args:
    model_function_name: The name of the model function.
    weights_name: The name of the weights.
    pre_processor_name: The name of the pre-processor function.
    post_processor_name: The name of the post-processor function.

  Returns:
    An `UnstructuredData` containing in its `inlined_bytes` field a
    `SimpleOrchestration` proto.
  """
  orch = simple_orchestration_pb2.SimpleOrchestration()
  orch.model_function_name = model_function_name
  orch.weights_name = weights_name
  if pre_processor_name is not None:
    orch.pre_processor_name = pre_processor_name
  if post_processor_name is not None:
    orch.post_processor_name = post_processor_name
  return serialize_simple_orchestration_proto(orch)


ORCHESTRATION_SUPPLEMENTAL_NAME = "orchestration"
