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

from typing import Any
from orbax.experimental.model import core as obm

VOXEL_PROCESSOR_MIME_TYPE = 'application/protobuf; type=voxel.PlanProto'
VOXEL_PROCESSOR_VERSION = '0.0.1'
DEFAULT_VOXEL_MODULE_FOLDER = 'voxel_module'


def voxel_plan_to_obm(
    # TODO(b/447200841): use the true type hint after voxel module is
    # implemented.
    voxel_module: Any,
    input_signature: obm.Tree[obm.ShloTensorSpec],
    output_signature: obm.Tree[obm.ShloTensorSpec],
    subfolder: str = DEFAULT_VOXEL_MODULE_FOLDER,
) -> obm.SerializableFunction:
  """Converts a Voxel plan to an `obm.SerializableFunction`.

  Args:
    voxel_module: The Voxel module to be converted.
    input_signature: The input signature of the Voxel module.
    output_signature: The output signature of the Voxel module.
    subfolder: The name of the subfolder for the converted module.

  Returns:
    An `obm.SerializableFunction` representing the Voxel module.
  """
  plan_proto = voxel_module.export_plan()
  plan_proto_bytes = plan_proto.SerializeToString()

  unstructured_data = obm.manifest_pb2.UnstructuredData(
      inlined_bytes=plan_proto_bytes,
      mime_type=VOXEL_PROCESSOR_MIME_TYPE,
      version=VOXEL_PROCESSOR_VERSION,
  )

  obm_func = obm.SerializableFunction(
      body=obm.UnstructuredDataWithExtName(
          proto=unstructured_data,
          subfolder=subfolder,
          ext_name='pb',
      ),
      input_signature=input_signature,
      output_signature=output_signature,
  )
  return obm_func
