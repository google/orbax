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

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bundle orchestration export utilities."""

import dataclasses
import os
import shutil
from typing import Dict, List

from neptune_model.protos import bundle_orchestration_pb2


@dataclasses.dataclass
class SubModel:
  """Defines a unique model in the bundle.

  Attributes:
    name: Name of the sub-model subdirectory in the bundle.
    path: Path to the exported sub-model.
  """

  name: str
  path: str


@dataclasses.dataclass
class PipelineStep:
  """Defines one model's execution within the bundle.

  Attributes:
    model: Name of the sub-model (must match a SubModel.name).
    pipeline: Name of the pipeline in the sub-model to execute.
    repeated_times: Number of times to repeat this step.
    requires_h2d: Whether this step requires Host-to-Device transfer.
    requires_d2h: Whether this step requires Device-to-Host transfer.
  """

  model: str
  pipeline: str
  repeated_times: int = 1
  requires_h2d: bool = False
  requires_d2h: bool = False


@dataclasses.dataclass
class BundleDefinition:
  """Defines organization of sub-models in the bundle.

  Attributes:
    name: Human-readable name of the bundle.
    version: Version of the bundle.
    pipelines: Maps pipeline name to a sequential list of steps.
  """

  name: str
  version: int
  pipelines: Dict[str, List[PipelineStep]]


def create_bundle(
    output_path: str,
    bundle_def: BundleDefinition,
    models: List[SubModel],
    copy_models: bool = False,
):
  """Creates the bundle directory, symlinks/copies sub-models, and writes the proto.

  Args:
    output_path: Path where the bundle should be created.
    bundle_def: Definition of the bundle pipelines and metadata.
    models: List of sub-models to include in the bundle.
    copy_models: If True, copies the sub-models instead of symlinking them.
  """
  os.makedirs(output_path, exist_ok=True)

  for model in models:
    dest_path = os.path.join(output_path, model.name)
    if os.path.lexists(dest_path):
      if os.path.islink(dest_path):
        os.unlink(dest_path)
      elif os.path.isdir(dest_path):
        shutil.rmtree(dest_path)
      else:
        os.remove(dest_path)
    if copy_models:
      shutil.copytree(model.path, dest_path)
    else:
      os.symlink(model.path, dest_path)

  proto = bundle_orchestration_pb2.BundleOrchestration()
  proto.metadata.name = bundle_def.name
  proto.metadata.version = bundle_def.version

  for pipeline_name, steps in bundle_def.pipelines.items():
    bundle_pipeline = bundle_orchestration_pb2.BundlePipeline()
    for step in steps:
      component = bundle_pipeline.components.add()
      component.model_name = step.model
      component.pipeline_name = step.pipeline
      component.repeated_times = step.repeated_times
      component.requires_h2d = step.requires_h2d
      component.requires_d2h = step.requires_d2h

    proto.pipelines[pipeline_name].CopyFrom(bundle_pipeline)

  pb_path = os.path.join(output_path, "bundle_orchestration.pb")
  with open(pb_path, "wb") as f:
    f.write(proto.SerializeToString())
