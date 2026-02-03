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

"""Constants for the Orbax model CLI."""

CLI_VERSION = '0.0.1'

MIME_TYPE_JAX_SPECIFIC_INFO = (
    'application/protobuf; type=orbax_model_jax_supplemental.Function'
)
MIME_TYPE_MLIR_STABLEHLO = 'application/x.mlir-stablehlo'
MIME_TYPE_ORBAX_CHECKPOINT = 'application/x.orbax-checkpoint'
MIME_TYPE_TF_CONCRETE_FUNCTION = (
    'application/protobuf;'
    ' type=orbax_model_tf_concrete_function_handle.TfConcreteFunctionHandle'
)
MIME_TYPE_TF_SAVED_MODEL = 'application/x.tensorflow-saved-model'


WELL_KNOWN_MIME_TYPE_DESCRIPTIONS = {
    MIME_TYPE_JAX_SPECIFIC_INFO: 'JAX Supplemental Function',
    MIME_TYPE_MLIR_STABLEHLO: 'MLIR StableHLO',
    MIME_TYPE_ORBAX_CHECKPOINT: 'Orbax Checkpoint',
    MIME_TYPE_TF_CONCRETE_FUNCTION: 'TensorFlow ConcreteFunction',
    MIME_TYPE_TF_SAVED_MODEL: 'TensorFlow SavedModel',
}

# Register flags for printing details for some types, typically
# ones that don't fit into the default details output.
MIME_TYPE_FLAGS = {
    MIME_TYPE_JAX_SPECIFIC_INFO: 'jax_specific_info',
}
