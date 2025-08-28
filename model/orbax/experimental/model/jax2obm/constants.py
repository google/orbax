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

"""Constants used in jax2obm."""
# XLA compilation options.
XLA_COMPILE_OPTIONS = 'xla_compile_options'

# XLA compile options mime type.
XLA_COMPILE_OPTIONS_MIME_TYPE = (
    'application/protobuf; type=orbax_model_manifest.CompileOptionsProtoMap'
)
