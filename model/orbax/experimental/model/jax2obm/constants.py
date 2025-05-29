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

import enum


# LINT.IfChange
class OrbaxNativeSerializationType(enum.Enum):
  """Defines the native serialization types available for a JAX model through Orbax."""

  CPU = 'cpu'
  CUDA = 'cuda'
  ROCM = 'rocm'
  TPU = 'tpu'


# LINT.ThenChange(//depot//orbax/export/constants.py)

# XLA compilation options.
XLA_COMPILE_OPTIONS = 'xla_compile_options'
