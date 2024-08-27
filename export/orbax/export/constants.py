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

"""Constants used in Orbax export."""

import enum


class ExportModelType(enum.Enum):
  """Defines the export types available for serializing a JAX model through Orbax."""

  TF_SAVEDMODEL = 'TF_SAVEDMODEL',
  ORBAX_MODEL = 'ORBAX_MODEL',


# Default method key for Orbax Export modules if apply_fn is not a mapping.
DEFAULT_METHOD_KEY = 'jax_module_default_method'
