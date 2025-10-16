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

# pylint: disable=missing-module-docstring
# pylint: disable=g-importing-member
from orbax.experimental.model.jax2obm import jax_supplemental_pb2
from orbax.experimental.model.jax2obm.constants import XLA_COMPILE_OPTIONS
from orbax.experimental.model.jax2obm.constants import XLA_COMPILE_OPTIONS_MIME_TYPE
from orbax.experimental.model.jax2obm.jax_specific_info import CURRENT_JAX_SUPPLEMENTAL_MIME_TYPE
from orbax.experimental.model.jax2obm.jax_specific_info import CURRENT_JAX_SUPPLEMENTAL_VERSION
from orbax.experimental.model.jax2obm.main_lib import *
from orbax.experimental.model.jax2obm.obm_to_jax import obm_functions_to_jax_function
