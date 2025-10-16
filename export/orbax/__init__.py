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

"""Defines exported symbols for Orbax Export."""

# pylint: disable=g-importing-member
from orbax.export import config
from orbax.export import constants
from orbax.export import obm_configs
from orbax.export import typing
from orbax.export import utils
from orbax.export.dtensor_utils import dtensor_initialized
from orbax.export.dtensor_utils import get_current_dtensor_mesh
from orbax.export.dtensor_utils import get_current_mesh
from orbax.export.dtensor_utils import get_pspec_from_jax_arrays
from orbax.export.dtensor_utils import initialize_dtensor
from orbax.export.dtensor_utils import maybe_enable_dtensor_export_on
from orbax.export.dtensor_utils import shutdown_dtensor
from orbax.export.export_manager import ExportManager
from orbax.export.jax_module import JaxModule
from orbax.export.serving_config import ServingConfig
# TODO(dinghua): remove them after we change all references to
# utils.remove_signature_defaults.
from orbax.export.utils import remove_signature_defaults
from orbax.export.utils import TensorSpecWithDefault


# A new PyPI release will be pushed everytime `__version__` is increased.
__version__ = '0.0.8'
