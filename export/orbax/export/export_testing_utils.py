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

"""Testing utils for orbax.export."""
import os
from typing import cast
import jax
from jax import sharding
from jax.experimental import mesh_utils
from jax.experimental.topologies import get_topology_desc
import jax.numpy as jnp
import numpy as np
from orbax.export import constants
from orbax.export import jax_module
from orbax.export import obm_configs
from orbax.export import serving_config as osc
import tensorflow as tf

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
