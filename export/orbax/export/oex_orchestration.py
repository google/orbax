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

"""Pipeline: pre-processor + model-function + post-processor."""
from collections.abc import Sequence
import dataclasses
from typing import Any

from absl import logging
import jax
import jaxtyping
# TODO: b/448900820 - remove this unused import.
from orbax.export.data_processors import data_processor_base
from orbax.export.modules import obm_module
