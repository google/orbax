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

"""Prometheus metrics exporter for Orbax.

This module enables Orbax to export performance and operational metrics to
Prometheus, allowing users to monitor checkpointing behavior in open-source
environments.

To enable telemetry collection, set the environment variable
`ENABLE_ORBAX_PROMETHEUS_TELEMETRY=true`. By default, telemetry collection
is disabled.
"""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.logging import prometheus_monitoring as _mod
from orbax.checkpoint.v0._src.logging.prometheus_monitoring import *
from orbax.checkpoint.v0._src.logging.prometheus_monitoring import _prometheus_multiproc_dir
from orbax.checkpoint.v0._src.logging.prometheus_monitoring import _prom_counter
from orbax.checkpoint.v0._src.logging.prometheus_monitoring import _prom_histogram

sys.modules[__name__] = _mod
