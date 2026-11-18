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

"""Helpers for transporting values through colocated Python."""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.multihost import colocated_transport as _mod
from orbax.checkpoint.v0._src.multihost.colocated_transport import *
from orbax.checkpoint.v0._src.multihost.colocated_transport import _PATHWAYS_SERIALIZATION_PATCH_INSTALLED
from orbax.checkpoint.v0._src.multihost.colocated_transport import _PJRT_IFRT_DEVICE_ID_RE
from orbax.checkpoint.v0._src.multihost.colocated_transport import _original_make_prng_wrapped_fun
from orbax.checkpoint.v0._src.multihost.colocated_transport import _make_prng_wrapped_fun_without_copied_lock
from orbax.checkpoint.v0._src.multihost.colocated_transport import _to_serializable_cpu_device
from orbax.checkpoint.v0._src.multihost.colocated_transport import _device_platform
from orbax.checkpoint.v0._src.multihost.colocated_transport import _device_identity_key
from orbax.checkpoint.v0._src.multihost.colocated_transport import _extract_pjrt_ifrt_device_id
from orbax.checkpoint.v0._src.multihost.colocated_transport import _all_backend_devices
from orbax.checkpoint.v0._src.multihost.colocated_transport import _get_cpu_device_map
from orbax.checkpoint.v0._src.multihost.colocated_transport import _normalize_mesh_to_colocated_cpu
from orbax.checkpoint.v0._src.multihost.colocated_transport import _normalize_device_list_to_colocated_cpu
from orbax.checkpoint.v0._src.multihost.colocated_transport import _normalize_single_device_sharding_to_colocated_cpu

sys.modules[__name__] = _mod
