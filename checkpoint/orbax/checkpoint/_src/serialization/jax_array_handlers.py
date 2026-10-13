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

"""Provides handlers for Jax Arrays."""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.serialization import jax_array_handlers as _mod
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import *
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _SHARDING_FILE_NAME
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _has_prng_key_dtype
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _physical_read_spec
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _is_replicated_sharding
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _async_serialize_shardings
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _get_replica_slices
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _record_logical_metrics
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _record_compression_metrics
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _record_raw_metrics
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _log_io_metrics
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _worker_serialize_arrays
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _get_deprioritized_batches_to_serialize
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _on_batch_callback
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _serialize_arrays_batches_without_dispatcher
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _serialize_arrays
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _async_serialize_replica_slices
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _wrap_random_key_data
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _validate_ocdbt_settings
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _validate_non_ocdbt_files
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _deserialize_shardings
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _deserialize_arrays
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _sync_deserialize_arrays
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _get_abstract_arrays
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _get_default_use_replica_parallel
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _is_host_for_primary_replica
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _validate_sharding_and_get_primary_replica_processes
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _single_replica_abstract_arrays
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _single_replica_deserialize_and_broadcast
from orbax.checkpoint.v0._src.serialization.jax_array_handlers import _single_replica_deserialize_on_worker

sys.modules[__name__] = _mod
