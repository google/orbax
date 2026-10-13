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

"""TensorStore serialization helper functions."""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.serialization import tensorstore_utils as _mod
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import *
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _GCS_OCDBT_TARGET_DATA_FILE_SIZE
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _OCDBT_SPLIT_VALUE_DATA_PREFIX
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _OCDBT_SPLIT_META_DATA_PREFIX
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _OCDBT_TMP_METADATA_PREFIX
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _GCS_PATH_RE
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _BASE_TS_CONTEXT
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _DEFAULT_OCDBT_TS_CONTEXT
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _REMOTE_URL_PREFIXES
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _REMOTE_DRIVER_VALIDATIONS
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _get_kvstore_for_gcs
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _normalize_path
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _OcdbtKvSpecParameters
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _override_ocdbt_kvspec_parameters_for_temporary_metadata
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _build_ocdbt_kvstore_tspec
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _build_non_ocdbt_kvstore_tspec
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _get_backend_ocdbt_target_data_file_size
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _add_ocdbt_write_options
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _build_zarr2_metadata
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _build_zarr3_metadata
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _build_zarr_shard_and_chunk_metadata
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _maybe_add_cast_to_write_spec
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _maybe_add_cast_to_read_spec
from orbax.checkpoint.v0._src.serialization.tensorstore_utils import _get_json_tspec

sys.modules[__name__] = _mod
