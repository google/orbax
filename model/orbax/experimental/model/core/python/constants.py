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

"""SavedModel format constants."""

# SavedModel assets directory.
ASSETS_DIRECTORY = 'assets'

# SavedModel assets.extra directory.
EXTRA_ASSETS_DIRECTORY = 'assets.extra'

# SavedModel assets key for graph collection-def.
ASSETS_KEY = 'saved_model_assets'

# SavedModel proto filename prefix.
SAVED_MODEL_FILENAME_PREFIX = 'saved_model'
# SavedModel proto filename.
SAVED_MODEL_FILENAME_PB = 'saved_model.pb'

# SavedModel text format proto filename.
SAVED_MODEL_FILENAME_PBTXT = 'saved_model.pbtxt'

# Directory in which to save the SavedModel variables.
VARIABLES_DIRECTORY = 'variables'

# SavedModel variables filename.
VARIABLES_FILENAME = 'variables'

# SavedModel SignatureDef keys for the initialization and train ops. Used in
# V2 SavedModels.
INIT_OP_SIGNATURE_KEY = '__saved_model_init_op'

# Filename for the FingerprintDef protocol buffer.
FINGERPRINT_FILENAME = 'fingerprint.pb'

# Needs to be identical to `tensorflow::kSavedModelTagServe`.
SERVE_TAG = 'serve'

# Supplemental info key for JAX-specific info.
JAX_SPECIFIC_INFO = 'jax_specific_info'
