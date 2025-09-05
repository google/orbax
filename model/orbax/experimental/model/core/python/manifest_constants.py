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

"""Manifest model format constants."""

MANIFEST_VERSION_FILENAME = 'orbax_model_version.txt'

# The file path of the manifest proto file
MANIFEST_FILE_PATH_KEY = 'manifest_file_path'
# TODO(b/439870164): Update the `MANIFEST_FILENAME` to be `MANIFEST_FILE_PATH`
# and treat it as a configurable path
MANIFEST_FILENAME = 'manifest.pb'

# The version of the manifest
VERSION_KEY = 'version'
MANIFEST_VERSION = '0.0.1'

# The mime type of the manifest proto file
MIME_TYPE_KEY = 'mime_type'
MANIFEST_MIME_TYPE = 'application/protobuf; type=orbax_model_manifest.Manifest'
