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

"""Utilities for working with arrays."""

# pylint: disable=g-importing-member, g-bad-import-order, unused-import, g-multiple-import

from orbax.checkpoint._src.arrays.abstract_arrays import to_shape_dtype_struct

from orbax.checkpoint._src.arrays.types import (
    Index,
    Shape,
)
