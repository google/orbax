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

"""Async wrappers for path operations."""

import asyncio
from orbax.checkpoint.experimental.v1._src.path import types as path_types


async def exists(path: path_types.Path) -> bool:
  return await asyncio.to_thread(path.exists)


async def rmtree(path: path_types.Path) -> None:
  return await asyncio.to_thread(path.rmtree)
