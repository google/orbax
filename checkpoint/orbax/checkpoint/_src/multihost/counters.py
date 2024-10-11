# Copyright 2024 The Orbax Authors.
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

"""Counters used to make barrier names unique.

Do not add any functions unless they conform precisely to the existing pattern.
"""

import itertools


_async_save_counter = itertools.count()
_composite_save_counter = itertools.count()
_tmp_directory_counter = itertools.count()
_model_surgery_loading_counter = itertools.count()

_global_max_broadcast_counter = itertools.count()
_local_all_steps_broadcast_counter = itertools.count()
_find_complete_slice_broadcast_counter = itertools.count()


def async_save_counter() -> str:
  return str(next(_async_save_counter))


def composite_save_counter() -> str:
  return str(next(_composite_save_counter))


def tmp_directory_counter() -> str:
  return str(next(_tmp_directory_counter))


def model_surgery_loading_counter() -> str:
  return str(next(_model_surgery_loading_counter))


# Emergency checkpointing counters.


def global_max_broadcast_counter() -> str:
  return str(next(_global_max_broadcast_counter))


def local_all_steps_broadcast_counter() -> str:
  return str(next(_local_all_steps_broadcast_counter))


def find_complete_slice_broadcast_counter() -> str:
  return str(next(_find_complete_slice_broadcast_counter))
