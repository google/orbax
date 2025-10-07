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

"""Base class for data processors."""

# pylint: disable=g-importing-member

import abc


class DataProcessor(abc.ABC):
  """An interface for processing data as part of a model export.

  A `DataProcessor` can be used to represent pre-processing or post-processing
  functions. After being "prepared", it provides input/output signatures and an
  `obm.Function` that can be composed with other functions.

  The properties of any DataProcessor class are available only after `prepare()`
  is called.
  If a property is accessed before `prepare()` is called, it will raise a
  RuntimeError.
  """
