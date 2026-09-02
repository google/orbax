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

"""Interface to retrieve the current step number.

The current step number can be retrieved in various ways depending on the
library/framework in use. Define the interface that the hang detector will
use to retrieve the current step number.
"""

import abc


class StepNumber(abc.ABC):

  @abc.abstractmethod
  def get(self) -> int:
    """Return the current step number or 0 if training has not started."""
    return 0
