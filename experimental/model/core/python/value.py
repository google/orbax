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

"""Values (i.e. non-function objects)."""

# pylint: disable=g-importing-member
from abc import ABC
from dataclasses import dataclass
from typing import Optional

from orbax.experimental.model.core.python import signature
from orbax.experimental.model.core.python import unstructured_data


class Value(ABC):
  pass


@dataclass
class ExternalValue(Value):
  data: unstructured_data.UnstructuredData
  type: Optional[signature.TreeOfTensorSpecs] = None
