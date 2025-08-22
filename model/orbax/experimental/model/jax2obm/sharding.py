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

"""Sharding related utils."""
from typing import Any

import jax
from jax.extend import sharding as jex_sharding
from orbax.experimental.model import core as obm

def hlo_sharding_to_op_sharding(
    hlo_sharding: Any | None,
) -> obm.OpSharding | None:
  """Converts `HloSharding` to proto `OpSharding`.

  `to_proto` is defined
  [here](

  Args:
    hlo_sharding: an `HloSharding`.

  Returns:
    An `OpSharding` proto.
  """
  if hlo_sharding is None:
    return None
  output = obm.OpSharding()
  # Note: `to_proto(self) -> xla_extension.OpSharding` so we need to serialize
  # and then parse.
  # TODO(b/364954510): See if
  # help us share messages between Python and C++.
  output.ParseFromString(
      jex_sharding.get_serialized_proto_from_hlo_sharding(hlo_sharding)
  )
  return output
