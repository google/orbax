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

"""Nesting utilities for model surgery."""

import logging
from typing import Any

from orbax.checkpoint._src.tree import utils as tree_utils
from orbax.checkpoint.experimental.model_surgery.transformations import types

Transformation = types.Transformation


def unflatten(
    separator: str = '.',
    *,
    inplace: bool = False,
    target: Any = None,
) -> Transformation:
  """Converts a flat dictionary with separated keys into a nested PyTree.

  Example:
      params = {
          'linear1.kernel.qvalue': arr1,
          'linear1.kernel.scale': arr2,
      }
      transform = unflatten()
      result = transform(params)
      # result = {
      #     'linear1': {
      #         'kernel': {
      #             'qvalue': arr1,
      #             'scale': arr2,
      #         }
      #     }
      # }

  Args:
      separator: The string used to separate keys.
      inplace: If True, deletes matched keys from input params to save memory.
        Requires input params to be a dict.
      target: A reference PyTree. If provided, the returned value will conform
        to this structure, and keys not in the target will be filtered out.

  Returns:
      A Transformation function.
  """

  def transform(
      *params: types.PyTreeOf[Any],
  ) -> types.PyTreeOf[Any]:
    assert (
        len(params) == 1
    ), 'Can only unflatten parameters in a single parameter structure.'
    p = params[0]
    if target is not None:
      flat_target = tree_utils.to_flat_dict(target, sep=separator)
      flat_p = p if isinstance(p, dict) else dict(p)
      missing_keys = set(flat_target.keys()) - set(flat_p.keys())
      if missing_keys:
        logging.warning(
            'The following %d keys were missing in the checkpoint and will'
            ' retain their default values: %s',
            len(missing_keys),
            list(missing_keys)[:10],
        )
      if inplace and isinstance(p, dict):
        for k, v in flat_target.items():
          if k not in p:
            p[k] = v
      else:
        p = dict(p)
        for k, v in flat_target.items():
          if k not in p:
            p[k] = v
    return tree_utils.from_flat_dict(
        p, target=target, sep=separator, inplace=inplace
    )

  return transform
