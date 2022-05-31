# Copyright 2022 The Orbax Authors.
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

"""Provides utils for transforming PyTrees from one version to another."""

from typing import Any, Callable, Optional, Tuple, Union

import flax
from flax import serialization
from flax import traverse_util
import jax

PyTree = type(jax.tree_structure(None))
ValueTransformFunction = Callable[[PyTree], Any]


@flax.struct.dataclass
class Transform:
  """A representation of a transformation applied to pytree keys/values.

  See `apply_transformations` for usage examples. Transform represents an
  operation on a single key/value pair. For example, the following mapping:

  {'a': Transform(original_key='b')}

  This denotes that the original key was named 'b', but we are changing it to
  'a'.

  Similarly, we have the following example:

  {'a': Transform(multi_value_fn=lambda kv: kv['b'] * 2)}

  This signifies that the new key 'a' is the old key 'b' multiplied by two.

  original_key: Denotes the original name of the key. Represented using a tuple,
    where each successive element represents a nested key in a nested
    dictionary. May also provide a string, which will be interpreted as a tuple
    of length 1 (no nesting). Note: not needed if multi_value_fn is provided.
  in_original: Indicates whether a parameter is expected to be present in the
    saved checkpointransform. Will raise an error if the parameter was expected,
    but is not present.
  value_fn: A function accepting a single value and returning a single value.
    The value provided as an argument is the value of the transformation key in
    the original PyTree.
  multi_value_fn: A function accepting a PyTree and returning any value. The
    PyTree argument will be the original PyTree, and the function should return
    the value of the key in the new PyTree.
  """
  original_key: Optional[Union[str, Tuple[str]]] = None
  in_original: bool = True
  value_fn: Optional[Callable[[Any], Any]] = None
  multi_value_fn: Optional[ValueTransformFunction] = None


def _is_leaf(x):
  if isinstance(x, dict):
    return set(x.keys()) >= {'original_key', 'in_original', 'value_fn'}
  return False


def _to_transform(x):
  t = serialization.from_state_dict(Transform(), x)
  if isinstance(t.original_key, str) or t.original_key is None:
    return t
  return t.replace(original_key=tuple([v for k, v in t.original_key.items()]))


# TODO(b/233406904) Add regex support.
# TODO(b/233407026) Add additional error checking.
def apply_transformations(original_tree: PyTree, transformations: PyTree,
                          new_tree: PyTree) -> PyTree:
  """Applies transformations to a pytree.

  Also uses `transformations` to provide structure to the output tree.

  Example:

  original_tree = {
    'a': 1,
    'b': {
      'c': 5,
      'd': [0, 1, 2, 3]
    },
    'f': 2,
  }
  transformations = {
    'a1': Transform(original_key='a'),  # rename
    'a1': Transform(multi_value_fn=lambda kv: kv['a']),  # another way of doing
    above
    'b': {
      'c': Transform(multi_value_fn=lambda kv: kv['b']['c'] * 2)  # doubled
      original
      # drop b/d
    },
     # Copy original into multiple new keys
    'c1': Transform(original_key=('b', 'c')),
    'c2': Transform(original_key=('b', 'c')),
    # one to many mapping
    'x': Transform(multi_value_fn=lambda kv: kv['b']['d'][0]),
    'y': Transform(multi_value_fn=lambda kv: kv['b']['d'][1:]),
    # many to one mapping
    'z': Transform(multi_value_fn=lambda kv: kv['a'] * 2 + sum(kv['b']['d'])),
  }

  # defines the structure of the result
  new_tree = {
    'a1': ...,
    'a1': ...,
    'b': {
      'c': ...,
    },
    'c1': ...,
    'c2': ...,
    'x': ...,
    'y': ...,
    'z': ...,
    # defined in original_tree and new_tree, but not in transforms. Value
    carried over from original_tree.
    'f': ...,
    # This value matters since it is not present in original_tree or
    transformations, so the value here will simply be preserved in the result.
    'g': 5,
  }

  Args:
    original_tree: a PyTree to be transformed.
    transformations: a PyTree of Transform objects that should have the same
      structure as the desired output tree.
    new_tree: a PyTree defining the structure of the output. A leaf value is
      only relevant if the key is not present in transformations or
      original_tree.

  Returns:
    a transformed PyTree with the structure of `new_tree`
  """
  if not new_tree:
    return {}
  original = serialization.to_state_dict(original_tree)
  new = serialization.to_state_dict(new_tree)
  # convert transformations to structure matching original
  transforms = serialization.to_state_dict(transformations)

  # Must recover Transform objects, while maintaining state dict structure.
  transforms = jax.tree_map(_to_transform, transforms, is_leaf=_is_leaf)

  original = traverse_util.flatten_dict(original, keep_empty_nodes=True)
  new = traverse_util.flatten_dict(new, keep_empty_nodes=True)
  transforms = traverse_util.flatten_dict(transforms)

  for key in new:
    if key in transforms:
      transform = transforms[key]
      if not transform.in_original:
        continue  # do not override existing value of key in new
      if not (transform.multi_value_fn is None or transform.value_fn is None):
        raise ValueError(
            f'Cannot provide both multi_value_fn and value_fn in {transform}')
      if transform.multi_value_fn is None:
        if transform.original_key is None:
          original_key = key
        else:
          original_key = transform.original_key
        if isinstance(original_key, str):
          original_key = (original_key,)
        if original_key not in original:
          raise ValueError(
              f'Transformation key {original_key} not found in origin tree (in_original=True)'
          )
        if transform.value_fn is None:
          value_fn = lambda x: x
        else:
          value_fn = transform.value_fn
        new[key] = value_fn(original[original_key])
      else:
        new[key] = transform.multi_value_fn(original_tree)
    else:
      # carry over directly from original, otherwise use value from new
      if key in original:
        new[key] = original[key]

  new = traverse_util.unflatten_dict(new)
  return serialization.from_state_dict(new_tree, new)
