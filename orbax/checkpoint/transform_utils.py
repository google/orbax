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

  {'a': Transform(origin_name='b')}

  This denotes that the original key was named 'b', but we are changing it to
  'a'.

  Similarly, we have the following example:

  {'a': Transform(value_fn=lambda kv: kv['b'] * 2)}

  This signifies that the new key 'a' is the old key 'b' multiplied by two.

  origin_name: Denotes the original name of the key. Represented using a tuple,
    where each successive element represents a nested key in a nested
    dictionary. May also provide a string, which will be interpreted as a tuple
    of length 1 (no nesting). Note: not needed if value_fn is provided.
  in_checkpoint: Indicates whether a parameter is expected to be present in the
    saved checkpoint. Will raise an error if the parameter was expected, but is
    not present.
  value_fn: A function accepting a PyTree and returning any value. The PyTree
    argument will be the original PyTree, and the function should return the
    value of the key in the new PyTree.
  """
  origin_name: Optional[Union[str, Tuple[str]]] = None
  # TODO(cpgaffney) consider supporting custom init fn for new keys.
  in_checkpoint: bool = True
  value_fn: Optional[ValueTransformFunction] = None


def apply_transformations(original_tree: PyTree,
                          transformations: PyTree) -> PyTree:
  """Applies transformations to a pytree.

  Also uses `transformations` to provide structure to the output tree.

  Example:

  original_tree = {
    'a': 1,
    'b': {
      'c': 5,
      'd': [0, 1, 2, 3]
    },
  }
  transforms = {
    'a1': Transform(origin_name='a'),  # rename
    'a1': Transform(value_fn=lambda kv: kv['a']),  # another way of doing above
    'b': {
      'c': Transform(value_fn=lambda kv: kv['b']['c'] * 2)  # doubled original
      # drop b/d
    },
     # Copy original into multiple new keys
    'c1': Transform(origin_name=('b', 'c')),
    'c2': Transform(origin_name=('b', 'c')),
    # one to many mapping
    'x': Transform(value_fn=lambda kv: kv['b']['d'][0]),
    'y': Transform(value_fn=lambda kv: kv['b']['d'][1:]),
    # many to one mapping
    'z': Transform(value_fn=lambda kv: kv['a'] * 2 + sum(kv['b']['d'])),
    # create a new key not in original
    'new': Transform(in_checkpoint=False),
  }

  Args:
    original_tree: a PyTree to be transformed.
    transformations: a PyTree of Transform objects that should have the same
      structure as the desired output tree.

  Returns:
    a transformed PyTree with the structure of `transformations`
  """
  original = serialization.to_state_dict(original_tree)
  # convert transformations to structure matching original
  transforms = serialization.to_state_dict(transformations)

  def is_leaf(x):
    if isinstance(x, dict):
      return set(x.keys()) >= {'origin_name', 'in_checkpoint', 'value_fn'}
    return False

  def to_transform(x):
    t = serialization.from_state_dict(Transform(), x)
    if isinstance(t.origin_name, str) or t.origin_name is None:
      return t
    return t.replace(origin_name=tuple([v for k, v in t.origin_name.items()]))

  # Must recover Transform objects, while maintaining state dict structure.
  transforms = jax.tree_map(to_transform, transforms, is_leaf=is_leaf)

  original = traverse_util.flatten_dict(original)
  transforms = traverse_util.flatten_dict(transforms)

  new = {}
  for k, t in transforms.items():
    if t.in_checkpoint:
      if t.value_fn is None:
        origin_name = k
        if t.origin_name is not None:
          origin_name = t.origin_name
        if isinstance(origin_name, str):
          origin_name = (origin_name,)
        if origin_name not in original:
          raise ValueError(
              f'Transformation key {origin_name} not found in origin tree (in_checkpoint=True)'
          )
        new[k] = original[origin_name]
      else:
        new[k] = t.value_fn(original_tree)
    else:
      new[k] = None

  new = traverse_util.unflatten_dict(new)
  structure = jax.tree_map(
      lambda x: object(),
      transformations,
      is_leaf=lambda x: isinstance(x, Transform))
  return serialization.from_state_dict(structure, new)
