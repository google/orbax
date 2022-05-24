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

  {'a': Transform(value_fn=lambda kv: kv['b'] * 2)}

  This signifies that the new key 'a' is the old key 'b' multiplied by two.

  original_key: Denotes the original name of the key. Represented using a tuple,
    where each successive element represents a nested key in a nested
    dictionary. May also provide a string, which will be interpreted as a tuple
    of length 1 (no nesting). Note: not needed if value_fn is provided.
  in_checkpoint: Indicates whether a parameter is expected to be present in the
    saved checkpointransform. Will raise an error if the parameter was expected,
    but is not present.
  init_value: Can only be used in conjunction with `in_checkpoint`. If
    `in_checkpoint` is False, will use the value provided by `init_value`.
  value_fn: A function accepting a PyTree and returning any value. The PyTree
    argument will be the original PyTree, and the function should return the
    value of the key in the new PyTree.
  """
  original_key: Optional[Union[str, Tuple[str]]] = None
  in_checkpoint: bool = True
  init_value: Optional[Any] = None
  value_fn: Optional[ValueTransformFunction] = None


def _is_leaf(x):
  if isinstance(x, dict):
    return set(x.keys()) >= {'original_key', 'in_checkpoint', 'value_fn'}
  return False


def _to_transform(x):
  t = serialization.from_state_dict(Transform(), x)
  if isinstance(t.original_key, str) or t.original_key is None:
    return t
  return t.replace(
      original_key=tuple([v for k, v in t.original_key.items()]))


def construct_transformations_from_fallback(original_tree: PyTree,
                                            fallback_tree: PyTree) -> PyTree:
  """Constructs a tree of transformations matching `fallback_tree`.

  Given an `original_tree` and a `fallback_tree` with some keys that are not
  present in the original, constructs a PyTree of transformations matching
  `fallback_tree` where the values are `Transform` objects. The resulting
  transformations PyTree can be used with apply_transformations to initialize
  elements present in `fallback_tree`.

  For example:
    original_tree: {
      'a': 0,
      'b': 1,
    }
    fallback_tree: {
      'a': 100,
      'b': 101,
      'c': 102,
    }
    returns: {
      'a': Transform(),
      'b': Transform(),
      'c': Transform(in_checkpoint=False, init_value=102),
    }

  In a typical example, `original_tree` is restored from an existing checkpoint
  with no modification, while `fallback_tree` is a tree with additional keys
  with randomly initialized values.

  Args:
    original_tree: a PyTree representing the original tree that must be
      transformed.
    fallback_tree: a PyTree with additional values that may be used to
      initialize values not present in the original.

  Returns:
    A PyTree of `Transform`.
  """
  original = traverse_util.flatten_dict(
      serialization.to_state_dict(original_tree), keep_empty_nodes=True)
  fallback = traverse_util.flatten_dict(
      serialization.to_state_dict(fallback_tree), keep_empty_nodes=True)

  transforms = {}
  for k, v in fallback.items():
    if isinstance(v, type(traverse_util.empty_node)):
      transforms[k] = v
    elif k not in original:
      transforms[k] = Transform(in_checkpoint=False, init_value=v)
    else:
      transforms[k] = Transform()

  return serialization.from_state_dict(fallback_tree,
                                       traverse_util.unflatten_dict(transforms))


# TODO(b/233406904) Add regex support.
# TODO(b/233407026) Add additional error checking.
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
    'a1': Transform(original_key='a'),  # rename
    'a1': Transform(value_fn=lambda kv: kv['a']),  # another way of doing above
    'b': {
      'c': Transform(value_fn=lambda kv: kv['b']['c'] * 2)  # doubled original
      # drop b/d
    },
     # Copy original into multiple new keys
    'c1': Transform(original_key=('b', 'c')),
    'c2': Transform(original_key=('b', 'c')),
    # one to many mapping
    'x': Transform(value_fn=lambda kv: kv['b']['d'][0]),
    'y': Transform(value_fn=lambda kv: kv['b']['d'][1:]),
    # many to one mapping
    'z': Transform(value_fn=lambda kv: kv['a'] * 2 + sum(kv['b']['d'])),
    # create a new key not in original
    'new': Transform(in_checkpoint=False, init_value=[0, 1, 2]),
  }

  Args:
    original_tree: a PyTree to be transformed.
    transformations: a PyTree of Transform objects that should have the same
      structure as the desired output tree.

  Returns:
    a transformed PyTree with the structure of `transformations`
  """
  if not transformations:
    return {}
  original = serialization.to_state_dict(original_tree)
  # convert transformations to structure matching original
  transforms = serialization.to_state_dict(transformations)

  # Must recover Transform objects, while maintaining state dict structure.
  transforms = jax.tree_map(_to_transform, transforms, is_leaf=_is_leaf)

  original = traverse_util.flatten_dict(original, keep_empty_nodes=True)
  transforms = traverse_util.flatten_dict(transforms, keep_empty_nodes=True)

  new = {}
  for key, transform in transforms.items():
    if isinstance(transform, type(traverse_util.empty_node)):
      new[key] = transform
    elif transform.in_checkpoint:
      if transform.value_fn is None:
        original_key = key
        if transform.original_key is not None:
          original_key = transform.original_key
        if isinstance(original_key, str):
          original_key = (original_key,)
        if original_key not in original:
          raise ValueError(
              f'Transformation key {original_key} not found in origin tree',
              '(in_checkpoint=True)')
        new[key] = original[original_key]
      else:
        new[key] = transform.value_fn(original_tree)
    else:
      new[key] = transform.init_value

  new = traverse_util.unflatten_dict(new)
  structure = jax.tree_map(
      lambda x: object(),
      transformations,
      is_leaf=lambda x: isinstance(x, Transform))
  return serialization.from_state_dict(structure, new)
