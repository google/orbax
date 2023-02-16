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

import re
from typing import Any, Callable, Optional, Tuple, Union

from absl import logging
import flax
from flax import serialization
from flax import traverse_util
import jax
from orbax.checkpoint import utils

PyTree = type(jax.tree_util.tree_structure(None))
ValueTransformFunction = Callable[[PyTree], Any]


@flax.struct.dataclass
class Transform:
  r"""A representation of a transformation applied to pytree keys/values.

  See `apply_transformations` for usage examples. Transform represents an
  operation on a single key/value pair. For example, the following mapping:

  {'a': Transform(original_key='b')}

  This denotes that the original key was named 'b', but we are changing it to
  'a'. A regex can also be used as follows:

  {r'(.*)a(.*)': Transform(original_key=r'\1b\2'}

  This denotes that the key 'b' should be renamed to 'a'. This may apply to
  multiple different keys at different levels of nesting. The '/' character
  denotes a successive level of nesting.

  We also have the following example:

  {'a': Transform(multi_value_fn=lambda kv: kv['b'] * 2)}

  This signifies that the new key 'a' is the old key 'b' multiplied by two.

  original_key: Denotes the original name of the key. Represented as a string
    with '/' denoting successive levels of nesting. If the key corresponding to
    this Transform is a regex, backreferences (such as \1) will be replaced with
    the appropriate matched group in the regex. Note: not needed if
    multi_value_fn is provided.
  use_fallback: if True, takes the value from the fallback tree. If
    `default_to_original=True` in `apply_transformations`, the fallback tree is
    `new_tree`. If `default_to_original=False` in `apply_transformations`, the
    fallback tree is `original_tree`.
  value_fn: A function accepting a single value and returning a single value.
    The value provided as an argument is the value of the transformation key in
    the original PyTree.
  multi_value_fn: A function accepting a PyTree and returning any value. The
    PyTree argument will be the original PyTree, and the function should return
    the value of the key in the new PyTree.
  """
  original_key: Optional[Union[str, Tuple[str]]] = None
  use_fallback: bool = False
  value_fn: Optional[Callable[[Any], Any]] = None
  multi_value_fn: Optional[ValueTransformFunction] = None


def _is_leaf(x):
  if isinstance(x, dict):
    return set(x.keys()) >= {'original_key', 'value_fn', 'multi_value_fn'}
  return False


def _to_transform(x):
  t = serialization.from_state_dict(Transform(), x)
  if isinstance(t.original_key, str) or t.original_key is None:
    return t
  return t.replace(original_key=tuple([v for k, v in t.original_key.items()]))


def has_value_functions(pytree: PyTree) -> bool:
  """Returns True if a PyTree contains any Transform elements with value_fn or multi_value_fn."""
  result = False

  def elem_has_value_functions(transform: Transform):
    nonlocal result
    if transform.value_fn is not None or transform.multi_value_fn is not None:
      result = True

  jax.tree_util.tree_map(
      elem_has_value_functions,
      pytree,
      is_leaf=lambda x: isinstance(x, Transform))
  return result


# TODO(b/233407026) Add additional error checking.
def apply_transformations(original_tree: PyTree,
                          transformations: PyTree,
                          new_tree: PyTree,
                          default_to_original: Optional[bool] = True) -> PyTree:
  r"""Applies transformations to a pytree.

  Also uses `transformations` to provide structure to the output tree.

  Example:

  original_tree = {
    'a': 1,
    'b': {
      'c': 5,
      'd': [0, 1, 2, 3]
    },
    'f': 2,
    'b1': {
      'c': 2,
    },
    'b2': {
      'c': 3,
    },
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
    'c1': Transform(original_key='b/c'),
    'c2': Transform(original_key='b/c'),
    # one to many mapping
    'x': Transform(multi_value_fn=lambda kv: kv['b']['d'][0]),
    'y': Transform(multi_value_fn=lambda kv: kv['b']['d'][1:]),
    # many to one mapping
    'z': Transform(multi_value_fn=lambda kv: kv['a'] * 2 + sum(kv['b']['d'])),
    r'x(\d.*)': Transform(original_key=r'b\1')
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
    # These are just 'b1', 'b2', but renamed to 'x1', 'x2', with all values
    copied over.
    'x1': {
      'c': 2,
    }
    'x2': {
      'c': 3,
    }
  }

  Args:
    original_tree: a PyTree to be transformed.
    transformations: a PyTree of Transform objects.
    new_tree: a PyTree defining the structure of the output. A leaf value is
      only relevant if the key is not present in transformations or
      original_tree.
    default_to_original: If True, the values of keys unspecified in
      transformations will be taken from `original_tree`. If False, they will be
      taken from `new_tree`.

  Returns:
    a transformed PyTree with the structure of `new_tree`
  """
  if not new_tree:
    return {}
  original = utils.to_state_dict(original_tree)
  new = utils.to_state_dict(new_tree)
  # convert transformations to state dict
  transforms = utils.to_state_dict(transformations)

  # Must recover Transform objects, while maintaining state dict structure.
  transforms = jax.tree_util.tree_map(
      _to_transform, transforms, is_leaf=_is_leaf)

  original = traverse_util.flatten_dict(
      original, keep_empty_nodes=True, sep='/')
  new = traverse_util.flatten_dict(new, keep_empty_nodes=True, sep='/')
  transforms = traverse_util.flatten_dict(transforms, sep='/')

  unmatched_new_keys = []

  for key in new:
    transform_found = False
    for transform_key, transform in transforms.items():
      match = re.fullmatch(transform_key, key)
      if match:
        transform_found = True
        if transform.use_fallback:
          if not default_to_original:
            if key not in original:
              raise ValueError(
                  f'{key} not found in origin tree (`use_fallback` requested).')
            new[key] = original[key]
          # else simply retain new[key]
          continue
        if not (transform.multi_value_fn is None or transform.value_fn is None):
          raise ValueError(
              f'Cannot provide both multi_value_fn and value_fn in {transform}')
        if transform.multi_value_fn is None:
          if transform.original_key is None:
            original_key = key
          else:
            original_key = match.expand(transform.original_key)
          if original_key not in original:
            raise ValueError(
                f'Transformation key "{original_key}" not found in origin tree.'
            )
          if transform.value_fn is None:
            value_fn = lambda x: x
          else:
            value_fn = transform.value_fn
          new[key] = value_fn(original[original_key])
        else:
          new[key] = transform.multi_value_fn(original_tree)
    if not transform_found:
      if key in original:
        if default_to_original:
          # carry over directly from original, otherwise use value from new
          new[key] = original[key]
        # if default_to_new, do not carry over key from original
      else:
        unmatched_new_keys.append(key)

  if unmatched_new_keys:
    logging.info('The following keys are not loaded from the original tree '
                 'after applying specified transforms: %s',
                 ', '.join(unmatched_new_keys))

  new = traverse_util.unflatten_dict(new, sep='/')
  return serialization.from_state_dict(new_tree, new)
