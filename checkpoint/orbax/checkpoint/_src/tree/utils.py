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

"""Tree utilities."""

from typing import Any, Callable, Mapping, Optional, Tuple, TypeVar, Union

import jax
import jax.tree_util as jtu
from orbax.checkpoint._src.arrays import abstract_arrays


T = TypeVar('T')

PyTree = Any
# This won't help the type checker but at least allows us to use types to
# document things like `PyTreeOf[ArrayDesc]`.
PyTreeOf = PyTree | T
PyTreeKey = (
    jax.tree_util.SequenceKey
    | jax.tree_util.DictKey
    | jax.tree_util.GetAttrKey
    | jax.tree_util.FlattenedIndexKey
)
PyTreePath = tuple[PyTreeKey, ...]

# These types are useful for specifying PyTree paths using user-friendly types,
# rather than PyTreeKeys. For example, we may specify dictionary keys as
# strings, and sequence indices as integers.
TreeKey = str | int
TreePath = tuple[TreeKey, ...]

to_shape_dtype_struct = abstract_arrays.to_shape_dtype_struct


def isinstance_of_namedtuple(value: Any) -> bool:
  """Determines if the `value` is a NamedTuple."""
  return isinstance(value, tuple) and hasattr(value, '_fields')


def issubclass_of_namedtuple(t: Any) -> bool:
  """Determines if the `type` is a subclass of NamedTuple."""
  return issubclass(t, tuple) and hasattr(t, '_fields')


def is_empty_node(x: Any) -> bool:
  try:
    children, _ = jax._src.tree_util.flatten_one_level(x)  # pylint: disable=protected-access
  except ValueError:
    return False  # non-empty leaf, otherwise flatten would return self.
  return not children


def is_leaf_node(t: Any) -> bool:
  """The default value of the `is_leaf` predicate."""
  return jax.tree_util.all_leaves([t])


def is_leaf_node_or_none(t: Any) -> bool:
  return is_leaf_node(t) or t is None


def is_jax_internal_node(x: Any) -> bool:
  return not is_leaf_node(x)


def _internal_node_as_dict(x: Any) -> Mapping[str, Any]:
  keys_and_children, _ = tree_flatten_with_path_one_level(x)
  return {jax.tree_util.keystr(k): v for k, v in keys_and_children}


def is_empty_or_leaf(x: Any) -> bool:
  try:
    children, _ = jax._src.tree_util.flatten_one_level(x)  # pylint: disable=protected-access
  except ValueError:
    return True  # Cannot flatten x; means it must be a leaf
  return not children


def get_key_name(key: Any) -> Union[int, str]:
  """Returns the name of a JAX Key."""
  if isinstance(key, jax.tree_util.SequenceKey):
    return key.idx
  elif isinstance(key, jax.tree_util.DictKey):
    return key.key
  elif isinstance(key, jax.tree_util.GetAttrKey):
    return key.name
  elif isinstance(key, jax.tree_util.FlattenedIndexKey):
    return key.key
  else:
    raise ValueError(f'Unsupported KeyEntry: {type(key)}: "{key}"')


def tuple_path_from_keypath(keypath: Tuple[Any, ...]) -> Tuple[str, ...]:
  """Converts JAX keypath tuple (from tree_map_with_path) to string tuple."""
  return tuple([str(get_key_name(k)) for k in keypath])


def is_dict_key(key) -> bool:
  return isinstance(key, (jax.tree_util.DictKey, jax.tree_util.GetAttrKey))


def is_sequence_key(key) -> bool:
  return isinstance(
      key, (jax.tree_util.FlattenedIndexKey, jax.tree_util.SequenceKey)
  )


def _raise_unsupported_key_error(key):
  raise ValueError(f'Unsupported KeyEntry: {key}.')


def _extend_list(ls, idx, nextvalue):
  assert idx <= len(ls)
  if idx == len(ls):
    ls.append(nextvalue)
  return ls


def from_flattened_with_keypath(
    flat_with_keys: list[tuple[tuple[Any, ...], Any]],
) -> PyTree:
  """Returns a tree for the given list of (KeyPath, value) pairs.

  IMPORTANT: The returned tree replaces tuple container nodes with list nodes,
  even though the input KeyPath had originated from a tuple.

  IMPORTANT: The returned tree replaces NamedTuple container nodes with dict
  nodes, even though the input KeyPath had originated from a NamedTuple.

  Args:
    flat_with_keys: A list of pair of Keypath and values.
  """
  if not flat_with_keys:
    raise ValueError(
        'Unable to uniquely reconstruct tree from empty flattened list '
        '(it could be {} or []).'
    )
  first_el = flat_with_keys[0]
  assert first_el, f'Invalid data format: expected a pair, got {first_el=}'
  if not first_el[0]:
    # The tree is a single element (the path is empty), just return it.
    return first_el[1]
  # Accesses the first path element (arbitrary), first tuple element
  # (keypath tuple), first key in keypath (outermost key in the PyTree).
  outerkey = first_el[0][0]
  if is_dict_key(outerkey):
    result = {}
  elif is_sequence_key(outerkey):
    result = []
  else:
    result = None
    _raise_unsupported_key_error(outerkey)

  for keypath, value in flat_with_keys:
    subtree = result
    for i, key in enumerate(keypath):
      if i == 0:
        assert isinstance(key, type(outerkey))
      if i == len(keypath) - 1:
        if is_dict_key(key):
          assert isinstance(subtree, dict)
          subtree[get_key_name(key)] = value
        elif is_sequence_key(key):
          assert isinstance(subtree, list)
          idx = get_key_name(key)
          subtree = _extend_list(subtree, idx, value)
      else:
        nextkey = keypath[i + 1]
        if is_dict_key(nextkey):
          nextvalue = {}
        elif is_sequence_key(nextkey):
          nextvalue = []
        else:
          nextvalue = None
          _raise_unsupported_key_error(nextkey)

        if is_dict_key(key):
          assert isinstance(subtree, dict)
          name = get_key_name(key)
          if name not in subtree:
            subtree[name] = nextvalue
          subtree = subtree[name]
        elif is_sequence_key(key):
          assert isinstance(subtree, list)
          idx = get_key_name(key)
          subtree = _extend_list(subtree, idx, nextvalue)
          subtree = subtree[idx]
        else:
          _raise_unsupported_key_error(key)

  return result


def serialize_tree(tree: PyTree, keep_empty_nodes: bool = False) -> PyTree:
  """Transforms a PyTree to a serializable format.

  IMPORTANT: The returned tree replaces tuple container nodes with list nodes.

  IMPORTANT: The returned tree replaces NamedTuple container nodes with dict
  nodes.


  Args:
    tree: The tree to serialize, if tree is empty and keep_empty_nodes is False,
      an error is raised as there is no valid representation.
    keep_empty_nodes: If true, does not filter out empty nodes.

  Returns:
    The serialized PyTree.
  """
  flat_with_keys, _ = jax.tree_util.tree_flatten_with_path(
      tree, is_leaf=is_empty_or_leaf if keep_empty_nodes else None
  )
  return from_flattened_with_keypath(flat_with_keys)


def deserialize_tree(
    serialized: PyTree, target: PyTree, keep_empty_nodes: bool = False
) -> PyTree:
  """Deserializes a PyTree to the same structure as `target`."""

  def _reconstruct_from_keypath(keypath, _):
    result = serialized
    for key in keypath:
      key_name = get_key_name(key)
      if isinstance(key, jax.tree_util.GetAttrKey) and isinstance_of_namedtuple(
          result
      ):
        result = getattr(result, key_name)
      else:
        # Special case to support Pax.
        if not isinstance(result, (list, tuple)) and key_name not in result:
          key_name = str(key_name)
        result = result[key_name]
    return result

  return jax.tree_util.tree_map_with_path(
      _reconstruct_from_keypath,
      target,
      is_leaf=is_empty_or_leaf if keep_empty_nodes else None,
  )


def is_flat_dict(tree: PyTree) -> bool:
  """Returns True if the tree is a flat dict."""
  return isinstance(tree, dict) and all(is_leaf_node(v) for v in tree.values())


def to_flat_dict(
    tree: PyTree,
    sep: Optional[str] = None,
    keep_empty_nodes: bool = False,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> PyTree:
  """Converts a tree into a flattened dictionary.

  The nested keys are flattened to a tuple.

  Example::

    tree = {'foo': 1, 'bar': {'a': 2, 'b': {}}}
    to_flat_dict(tree)
    {
      ('foo',): 1,
      ('bar', 'a'): 2,
    }

  Args:
    tree: A PyTree to be flattened.
    sep: If provided, keys will be returned as `sep`-separated strings.
      Otherwise, keys are returned as tuples.
    keep_empty_nodes: If True, empty nodes are not filtered out.
    is_leaf: If provided, a function that returns True if a value is a leaf.
      Overrides `keep_empty_nodes` if that is also provided.

  Returns:
    A flattened dictionary and the tree structure.
  """
  is_leaf = is_leaf or (is_empty_or_leaf if keep_empty_nodes else None)
  flat_with_keys, _ = jax.tree_util.tree_flatten_with_path(
      tree, is_leaf=is_leaf
  )
  flat_dict = {tuple_path_from_keypath(k): v for k, v in flat_with_keys}
  if sep is not None:
    flat_dict = {sep.join(k): v for k, v in flat_dict.items()}
  return flat_dict


def from_flat_dict(
    flat_dict: PyTree,
    target: Optional[PyTree] = None,
    sep: Optional[str] = None,
) -> PyTree:
  """Reconstructs the original tree object from a flattened dictionary.

  Args:
    flat_dict: A dictionary conforming to the return value of `to_flat_dict`.
    target: A reference PyTree. The returned value will conform to this
      structure. If not provided, an unflattened dict will be returned with the
      inferred structure of the original tree, without necessarily matching it
      exactly. Note, if not provided, the keys in `flat_dict` need to match
      `sep`.
    sep: separator used for nested keys in `flat_dict`.

  Returns:
    A dict matching the structure of `tree` with the values of `flat_dict`.
  """
  if target is None:
    result = {}
    for k, v in flat_dict.items():
      subtree = result
      if sep is None:
        assert isinstance(k, tuple)
        tuple_k = k
      else:
        tuple_k = tuple(k.split(sep))
      for i, name in enumerate(tuple_k):
        if i == len(tuple_k) - 1:
          assert name not in subtree
          subtree[name] = v
        else:
          if name not in subtree:
            subtree[name] = {}
          subtree = subtree[name]
    return result
  else:
    flat_structure = to_flat_dict(target, sep=sep)
    # Ensure that the ordering of `flat_dict` keys matches that of `target`.
    # Necessary for later unflattening.
    flat_dict = {k: flat_dict[k] for k in flat_structure.keys()}
    return jax.tree.unflatten(jax.tree.structure(target), flat_dict.values())


def str_keypath(keypath: Tuple[Any, ...]) -> Tuple[str, ...]:
  """Returns the parameter name for a keypath."""
  return tuple([str(get_key_name(k)) for k in keypath])


def param_name_from_keypath(keypath: Tuple[Any, ...]) -> str:
  """Returns the parameter name for a keypath."""
  return '.'.join(str_keypath(keypath))


def get_param_names(
    item: PyTree, *, include_empty_nodes: bool = True
) -> PyTree:
  """Gets parameter names for PyTree elements."""
  is_leaf = is_empty_or_leaf if include_empty_nodes else None
  return jax.tree_util.tree_map_with_path(
      lambda kp, _: param_name_from_keypath(kp),
      item,
      is_leaf=is_leaf,
  )


def tree_flatten_with_path_one_level(
    x: Any,
) -> tuple[list[tuple[PyTreePath, Any]], jax.tree_util.PyTreeDef]:
  return jax.tree_util.tree_flatten_with_path(x, is_leaf=lambda y: y is not x)


def look_up_pytree_key(pytree: PyTreeOf[T], key: PyTreeKey) -> PyTreeOf[T]:
  """Resolves a single PyTree key (i.e. one element of a PyTreePath).

  Args:
    pytree: The PyTree node with respect to which to resolve the key.
    key: The key to resolve.

  Returns:
    An element of the given collection or custom PyTree node.

  Raises:
    KeyError: If the given key is not present in the given node.
  """
  match key:
    case jtu.SequenceKey(idx=k) | jtu.DictKey(key=k):
      try:
        return pytree[k]
      except IndexError as e:
        raise KeyError(f'Key {key!r} not found in {pytree!r}') from e
    case jtu.FlattenedIndexKey(key=k):
      leaves, _ = tree_flatten_with_path_one_level(pytree)
      try:
        idx_and_leaf = leaves[k]
      except IndexError as e:
        raise KeyError(f'Key {key!r} not found in {pytree!r}') from e
      else:
        return idx_and_leaf[1]
    case jtu.GetAttrKey(name=k):
      try:
        return getattr(pytree, k)
      except AttributeError as e:
        raise KeyError(f'Key {key!r} not found in {pytree!r}') from e
  raise KeyError(f'Key {key!r} not found in {pytree!r}')


def _pytree_key_as_tree_key(key: PyTreeKey) -> TreeKey:
  match key:
    case (jtu.SequenceKey(idx=k)
          | jtu.DictKey(key=k)
          | jtu.FlattenedIndexKey(key=k)
          | jtu.GetAttrKey(name=k)):
      return k  # pytype: disable=bad-return-type
  raise KeyError(f'Cannot convert unexpected PyTreeKey to TreeKey: {key!r}')


def pytree_path_as_tree_path(path: PyTreePath) -> TreePath:
  """Converts a PyTreePath to a legacy tree path.

  Each PyTreeKey in a PyTreePath has a type specific to the type into which it
  indexes. Legacy tree keys are either strings or integers.

  Args:
    path: The path to be converted.

  Returns:
    A legacy representation of the path.
  """
  return tuple(_pytree_key_as_tree_key(k) for k in path)


def select_by_pytree_path(
    tree: PyTreeOf[T],
    path: PyTreePath,
) -> PyTreeOf[T]:
  """Extracts a subtree from a PyTree, using a PyTreePath to describe the path.

  Args:
    tree: The PyTree from which to extract the subtree.
    path: A tuple of PyTreeKeys specifying the path to the subtree.

  Returns:
    The subtree.
  """
  match path:
    case (key, *rest):
      try:
        return select_by_pytree_path(look_up_pytree_key(tree, key), tuple(rest))
      except KeyError as e:
        raise ValueError(f'Path {path} does not exist in {tree=}.') from e
    case ():
      return tree


def select_by_tree_path(
    tree: PyTreeOf[T],
    path: TreePath,
) -> PyTreeOf[T]:
  """Extracts a subtree from a PyTree, using a tree path to describe the path.

  Args:
    tree: The PyTree from which to extract the subtree.
    path: A tuple of keys (str | int) specifying the path to the subtree.

  Returns:
    The subtree.
  """
  match path:
    case (key, *rest):
      keys_and_children, _ = tree_flatten_with_path_one_level(tree)
      for (pytree_key,), child in keys_and_children:
        if _pytree_key_as_tree_key(pytree_key) == key:
          return select_by_tree_path(child, tuple(rest))
      else:
        raise ValueError(f'Path {path} does not exist in {tree=}.')
    case ():
      return tree
