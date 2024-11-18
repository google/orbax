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

"""Storage for PyTree checkpoint metadata with rich types."""

from __future__ import annotations

import collections
import dataclasses
import functools
from typing import Any, Iterable, Mapping, Sequence, Type, TypeAlias

from orbax.checkpoint._src.metadata import tree_base
from orbax.checkpoint._src.serialization import type_handlers
import simplejson

PyTree: TypeAlias = Any


@functools.lru_cache()
def _new_namedtuple_type(
    module_name: str,
    class_name: str,
    fields: Sequence[str],
) -> Type[tuple[Any, ...]]:
  # TODO: b/365169723 - Return concrete NamedTuple if available in given module.
  # Fix module name to be prefixed to the class name.
  fixed_module_name = module_name.replace('.', '_')
  arity = len(fields)
  unique_class_name = f'{fixed_module_name}_{class_name}_{arity}'
  return collections.namedtuple(unique_class_name, fields)


def _create_namedtuple(
    *, module_name: str, class_name: str, attrs: Iterable[tuple[str, Any]]
) -> tuple[Any, ...]:
  ks, vs = [*zip(*attrs)] or ((), ())
  result = _new_namedtuple_type(module_name, class_name, ks)(*vs)
  return result


def _module_and_class_name(cls) -> tuple[str, str]:
  """Returns the module and class name of the given class instance."""
  return cls.__module__, cls.__qualname__


def _value_metadata_tree_for_json_dumps(obj: Any) -> Any:
  """Callback for `simplejson.dumps` to convert a PyTree to JSON object."""
  if dataclasses.is_dataclass(obj):
    return dict(
        category='dataclass',
        clazz=_module_and_class_name(obj.__class__),
        data={
            field.name: _value_metadata_tree_for_json_dumps(
                getattr(obj, field.name)
            )
            for field in dataclasses.fields(obj)
        },
    )

  if type_handlers.isinstance_of_namedtuple(obj):
    return dict(
        category='namedtuple',
        clazz=_module_and_class_name(obj.__class__),
        entries=[
            dict(key=k, value=_value_metadata_tree_for_json_dumps(v))
            for k, v in zip(obj._fields, obj)
        ],
    )

  if isinstance(obj, tuple):
    return dict(
        category='custom',
        clazz='tuple',
        entries=[_value_metadata_tree_for_json_dumps(e) for e in obj],
    )

  return obj


def _value_metadata_tree_for_json_loads(obj):
  """Callback for `simplejson.loads` to convert JSON object to a PyTree."""
  if not isinstance(obj, Mapping):
    return obj

  if 'category' in obj:
    if obj['category'] == 'dataclass':
      class_name = obj['clazz'][1]
      if class_name == tree_base.ValueMetadataEntry.__name__:
        return tree_base.ValueMetadataEntry(**obj['data'])
      else:
        return {
            k: _value_metadata_tree_for_json_loads(v)
            for k, v in obj['data'].items()
        }

    if obj['category'] == 'namedtuple':
      return _create_namedtuple(
          module_name=obj['clazz'][0],
          class_name=obj['clazz'][1],
          attrs=[
              (
                  e['key'],
                  _value_metadata_tree_for_json_loads(e['value']),
              )
              for e in obj['entries']
          ],
      )

    if obj['category'] == 'custom':
      if obj['clazz'] == 'tuple':
        return tuple(
            [(_value_metadata_tree_for_json_loads(v)) for v in obj['entries']]
        )
      else:
        raise ValueError(
            f'Unsupported custom object in JSON deserialization: {obj}'
        )
  return {k: _value_metadata_tree_for_json_loads(v) for k, v in obj.items()}


def value_metadata_tree_to_json_str(tree: PyTree) -> str:
  return simplejson.dumps(
      tree,
      default=_value_metadata_tree_for_json_dumps,
      tuple_as_array=False,  # Must be False to preserve tuples.
      namedtuple_as_object=False,  # Must be False to preserve namedtuples.
  )


def value_metadata_tree_from_json_str(json_str: str) -> PyTree:
  return simplejson.loads(
      json_str,
      object_hook=_value_metadata_tree_for_json_loads,
  )
