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

"""Helper functions for printing Orbax model objects."""

from collections.abc import MutableSequence

from google.protobuf import text_format
from orbax.experimental.model import core as obm
from orbax.experimental.model.cli import constants

_BatchComponent = oex_orchestration_pb2.BatchOptions.BatchComponent


def top_level_object(obj: obm.manifest_pb2.TopLevelObject) -> str:
  if obj.HasField('function'):
    return function(obj.function)
  if obj.HasField('value'):
    return value_type(obj.value)
  if obj.HasField('poly_fn'):
    return 'polymorphic function'
  return 'unknown'


def function(fn: obm.manifest_pb2.Function) -> str:
  if fn.body.HasField('stable_hlo_body'):
    return unstructured_data(fn.body.stable_hlo_body.stable_hlo)
  if fn.body.HasField('other'):
    return unstructured_data(fn.body.other)
  return 'function'


def value_type(value: obm.manifest_pb2.Value) -> str:
  if value.HasField('external'):
    return unstructured_data(value.external.data)
  if value.HasField('tuple'):
    return 'tuple'
  if value.HasField('named_tuple'):
    return 'named tuple'
  return 'unknown'


def unstructured_data(
    data: obm.manifest_pb2.UnstructuredData,
) -> str:
  """Generates a human-readable string for UnstructuredData details."""
  details: MutableSequence[str] = []
  mime_type_and_version: MutableSequence[str] = []

  if data.mime_type:
    mime_type_and_version.append(
        constants.WELL_KNOWN_MIME_TYPE_DESCRIPTIONS.get(
            data.mime_type, data.mime_type
        )
    )
  if data.version:
    mime_type_and_version.append(f'v{data.version}')

  if mime_type_and_version:
    details.append(f'{' '.join(mime_type_and_version)}')

  if data.HasField('file_system_location'):
    details.append(f'file: {data.file_system_location.string_path}')
  elif data.HasField('inlined_string'):
    details.append(f'inlined: {len(data.inlined_string)} characters')
  elif data.HasField('inlined_bytes'):
    details.append(f'inlined: {len(data.inlined_bytes)} bytes')

  return '\n'.join(details)


def obm_type(t: obm.type_pb2.Type) -> str:
  """Generates a human-readable string for a model type."""
  if t.HasField('leaf'):
    if t.leaf.HasField('tensor_type'):
      return tensor_type(t.leaf.tensor_type)
    elif t.leaf.HasField('token_type'):
      return 'token_type'

  if t.HasField('tuple'):
    if not t.tuple.elements:
      return '()'

    return (
        '(\n'
        + pad(
            ',\n'.join([obm_type(e) for e in t.tuple.elements]),
            padding='  ',
        )
        + '\n)'
    )

  if t.HasField('list'):
    if not t.list.elements:
      return '[]'

    return (
        '[\n'
        + pad(
            ',\n'.join([obm_type(e) for e in t.list.elements]),
            padding='  ',
        )
        + '\n]'
    )

  if t.HasField('none'):
    return 'none'

  if t.HasField('dict'):
    if not t.dict.string_to_type:
      return '{}'

    return (
        '{\n'
        + pad(
            ',\n'.join([
                f'{k}: {obm_type(v)}'
                for k, v in sorted(t.dict.string_to_type.items())
            ]),
            padding='  ',
        )
        + '\n}'
    )

  if t.HasField('string_type_pairs'):
    if not t.string_type_pairs.elements:
      return '{}'

    return (
        '{\n'
        + pad(
            ',\n'.join([
                f'{pair.fst}: {obm_type(pair.snd)}'
                for pair in t.string_type_pairs.elements
            ]),
            padding='  ',
        )
        + '\n}'
    )

  return ''


def tensor_type(tt: obm.type_pb2.TensorType) -> str:
  if not tt.shape.shape_with_known_rank.dimension_sizes:
    return f'{obm.type_pb2.DType.Name(tt.dtype)}'

  dims = ', '.join([
      str(s.size if s.size != 0 else '_')
      for s in tt.shape.shape_with_known_rank.dimension_sizes
  ])
  return f'[{dims}, {obm.type_pb2.DType.Name(tt.dtype)}]'


def pad(text: str, padding: str = '  ') -> str:
  return '\n'.join([padding + line for line in text.splitlines()])
