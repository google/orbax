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

"""Tests for OBM CLI printer helpers."""

from absl.testing import absltest
from absl.testing import parameterized
from orbax.experimental.model.cli import printer
from orbax.experimental.model.cli import redaction_test_pb2
from orbax.experimental.model.core.protos import manifest_pb2
from orbax.experimental.model.core.protos import type_pb2


class PrinterTest(parameterized.TestCase):

  def test_pad(self):
    text = 'line1\nline2'
    padded_text = printer.pad(text, padding='  ')
    self.assertEqual(padded_text, '  line1\n  line2')

  def test_tensor_type(self):
    tt = type_pb2.TensorType(
        shape=type_pb2.Shape(
            shape_with_known_rank=type_pb2.ShapeWithKnownRank(
                dimension_sizes=[
                    type_pb2.DimensionSize(size=1),
                    type_pb2.DimensionSize(size=2),
                ]
            )
        ),
        dtype=type_pb2.DType.f32,
    )
    self.assertEqual(printer.tensor_type(tt), '[1, 2, f32]')

  @parameterized.named_parameters(
      (
          'tensor',
          type_pb2.Type(
              leaf=type_pb2.LeafType(
                  tensor_type=type_pb2.TensorType(
                      shape=type_pb2.Shape(
                          shape_with_known_rank=type_pb2.ShapeWithKnownRank(
                              dimension_sizes=[
                                  type_pb2.DimensionSize(size=1),
                              ]
                          )
                      ),
                      dtype=type_pb2.DType.f32,
                  )
              )
          ),
          '[1, f32]',
      ),
      (
          'token',
          type_pb2.Type(
              leaf=type_pb2.LeafType(token_type=type_pb2.TokenType())
          ),
          'token_type',
      ),
      (
          'tuple',
          type_pb2.Type(
              tuple=type_pb2.Tuple(
                  elements=[
                      type_pb2.Type(
                          leaf=type_pb2.LeafType(
                              tensor_type=type_pb2.TensorType(
                                  shape=type_pb2.Shape(
                                      shape_with_known_rank=type_pb2.ShapeWithKnownRank()
                                  ),
                                  dtype=type_pb2.DType.si8,
                              )
                          )
                      ),
                      type_pb2.Type(none=type_pb2.NoneType()),
                  ]
              )
          ),
          '(\n  si8,\n  none\n)',
      ),
      (
          'list',
          type_pb2.Type(
              list=type_pb2.List(
                  elements=[
                      type_pb2.Type(
                          leaf=type_pb2.LeafType(
                              tensor_type=type_pb2.TensorType(
                                  shape=type_pb2.Shape(
                                      shape_with_known_rank=type_pb2.ShapeWithKnownRank()
                                  ),
                                  dtype=type_pb2.DType.si8,
                              )
                          )
                      ),
                      type_pb2.Type(none=type_pb2.NoneType()),
                  ]
              )
          ),
          '[\n  si8,\n  none\n]',
      ),
      ('none', type_pb2.Type(none=type_pb2.NoneType()), 'none'),
      (
          'dict',
          type_pb2.Type(
              dict=type_pb2.Dict(
                  string_to_type={
                      'b': type_pb2.Type(none=type_pb2.NoneType()),
                      'a': type_pb2.Type(none=type_pb2.NoneType()),
                  }
              )
          ),
          '{\n  a: none,\n  b: none\n}',
      ),
      (
          'string_type_pairs',
          type_pb2.Type(
              string_type_pairs=type_pb2.StringTypePairs(
                  elements=[
                      type_pb2.StringTypePair(
                          fst='b', snd=type_pb2.Type(none=type_pb2.NoneType())
                      ),
                      type_pb2.StringTypePair(
                          fst='a', snd=type_pb2.Type(none=type_pb2.NoneType())
                      ),
                  ]
              )
          ),
          '{\n  b: none,\n  a: none\n}',
      ),
      (
          'empty_tuple',
          type_pb2.Type(tuple=type_pb2.Tuple()),
          '()',
      ),
      (
          'empty_list',
          type_pb2.Type(list=type_pb2.List()),
          '[]',
      ),
      (
          'empty_dict',
          type_pb2.Type(dict=type_pb2.Dict()),
          '{}',
      ),
      (
          'empty_string_type_pairs',
          type_pb2.Type(string_type_pairs=type_pb2.StringTypePairs()),
          '{}',
      ),
      ('empty', type_pb2.Type(), ''),
  )
  def test_type(self, t, expected_str):
    self.assertEqual(printer.obm_type(t), expected_str)

  @parameterized.named_parameters(
      (
          'all_fields',
          manifest_pb2.UnstructuredData(
              mime_type='application/x.mlir-stablehlo',
              version='1.0',
              file_system_location=manifest_pb2.FileSystemLocation(
                  string_path='/tmp/foo'
              ),
          ),
          'MLIR StableHLO v1.0\nfile: /tmp/foo',
      ),
      (
          'unknown_mime_type',
          manifest_pb2.UnstructuredData(
              mime_type='unknown',
              version='1.0',
              file_system_location=manifest_pb2.FileSystemLocation(
                  string_path='/tmp/foo'
              ),
          ),
          'unknown v1.0\nfile: /tmp/foo',
      ),
      (
          'inlined_string',
          manifest_pb2.UnstructuredData(inlined_string='test'),
          'inlined: 4 characters',
      ),
      (
          'inlined_bytes',
          manifest_pb2.UnstructuredData(inlined_bytes=b'test'),
          'inlined: 4 bytes',
      ),
  )
  def test_unstructured_data(self, data, expected_str):
    self.assertEqual(printer.unstructured_data(data), expected_str)

  @parameterized.named_parameters(
      (
          'external',
          manifest_pb2.Value(
              external=manifest_pb2.ExternalValue(
                  data=manifest_pb2.UnstructuredData(inlined_bytes=b'test')
              )
          ),
          'inlined: 4 bytes',
      ),
      ('tuple', manifest_pb2.Value(tuple=manifest_pb2.TupleValue()), 'tuple'),
      (
          'named_tuple',
          manifest_pb2.Value(named_tuple=manifest_pb2.NamedTupleValue()),
          'named tuple',
      ),
      ('unknown', manifest_pb2.Value(), 'unknown'),
  )
  def test_value_type(self, value, expected_str):
    self.assertEqual(printer.value_type(value), expected_str)

  @parameterized.named_parameters(
      (
          'stable_hlo',
          manifest_pb2.Function(
              body=manifest_pb2.FunctionBody(
                  stable_hlo_body=manifest_pb2.StableHloFunctionBody(
                      stable_hlo=manifest_pb2.UnstructuredData(
                          inlined_bytes=b'test'
                      )
                  )
              )
          ),
          'inlined: 4 bytes',
      ),
      (
          'other',
          manifest_pb2.Function(
              body=manifest_pb2.FunctionBody(
                  other=manifest_pb2.UnstructuredData(inlined_bytes=b'test')
              )
          ),
          'inlined: 4 bytes',
      ),
      ('empty', manifest_pb2.Function(), 'function'),
  )
  def test_function(self, fn, expected_str):
    self.assertEqual(printer.function(fn), expected_str)

  @parameterized.named_parameters(
      (
          'function',
          manifest_pb2.TopLevelObject(function=manifest_pb2.Function()),
          'function',
      ),
      (
          'value',
          manifest_pb2.TopLevelObject(
              value=manifest_pb2.Value(
                  external=manifest_pb2.ExternalValue(
                      data=manifest_pb2.UnstructuredData(inlined_bytes=b'test')
                  )
              )
          ),
          'inlined: 4 bytes',
      ),
      (
          'poly_fn',
          manifest_pb2.TopLevelObject(
              poly_fn=manifest_pb2.PolymorphicFunction()
          ),
          'polymorphic function',
      ),
      ('unknown', manifest_pb2.TopLevelObject(), 'unknown'),
  )
  def test_object(self, obj, expected_str):
    self.assertEqual(printer.top_level_object(obj), expected_str)

  def test_redact_byte_fields(self):
    msg = redaction_test_pb2.RedactionTestMessage(
        string_field='foo',
        bytes_field=b'123',
        int_field=10,
        repeated_bytes_field=[b'45', b'678'],
        nested_message=redaction_test_pb2.NestedMessageWithBytes(
            nested_bytes=b'bar', nested_string='baz'
        ),
        map_string_field={'a': 'b'},
    )
    msg.map_bytes_field['b1'] = b'90'
    msg.map_message_field['m1'].nested_bytes = b'quux'
    msg.map_message_field['m1'].nested_string = 'quux2'
    msg.map_message_field['m2'].nested_bytes = b''

    printer.redact_byte_fields(msg)

    self.assertEqual(msg.string_field, 'foo')
    self.assertEqual(msg.int_field, 10)
    self.assertEqual(msg.bytes_field, b'[bold]<3 bytes...>[/bold]')
    self.assertEqual(msg.repeated_bytes_field[0], b'[bold]<2 bytes...>[/bold]')
    self.assertEqual(msg.repeated_bytes_field[1], b'[bold]<3 bytes...>[/bold]')
    self.assertEqual(
        msg.nested_message.nested_bytes, b'[bold]<3 bytes...>[/bold]'
    )
    self.assertEqual(msg.nested_message.nested_string, 'baz')
    self.assertEqual(msg.map_bytes_field['b1'], b'[bold]<2 bytes...>[/bold]')
    self.assertEqual(
        msg.map_message_field['m1'].nested_bytes, b'[bold]<4 bytes...>[/bold]'
    )
    self.assertEqual(msg.map_message_field['m1'].nested_string, 'quux2')
    self.assertEqual(msg.map_message_field['m2'].nested_bytes, b'')
    self.assertEqual(msg.map_string_field['a'], 'b')


if __name__ == '__main__':
  absltest.main()
