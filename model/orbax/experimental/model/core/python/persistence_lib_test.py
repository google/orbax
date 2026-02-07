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

"""Persistence tests (saving/loading the model)."""

import os

from absl.testing import absltest
from orbax.experimental.model.core.protos import manifest_pb2
from orbax.experimental.model.core.protos import type_pb2
from orbax.experimental.model.core.python import device_assignment
from orbax.experimental.model.core.python import file_utils
from orbax.experimental.model.core.python import function
from orbax.experimental.model.core.python import manifest_constants
from orbax.experimental.model.core.python import persistence_lib
from orbax.experimental.model.core.python import serializable_function
from orbax.experimental.model.core.python import shlo_function
from orbax.experimental.model.core.python import unstructured_data
from orbax.experimental.model.core.python import value

from tensorflow.python.util.protobuf import compare


class PersistenceLibTest(absltest.TestCase):

  def test_save_succeeds_with_all_fields(self):
    target_dir: str = self.create_tempdir().full_path

    func1 = TestShloFunction(b'func1')
    func2 = TestShloFunction(b'func2')
    func3 = TestSerializableFunction(b'func3')

    persistence_lib.save(
        module={
            'my_func': func1,
            'my_seq_of_funcs': [func2, func3],
            'my_value': value.ExternalValue(
                data=manifest_pb2.UnstructuredData(inlined_string='my_value'),
            ),
        },
        target_dir=target_dir,
        options=persistence_lib.SaveOptions(
            version=2,
            supplementals={
                'supp_a': persistence_lib.GlobalSupplemental(
                    data=manifest_pb2.UnstructuredData(inlined_string='supp_a'),
                    save_as='supp_a.pb',
                ),
                'supp_b': persistence_lib.GlobalSupplemental(
                    data=manifest_pb2.UnstructuredData(inlined_string='supp_b')
                ),
            },
            visibility={
                'my_func': manifest_pb2.PRIVATE,
                'my_seq_of_funcs': manifest_pb2.PUBLIC,
            },
            device_assignment_by_coords=[
                device_assignment.DeviceAssignment(
                    id=0,
                    coords=[0, 1, 2],
                    core_on_chip=0,
                ),
                device_assignment.DeviceAssignment(
                    id=1,
                    coords=[1, 2, 3],
                    core_on_chip=1,
                ),
            ],
        ),
    )

    model_version_path = os.path.join(
        target_dir, manifest_constants.MODEL_VERSION_FILENAME
    )
    self.assertTrue(os.path.exists(model_version_path))

    actual_manifest = persistence_lib.load(target_dir)
    expected_manifest = manifest_pb2.Manifest(
        objects={
            'my_func': manifest_pb2.TopLevelObject(
                function=func1.manifest_repr('my_func', manifest_pb2.PRIVATE)
            ),
            'my_seq_of_funcs': manifest_pb2.TopLevelObject(
                poly_fn=manifest_pb2.PolymorphicFunction(
                    concrete_functions=[
                        func2.manifest_repr(
                            '__my_seq_of_funcs_0', manifest_pb2.PUBLIC
                        ),
                        func3.manifest_repr(
                            '__my_seq_of_funcs_1',
                            manifest_pb2.PUBLIC,
                        ),
                    ],
                )
            ),
            'my_value': manifest_pb2.TopLevelObject(
                value=manifest_pb2.Value(
                    external=manifest_pb2.ExternalValue(
                        data=manifest_pb2.UnstructuredData(
                            inlined_string='my_value'
                        )
                    )
                )
            ),
        },
        supplemental_info={
            'supp_a': manifest_pb2.UnstructuredData(
                file_system_location=manifest_pb2.FileSystemLocation(
                    string_path='supp_a.pb'
                ),
                mime_type='',
                version='',
            ),
            'supp_b': manifest_pb2.UnstructuredData(
                inlined_string='supp_b',
            ),
        },
        device_assignment_by_coords=manifest_pb2.DeviceAssignmentByCoords(
            devices=[
                manifest_pb2.DeviceAssignmentByCoords.Device(
                    id=0, coords=[0, 1, 2], core_on_chip=0
                ),
                manifest_pb2.DeviceAssignmentByCoords.Device(
                    id=1, coords=[1, 2, 3], core_on_chip=1
                ),
            ]
        ),
    )

    compare.assertProtoEqual(
        self,
        expected_manifest,
        actual_manifest,
    )

    # Check that the supplemental files are written only for the functions
    # that don't have them inlined.
    self.assertTrue(os.path.exists(os.path.join(target_dir, 'supp_a.pb')))
    self.assertTrue(
        os.path.exists(
            os.path.join(target_dir, 'my_func_supp_a_supplemental.pb')
        )
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(target_dir, 'my_func_supp_b_supplemental.pb')
        )
    )
    self.assertFalse(
        os.path.exists(os.path.join(target_dir, '__my_seq_of_funcs_0.pb'))
    )
    self.assertTrue(
        os.path.exists(os.path.join(target_dir, '__my_seq_of_funcs_1.pb'))
    )

  def test_load_fails_version_file_missing(self):
    target_dir: str = self.create_tempdir().full_path

    with self.assertRaisesRegex(Exception, 'orbax_model_version.txt'):
      persistence_lib.load(target_dir)

  def test_load_fails_with_unsupported_version(self):
    target_dir: str = self.create_tempdir().full_path

    path = os.path.join(target_dir, 'orbax_model_version.txt')
    file_content = """
    version: "42"
    manifest_file_path: "test/path"
    mime_type: "test_mime_type; application/foo"
    """
    with file_utils.open_file(path, 'w') as f:
      f.write(file_content)

    with self.assertRaisesRegex(
        ValueError, 'Unsupported manifest version "42"'
    ):
      persistence_lib.load(target_dir)

  def test_load_fails_with_unsupported_mime_type(self):
    target_dir: str = self.create_tempdir().full_path

    path = os.path.join(target_dir, 'orbax_model_version.txt')
    file_content = """
    version: "0.0.1"
    mime_type: "text/plain"
    manifest_file_path: "test/path"
    """
    with file_utils.open_file(path, 'w') as f:
      f.write(file_content)

    with self.assertRaisesRegex(
        ValueError, 'Unsupported manifest type "text/plain"'
    ):
      persistence_lib.load(target_dir)


class TestShloFunctionSupplementalInfo(
    shlo_function.ShloFunctionSupplementalInfo
):

  def __init__(self, b: bytes):
    self._bytes = b

  def serializable_to_proto(
      self,
  ) -> unstructured_data.UnstructuredDataWithExtName:
    return unstructured_data.UnstructuredDataWithExtName(
        proto=manifest_pb2.UnstructuredData(inlined_bytes=self._bytes),
        ext_name='pb',
    )


class TestShloFunction(shlo_function.ShloFunction):

  def __init__(self, b: bytes):
    self.bytes = b
    self.input_signature = function.ShloTensorSpec((1,), function.ShloDType.f32)
    self.output_signature = function.ShloTensorSpec(
        (
            1,
            2,
        ),
        function.ShloDType.f16,
    )
    self.mlir_module_serialized = b
    self.calling_convention_version = 42
    self.lowering_platforms = ('a', 'b')
    self.module_kept_var_idx = (0, 1)
    self.supplemental_info = {
        'supp_a': TestShloFunctionSupplementalInfo(b'supp_a'),
        'supp_b': TestShloFunctionSupplementalInfo(b'supp_b'),
    }
    self.physical_in_dtypes = (None, function.ShloDType.f32)
    self.physical_out_dtypes = (function.ShloDType.f16, None)

  def manifest_repr(
      self, key: str, visibility: manifest_pb2.Visibility
  ) -> manifest_pb2.Function:
    return manifest_pb2.Function(
        signature=type_pb2.FunctionSignature(
            input=type_pb2.Type(
                leaf=type_pb2.LeafType(
                    tensor_type=type_pb2.TensorType(
                        shape=type_pb2.Shape(
                            shape_with_known_rank=type_pb2.ShapeWithKnownRank(
                                dimension_sizes=[type_pb2.DimensionSize(size=1)]
                            )
                        ),
                        dtype=type_pb2.DType.f32,
                    )
                )
            ),
            output=type_pb2.Type(
                leaf=type_pb2.LeafType(
                    tensor_type=type_pb2.TensorType(
                        shape=type_pb2.Shape(
                            shape_with_known_rank=type_pb2.ShapeWithKnownRank(
                                dimension_sizes=[
                                    type_pb2.DimensionSize(size=1),
                                    type_pb2.DimensionSize(size=2),
                                ]
                            )
                        ),
                        dtype=type_pb2.DType.f16,
                    )
                )
            ),
        ),
        body=manifest_pb2.FunctionBody(
            stable_hlo_body=manifest_pb2.StableHloFunctionBody(
                stable_hlo=manifest_pb2.UnstructuredData(
                    inlined_bytes=self.bytes,
                    mime_type='application/x.mlir-stablehlo',
                    version='1.0',
                ),
                calling_convention_version=42,
                lowering_platforms=['a', 'b'],
                module_kept_var_idx=[0, 1],
                supplemental_info={
                    'supp_a': manifest_pb2.UnstructuredData(
                        file_system_location=manifest_pb2.FileSystemLocation(
                            string_path=f'{key}_supp_a_supplemental.pb'
                        ),
                        mime_type='',
                        version='',
                    ),
                    'supp_b': manifest_pb2.UnstructuredData(
                        file_system_location=manifest_pb2.FileSystemLocation(
                            string_path=f'{key}_supp_b_supplemental.pb'
                        ),
                        mime_type='',
                        version='',
                    ),
                },
            )
        ),
        visibility=visibility,
    )


class TestSerializableFunction(serializable_function.SerializableFunction):

  def __init__(self, b: bytes):
    self.input_signature = function.ShloTensorSpec((1,), function.ShloDType.f32)
    self.output_signature = function.ShloTensorSpec(
        (
            1,
            2,
        ),
        function.ShloDType.f16,
    )
    self.body = unstructured_data.UnstructuredDataWithExtName(
        proto=manifest_pb2.UnstructuredData(inlined_bytes=b),
        ext_name='pb',
    )

  def manifest_repr(
      self, key: str, visibility: manifest_pb2.Visibility
  ) -> manifest_pb2.Function:
    return manifest_pb2.Function(
        signature=type_pb2.FunctionSignature(
            input=type_pb2.Type(
                leaf=type_pb2.LeafType(
                    tensor_type=type_pb2.TensorType(
                        shape=type_pb2.Shape(
                            shape_with_known_rank=type_pb2.ShapeWithKnownRank(
                                dimension_sizes=[type_pb2.DimensionSize(size=1)]
                            )
                        ),
                        dtype=type_pb2.DType.f32,
                    )
                )
            ),
            output=type_pb2.Type(
                leaf=type_pb2.LeafType(
                    tensor_type=type_pb2.TensorType(
                        shape=type_pb2.Shape(
                            shape_with_known_rank=type_pb2.ShapeWithKnownRank(
                                dimension_sizes=[
                                    type_pb2.DimensionSize(size=1),
                                    type_pb2.DimensionSize(size=2),
                                ]
                            )
                        ),
                        dtype=type_pb2.DType.f16,
                    )
                )
            ),
        ),
        body=manifest_pb2.FunctionBody(
            other=manifest_pb2.UnstructuredData(
                file_system_location=manifest_pb2.FileSystemLocation(
                    string_path=f'{key}.pb',
                ),
                mime_type='',
                version='',
            )
        ),
        visibility=visibility,
    )


if __name__ == '__main__':
  absltest.main()
