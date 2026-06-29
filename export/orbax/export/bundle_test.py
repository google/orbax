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

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test suite for model bundle execution."""

import os

from absl.testing import absltest
from orbax.export import bundle

from neptune_model.protos import bundle_orchestration_pb2


class BundleTest(absltest.TestCase):

  def test_create_bundle_copy_models(self):
    src_workspace = self.create_tempdir('source_workspace').full_path
    bundle_dest = self.create_tempdir('bundle_destination').full_path

    src_model_dir = os.path.join(src_workspace, 'src_model')
    self._create_dummy_orbax_model(src_model_dir)

    bundle_def = bundle.BundleDefinition(
        name='test_bundle',
        version=1,
        pipelines={
            'pipeline1': [
                bundle.PipelineStep(model='model1', pipeline='step1'),
            ]
        }
    )
    models = [
        bundle.SubModel(name='model1', path=src_model_dir),
    ]

    bundle.create_bundle(
        bundle_dest, bundle_def, models, copy_models=True
    )

    self.assertTrue(os.path.exists(bundle_dest))

    dest_model_dir = os.path.join(bundle_dest, 'model1')
    self.assertTrue(os.path.exists(dest_model_dir))
    self.assertTrue(os.path.isdir(dest_model_dir))

    self._assert_model_files_exist(dest_model_dir)

    proto_path = os.path.join(bundle_dest, 'bundle_orchestration.pb')
    self.assertTrue(os.path.exists(proto_path))

    proto = bundle_orchestration_pb2.BundleOrchestration()
    with open(proto_path, 'rb') as f:
      proto.ParseFromString(f.read())

    self.assertEqual(proto.metadata.name, 'test_bundle')
    self.assertEqual(proto.metadata.version, 1)
    self.assertIn('pipeline1', proto.pipelines)

    pipeline = proto.pipelines['pipeline1']
    self.assertLen(pipeline.components, 1)
    self.assertEqual(pipeline.components[0].model_name, 'model1')
    self.assertEqual(pipeline.components[0].pipeline_name, 'step1')
    self.assertEqual(pipeline.components[0].repeated_times, 1)
    self.assertFalse(pipeline.components[0].requires_h2d)
    self.assertFalse(pipeline.components[0].requires_d2h)

  def test_create_bundle_copy_multiple_nested_models(self):
    src_workspace = self.create_tempdir('source_workspace').full_path
    bundle_dest = self.create_tempdir('bundle_destination').full_path

    src_model1_dir = os.path.join(src_workspace, 'src_model1')
    self._create_dummy_orbax_model(src_model1_dir)

    src_model2_dir = os.path.join(src_workspace, 'src_model2')
    os.makedirs(src_model2_dir)
    with open(os.path.join(src_model2_dir, 'model.json'), 'w') as f:
      f.write('model2_json_content')

    bundle_def = bundle.BundleDefinition(
        name='nested_bundle',
        version=2,
        pipelines={
            'pipeline1': [
                bundle.PipelineStep(model='model1', pipeline='step1'),
                bundle.PipelineStep(model='model2', pipeline='step2'),
            ]
        }
    )
    models = [
        bundle.SubModel(name='model1', path=src_model1_dir),
        bundle.SubModel(name='model2', path=src_model2_dir),
    ]

    bundle.create_bundle(
        bundle_dest, bundle_def, models, copy_models=True
    )

    dest_model1 = os.path.join(bundle_dest, 'model1')
    self.assertTrue(os.path.exists(dest_model1))
    self.assertTrue(os.path.isdir(dest_model1))
    self.assertFalse(os.path.islink(dest_model1))

    self._assert_model_files_exist(dest_model1)

    dest_model2 = os.path.join(bundle_dest, 'model2')
    self.assertTrue(os.path.exists(dest_model2))
    self.assertTrue(os.path.isdir(dest_model2))

    self.assertTrue(os.path.exists(os.path.join(dest_model2, 'model.json')))
    with open(os.path.join(dest_model2, 'model.json'), 'r') as f:
      self.assertEqual(f.read(), 'model2_json_content')

    proto_path = os.path.join(bundle_dest, 'bundle_orchestration.pb')
    self.assertTrue(os.path.exists(proto_path))

    proto = bundle_orchestration_pb2.BundleOrchestration()
    with open(proto_path, 'rb') as f:
      proto.ParseFromString(f.read())

    self.assertEqual(proto.metadata.name, 'nested_bundle')
    self.assertEqual(proto.metadata.version, 2)
    self.assertIn('pipeline1', proto.pipelines)

    pipeline = proto.pipelines['pipeline1']
    self.assertLen(pipeline.components, 2)

    self.assertEqual(pipeline.components[0].model_name, 'model1')
    self.assertEqual(pipeline.components[0].pipeline_name, 'step1')
    self.assertEqual(pipeline.components[0].repeated_times, 1)
    self.assertFalse(pipeline.components[0].requires_h2d)
    self.assertFalse(pipeline.components[0].requires_d2h)

    self.assertEqual(pipeline.components[1].model_name, 'model2')
    self.assertEqual(pipeline.components[1].pipeline_name, 'step2')
    self.assertEqual(pipeline.components[1].repeated_times, 1)
    self.assertFalse(pipeline.components[1].requires_h2d)
    self.assertFalse(pipeline.components[1].requires_d2h)

  def test_create_bundle_symlink_models(self):
    src_workspace = self.create_tempdir('source_workspace').full_path
    bundle_dest = self.create_tempdir('bundle_destination').full_path

    src_model_dir = os.path.join(src_workspace, 'src_model')
    self._create_dummy_orbax_model(src_model_dir)

    bundle_def = bundle.BundleDefinition(
        name='symlink_bundle',
        version=1,
        pipelines={
            'pipeline1': [
                bundle.PipelineStep(model='model1', pipeline='step1'),
            ]
        }
    )
    models = [
        bundle.SubModel(name='model1', path=src_model_dir),
    ]

    bundle.create_bundle(
        bundle_dest, bundle_def, models, copy_models=False
    )

    self.assertTrue(os.path.exists(bundle_dest))

    dest_model_dir = os.path.join(bundle_dest, 'model1')
    self.assertTrue(os.path.exists(dest_model_dir))
    self.assertTrue(os.path.islink(dest_model_dir))
    self.assertEqual(os.readlink(dest_model_dir), src_model_dir)

    proto_path = os.path.join(bundle_dest, 'bundle_orchestration.pb')
    self.assertTrue(os.path.exists(proto_path))

    proto = bundle_orchestration_pb2.BundleOrchestration()
    with open(proto_path, 'rb') as f:
      proto.ParseFromString(f.read())

    self.assertEqual(proto.metadata.name, 'symlink_bundle')
    self.assertEqual(proto.metadata.version, 1)
    self.assertIn('pipeline1', proto.pipelines)

    pipeline = proto.pipelines['pipeline1']
    self.assertLen(pipeline.components, 1)
    self.assertEqual(pipeline.components[0].model_name, 'model1')
    self.assertEqual(pipeline.components[0].pipeline_name, 'step1')

  def _create_dummy_orbax_model(self, model_dir: str):
    """Creates a dummy Orbax Model structure for testing."""
    os.makedirs(os.path.join(model_dir, 'checkpoint'))
    with open(os.path.join(model_dir, 'manifest.pb'), 'w') as f:
      f.write('dummy_manifest_pb')
    with open(os.path.join(model_dir, 'neptune_model_version.txt'), 'w') as f:
      f.write('dummy_version')
    with open(os.path.join(model_dir, 'orchestration.pb'), 'w') as f:
      f.write('dummy_orchestration')
    with open(os.path.join(model_dir, 'predict.shlo'), 'w') as f:
      f.write('dummy_shlo')
    for i in range(1, 4):
      checkpoint_path = os.path.join(
          model_dir, 'checkpoint', f'checkpoint_{i}'
      )
      with open(checkpoint_path, 'w') as f:
        f.write(f'dummy_checkpoint_{i}')
    with open(os.path.join(model_dir, 'checkpoint', 'metadata'), 'w') as f:
      f.write('dummy_metadata')

  def _assert_model_files_exist(self, model_dir: str):
    """Asserts that the expected dummy Orbax Model files exist."""
    for filename in [
        'manifest.pb',
        'neptune_model_version.txt',
        'predict.shlo',
    ]:
      self.assertTrue(os.path.exists(os.path.join(model_dir, filename)))
    for i in range(1, 4):
      self.assertTrue(
          os.path.exists(
              os.path.join(model_dir, 'checkpoint', f'checkpoint_{i}')
          )
      )


if __name__ == '__main__':
  absltest.main()
