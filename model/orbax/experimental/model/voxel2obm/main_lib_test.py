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

"""Tests for main_lib."""

from unittest import mock
from absl.testing import absltest
from orbax.experimental.model.voxel2obm import main_lib


class MainLibTest(absltest.TestCase):

  def test_voxel_plan_to_obm_with_plan(self):
    # TODO(b/447200841): Replace with a real Voxel module.
    class FakeVoxelModule:

      def export_plan(self):
        plan_proto = mock.Mock()
        plan_proto.SerializeToString.return_value = b'test plan'
        return plan_proto

    obm_fn = main_lib.voxel_plan_to_obm(
        FakeVoxelModule(), input_signature={}, output_signature={}
    )
    self.assertEqual(obm_fn.body.proto.inlined_bytes, b'test plan')
    self.assertEqual(
        obm_fn.body.proto.mime_type, main_lib.VOXEL_PROCESSOR_MIME_TYPE
    )
    self.assertEqual(
        obm_fn.body.proto.version, main_lib.VOXEL_PROCESSOR_VERSION
    )


if __name__ == '__main__':
  absltest.main()
