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

"""Tests for RocLayout integration with TFHub."""

import jax
import numpy as np
import orbax.checkpoint.experimental.v1 as ocp
from orbax.checkpoint.google.path import tfhub_test_lib

from .learning.brain.contrib.hub.public.proto import metadata_pb2
from .learning.brain.contrib.hub.public.proto import options_pb2
from .learning.brain.contrib.hub.public.proto import realm_pb2
from .learning.brain.contrib.hub.public.python import handle as handle_lib
from .learning.deepmind.jax.roc import roc
from absl.testing import absltest
from .testing.pybase import parameterized


def setUpModule():
  tfhub_test_lib.setUpModule()


class RocLayoutTfhubTest(tfhub_test_lib.TfHubTestBase, parameterized.TestCase):

  def _write_roc_checkpoint(
      self,
      state: dict[str, np.ndarray],
      checkpoint_format: roc.checkpoint.Format,
  ) -> str:
    base_path = roc.checkpoint.Path(self.create_tempdir().full_path)
    ckpt_path = base_path / 'state'
    coordinator = roc.coordination.OneShotWriterCoordinator(
        ckpt_path,
        checkpoint_format=checkpoint_format,
    )
    with coordinator.sole_checkpoint_host_context() as host_context:
      roc.save(
          host_context,
          state,
          shard_desc_fn=roc.sharding.identity_shard_desc_fn,
      )
    return str(ckpt_path)

  def _publish_roc_checkpoint(
      self,
      state: dict[str, np.ndarray],
      checkpoint_format: roc.checkpoint.Format,
  ) -> str:
    ckpt_path = self._write_roc_checkpoint(state, checkpoint_format)
    ckpt_name = 'test_roc_checkpoint'
    ckpt_version = 1
    description = 'Test Roc checkpoint.'
    constant_version_label = '1.step_1'

    versioned_handle = handle_lib.Handle.from_parts(
        tfhub_test_lib.PUBLISHER, ckpt_name, ckpt_version
    )

    metadata = metadata_pb2.ModelDescriptorProto()
    metadata.asset_descriptor.documentation.description = description
    metadata.asset_descriptor.labels.append(
        metadata_pb2.LabelProto(
            value=constant_version_label,
            type=metadata_pb2.LabelType.LABEL_TYPE_CONSTANT,
        )
    )
    self._client.create_model(
        options_pb2.CreateOptionsProto(),
        versioned_handle,
        metadata,
        ckpt_path,
    )

    tfhub_path = versioned_handle.to_gfile_filepath(realm_pb2.ISOLATED_REALM)
    return tfhub_path

  def test_load_from_tfhub_using_orbax(self):
    state_to_save = {'a': np.array([1, 2, 3], dtype=np.int32)}
    checkpoint_format = roc.checkpoint.FormatEnum.EINSHAPE_NUMPY_PROTO

    tfhub_path = self._publish_roc_checkpoint(state_to_save, checkpoint_format)

    devices = jax.local_devices()
    sharding = jax.sharding.SingleDeviceSharding(devices[0])
    abstract_pytree = {
        'a': jax.ShapeDtypeStruct(shape=(3,), dtype=np.int32, sharding=sharding)
    }

    with ocp.Context(checkpoint_layout=ocp.options.CheckpointLayout.ROC):
      result = ocp.load(
          tfhub_path,
          abstract_state=abstract_pytree,
          checkpointable_name=None,
      )
      jax.tree.map(np.testing.assert_array_equal, result, state_to_save)
      self.assertIsInstance(result['a'], jax.Array)


if __name__ == '__main__':
  googletest.main()
