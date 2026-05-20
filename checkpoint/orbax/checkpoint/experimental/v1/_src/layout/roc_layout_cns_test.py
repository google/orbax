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

"""Tests for RocLayout integration with CNS."""

from etils import epath
import jax
import numpy as np
import orbax.checkpoint.experimental.v1 as ocp

from .file.colossus.testing.public.python import temporary_colossus_filesystem
from .learning.deepmind.jax.roc import roc
from absl.testing import absltest
from .testing.pybase import parameterized


def setUpModule():
  temporary_colossus_filesystem.setUpModule()


class RocLayoutCnsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    fs = temporary_colossus_filesystem.TemporaryFilesystem(user='someone')
    self.enter_context(fs)
    self.root = epath.Path(fs.cns_root / f'home/someone/{self.id()}/')
    self.root.mkdir(parents=True, exist_ok=True)

  def _write_roc_checkpoint(
      self,
      state: dict[str, np.ndarray],
      checkpoint_format: roc.checkpoint.Format,
  ) -> epath.Path:
    ckpt_path = self.root / 'state'
    coordinator = roc.coordination.OneShotWriterCoordinator(
        roc.checkpoint.Path(ckpt_path.as_posix()),
        checkpoint_format=checkpoint_format,
    )
    with coordinator.sole_checkpoint_host_context() as host_context:
      roc.save(
          host_context,
          state,
          shard_desc_fn=roc.sharding.identity_shard_desc_fn,
      )
    return ckpt_path

  def test_load_from_cns_using_orbax(self):
    state_to_save = {'a': np.array([1, 2, 3], dtype=np.int32)}
    checkpoint_format = roc.checkpoint.FormatEnum.EINSHAPE_NUMPY_PROTO

    cns_path = self._write_roc_checkpoint(state_to_save, checkpoint_format)

    devices = jax.local_devices()
    sharding = jax.sharding.SingleDeviceSharding(devices[0])
    abstract_state = {
        'a': jax.ShapeDtypeStruct(shape=(3,), dtype=np.int32, sharding=sharding)
    }

    with ocp.Context(checkpoint_layout=ocp.options.CheckpointLayout.ROC):
      result = ocp.load(
          cns_path,
          abstract_state=abstract_state,
          checkpointable_name=None,
      )
      jax.tree.map(np.testing.assert_array_equal, result, state_to_save)
      self.assertIsInstance(result['a'], jax.Array)


if __name__ == '__main__':
  googletest.main()
