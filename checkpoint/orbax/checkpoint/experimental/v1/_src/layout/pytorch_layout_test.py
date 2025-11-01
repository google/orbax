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

import asyncio
import os
import zipfile

from absl.testing import absltest
import jax
import numpy as np
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import pytorch_layout
from orbax.checkpoint.experimental.v1._src.path import types
import torch

Path = types.Path
InvalidLayoutError = checkpoint_layout.InvalidLayoutError
PyTorchLayout = pytorch_layout.PyTorchLayout


class PyTorchLayoutTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    if torch is None:
      self.skipTest('torch is not installed')
    self.directory = self.create_tempdir().full_path
    self._test_ckpt_path = os.path.join(self.directory, 'ckpt.pt')
    self._test_data = {'alice': torch.arange(2), 'bob': torch.ones(5)}
    torch.save(self._test_data, self._test_ckpt_path)

  def test_validate_valid_checkpoint(self):
    async def run_test():
      layout = PyTorchLayout(Path(self._test_ckpt_path))
      await layout.validate()
    asyncio.run(run_test())

  def test_validate_invalid_not_zip(self):
    async def run_test():
      bad_path = os.path.join(self.directory, 'bad.pt')
      with open(bad_path, 'w') as f:
        f.write('not a zip')
      layout = PyTorchLayout(Path(bad_path))
      with self.assertRaisesRegex(InvalidLayoutError, 'not a valid ZIP file'):
        await layout.validate()
    asyncio.run(run_test())

  def test_validate_invalid_no_pickle(self):
    async def run_test():
      bad_path = os.path.join(self.directory, 'bad.pt')
      with zipfile.ZipFile(bad_path, 'w') as zf:
        zf.writestr('foo.txt', b'bar')
      layout = PyTorchLayout(Path(bad_path))
      with self.assertRaisesRegex(InvalidLayoutError, 'missing data.pkl'):
        await layout.validate()
    asyncio.run(run_test())

  def test_metadata(self):
    async def run_test():
      layout = PyTorchLayout(Path(self._test_ckpt_path))
      await layout.validate()
      metadata = await layout.metadata()
      self.assertIn(
          checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY, metadata.metadata
      )
      pt_metadata = metadata.metadata[
          checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY
      ]
      self.assertIsInstance(pt_metadata['alice'], jax.ShapeDtypeStruct)
      self.assertEqual(pt_metadata['alice'].shape, (2,))
      self.assertEqual(pt_metadata['alice'].dtype, np.int64)
      self.assertIsInstance(pt_metadata['bob'], jax.ShapeDtypeStruct)
      self.assertEqual(pt_metadata['bob'].shape, (5,))
      self.assertEqual(pt_metadata['bob'].dtype, np.float32)

    asyncio.run(run_test())

  def test_load_numpy(self):
    async def run_test():
      layout = PyTorchLayout(Path(self._test_ckpt_path))
      await layout.validate()
      restored = await (await layout.load())
      restored_pytree = restored[checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY]
      np.testing.assert_array_equal(
          restored_pytree['alice'], self._test_data['alice'].numpy()
      )
      np.testing.assert_array_equal(
          restored_pytree['bob'], self._test_data['bob'].numpy()
      )
    asyncio.run(run_test())


if __name__ == '__main__':
  absltest.main()
