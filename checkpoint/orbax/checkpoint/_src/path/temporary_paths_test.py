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

import asyncio
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import atomicity
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint._src.path import gcs_utils
from orbax.checkpoint._src.path import temporary_paths


class TemporaryPathsTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir('ckpt').full_path)

  @parameterized.parameters(
      (atomicity.AtomicRenameTemporaryPath,),
      (atomicity.CommitFileTemporaryPath,),
  )
  async def test_temporary_path(
      self, tmp_path_cls: type[atomicity_types.TemporaryPath]
  ):
    tmp_path = tmp_path_cls.from_final(self.directory / 'ckpt')
    await tmp_path.create()
    self.assertTrue(
        await temporary_paths.is_path_temporary(
            tmp_path.get(), temporary_path_cls=tmp_path_cls
        )
    )
    self.assertFalse(
        await temporary_paths.is_path_finalized(
            tmp_path.get(), temporary_path_cls=tmp_path_cls
        )
    )

  @parameterized.parameters(
      (atomicity.AtomicRenameTemporaryPath,),
      (atomicity.CommitFileTemporaryPath,),
  )
  async def test_finalized_path(
      self, tmp_path_cls: type[atomicity_types.TemporaryPath]
  ):
    tmp_path = tmp_path_cls.from_final(self.directory / 'ckpt')
    await tmp_path.create()
    await tmp_path.finalize()
    self.assertFalse(
        await temporary_paths.is_path_temporary(tmp_path.get_final())
    )
    self.assertTrue(
        await temporary_paths.is_path_finalized(
            tmp_path.get_final(), temporary_path_cls=tmp_path_cls
        )
    )

  async def test_incorrect_path_class(self):
    tmp_path = atomicity.AtomicRenameTemporaryPath.from_final(
        self.directory / 'ckpt'
    )
    await tmp_path.create()
    # Missing commit_success, so looks like a valid temporary path.
    self.assertTrue(
        await temporary_paths.is_path_temporary(
            tmp_path.get(), temporary_path_cls=atomicity.CommitFileTemporaryPath
        )
    )
    self.assertFalse(
        await temporary_paths.is_path_finalized(
            tmp_path.get(), temporary_path_cls=atomicity.CommitFileTemporaryPath
        )
    )
    await tmp_path.finalize()

    self.assertFalse(
        await temporary_paths.is_path_temporary(
            tmp_path.get_final(),
            temporary_path_cls=atomicity.CommitFileTemporaryPath,
        )
    )
    self.assertTrue(
        await temporary_paths.is_path_finalized(
            tmp_path.get_final(),
            temporary_path_cls=atomicity.CommitFileTemporaryPath,
        )
    )

    tmp_path = atomicity.CommitFileTemporaryPath.from_final(
        self.directory / 'ckpt2'
    )
    await tmp_path.create()
    self.assertFalse(
        await temporary_paths.is_path_temporary(
            tmp_path.get(),
            temporary_path_cls=atomicity.AtomicRenameTemporaryPath,
        )
    )
    # Looks finalized, because does not contain `orbax-checkpoint-tmp`.
    self.assertTrue(
        await temporary_paths.is_path_finalized(
            tmp_path.get(),
            temporary_path_cls=atomicity.AtomicRenameTemporaryPath,
        )
    )
    await tmp_path.finalize()
    self.assertFalse(
        await temporary_paths.is_path_temporary(
            tmp_path.get_final(),
            temporary_path_cls=atomicity.AtomicRenameTemporaryPath,
        )
    )
    self.assertTrue(
        await temporary_paths.is_path_finalized(
            tmp_path.get_final(),
            temporary_path_cls=atomicity.AtomicRenameTemporaryPath,
        )
    )

  @parameterized.parameters(
      (atomicity.AtomicRenameTemporaryPath,),
      (atomicity.CommitFileTemporaryPath,),
  )
  async def test_all_temporary_paths(
      self, tmp_path_cls: type[atomicity_types.TemporaryPath]
  ):
    num_paths = 3
    tmp_paths = [
        tmp_path_cls.from_final(self.directory / str(i))
        for i in range(num_paths)
    ]
    await asyncio.gather(*[tmp_path.create() for tmp_path in tmp_paths])
    self.assertSameElements(
        [
            p.get()
            for p in await temporary_paths.all_temporary_paths(
                self.directory, temporary_path_cls=tmp_path_cls
            )
        ],
        [tmp_path.get() for tmp_path in tmp_paths],
    )

    await tmp_paths[0].finalize()
    self.assertSameElements(
        [
            p.get()
            for p in await temporary_paths.all_temporary_paths(
                self.directory, temporary_path_cls=tmp_path_cls
            )
        ],
        [tmp_path.get() for tmp_path in tmp_paths[1:]],
    )

  @parameterized.parameters(
      (atomicity.AtomicRenameTemporaryPath,),
      (atomicity.CommitFileTemporaryPath,),
  )
  async def test_cleanup_temporary_paths(
      self, tmp_path_cls: type[atomicity_types.TemporaryPath]
  ):
    num_paths = 3
    tmp_paths = [
        tmp_path_cls.from_final(self.directory / str(i))
        for i in range(num_paths)
    ]
    await asyncio.gather(*[tmp_path.create() for tmp_path in tmp_paths])
    await tmp_paths[0].finalize()
    await temporary_paths.cleanup_temporary_paths(
        self.directory, temporary_path_cls=tmp_path_cls
    )

    self.assertEmpty(
        await temporary_paths.all_temporary_paths(
            self.directory, temporary_path_cls=tmp_path_cls
        )
    )

    for i, path in enumerate(tmp_paths):
      if i == 0:
        self.assertTrue(await async_path.exists(path.get_final()))
      else:
        self.assertFalse(await async_path.exists(path.get_final()))


class LegacyAtomicRenameFallbackTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir('ckpt').full_path)

  async def _create_legacy_finalized(self, name: str) -> epath.Path:
    """Creates a finalized rename-style checkpoint without a commit marker."""
    tmp_path = atomicity.AtomicRenameTemporaryPath.from_final(
        self.directory / name
    )
    await tmp_path.create()
    await tmp_path.finalize()
    final_path = tmp_path.get_final()
    (final_path / atomicity_types.COMMIT_SUCCESS_FILE).unlink()
    return final_path

  @parameterized.parameters(
      (None,),
      (options_lib.AtomicityOptions(),),
      (
          options_lib.AtomicityOptions(
              mode=options_lib.AtomicityMode.COMMIT_FILE
          ),
      ),
  )
  async def test_gcsfuse_does_not_read_legacy_checkpoint_without_flag(
      self, atomicity_options
  ):
    final_path = await self._create_legacy_finalized('step_1')
    with mock.patch.object(gcs_utils, 'is_gcsfuse_path', return_value=True):
      self.assertFalse(
          await temporary_paths.is_path_finalized(
              final_path, atomicity_options=atomicity_options
          )
      )
      # Without the marker the directory reads as an in-progress commit-file
      # save, so cleanup passes will remove it.
      self.assertTrue(
          await temporary_paths.is_path_temporary(
              final_path, atomicity_options=atomicity_options
          )
      )

  async def test_gcsfuse_treats_rename_tmp_dir_as_temporary(self):
    tmp_path = atomicity.AtomicRenameTemporaryPath.from_final(
        self.directory / 'step_1'
    )
    await tmp_path.create()
    with mock.patch.object(gcs_utils, 'is_gcsfuse_path', return_value=True):
      self.assertTrue(await temporary_paths.is_path_temporary(tmp_path.get()))
      self.assertFalse(await temporary_paths.is_path_finalized(tmp_path.get()))

  @parameterized.parameters((True,), (False,))
  async def test_explicit_flag_enables_fallback(self, gcsfuse_detected):
    final_path = await self._create_legacy_finalized('step_1')
    atomicity_options = options_lib.AtomicityOptions(
        mode=options_lib.AtomicityMode.COMMIT_FILE,
        allow_legacy_atomic_rename=True,
    )
    with mock.patch.object(
        gcs_utils, 'is_gcsfuse_path', return_value=gcsfuse_detected
    ):
      self.assertTrue(
          await temporary_paths.is_path_finalized(
              final_path, atomicity_options=atomicity_options
          )
      )
      self.assertFalse(
          await temporary_paths.is_path_temporary(
              final_path, atomicity_options=atomicity_options
          )
      )


if __name__ == '__main__':
  absltest.main()
