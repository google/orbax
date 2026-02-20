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

from absl.testing import absltest
from etils import epath
from orbax.checkpoint._src.path import types as path_types
from orbax.checkpoint.google.path import tfhub_atomicity


class AwaitAndResolveAsPosixTest(absltest.TestCase):

  def test_resolves_epath_immediately(self):
    async def _test():
      path = epath.Path('/tmp/test/dir')
      result = await path_types.await_and_resolve_as_posix(path)
      self.assertEqual(result, '/tmp/test/dir')

    asyncio.run(_test())

  def test_resolves_deferred_path_after_set(self):
    async def _test():
      directory = self.create_tempdir().full_path
      deferred = tfhub_atomicity.DeferredPath()
      deferred.set_path(epath.Path(directory))
      result = await path_types.await_and_resolve_as_posix(deferred)
      self.assertEqual(result, directory)

    asyncio.run(_test())

  def test_blocks_until_deferred_path_set(self):
    async def _test():
      directory = self.create_tempdir().full_path
      deferred = tfhub_atomicity.DeferredPath()
      task = asyncio.create_task(
          path_types.await_and_resolve_as_posix(deferred)
      )

      await asyncio.sleep(0.1)
      self.assertFalse(task.done())

      deferred.set_path(epath.Path(directory))
      result = await task
      self.assertEqual(result, directory)

    asyncio.run(_test())

  def test_resolves_child_deferred_path(self):
    async def _test():
      directory = self.create_tempdir().full_path
      deferred = tfhub_atomicity.DeferredPath()
      child = deferred / 'subdir'
      task = asyncio.create_task(path_types.await_and_resolve_as_posix(child))

      await asyncio.sleep(0.1)
      self.assertFalse(task.done())

      deferred.set_path(epath.Path(directory))
      result = await task
      self.assertEqual(result, directory + '/subdir')

    asyncio.run(_test())


if __name__ == '__main__':
  absltest.main()
