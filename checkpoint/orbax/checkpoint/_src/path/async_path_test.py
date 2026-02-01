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
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint._src.path import async_path


class AsyncPathTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = epath.Path(self.create_tempdir().full_path)

  def test_open_read(self):
    test_file = self.test_dir / 'test.txt'
    test_file.write_text('hello world')

    async def _test():
      async with async_path.open_file(test_file, 'r') as f:
        content = await f.read()
        self.assertEqual(content, 'hello world')

        await f.seek(0)
        content_partial = await f.read(5)
        self.assertEqual(content_partial, 'hello')

    asyncio.run(_test())

  def test_open_binary_read(self):
    test_file = self.test_dir / 'test.bin'
    data = b'\x00\x01\x02\x03'
    test_file.write_bytes(data)

    async def _test():
      async with async_path.open_file(test_file, 'rb') as f:
        content = await f.read()
        self.assertEqual(content, data)

        await f.seek(1)
        content_partial = await f.read(2)
        self.assertEqual(content_partial, b'\x01\x02')

    asyncio.run(_test())

  def test_seek(self):
    test_file = self.test_dir / 'seek.txt'
    test_file.write_bytes(b'0123456789')

    async def _test():
      async with async_path.open_file(test_file, 'rb') as f:
        await f.seek(2)
        self.assertEqual(await f.read(1), b'2')

        await f.seek(2, 1)  # Seek from current (3) + 2 = 5
        self.assertEqual(await f.read(1), b'5')

        await f.seek(-2, 2)  # Seek from end (10) - 2 = 8
        self.assertEqual(await f.read(1), b'8')

    asyncio.run(_test())

  def test_tell(self):
    test_file = self.test_dir / 'tell.txt'
    test_file.write_text('0123456789')

    async def _test():
      async with async_path.open_file(test_file, 'r') as f:
        self.assertEqual(await f.tell(), 0)
        await f.read(5)
        self.assertEqual(await f.tell(), 5)
        await f.seek(2)
        self.assertEqual(await f.tell(), 2)

    asyncio.run(_test())

  def test_write(self):
    test_file = self.test_dir / 'write.txt'

    async def _test():
      async with async_path.open_file(test_file, 'w') as f:
        await f.write('hello')
        await f.flush()
        self.assertTrue(test_file.exists())
        self.assertEqual(test_file.read_text(), 'hello')

    asyncio.run(_test())

  def test_close(self):
    test_file = self.test_dir / 'close.txt'
    test_file.write_text('content')

    async def _test():
      async with async_path.open_file(test_file, 'r') as f:
        await f.close()
        # Verify we can't read after close.
        # Note: The underlying file object raises ValueError on I/O operation
        # on closed file. Since we wrap it in run_in_executor, the exception
        # should be propagated.
        with self.assertRaises(ValueError):
          await f.read()

    asyncio.run(_test())

  def test_concurrent_reads(self):
    test_file = self.test_dir / 'concurrent_reads.txt'
    test_file.write_text('0123456789')

    async def _test():
      async def read_chunk(offset, size):
        async with async_path.open_file(test_file, 'r') as f:
          await f.seek(offset)
          return await f.read(size)

      results = await asyncio.gather(
          read_chunk(0, 5),
          read_chunk(5, 5),
          read_chunk(2, 7),
      )
      self.assertEqual(results, ['01234', '56789', '2345678'])

    asyncio.run(_test())


if __name__ == '__main__':
  absltest.main()
