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
from typing import Any, Awaitable

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint.experimental.v1._src.handlers import descriptor_handler
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.testing import path_utils as path_test_utils
from orbax.google.proto import descriptor_pb2

PathAwaitingCreation = path_types.PathAwaitingCreation
PathLike = path_types.PathLike
Path = path_types.Path


async def _run_awaitable(awaitable: Awaitable[Any]) -> Any:
  """Runs and returns the result of an awaitable."""
  return await awaitable


class DescriptorHandler:
  """Wrapper around DescriptorHandler that can block on save and load."""

  def __init__(self, **kwargs):
    self._handler = descriptor_handler.DescriptorHandler(**kwargs)

  def save(self, directory: Path, checkpointable: descriptor_pb2.Descriptor):
    path = path_test_utils.PathAwaitingCreationWrapper(directory)
    awaitable = asyncio.run(self._handler.save(path, checkpointable))
    return asyncio.run(_run_awaitable(awaitable))

  def load(
      self,
      path: Path,
  ) -> descriptor_pb2.Descriptor:
    awaitable = self.load_async(path)
    return asyncio.run(_run_awaitable(awaitable))

  def load_async(
      self,
      path: Path,
  ) -> Awaitable[descriptor_pb2.Descriptor]:
    return asyncio.run(self._handler.load(path))


class DescriptorHandlerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='checkpointing_test').full_path
    )

  def test_save_load(self):
    proto = descriptor_pb2.Descriptor(uuid='test_uuid')
    handler = DescriptorHandler()
    handler.save(
        directory=self.directory,
        checkpointable=proto,
    )
    self.assertTrue((self.directory / 'descriptor.pbtxt').exists())
    self.assertTrue((self.directory / 'uuid-test_uuid').exists())

    restored = handler.load(self.directory)
    self.assertEqual(proto, restored)

  def test_is_handleable(self):
    handler = descriptor_handler.DescriptorHandler()
    self.assertTrue(handler.is_handleable(descriptor_pb2.Descriptor()))
    self.assertFalse(handler.is_handleable({'a': 1}))
    self.assertFalse(handler.is_handleable(1))
    self.assertFalse(handler.is_handleable('str'))

  def test_is_abstract_handleable(self):
    handler = descriptor_handler.DescriptorHandler()
    self.assertTrue(handler.is_abstract_handleable(descriptor_pb2.Descriptor))
    self.assertFalse(
        handler.is_abstract_handleable(descriptor_pb2.Descriptor())
    )
    self.assertFalse(handler.is_abstract_handleable(dict))
    self.assertFalse(handler.is_abstract_handleable(1))


if __name__ == '__main__':
  absltest.main()
