# Copyright 2023 The Orbax Authors.
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

"""Tests custom handler registration and checkpoint saving."""

import asyncio
from concurrent import futures
import dataclasses
import functools
import glob
import json
import os
from typing import List, Optional, Sequence

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np
import orbax.checkpoint as ocp
from orbax.checkpoint import test_utils

ParamInfo = ocp.pytree_checkpoint_handler.ParamInfo
Metadata = ocp.value_metadata.Metadata


@dataclasses.dataclass
class MyState:
  a: np.array
  b: np.array

  def __eq__(self, other):
    return (self.a == other.a).all() and (self.b == other.b).all()


@dataclasses.dataclass
class MyStateSaveArgs(ocp.SaveArgs):
  # When False, saves as JSON instead of numpy.
  use_npz: bool = True


@dataclasses.dataclass
class MyStateRestoreArgs(ocp.RestoreArgs):
  # When true, restores as `dict` instead of creating a new `MyState`.
  restore_as_dict: bool = False


class MyStateHandler(ocp.pytree_checkpoint_handler.TypeHandler):
  """Serializes MyState to the numpy npz or JSON format."""

  def __init__(self):
    self._executor = futures.ThreadPoolExecutor(max_workers=1)

  def typestr(self) -> str:
    return 'MyState'

  async def serialize(
      self,
      values: Sequence[MyState],
      infos: Sequence[ParamInfo],
      args: Optional[Sequence[MyStateSaveArgs]],
  ) -> List[futures.Future[str]]:
    ret = []
    for value, info, arg in zip(values, infos, args):
      if arg is None:
        arg = MyStateSaveArgs()
      ret.append(
          self._executor.submit(
              functools.partial(_write_state, value, info.path, arg)
          )
      )
    return ret

  async def deserialize(
      self,
      infos: Sequence[ParamInfo],
      args: Optional[Sequence[MyStateRestoreArgs]] = None,
  ) -> MyState:
    ret = []
    for info, arg in zip(infos, args):
      if arg is None:
        arg = MyStateRestoreArgs()
      ret.append(
          await asyncio.get_event_loop().run_in_executor(
              self._executor, functools.partial(_from_state, info.path, arg)
          )
      )
    return await asyncio.gather(*ret)

  async def metadata(self, infos: Sequence[ParamInfo]) -> Sequence[Metadata]:
    return [Metadata()] * len(infos)


def _write_state(state: MyState, path: epath.Path, arg: MyStateSaveArgs) -> str:
  """Writes MyState to the path with the specified args."""
  if arg.use_npz:
    np.savez(path / 'my_state.npz', a=state.a, b=state.b)
  else:
    with open(path / 'my_state.json', 'w') as f:
      f.write(json.dumps(dict(a=state.a.tolist(), b=state.b.tolist())))
  return path


async def _from_state(
    path: epath.Path, arg: Optional[MyStateRestoreArgs] = None
) -> MyState:
  # Detect whether the state is saved to numpy or json
  files = glob.glob(os.fspath(path / '*.*'))
  is_npz = '.npz' in files[0]
  if is_npz:
    with np.load(path / 'my_state.npz') as f:
      data = {'a': f['a'], 'b': f['b']}
  else:
    with open(path / 'my_state.json', 'r') as f:
      json_data = json.loads(f.read())
      data = {'a': np.array(json_data['a']), 'b': np.array(json_data['b'])}
  if arg.restore_as_dict:
    return data
  else:
    return MyState(**data)


ocp.type_handlers.register_type_handler(
    MyState,
    MyStateHandler(),
    override=True
)

_TYPE = MyState
_TYPE_STR = MyStateHandler().typestr()


class CustomTypeHandlerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='checkpointing_test').full_path
    )

  @parameterized.named_parameters(('with_type', _TYPE), ('type_str', _TYPE_STR))
  def test_registration(self, obj_type):
    self.assertIsInstance(
        ocp.type_handlers.get_default_restore_args(obj_type), MyStateRestoreArgs
    )
    self.assertIsInstance(
        ocp.type_handlers.get_default_save_args(obj_type), MyStateSaveArgs
    )
    self.assertIsInstance(
        ocp.type_handlers.get_type_handler(obj_type), MyStateHandler
    )

  def test_no_registration(self):
    # Create fake temporary class with the same name as a registered class.
    class MyState:  # pylint: disable=redefined-outer-name
      pass

    with self.assertRaisesRegex(ValueError, 'Unknown type'):
      ocp.type_handlers.get_type_handler(MyState)

  @parameterized.named_parameters(
      ('save_default_no_restore_args', None, False),
      ('save_default_with_restore_args', None, True),
      ('save_npz_no_restore_args', MyStateSaveArgs(use_npz=True), True),
      ('save_json_with_restore_args', MyStateSaveArgs(use_npz=False), True),
  )
  def test_save(self, my_state_save_args, use_restore_args):
    my_tree = {
        'state': {'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])},
        'my_state': MyState(a=np.array([10, 20, 30]), b=np.array([40, 50, 60])),
    }

    checkpointer = ocp.Checkpointer(
        ocp.PyTreeCheckpointHandler(write_tree_metadata=True)
    )
    path = epath.Path(self.directory / 'my_tree')

    save_args = None
    if my_state_save_args is not None:
      save_args = {
          'my_state': my_state_save_args,
          'state': {'a': ocp.SaveArgs(), 'b': ocp.SaveArgs()},
      }
    checkpointer.save(path / 'my_tree2', my_tree, save_args=save_args)

    if use_restore_args:
      restored = checkpointer.restore(
          path / 'my_tree2',
          restore_args={
              'my_state': MyStateRestoreArgs(restore_as_dict=True),
              'state': {'a': ocp.RestoreArgs(), 'b': ocp.RestoreArgs()},
          },
      )
      test_utils.assert_tree_equal(self, my_tree['state'], restored['state'])
      test_utils.assert_tree_equal(
          self,
          {'a': np.array([10, 20, 30]), 'b': np.array([40, 50, 60])},
          restored['my_state'],
      )
    else:
      restored = checkpointer.restore(path / 'my_tree2')
      test_utils.assert_tree_equal(self, my_tree, restored)


if __name__ == '__main__':
  absltest.main()
