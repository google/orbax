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

"""Tests for checkpoint_info module."""

from concurrent import futures
import datetime
import time
from absl.testing import absltest
from orbax.checkpoint._src.metadata import checkpoint_info

CheckpointInfo = checkpoint_info.CheckpointInfo


def build_info(step: int) -> CheckpointInfo:
  return CheckpointInfo(step=step, time=datetime.datetime.now(), metrics=None)


class CheckpointInfoTest(absltest.TestCase):

  def test_latest(self):
    info_list = [build_info(i) for i in range(10)]
    infos = checkpoint_info.CheckpointInfos(info_list)

    self.assertEqual(infos.latest(), info_list[-1])

  def test_latest_empty(self):
    infos = checkpoint_info.CheckpointInfos([])

    self.assertIsNone(infos.latest())

  def test_delete_if(self):
    info_list = [build_info(i) for i in range(10)]
    delete_fn = lambda info: info.step % 2 != 0
    expected_remaining_steps = [i for i in range(10) if i % 2 == 0]
    infos = checkpoint_info.CheckpointInfos(info_list)

    infos.delete_if(delete_fn)

    self.assertEqual([info.step for info in infos], expected_remaining_steps)

  def test_delete_if_no_match(self):
    info_list = [build_info(i) for i in range(10)]
    delete_fn = lambda info: False
    expected_remaining_steps = list(range(10))
    infos = checkpoint_info.CheckpointInfos(info_list)

    infos.delete_if(delete_fn)

    self.assertEqual([info.step for info in infos], expected_remaining_steps)


class CheckpointInfoThreadSafetyTest(absltest.TestCase):

  def test_set_with_iter(self):
    infos = checkpoint_info.CheckpointInfos()
    self.assertEmpty(infos)

    num_repeat = 50
    num_threads = 10

    def _set(steps):
      for _ in range(num_repeat):
        infos.set([build_info(i) for i in steps])

    def _iter():
      for _ in range(num_repeat):
        for _ in infos:
          pass

    fs = []
    with futures.ThreadPoolExecutor() as executor:
      for _ in range(num_threads):
        fs.append(executor.submit(_set, range(5)))
        fs.append(executor.submit(_set, range(6, 10)))
        fs.append(executor.submit(_iter))
      futures.wait(fs)

    steps = [info.step for info in infos]
    self.assertTrue(steps == list(range(5)) or steps == list(range(6, 10)))

  def test_append_with_iter(self):
    infos = checkpoint_info.CheckpointInfos()
    self.assertEmpty(infos)

    num_repeat = 50
    num_threads = 10

    def _append(steps):
      for _ in range(num_repeat):
        infos.set([])
        for i in steps:
          infos.append(build_info(i))

    def _iter():
      for _ in range(num_repeat):
        for _ in infos:
          pass

    fs = []
    with futures.ThreadPoolExecutor() as executor:
      for _ in range(num_threads):
        fs.append(executor.submit(_append, range(5)))
        fs.append(executor.submit(_append, range(6, 10)))
        fs.append(executor.submit(_iter))
      futures.wait(fs)

    steps = [info.step for info in infos]
    self.assertTrue(
        steps == list(range(5)) or steps == list(range(6, 10)), f"Found {steps}"
    )

  def test_append_with_delete_if(self):
    infos = checkpoint_info.CheckpointInfos()
    self.assertEmpty(infos)

    num_repeat = 50
    num_threads = 10

    def _append(steps):
      for _ in range(num_repeat):
        infos.set([])
        for i in steps:
          infos.append(build_info(i))

    def _delete():
      for _ in range(num_repeat):
        infos.delete_if(lambda info: True)

    fs = []
    with futures.ThreadPoolExecutor() as executor:
      for _ in range(num_threads):
        fs.append(executor.submit(_append, range(5)))
        fs.append(executor.submit(_append, range(6, 10)))
        fs.append(executor.submit(_delete))
      futures.wait(fs)

    steps = [info.step for info in infos]
    self.assertTrue(
        steps == list(range(5)) or steps == list(range(6, 10)) or not steps,
        f"Found steps={steps}",
    )

  def test_append_with_delete_if_with_iter(self):
    infos = checkpoint_info.CheckpointInfos()
    self.assertEmpty(infos)

    num_repeat = 50
    num_threads = 10

    def _append(steps):
      for _ in range(num_repeat):
        new_infos = [build_info(i) for i in steps]
        infos.set(new_infos)

    def _delete():
      for _ in range(num_repeat):
        infos.delete_if(lambda info: True)

    def _iter():
      for _ in range(num_repeat):
        for _ in infos:
          pass

    fs = []
    with futures.ThreadPoolExecutor() as executor:
      for _ in range(num_threads):
        fs.append(executor.submit(_append, range(5)))
        fs.append(executor.submit(_append, range(6, 10)))
        fs.append(executor.submit(_delete))
        fs.append(executor.submit(_iter))
      futures.wait(fs)

    steps = [info.step for info in infos]
    self.assertTrue(
        steps == list(range(5)) or steps == list(range(6, 10)) or not steps,
        f"Found steps={steps}",
    )

  def test_lock_timeout(self):
    infos = checkpoint_info.CheckpointInfos(
        [build_info(i) for i in range(2)], timeout_sec=0.01
    )

    def _do_with_lock():
      for _ in infos:
        time.sleep(0.5)  # delay longer than timeout
        pass

    with futures.ThreadPoolExecutor() as executor:
      f1 = executor.submit(_do_with_lock)
      f2 = executor.submit(_do_with_lock)

      with self.assertRaises(TimeoutError):
        f1.result()
        f2.result()


if __name__ == "__main__":
  absltest.main()
