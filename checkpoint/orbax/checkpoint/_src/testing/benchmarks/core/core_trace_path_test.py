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

"""Unit tests for TestContext.trace_path — the TB Profile-plugin layout routing."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint._src.testing.benchmarks.core import core


@dataclasses.dataclass(frozen=True)
class _TraceOpts:
  enable_trace: bool = True
  trace_every_repeat: bool = False


class TestContextTracePathTest(parameterized.TestCase):

  _OUTPUT_DIR = epath.Path('/tmp/bench')
  _NAME = 'MyBench_abc123'
  _EXPECTED_SAVE = (
      _OUTPUT_DIR / 'tensorboard' / 'plugins' / 'profile' / f'{_NAME}__save'
  )
  _EXPECTED_LOAD = (
      _OUTPUT_DIR / 'tensorboard' / 'plugins' / 'profile' / f'{_NAME}__load'
  )

  def _ctx(
      self,
      *,
      repeat_index=None,
      enable_trace=True,
      trace_every_repeat=False,
  ):
    path = self._OUTPUT_DIR / self._NAME
    if repeat_index is not None:
      path = path / f'repeat_{repeat_index}'
    return core.TestContext(
        pytree=None,
        path=path,
        options=_TraceOpts(
            enable_trace=enable_trace,
            trace_every_repeat=trace_every_repeat,
        ),
        mesh=None,
        repeat_index=repeat_index,
        output_dir=self._OUTPUT_DIR,
        name=self._NAME,
    )

  def test_no_repeat_returns_tb_profile_layout(self):
    self.assertEqual(self._ctx().trace_path('save'), self._EXPECTED_SAVE)

  def test_first_repeat_captures(self):
    self.assertEqual(
        self._ctx(repeat_index=0).trace_path('load'), self._EXPECTED_LOAD
    )

  def test_non_first_repeat_skipped_by_default(self):
    self.assertIsNone(self._ctx(repeat_index=2).trace_path('save'))

  def test_trace_every_repeat_captures_all(self):
    self.assertEqual(
        self._ctx(repeat_index=2, trace_every_repeat=True).trace_path('save'),
        self._EXPECTED_SAVE,
    )

  def test_disabled_returns_none(self):
    self.assertIsNone(self._ctx(enable_trace=False).trace_path('save'))

  def test_disabled_overrides_trace_every_repeat(self):
    self.assertIsNone(
        self._ctx(
            repeat_index=0,
            enable_trace=False,
            trace_every_repeat=True,
        ).trace_path('save')
    )

  def test_missing_output_dir_returns_none(self):
    ctx = core.TestContext(
        pytree=None,
        path=self._OUTPUT_DIR / self._NAME,
        options=_TraceOpts(),
        mesh=None,
    )
    self.assertIsNone(ctx.trace_path('save'))


if __name__ == '__main__':
  absltest.main()
