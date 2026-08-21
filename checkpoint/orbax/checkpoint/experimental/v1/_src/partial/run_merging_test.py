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

from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from etils import epath
import jax
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.partial import merging
from orbax.checkpoint.experimental.v1._src.partial import run_merging


class RunMergingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.out_path = self.create_tempdir().full_path
    self.in_paths = [self.create_tempdir().full_path]

  @mock.patch.object(
      orbax_layout.OrbaxLayout, 'validate', new_callable=mock.AsyncMock
  )
  @mock.patch.object(merging, 'merge_checkpoints', autospec=True)
  @mock.patch.object(jax, 'process_index', return_value=0)
  def test_main_success(self, _, mock_merge, mock_validate):
    with flagsaver.flagsaver(
        in_paths=self.in_paths,
        out_path=self.out_path,
        per_host_memory_limit_bytes=1024,
    ):
      run_merging.main([])

    mock_validate.assert_called()
    mock_merge.assert_called_once()

  @mock.patch.object(
      orbax_layout.OrbaxLayout, 'validate', new_callable=mock.AsyncMock
  )
  @mock.patch.object(jax, 'process_index', return_value=0)
  def test_main_invalid_output_not_empty(self, *_):
    out_path = epath.Path(self.out_path)
    (out_path / 'some_file').write_text('content')

    with flagsaver.flagsaver(
        in_paths=self.in_paths,
        out_path=self.out_path,
        per_host_memory_limit_bytes=1024,
    ):
      with self.assertRaisesRegex(ValueError, 'not empty'):
        run_merging.main([])

  @mock.patch.object(
      orbax_layout.OrbaxLayout, 'validate', new_callable=mock.AsyncMock
  )
  @mock.patch.object(jax, 'process_index', return_value=0)
  def test_main_invalid_input(self, _, mock_validate):
    mock_validate.side_effect = ValueError('Invalid checkpoint')

    with flagsaver.flagsaver(
        in_paths=self.in_paths,
        out_path=self.out_path,
        per_host_memory_limit_bytes=1024,
    ):
      with self.assertRaisesRegex(ValueError, 'is not a valid checkpoint'):
        run_merging.main([])


if __name__ == '__main__':
  absltest.main()
