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

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.serialization import compression_options
from orbax.checkpoint._src.serialization import types


class CompressionOptionsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='defaults_to_zstd',
          compression=None,
          use_compression=None,
          expected=options_lib.CompressionPreset.ZSTD,
      ),
      dict(
          testcase_name='use_compression_true',
          compression=None,
          use_compression=True,
          expected=options_lib.CompressionPreset.ZSTD,
      ),
      dict(
          testcase_name='use_compression_false',
          compression=None,
          use_compression=False,
          expected=options_lib.CompressionPreset.NONE,
      ),
      dict(
          testcase_name='explicit_none',
          compression=options_lib.CompressionPreset.NONE,
          use_compression=None,
          expected=options_lib.CompressionPreset.NONE,
      ),
      dict(
          testcase_name='explicit_zstd',
          compression=options_lib.CompressionPreset.ZSTD,
          use_compression=None,
          expected=options_lib.CompressionPreset.ZSTD,
      ),
  )
  def test_resolve_compression_preset(
      self,
      compression: options_lib.CompressionPreset | None,
      use_compression: bool | None,
      expected: options_lib.CompressionPreset,
  ):
    self.assertEqual(
        compression_options.resolve_compression_preset(
            compression=compression,
            use_compression=use_compression,
        ),
        expected,
    )

  def test_resolve_compression_preset_rejects_both(self):
    with self.assertRaisesRegex(
        ValueError, 'Cannot specify both use_compression and compression'
    ):
      compression_options.resolve_compression_preset(
          compression=options_lib.CompressionPreset.ZSTD,
          use_compression=False,
      )

  @parameterized.parameters('zstd', 'invalid', 123)
  def test_resolve_compression_preset_rejects_invalid_type(self, invalid: Any):
    with self.assertRaisesRegex(ValueError, 'Unsupported compression preset'):
      compression_options.resolve_compression_preset(compression=invalid)

  @parameterized.named_parameters(
      dict(
          testcase_name='none',
          preset=options_lib.CompressionPreset.NONE,
          expected=types.NoCompression(),
      ),
      dict(
          testcase_name='zstd',
          preset=options_lib.CompressionPreset.ZSTD,
          expected=types.ZstdCompression(),
      ),
  )
  def test_compression_spec_from_preset(
      self,
      preset: options_lib.CompressionPreset,
      expected: types.CompressionSpec,
  ):
    self.assertEqual(
        compression_options.compression_spec_from_preset(preset),
        expected,
    )

  def test_compression_spec_from_preset_rejects_invalid(self):
    invalid_preset: Any = 'invalid'
    with self.assertRaisesRegex(ValueError, 'Unsupported compression preset'):
      compression_options.compression_spec_from_preset(invalid_preset)


if __name__ == '__main__':
  absltest.main()
