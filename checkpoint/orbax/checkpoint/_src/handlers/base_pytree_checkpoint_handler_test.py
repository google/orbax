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

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.handlers import base_pytree_checkpoint_handler
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.serialization import types
import tensorstore as ts


BasePyTreeCheckpointHandler = (
    base_pytree_checkpoint_handler.BasePyTreeCheckpointHandler
)
BasePyTreeSaveArgs = base_pytree_checkpoint_handler.BasePyTreeSaveArgs
BasePyTreeRestoreArgs = base_pytree_checkpoint_handler.BasePyTreeRestoreArgs


def _read_zarr_compression(
    checkpoint_directory: epath.Path,
    param_name: str,
    *,
    use_zarr3: bool,
    use_ocdbt: bool,
) -> value_metadata.Compression | None:
  ts_spec = test_utils._get_simple_tensorstore_read_spec(  # pylint: disable=protected-access
      checkpoint_directory, param_name, use_zarr3, use_ocdbt
  )
  zarr_meta = ts_utils.zarr_metadata_from_tensorstore(ts.open(ts_spec).result())
  assert zarr_meta is not None
  return zarr_meta.compression


class BasePyTreeCheckpointHandlerCompressionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    test_utils.set_tensorstore_driver_for_test()
    self.directory = epath.Path(self.create_tempdir().full_path)

  @parameterized.product(
      use_ocdbt=(True, False),
      use_zarr3=(False, True),
      spec_and_expected=(
          (types.NoCompression(), None),
          (
              types.ZstdCompression(level=5),
              value_metadata.Compression('zstd', 5),
          ),
          (
              types.BloscCompression(cname='lz4', clevel=3),
              value_metadata.Compression('blosc_lz4', 3),
          ),
      ),
  )
  def test_save_and_restore_with_compression_spec(
      self,
      use_ocdbt: bool,
      use_zarr3: bool,
      spec_and_expected: tuple[
          types.CompressionSpec, value_metadata.Compression | None
      ],
  ):
    compression_spec, expected_compression = spec_and_expected
    item = {
        'a': np.arange(16, dtype=np.float32),
        'b': {'c': np.ones((4, 4), dtype=np.int32)},
    }
    handler = BasePyTreeCheckpointHandler(
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
        compression_spec=compression_spec,
    )
    self.addCleanup(handler.close)

    handler.save(self.directory, args=BasePyTreeSaveArgs(item))
    handler.finalize(self.directory)
    restored = handler.restore(self.directory, args=BasePyTreeRestoreArgs())

    np.testing.assert_array_equal(restored['a'], item['a'])
    np.testing.assert_array_equal(restored['b']['c'], item['b']['c'])
    self.assertEqual(
        _read_zarr_compression(
            self.directory,
            'a',
            use_zarr3=use_zarr3,
            use_ocdbt=use_ocdbt,
        ),
        expected_compression,
    )

  @parameterized.product(
      use_zarr3=(False, True),
      use_compression=(True, False),
  )
  def test_save_and_restore_with_deprecated_use_compression(
      self,
      use_zarr3: bool,
      use_compression: bool,
  ):
    item = {'a': np.arange(16, dtype=np.float32)}
    handler = BasePyTreeCheckpointHandler(
        use_ocdbt=True,
        use_zarr3=use_zarr3,
        use_compression=use_compression,
    )
    self.addCleanup(handler.close)

    handler.save(self.directory, args=BasePyTreeSaveArgs(item))
    handler.finalize(self.directory)
    restored = handler.restore(self.directory, args=BasePyTreeRestoreArgs())

    np.testing.assert_array_equal(restored['a'], item['a'])
    self.assertEqual(
        test_utils.is_compression_used(
            self.directory,
            'a',
            use_zarr3=use_zarr3,
            use_ocdbt=True,
            expected_codec='zstd' if use_compression else None,
        ),
        use_compression,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='false_and_zstd',
          use_compression=False,
          compression_spec=types.ZstdCompression(),
      ),
      dict(
          testcase_name='true_and_none',
          use_compression=True,
          compression_spec=types.NoCompression(),
      ),
  )
  def test_rejects_both_use_compression_and_compression_spec(
      self,
      use_compression: bool,
      compression_spec: types.CompressionSpec,
  ):
    with self.assertRaisesRegex(
        ValueError, 'Cannot specify both use_compression and compression_spec'
    ):
      BasePyTreeCheckpointHandler(
          use_compression=use_compression,
          compression_spec=compression_spec,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='no_compression',
          compression_spec=types.NoCompression(),
      ),
      dict(
          testcase_name='zstd_level_5',
          compression_spec=types.ZstdCompression(level=5),
      ),
  )
  def test_get_param_infos_defaults_to_handler_compression_spec(
      self,
      compression_spec: types.CompressionSpec,
  ):
    item = {'a': np.arange(4, dtype=np.float32)}
    handler = BasePyTreeCheckpointHandler(compression_spec=compression_spec)
    self.addCleanup(handler.close)

    param_infos = handler._get_param_infos(item, self.directory)
    self.assertEqual(param_infos['a'].compression_spec, compression_spec)

    override_spec = types.BloscCompression(cname='lz4', clevel=3)
    overridden = handler._get_param_infos(
        item, self.directory, compression_spec=override_spec
    )
    self.assertEqual(overridden['a'].compression_spec, override_spec)


if __name__ == '__main__':
  absltest.main()
