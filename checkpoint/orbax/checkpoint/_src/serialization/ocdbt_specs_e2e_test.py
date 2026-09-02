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

"""End-to-end tests for distributed arrays serialization with OCDBT."""

import asyncio
import dataclasses
from typing import TypeAlias
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np
from orbax.checkpoint._src.arrays import fragments as fragments_lib
from orbax.checkpoint._src.arrays import subchunking
from orbax.checkpoint._src.arrays import types as arrays_types
from orbax.checkpoint._src.serialization import ocdbt_process_spec
from orbax.checkpoint._src.serialization import ocdbt_utils
from orbax.checkpoint._src.serialization import tensorstore_utils
import tensorstore as ts


@dataclasses.dataclass(frozen=True)
class TestArray:
  """A test array.

  Attributes:
    name: The name of the array.
    value: The value (unsharded) of the array.
    fragments_by_process_id: A mapping describing which array fragments should
      be written by which test process.
  """
  name: str
  value: np.ndarray
  fragments_by_process_id: dict[str, fragments_lib.NpFragments]


TestData: TypeAlias = tuple[TestArray, ...]


def build_test_data() -> TestData:
  """Builds test arrays. Test setup simulates 4 test processes."""
  # Unsharded array.
  a = np.arange(10)

  # Array sharded across 2 out of 4 test processes.
  b = np.arange(4* 5).reshape(4, 5)
  b_sharded_shape = (1, 5)

  # Array sharded across 2 out of 4 test processes. Large enough to trigger
  # writing of an OCDBT "value" data file
  b_large = np.arange(4 * 10 * 1024).reshape(4, 10, 1024)
  b_large_sharded_shape = (1, 10, 1024)

  # Array sharded across 3 out of 4 test processes.
  c = np.arange(6 * 13).reshape(6, 13)
  c_sharded_shape = (2, 13)

  # Array sharded across all 4 test processes.
  d = np.arange(3 * 12 * 8).reshape(3, 12, 8)
  d_sharded_shape = (3, 6, 4)

  def _generate_fragments_by_process_id(
      array: np.ndarray,
      sharded_shape: arrays_types.Shape,
      num_processes: int,
  ) -> dict[str, fragments_lib.NpFragments]:
    """Generates fragments_by_process_id for a given array."""
    sharded_fragments = subchunking.chunk_fragments(
        fragments_lib.NpFragments.all_of(array), sharded_shape
    )
    num_sharded_fragments = len(sharded_fragments.fragments)
    assert num_sharded_fragments % num_processes == 0
    fragments_per_process = num_sharded_fragments // num_processes
    return {
        f"h{i}": fragments_lib.NpFragments(
            shape=array.shape,
            dtype=array.dtype,
            fragments=sharded_fragments.fragments[
                (i * fragments_per_process) : (i + 1) * fragments_per_process
            ],
        )
        for i in range(num_processes)
    }

  return (
      TestArray(
          name="a",
          value=a,
          fragments_by_process_id={"h0": fragments_lib.NpFragments.all_of(a)},
      ),
      TestArray(
          name="b",
          value=b,
          fragments_by_process_id=_generate_fragments_by_process_id(
              b, b_sharded_shape, num_processes=2
          ),
      ),
      TestArray(
          name="b_large",
          value=b_large,
          fragments_by_process_id=_generate_fragments_by_process_id(
              b_large, b_large_sharded_shape, num_processes=2
          ),
      ),
      TestArray(
          name="c",
          value=c,
          fragments_by_process_id=_generate_fragments_by_process_id(
              c, c_sharded_shape, num_processes=3
          ),
      ),
      TestArray(
          name="d",
          value=d,
          fragments_by_process_id=_generate_fragments_by_process_id(
              d, d_sharded_shape, num_processes=4
          ),
      ),
  )


def all_process_ids(test_data: TestData) -> set[str]:
  """Returns all unique process ids in the given test data."""
  return set(
      [
          process_id  # pylint: disable=g-complex-comprehension
          for array in test_data
          for process_id in array.fragments_by_process_id
      ]
  )


def _should_create_ts(fragments: fragments_lib.NpFragments) -> bool:
  """Determines if TensorStore array metadata needs to be written."""
  # Only do this if the fragments contain the "leading" array element
  # (0th in array's flat index space).
  return any((fragment.start == 0).all() for fragment in fragments.fragments)


async def _write_array(
    name: str,
    path: epath.Path,
    array_fragments: fragments_lib.NpFragments,
    process_id: str,
    ts_context: ts.Context,
    *,
    store_ocdbt_metadata_and_values_separately: bool = False,
) -> None:
  """Writes array fragments to the given path with the given process id."""
  array_write_tspec = tensorstore_utils.ArrayWriteSpec(
      directory=path.as_posix(),
      relative_array_filename=name,
      global_shape=array_fragments.shape,
      write_shape=array_fragments.fragments[0].value.shape,
      dtype=array_fragments.dtype,
      use_ocdbt=True,
      process_id=process_id,
      store_ocdbt_metadata_and_values_separately=(
          store_ocdbt_metadata_and_values_separately
      ),
  ).json

  if _should_create_ts(array_fragments):
    # Open with create=True once (`should_create` is supposed to be True for
    # only one of the hosts), so that the metadata (.zarray) is written once.
    array_ts = await ts.open(
        array_write_tspec,
        context=ts_context,
        open=True,
        create=True,
    )
  else:
    array_ts = await ts.open(
        array_write_tspec,
        context=ts_context,
        open=True,
        assume_metadata=True,
    )

  write_futures = []
  for fragment in array_fragments.fragments:
    write_futures.append(array_ts[fragment.index].write(fragment.value))

  await asyncio.gather(*write_futures)


async def _read_array(
    name: str,
    path: epath.Path,
    ts_context: ts.Context,
) -> np.ndarray:
  """Reads an array with the given name from the given path."""
  array_ts = await ts.open(
      {
          "driver": "zarr",
          "kvstore": tensorstore_utils.build_kvstore_tspec(
              path.as_posix(),
              name,
              use_ocdbt=True,
          ),
      },
      context=ts_context,
      read=True,
  )
  return await array_ts.read()


async def _verify_array_data(test_data: TestData, path: epath.Path) -> None:
  """Verifies all test arrays' data in the given directory."""
  ts_context = tensorstore_utils.get_ts_context(use_ocdbt=True)
  for array in test_data:
    read_array_value = await _read_array(array.name, path, ts_context)
    np.testing.assert_array_equal(read_array_value, array.value)


class OcdbtSpecsE2eTest(
    unittest.IsolatedAsyncioTestCase, parameterized.TestCase
):

  def _verify_per_process_ocdbt_files(
      self,
      test_data: TestData,
      path: epath.Path,
      store_ocdbt_metadata_and_values_separately: bool,
  ) -> None:
    """Verifies key per-process OCDBT files at the given path."""
    for process_id in all_process_ids(test_data):
      process_spec = ocdbt_process_spec.OcdbtProcessSpec(process_id=process_id)
      process_dir = path / str(process_spec)
      self.assertTrue((process_dir / "manifest.ocdbt").exists())
      # Values and metadata directory naming depends on the
      # store_ocdbt_metadata_and_values_separately flag.
      if store_ocdbt_metadata_and_values_separately:
        self.assertFalse((process_dir / "d").is_dir())
        self.assertTrue((process_dir / "ocdbt_meta").is_dir())
        # b_large should have generated files written to ocdbt_data/ subdir.
        if process_id in ("h0", "h1"):
          self.assertTrue((process_dir / "ocdbt_data").is_dir())
      else:
        self.assertTrue((process_dir / "d").is_dir())
        self.assertFalse((process_dir / "ocdbt_meta").is_dir())
        self.assertFalse((process_dir / "ocdbt_data").is_dir())

  @parameterized.product(
      store_ocdbt_metadata_and_values_separately=(False, True),
  )
  async def test_write(self, store_ocdbt_metadata_and_values_separately: bool):
    test_dir = epath.Path(self.create_tempdir()) / "test_data"
    test_dir.mkdir(parents=True, exist_ok=True)

    test_data = build_test_data()

    # Create process-specific subdirectories.
    for process_id in all_process_ids(test_data):
      spec = ocdbt_process_spec.OcdbtProcessSpec(process_id=process_id)
      (test_dir / str(spec)).mkdir(parents=False, exist_ok=False)

    ts_context = tensorstore_utils.get_ts_context(use_ocdbt=True)

    async def _write():
      write_futures = []
      for array in test_data:
        for process_id, fragments in array.fragments_by_process_id.items():
          write_futures.append(
              _write_array(
                  array.name,
                  test_dir,
                  fragments,
                  process_id,
                  ts_context,
                  store_ocdbt_metadata_and_values_separately=(
                      store_ocdbt_metadata_and_values_separately
                  ),
              )
          )
      await asyncio.gather(*write_futures)

    await _write()
    self._verify_per_process_ocdbt_files(
        test_data,
        test_dir,
        store_ocdbt_metadata_and_values_separately,
    )

    await ocdbt_utils.merge_ocdbt_per_process_files(
        test_dir, ts_context, use_zarr3=False, enable_validation=False
    )
    await _verify_array_data(test_data, test_dir)


if __name__ == "__main__":
  absltest.main()
