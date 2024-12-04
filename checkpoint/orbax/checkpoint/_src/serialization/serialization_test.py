# Copyright 2024 The Orbax Authors.
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
import math
import os
import pathlib
import tracemalloc as tm
from typing import Any
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import dtypes as _dtypes
from jax.experimental import layout
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import future
from orbax.checkpoint import test_utils
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.serialization import serialization
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
import tensorstore as ts


GSPMDSharding = jax.sharding.GSPMDSharding
NamedSharding = jax.sharding.NamedSharding
P = jax.sharding.PartitionSpec
DLL = layout.DeviceLocalLayout
Layout = layout.Layout

jax.config.update('jax_enable_x64', True)


def _dtype(x):
  if hasattr(x, 'dtype'):
    return x.dtype
  elif type(x) in _dtypes.python_scalar_dtypes:
    return np.dtype(_dtypes.python_scalar_dtypes[type(x)])
  else:
    return np.asarray(x).dtype


def serialize(arrs, tspecs):
  async def _serialize():
    await asyncio.gather(*[
        serialization.async_serialize(arr, tspec)
        for arr, tspec in zip(arrs, tspecs)
    ])

  asyncio_utils.run_sync(_serialize())
  test_utils.sync_global_processes('serialization_complete')


def deserialize(
    shardings, tensorstore_specs, global_shapes=None, dtypes=None, strict=True
):
  if global_shapes is None:
    global_shapes = [None for _ in tensorstore_specs]
  if dtypes is None:
    dtypes = [None for _ in tensorstore_specs]

  async def _deserialize():
    return await asyncio.gather(*[
        serialization.async_deserialize(
            sharding, tspec, shape, dtype, strict=strict
        )
        for sharding, tspec, shape, dtype in zip(
            shardings, tensorstore_specs, global_shapes, dtypes
        )
    ])

  result = asyncio_utils.run_sync(_deserialize())
  test_utils.sync_global_processes('deserialization_complete')
  return result


class FutureWithSpeedbump(future.Future):

  def __init__(self, f, speedbump):
    self._f = f
    self._speedbump = speedbump
    assert self._speedbump >= 0

  def result(self, timeout: int | None = None) -> Any:
    raise NotImplementedError()

  async def _sleep_and_result(self):
    await asyncio.sleep(self._speedbump)
    return await self._f

  def __await__(self):
    return self._sleep_and_result().__await__()


def create_global_mesh(mesh_shape, axis_names):
  size = math.prod(mesh_shape)
  if len(jax.devices()) < size:
    raise unittest.SkipTest(f'Test requires {size} global devices.')
  devices = sorted(jax.devices(), key=lambda d: d.id)
  mesh_devices = np.array(devices[:size]).reshape(mesh_shape)
  global_mesh = jax.sharding.Mesh(mesh_devices, axis_names)
  return global_mesh


class CheckpointTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt_dir = pathlib.Path(self.create_tempdir('ckpt').full_path)
    test_utils.sync_global_processes('CheckpointTest:setup_complete')

  def tearDown(self):
    test_utils.sync_global_processes('CheckpointTest:tests_complete')
    super().tearDown()

  def assertArraysEqual(
      self,
      x,
      y,
      *,
      check_dtypes=True,
      err_msg='',
      allow_object_dtype=False,
      verbose=True,
  ):
    """Assert that x and y arrays are exactly equal."""
    if check_dtypes:
      self.assertDtypesMatch(x, y)
    x = np.asarray(x)
    y = np.asarray(y)

    if (not allow_object_dtype) and (x.dtype == object or y.dtype == object):
      # See https://github.com/google/jax/issues/17867
      raise TypeError(
          'assertArraysEqual may be poorly behaved when np.asarray casts to'
          ' dtype=object. If comparing PRNG keys, consider'
          ' random_test.KeyArrayTest.assertKeysEqual. If comparing collections'
          ' of arrays, consider using assertAllClose. To let this test proceed'
          ' anyway, pass allow_object_dtype=True.'
      )

    # Work around https://github.com/numpy/numpy/issues/18992
    with np.errstate(over='ignore'):
      np.testing.assert_array_equal(x, y, err_msg=err_msg, verbose=verbose)

  def assertDtypesMatch(self, x, y):
    self.assertEqual(_dtype(x), _dtype(y))

  def test_memory_consumption(self):
    global_mesh = create_global_mesh((2, 4), ('x', 'y'))
    inp_shape = (2_048, 4_096)
    pspec = P('x', 'y')
    num = math.prod(inp_shape)
    sharding = NamedSharding(global_mesh, pspec)
    src = jnp.arange(num, dtype=np.int32).reshape(inp_shape)  # 8e9
    inp = jax.make_array_from_callback(
        inp_shape, sharding, lambda idx: src[idx]
    )
    tspec = serialization.get_tensorstore_spec(str(self.ckpt_dir))

    serialize(
        [inp],
        [tspec],
    )

    async def deserialize_with_byte_limit():
      r = await serialization.async_deserialize(
          sharding,
          tspec,
          inp_shape,
          byte_limiter=serialization.LimitInFlightBytes(4_200_000),
      )
      r.block_until_ready()

    tm.start()
    _, start_memory_usage = tm.get_traced_memory()
    asyncio_utils.run_sync(deserialize_with_byte_limit())
    _, peak_memory_usage = tm.get_traced_memory()
    # NB: some padding + tensorstore overhead. It should always be
    # less than array size (2048 * 4096 * 4 = 32M)
    self.assertLess(peak_memory_usage - start_memory_usage, 10_000_000)
    deserialize_wo_limit = serialization.async_deserialize(
        sharding, tspec, inp_shape
    )
    tm.clear_traces()
    _, start_memory_usage = tm.get_traced_memory()
    # NB: call block_until_ready() is important here and above
    # because otherwise this leads to racing condition and segfault with
    # tensorstore attempting to dealloc using tracemalloc which is already
    # destroyed.
    asyncio_utils.run_sync(deserialize_wo_limit).block_until_ready()

    _, peak_memory_usage = tm.get_traced_memory()
    # We load entire array in memory here.
    self.assertGreater(peak_memory_usage - start_memory_usage, 30_000_000)
    tm.stop()

  def test_checkpointing_jax_array(self):
    global_mesh = create_global_mesh((4, 2), ('x', 'y'))
    inp_shape = (8, 2)
    pspec = P('x', 'y')
    num = math.prod(inp_shape)

    # First Array
    global_input_data1 = np.arange(num, dtype=np.int32).reshape(inp_shape)
    a1 = jax.make_array_from_callback(
        inp_shape,
        NamedSharding(global_mesh, pspec),
        lambda idx: global_input_data1[idx],
    )

    # Second Array
    global_input_data2 = np.arange(num, num + num, dtype=np.int32).reshape(
        inp_shape
    )
    a2 = jax.make_array_from_callback(
        inp_shape,
        NamedSharding(global_mesh, pspec),
        lambda idx: global_input_data2[idx],
    )

    # Third Array
    def cb3(_):
      return np.array([], dtype=np.float32)

    global_mesh1d = create_global_mesh((8,), ('x',))
    a3 = jax.make_array_from_callback(
        (0,), NamedSharding(global_mesh1d, P(None)), cb3
    )

    ckpt_paths = [
        self.create_tempdir(f'{self.ckpt_dir}/{i}').full_path for i in range(3)
    ]
    test_utils.sync_global_processes(
        'test_checkpointing_jax_array:create_arr_paths'
    )
    tspecs = jax.tree.map(serialization.get_tensorstore_spec, ckpt_paths)

    serialize([a1, a2, a3], tspecs)

    m1, m2, m3 = deserialize(
        [
            NamedSharding(global_mesh, pspec),
            NamedSharding(global_mesh, P('x')),
            NamedSharding(global_mesh1d, P(None)),
        ],
        tspecs,
    )

    self.assertIsInstance(m1, jax.Array)
    self.assertArraysEqual(
        np.asarray(m1.addressable_shards[0].data),
        np.array([[0], [2]], dtype=np.int32),
    )
    self.assertArraysEqual(
        np.asarray(m1.addressable_shards[1].data),
        np.array([[1], [3]], dtype=np.int32),
    )
    self.assertEqual(m1.addressable_shards[0].data.shape, (2, 1))
    self.assertEqual(m1.dtype, np.int32)

    self.assertIsInstance(m2, jax.Array)
    self.assertArraysEqual(
        np.asarray(m2.addressable_shards[0].data),
        np.array([[16, 17], [18, 19]], dtype=np.int32),
    )
    self.assertArraysEqual(
        np.asarray(m2.addressable_shards[1].data),
        np.array([[16, 17], [18, 19]], dtype=np.int32),
    )
    self.assertEqual(m2.addressable_shards[0].data.shape, (2, 2))
    self.assertEqual(m2.dtype, np.int32)

    self.assertIsInstance(m3, jax.Array)
    for i, s in enumerate(m3.addressable_shards):
      self.assertEqual(s.index, (slice(None),))
      self.assertEqual(s.replica_id, i)
      self.assertArraysEqual(np.asarray(s.data), np.array([], dtype=np.float32))
    self.assertEqual(m3.dtype, np.float32)

  @parameterized.product(input_dtype=[np.int32, jnp.bfloat16])
  def test_checkpointing_with_bigger_shape_jax_array(self, input_dtype):
    global_mesh = create_global_mesh((2, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    num = math.prod(global_input_shape)

    global_input_data1 = np.arange(num, dtype=input_dtype).reshape(
        global_input_shape
    )

    def cb1(index):
      return global_input_data1[index]

    arr = jax.make_array_from_callback(
        global_input_shape, NamedSharding(global_mesh, P('x', 'y')), cb1
    )
    ckpt_paths = [str(self.ckpt_dir)]
    tspecs = jax.tree.map(serialization.get_tensorstore_spec, ckpt_paths)

    serialize([arr], tspecs)

    ds = NamedSharding(create_global_mesh((4, 2), ('x', 'y')), P('x', 'y'))

    (m1,) = deserialize([ds], tspecs, [(12, 2)], [np.float32], strict=False)

    expected_data = {
        0: np.array([[0], [2], [4]], dtype=np.float32),
        1: np.array([[1], [3], [5]], dtype=np.float32),
        2: np.array([[6], [8], [10]], dtype=np.float32),
        3: np.array([[7], [9], [11]], dtype=np.float32),
        4: np.array([[12], [14], [0]], dtype=np.float32),
        5: np.array([[13], [15], [0]], dtype=np.float32),
        6: np.array([[0], [0], [0]], dtype=np.float32),
        7: np.array([[0], [0], [0]], dtype=np.float32),
    }

    for l in m1.addressable_shards:
      self.assertArraysEqual(np.asarray(l.data), expected_data[l.device.id])

    new_ds = GSPMDSharding.get_replicated(list(global_mesh.devices.flat))
    (m2,) = deserialize([new_ds], tspecs, [(8, 2)], [np.float32])
    for l in m2.addressable_shards:
      self.assertArraysEqual(l.data, global_input_data1.astype('float32'))

  @parameterized.product(input_dtype=[jnp.int4, jnp.int8])
  def test_checkpointing_with_int4(self, input_dtype):
    global_mesh = create_global_mesh((2, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    num = math.prod(global_input_shape)

    global_input_data = np.arange(num, dtype=input_dtype).reshape(
        global_input_shape
    )

    def cb(index):
      return global_input_data[index]

    arr = jax.make_array_from_callback(
        global_input_shape, NamedSharding(global_mesh, P('x', 'y')), cb
    )
    ckpt_paths = [str(self.ckpt_dir)]
    tspecs = jax.tree.map(serialization.get_tensorstore_spec, ckpt_paths)

    serialize([arr], tspecs)

    ds = NamedSharding(create_global_mesh((4, 2), ('x', 'y')), P('x', 'y'))

    target_dtype = jnp.dtype('int4')
    (m1,) = deserialize([ds], tspecs, [(12, 2)], [target_dtype], strict=False)

    # values bigger than 7 are converted properly.
    expected_data = {
        0: jnp.array([[0], [2], [4]], dtype=target_dtype),
        1: jnp.array([[1], [3], [5]], dtype=target_dtype),
        2: jnp.array([[6], [8], [10]], dtype=target_dtype),
        3: jnp.array([[7], [9], [11]], dtype=target_dtype),
        4: jnp.array([[12], [14], [0]], dtype=target_dtype),
        5: jnp.array([[13], [15], [0]], dtype=target_dtype),
        6: jnp.array([[0], [0], [0]], dtype=target_dtype),
        7: jnp.array([[0], [0], [0]], dtype=target_dtype),
    }

    for l in m1.addressable_shards:
      self.assertArraysEqual(np.asarray(l.data), expected_data[l.device.id])

    new_ds = GSPMDSharding.get_replicated(list(global_mesh.devices.flat))
    (m2,) = deserialize([new_ds], tspecs, [(8, 2)], [target_dtype])
    for l in m2.addressable_shards:
      self.assertArraysEqual(l.data, global_input_data.astype(target_dtype))

  def test_checkpointing_scalar_jax_array(self):
    global_mesh = create_global_mesh((2,), 'x')
    global_input_shape = ()
    data = np.array(4)
    s = NamedSharding(global_mesh, P(None))
    array1 = jax.make_array_from_callback(
        global_input_shape, s, lambda idx: data[idx]
    )
    ckpt_paths = [str(self.ckpt_dir)]
    tspecs = jax.tree.map(serialization.get_tensorstore_spec, ckpt_paths)

    serialize([array1], tspecs)

    ds = NamedSharding(global_mesh, P(None))

    (m1,) = deserialize([ds], tspecs, [()], [np.float32])

    for l in m1.addressable_shards:
      self.assertArraysEqual(np.asarray(l.data), data.astype(np.float32))

  def test_deserialize_tensorstore_array_jax_array(self):
    global_mesh = create_global_mesh((2,), 'x')
    data = np.arange(1024)
    tspec = ts.array(data).spec()
    (m1,) = deserialize([NamedSharding(global_mesh, P(None))], [tspec])
    for l in m1.addressable_shards:
      self.assertArraysEqual(np.asarray(l.data), data)

  def test_spec_has_metadata(self):
    spec = {
        'a': {
            'b': 1,
            'c': 2,
        },
        'd': 3,
        'e': {'a': 2, 'metadata': 3},
        'f': 4,
    }
    self.assertTrue(serialization._spec_has_metadata(spec))
    self.assertTrue(
        serialization._spec_has_metadata({
            'driver': 'zarr',
            'kvstore': 'gfile',
            'metadata': {'chunks': 4, 'shape': (32, 64)},
            'one_more': 'thing',
        })
    )

  def test_spec_has_no_metadata(self):
    spec = {
        'a': {
            'b': 1,
            'c': 2,
        },
        'd': 3,
        'e': {
            'a': 2,
        },
        'f': 4,
    }
    self.assertFalse(serialization._spec_has_metadata(spec))

  def test_empty_spec_has_no_metadata(self):
    spec = {}
    self.assertFalse(serialization._spec_has_metadata(spec))

  @parameterized.named_parameters(
      ('gcs', 'gs://my/ckpt/dir/path'), ('file', '/my/ckpt/dir/path')
  )
  def test_get_tensorstore_spec_ocdbt(self, path):
    spec = serialization.get_tensorstore_spec(path, ocdbt=True)
    is_gcs_path = path.startswith('gs://')
    if is_gcs_path:
      self.assertEqual(spec['kvstore']['base'], os.path.dirname(path))
    else:
      self.assertEqual(
          spec['kvstore']['base'],
          {
              'driver': ts_utils.DEFAULT_DRIVER,
              'path': os.path.dirname(path),
          },
      )
    self.assertEqual(spec['kvstore']['path'], 'path')

  def test_get_tensorstore_spec_not_absolute_path(self):
    path = 'my/ckpt/path'
    with self.assertRaisesRegex(
        ValueError, 'Checkpoint path should be absolute'
    ):
      serialization.get_tensorstore_spec(path, ocdbt=True)

  def test_maybe_cloud_storage(self):
    gs_path = 'gs://some-buck/path'
    gs_spec = serialization.get_tensorstore_spec(gs_path, ocdbt=True)
    self.assertTrue(serialization.is_remote_storage(gs_spec))

    local_path = '/tmp/checkpoint'
    local_spec = serialization.get_tensorstore_spec(local_path, ocdbt=True)
    self.assertFalse(serialization.is_remote_storage(local_spec))

    nested_tspec = {
        'driver': 'cast',
        'dtype': 'int32',
        'base': {
            'driver': 'zarr',
            'kvstore': {'driver': 'ocdbt', 'base': 's3://some-bucket/path'},
        },
    }
    self.assertTrue(serialization.is_remote_storage(nested_tspec))

  def test_deserialization_with_int4(self):
    dtype = jnp.int4
    shape = (8, 2)
    arr = jnp.arange(np.prod(shape)).reshape(shape).astype(dtype)

    # Run serialization.
    sharding = jax.sharding.GSPMDSharding.get_replicated(jax.devices())
    tspecs = jax.tree.map(serialization.get_tensorstore_spec, [self.ckpt_dir])

    serialize([arr], tspecs)

    # Run deserialization.
    (deserialized_arr,) = deserialize(
        shardings=[sharding],
        tensorstore_specs=tspecs,
        global_shapes=[shape],
        dtypes=[dtype],
    )

    out = deserialized_arr.astype(jnp.int8)  # doesn't crash
    self.assertEqual(out.dtype, jnp.int8)
    self.assertArraysEqual(out + out, out * 2)

  @parameterized.parameters((True,), (False,))
  def test_padding(self, strict: bool):
    data = np.arange(8)
    save_shape = data.shape
    global_mesh = create_global_mesh((2,), 'x')
    sharding = NamedSharding(global_mesh, P(None))
    array = jax.make_array_from_callback(
        save_shape, sharding, lambda idx: data[idx]
    )
    ckpt_paths = [str(self.ckpt_dir)]
    tspecs = jax.tree.map(serialization.get_tensorstore_spec, ckpt_paths)

    serialize([array], tspecs)

    restore_shape = (16,)
    if strict:
      with self.assertRaisesRegex(
          ValueError, 'is not compatible with the stored shape'
      ):
        deserialize([sharding], tspecs, [restore_shape], strict=strict)
    else:
      (restored,) = deserialize(
          [sharding], tspecs, [restore_shape], strict=strict
      )
      for shard in restored.addressable_shards:
        expected = np.arange(16)
        expected[8:] = 0
        self.assertArraysEqual(np.asarray(shard.data), expected)

  @parameterized.parameters((True,), (False,))
  def test_truncation(self, strict: bool):
    data = np.arange(16)
    save_shape = data.shape
    global_mesh = create_global_mesh((2,), 'x')
    sharding = NamedSharding(global_mesh, P(None))
    array = jax.make_array_from_callback(
        save_shape, sharding, lambda idx: data[idx]
    )
    ckpt_paths = [str(self.ckpt_dir)]
    tspecs = jax.tree.map(serialization.get_tensorstore_spec, ckpt_paths)

    serialize([array], tspecs)

    restore_shape = (8,)
    if strict:
      with self.assertRaisesRegex(
          ValueError, 'is not compatible with the stored shape'
      ):
        deserialize([sharding], tspecs, [restore_shape], strict=strict)
    else:
      (restored,) = deserialize(
          [sharding], tspecs, [restore_shape], strict=strict
      )
      for shard in restored.addressable_shards:
        self.assertArraysEqual(np.asarray(shard.data), np.arange(8))

  def test_odd_resharding(self):
    data = np.arange(12)
    global_shape = data.shape
    global_mesh = create_global_mesh((2,), 'x')
    sharding = NamedSharding(
        global_mesh,
        P(
            'x',
        ),
    )
    array = jax.make_array_from_callback(
        global_shape, sharding, lambda idx: data[idx]
    )
    ckpt_paths = [str(self.ckpt_dir)]
    tspecs = jax.tree.map(serialization.get_tensorstore_spec, ckpt_paths)

    serialize([array], tspecs)

    global_mesh = create_global_mesh((3,), 'x')
    sharding = NamedSharding(
        global_mesh,
        P(
            'x',
        ),
    )
    (restored,) = deserialize([sharding], tspecs, [global_shape])
    for i, shard in enumerate(restored.addressable_shards):
      self.assertArraysEqual(np.asarray(shard.data), np.arange(4) + (i * 4))

  def test_load_with_layout(self):
    mesh = create_global_mesh((4, 2), ('x', 'y'))
    np_inp = np.arange(32).reshape(8, 4)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    out_layout = Layout(
        device_local_layout=DLL(
            arr.layout.device_local_layout.major_to_minor[::-1],
            arr.layout.device_local_layout._tiling,
        ),
        sharding=arr.sharding,
    )

    ckpt_dir = pathlib.Path(self.create_tempdir('ckpt').full_path)
    ckpt_path = pathlib.Path(self.create_tempdir(f'{ckpt_dir}/first').full_path)
    tspecs = jax.tree.map(serialization.get_tensorstore_spec, [ckpt_path])

    serialize(
        [arr],
        tspecs,
    )

    (out,) = deserialize([out_layout], tspecs)

    self.assertEqual(out.layout, out_layout)
    self.assertIsInstance(out, jax.Array)
    self.assertArraysEqual(out, np_inp)
    for s in out.addressable_shards:
      self.assertArraysEqual(s.data, np_inp[s.index])

  def test_incomplete_write(self):
    data = np.arange(8)
    chunk_len = 4
    global_mesh = create_global_mesh((8,), 'x')
    sharding = NamedSharding(global_mesh, P(None))
    tspec = ts_utils.ArrayWriteSpec(
        self.ckpt_dir.as_posix(),
        'a',
        global_shape=data.shape,
        write_shape=(chunk_len,),
        dtype=data.dtype,
        use_ocdbt=False,
    ).json
    t = ts.open(
        ts.Spec(tspec),
        create=True,
        open=True,
    ).result()
    t[:chunk_len].write(data[:chunk_len]).result()

    # Enable raising error for incomplete chunk.
    tspec['fill_missing_data_reads'] = False
    with self.assertRaisesRegex(
        Exception, 'Encountered error while reading array index'
    ):
      deserialize([sharding], [tspec])

  @parameterized.named_parameters(
      dict(testcase_name='fully_replicated', pspec=(None, None)),
      dict(testcase_name='partially_replicated', pspec=('x', None)),
      dict(testcase_name='fully_sharded', pspec=('x', 'y')),
  )
  def test_dedup_loading(self, pspec):
    data = np.arange(2_048 * 4_096, dtype=np.float32).reshape(2_048, 4_096)
    global_shape = data.shape
    global_mesh = create_global_mesh((2, 2), ('x', 'y'))
    sharding = NamedSharding(global_mesh, P(*pspec))
    array = jax.make_array_from_callback(
        global_shape, sharding, lambda idx: data[idx]
    )
    ckpt_paths = [str(self.ckpt_dir)]
    tspecs = jax.tree.map(serialization.get_tensorstore_spec, ckpt_paths)
    serialize([array], tspecs)

    tm.start()
    _, start_memory_usage = tm.get_traced_memory()
    deserialize([sharding], tspecs, [global_shape])
    _, peak_memory_usage = tm.get_traced_memory()
    tm.clear_traces()
    # Array size (2048 * 4096 * 4 = 32M)
    delta = 2_000_000  # Empirically chosen wiggle room.
    self.assertLess(peak_memory_usage - start_memory_usage, 32_000_000 + delta)


if __name__ == '__main__':
  absltest.main()
