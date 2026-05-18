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

"""Tests for utility functions for serialization."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.serialization import options_resolution


def cb_overriding_global(k, v, s):
  s.dtype = np.int16


def cb_overriding_all(k, v, s):
  s.dtype = np.float32
  s.chunk_byte_size = 32_000_000
  s.shard_axes = (1,)


def cb_jnp_converter(k, v, s):
  s.dtype = jnp.bfloat16


def cb_empty_axes(k, v, s):
  s.shard_axes = ()


class OptionsResolutionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='callback_overriding_global',
          callback=cb_overriding_global,
          expected_storage_options=options_lib.ArrayOptions.Saving.StorageOptions(
              dtype=np.int16,
              chunk_byte_size=16_000_000,
              shard_axes=(0,),
          ),
      ),
      dict(
          testcase_name='callback_overriding_all',
          callback=cb_overriding_all,
          expected_storage_options=options_lib.ArrayOptions.Saving.StorageOptions(
              dtype=np.float32,
              chunk_byte_size=32_000_000,
              shard_axes=(1,),
          ),
      ),
      dict(
          testcase_name='without_callback_falls_back_to_global',
          callback=None,
          expected_storage_options=options_lib.ArrayOptions.Saving.StorageOptions(
              dtype=np.int32,
              chunk_byte_size=16_000_000,
              shard_axes=(0,),
          ),
      ),
      dict(
          testcase_name='jnp_dtype_converter',
          callback=cb_jnp_converter,
          expected_storage_options=options_lib.ArrayOptions.Saving.StorageOptions(
              dtype=jnp.bfloat16,
              chunk_byte_size=16_000_000,
              shard_axes=(0,),
          ),
      ),
      dict(
          testcase_name='empty_shard_axes_overrides_to_empty',
          callback=cb_empty_axes,
          expected_storage_options=options_lib.ArrayOptions.Saving.StorageOptions(
              dtype=np.int32,
              chunk_byte_size=16_000_000,
              shard_axes=(),
          ),
      ),
  )
  def test_resolve_storage_options(
      self,
      callback,
      expected_storage_options,
  ):
    # Global options
    global_storage = options_lib.ArrayOptions.Saving.StorageOptions(
        dtype=np.int32,
        chunk_byte_size=16_000_000,
        shard_axes=(0,),
    )

    saving_options = options_lib.ArrayOptions.Saving(
        storage_options=global_storage,
        scoped_storage_options_creator=callback,
    )

    keypath = (jax.tree_util.DictKey(key='foo'),)
    value = np.ones((2, 2))

    resolved_options = options_resolution.resolve_storage_options(
        keypath, value, saving_options
    )

    self.assertEqual(resolved_options, expected_storage_options)


if __name__ == '__main__':
  absltest.main()
