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
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.serialization import options_resolution


class OptionsResolutionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='callback_overriding_global',
          callback=lambda k, v: options_lib.ArrayOptions.Saving.StorageOptions(
              dtype=np.int16
          ),
          expected_storage_options=options_lib.ArrayOptions.Saving.StorageOptions(
              dtype=np.int16,
              chunk_byte_size=16_000_000,
              shard_axes=(0,),
          ),
      ),
      dict(
          testcase_name='callback_overriding_all',
          callback=lambda k, v: options_lib.ArrayOptions.Saving.StorageOptions(
              dtype=np.float32,
              chunk_byte_size=32_000_000,
              shard_axes=(1,),
          ),
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
          callback=lambda k, v: options_lib.ArrayOptions.Saving.StorageOptions(
              dtype=jnp.bfloat16,
          ),
          expected_storage_options=options_lib.ArrayOptions.Saving.StorageOptions(
              dtype=jnp.bfloat16,
              chunk_byte_size=16_000_000,
              shard_axes=(0,),
          ),
      ),
      dict(
          testcase_name='empty_shard_axes_overrides_to_empty',
          callback=lambda k, v: options_lib.ArrayOptions.Saving.StorageOptions(
              shard_axes=(),
          ),
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

    context = context_lib.Context(
        array_options=options_lib.ArrayOptions(
            saving=options_lib.ArrayOptions.Saving(
                storage_options=global_storage,
                scoped_storage_options_creator=callback,
            )
        ),
    )

    # Dummy param
    keypath = (jax.tree_util.DictKey(key='foo'),)
    value = np.ones((2, 2))

    resolved_options = options_resolution.resolve_storage_options(
        keypath, value, context.array_options.saving
    )

    self.assertEqual(resolved_options, expected_storage_options)


if __name__ == '__main__':
  absltest.main()
