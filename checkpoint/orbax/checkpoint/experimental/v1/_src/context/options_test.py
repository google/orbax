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

"""Tests for checkpoint file options."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from orbax.checkpoint import options as v0_options_lib
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as ocp_options
from orbax.checkpoint.experimental.v1._src.saving import saving



class FileOptionsTest(parameterized.TestCase):

  def test_v0_conversion_with_none_options(self):
    opts = ocp_options.FileOptions()
    v0_opts = opts.v0()
    self.assertIsInstance(v0_opts, v0_options_lib.FileOptions)
    self.assertIsNone(v0_opts.path_permission_mode)

  def test_v0_conversion_with_all_options(self):

    opts = ocp_options.FileOptions(
        path_permission_mode=0o777,
    )
    v0_opts = opts.v0()
    self.assertIsInstance(v0_opts, v0_options_lib.FileOptions)
    self.assertEqual(v0_opts.path_permission_mode, 0o777)


class MemoryOptionsTest(parameterized.TestCase):

  def test_memory_options_propagation(self):
    def is_prioritized_key_fn(path):
      del path
      return True

    memory_options = ocp_options.MemoryOptions(
        write_concurrent_bytes=1024,
        read_concurrent_bytes=2048,
        transfer_concurrent_bytes=512,
        is_prioritized_key_fn=is_prioritized_key_fn,
    )

    with context_lib.Context(memory_options=memory_options):
      with mock.patch(
          'orbax.checkpoint._src.handlers.base_pytree_checkpoint_handler.BasePyTreeCheckpointHandler',
          autospec=True,
      ) as mock_handler_class:
        # Mock async_save to return a future as expected by PyTreeHandler.save
        mock_handler_instance = mock_handler_class.return_value
        mock_handler_instance.async_save.return_value = (
            mock.MagicMock()
        )  # Awaitable

        pytree = {'a': jnp.ones((1,))}

        # save_pytree will eventually call BasePyTreeCheckpointHandler
        try:
          saving.save_pytree('/tmp/test', pytree)
        except Exception:  # pylint: disable=broad-except
          # We might get some errors because we mocked too much,
          # but we check if mock_handler_class was called.
          pass

        mock_handler_class.assert_called()
        # Find the call that has our expected arguments
        found = False
        for call in mock_handler_class.call_args_list:
          kwargs = call.kwargs
          if (
              kwargs.get('save_concurrent_bytes') == 1024
              and kwargs.get('restore_concurrent_bytes') == 2048
              and kwargs.get('save_device_host_concurrent_bytes') == 512
              and kwargs.get('is_prioritized_key_fn') == is_prioritized_key_fn  # pylint: disable=comparison-with-callable
          ):
            found = True
            break
        self.assertTrue(
            found,
            f'Expected call not found in {mock_handler_class.call_args_list}',
        )


if __name__ == '__main__':
  absltest.main()
