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

"""Tests for PyTreeCheckpointHandler module."""

from absl import flags
import jax
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler_test_utils
from orbax.checkpoint._src.testing import multiprocess_test


FLAGS = flags.FLAGS

jax.config.update('jax_enable_x64', True)


class PyTreeCheckpointHandlerTest(
    pytree_checkpoint_handler_test_utils.PyTreeCheckpointHandlerTestBase.Test,
    multiprocess_test.MultiProcessTest,
):
  pass


if __name__ == '__main__':
  multiprocess_test.main()
