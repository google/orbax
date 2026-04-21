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

"""Tests for v0/v1 compatibility checkpointer in Pathways."""

from absl import flags
import jax
from orbax.checkpoint.experimental.v1._src.training import v0v1_compatibility_checkpointer_test_base

from .pyglib.contrib.g3_multiprocessing import g3_multiprocessing
from absl.testing import absltest
from .testing.pybase import parameterized


FLAGS = flags.FLAGS

jax.config.update('jax_enable_x64', True)

V0v1CompatibilityCheckpointerTestBase = (
    v0v1_compatibility_checkpointer_test_base.V0v1CompatibilityCheckpointerTestBase
)


class PathwaysV0v1CompatibilityCheckpointerTest(
    V0v1CompatibilityCheckpointerTestBase,
    parameterized.TestCase,
):

  pass


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  g3_multiprocessing.handle_test_main(googletest.main)
