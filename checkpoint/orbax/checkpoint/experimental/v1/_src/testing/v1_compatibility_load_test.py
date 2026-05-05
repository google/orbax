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

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import jax
from orbax.checkpoint.experimental.v1._src.testing.compatibility import checkpointables_metadata_compatibility_test_base
from orbax.checkpoint.experimental.v1._src.testing.compatibility import load_checkpointables_compatibility_test_base
from orbax.checkpoint.experimental.v1._src.testing.compatibility import load_pytree_compatibility_test_base
from orbax.checkpoint.experimental.v1._src.testing.compatibility import pytree_metadata_compatibility_test_base


FLAGS = flags.FLAGS

jax.config.update('jax_enable_x64', True)


class CheckpointablesMetadataTest(
    checkpointables_metadata_compatibility_test_base.CheckpointablesMetadataCompatibilityTestBase,
    parameterized.TestCase,
):
  pass


class LoadCheckpointablesTest(
    load_checkpointables_compatibility_test_base.LoadCheckpointablesCompatibilityTestBase,
    parameterized.TestCase,
):
  pass


class LoadPytreeTest(
    load_pytree_compatibility_test_base.LoadPytreeCompatibilityTestBase,
    parameterized.TestCase,
):
  pass


class PytreeMetadataTest(
    pytree_metadata_compatibility_test_base.PytreeMetadataCompatibilityTestBase,
    parameterized.TestCase,
):
  pass


if __name__ == '__main__':
  absltest.main()
