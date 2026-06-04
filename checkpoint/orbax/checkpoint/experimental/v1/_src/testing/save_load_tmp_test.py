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
import jax
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint.experimental.v1._src.testing import save_load_test_base


FLAGS = flags.FLAGS

jax.config.update('jax_enable_x64', True)


class SaveLoadTest(
    save_load_test_base.SaveLoadTestBase.SaveLoadTest,
    multiprocess_test.MultiProcessTest,
):
  pass


class SynchronizationTest(
    save_load_test_base.SaveLoadTestBase.SynchronizationTest,
    multiprocess_test.MultiProcessTest,
):
  pass


if __name__ == '__main__':
  multiprocess_test.main()
