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

import time

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
import orbax.checkpoint as ocp

_TEST_GCS_DIR = flags.DEFINE_string(
    'test_gcs_dir',
    None,
    'GCS directory for testing, e.g., gs://my-bucket/orbax-test.',
)


def get_max_object_size(gcs_path: epath.Path) -> int:
  """Returns the largest object size recursively in a GCS path."""
  max_size = 0
  for child_path in gcs_path.iterdir():
    if child_path.is_dir():
      size = get_max_object_size(child_path)
      if size > max_size:
        max_size = size
    else:
      file_stat = child_path.stat()
      if file_stat.length > max_size:
        max_size = file_stat.length
  return max_size


class GcsIntegrationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='small_tree',
          tree={'a': np.arange(10), 'b': np.ones((2, 3))},
          step=0,
      ),
      dict(
          testcase_name='large_array',
          tree={
              'large': np.random.randint(
                  np.iinfo(np.int8).min,
                  np.iinfo(np.int8).max,
                  2_000 * 2**20,
                  dtype=np.int8,
              )
          },  # 2GiB
          step=1,
      ),
  )
  def test_save_restore_ocdbt_gcs(self, tree, step):
    if _TEST_GCS_DIR.value is None:
      self.skipTest('Requires --test_gcs_dir to be set.')

    path = (
        epath.Path(_TEST_GCS_DIR.value)
        / f'orbax_test_{int(time.time())}_{step}'
    )
    print(f'Using GCS path for test: {path}')

    mngr = ocp.CheckpointManager(path)
    mngr.save(step, args=ocp.args.StandardSave(tree))
    mngr.wait_until_finished()

    max_size = get_max_object_size(path)
    print(f'Max object size for {self.id()} test: {max_size}')
    # Assert max object size is <= 405MiB
    self.assertLessEqual(max_size, 405 * 2**20)

    restored_tree = mngr.restore(step)
    jax.tree.map(np.testing.assert_array_equal, tree, restored_tree)
    print(f'{self.id()} save/restore to GCS successful.')
    print(f'GCS integration test passed. Checkpoint data is in {path}.')
    print(f'Run `gcloud storage ls -r -l {path}` to view checkpoint files.')

  @parameterized.named_parameters(
      dict(
          testcase_name='200MiB',
          name='200MiB',
          target_size=200 * 2**20,
      ),
      dict(
          testcase_name='400MiB',
          name='400MiB',
          target_size=400 * 2**20,
      ),
      dict(
          testcase_name='2GiB',
          name='2GiB',
          target_size=2_000 * 2**20,
      ),
  )
  def test_benchmark_different_target_sizes(self, name, target_size):
    if _TEST_GCS_DIR.value is None:
      self.skipTest('Requires --test_gcs_dir to be set.')

    large_array = np.random.randint(
        np.iinfo(np.int8).min,
        np.iinfo(np.int8).max,
        2_000 * 1024 * 1024,
        dtype=np.int8,
    )  # 2GiB
    tree_large = {'large': large_array}

    print(f'Starting benchmark for target size: {name}')
    path = (
        epath.Path(_TEST_GCS_DIR.value)
        / f'orbax_test_bench_{name}_{int(time.time())}'
    )
    mngr = ocp.CheckpointManager(path)

    start = time.time()
    mngr.save(
        0,
        args=ocp.args.PyTreeSave(
            tree_large, ocdbt_target_data_file_size=target_size
        ),
    )
    mngr.wait_until_finished()
    save_time = time.time() - start

    max_size = get_max_object_size(path)
    print(f'Max object size for {name} test: {max_size}')
    # Assert max object size is <= target size + 5MiB
    self.assertLessEqual(max_size, target_size + 5 * 2**20)

    start = time.time()
    restored = mngr.restore(0)
    restore_time = time.time() - start

    jax.tree.map(np.testing.assert_array_equal, tree_large, restored)
    print(
        f'Target size: {name}, save time: {save_time:.2f}s, restore time:'
        f' {restore_time:.2f}s'
    )


if __name__ == '__main__':
  absltest.main()
