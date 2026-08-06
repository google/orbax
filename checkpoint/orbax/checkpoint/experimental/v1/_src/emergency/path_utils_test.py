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

import itertools
from absl import flags
import jax
import numpy as np
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import pathways as multihost_pathways
from orbax.checkpoint._src.path import step as step_lib
from orbax.checkpoint.experimental.v1._src.emergency import mesh_test_utils
from orbax.checkpoint.experimental.v1._src.emergency import path_utils
from orbax.checkpoint.testing import local_path as local_path_test_utils
from .pyglib.contrib.g3_multiprocessing import g3_multiprocessing
from absl.testing import absltest
from .testing.pybase import parameterized


FLAGS = flags.FLAGS
LocalPath = local_path_test_utils.LocalPath

jax.config.update('jax_enable_x64', True)

MeshConfig = mesh_test_utils.MeshConfig


def _mesh_configs() -> list[MeshConfig]:
  result = []
  for replica_count, replica_axis_index, use_device_count in itertools.product(
      [8, 4, 2, 1],
      [0],  # TODO(b/448471028): Fix replica_axis_index=1 behavior.
      [8, 4],
  ):
    try:
      cfg = MeshConfig(
          replica_count=replica_count,
          replica_axis_index=replica_axis_index,
          use_device_count=use_device_count,
      )
    except ValueError:
      continue
    result.append(cfg)
  return result


class PerReplicaLocalLatestStepsTest(
    parameterized.TestCase,
):

  def setUp(self):
    super().setUp()
    self.directory = local_path_test_utils.create_local_path_base(self)
    self.assertTrue(self.directory.exists())
    self.assertTrue(multihost.is_pathways_backend())
    self.local_directory = LocalPath(self.directory)
    self.assertEqual(jax.device_count(), 8)
    self.assertEqual(multihost_pathways.worker_count(None), 4)

  def assertLocalPathsExist(
      self, steps: list[int], global_mesh: jax.sharding.Mesh
  ):
    name_format = step_lib.standard_name_format()
    for w in range(multihost_pathways.worker_count(global_mesh)):
      for step in steps:
        path = self.directory / f'local_{w}' / name_format.build_name(step)
        self.assertTrue(path.exists(), f'Path {path} does not exist.')

  def save_local_steps(self, steps: list[int], global_mesh: jax.sharding.Mesh):
    name_format = step_lib.standard_name_format()
    local_directory = self.local_directory

    def _mkdirs():
      for step in steps:
        (local_directory / name_format.build_name(step)).mkdir(
            parents=True, exist_ok=False
        )

    jax.block_until_ready(
        dispatchers.RemotePythonDispatcher().dispatch(
            _mkdirs
        )
    )
    self.assertLocalPathsExist(steps, global_mesh)

  @parameterized.product(mesh_cfg=_mesh_configs())
  def test_worker_count(self, mesh_cfg: MeshConfig):
    mesh = mesh_cfg.mesh
    self.assertEqual(
        multihost_pathways.worker_count(mesh), len(mesh.devices.flatten()) // 2
    )

  @parameterized.product(
      steps=([0, 1, 2], [5, 0], [0], []), mesh_cfg=_mesh_configs()
  )
  def test_same_steps(self, steps: list[int], mesh_cfg: MeshConfig):
    replica_count = mesh_cfg.replica_count
    replica_axis_index = mesh_cfg.replica_axis_index
    mesh = mesh_cfg.mesh
    self.save_local_steps(steps, mesh)
    max_num_steps = path_utils._get_max_num_steps(
        self.local_directory,
        global_mesh=mesh,
        step_name_format=step_lib.standard_name_format(),
    )
    self.assertLen(steps, max_num_steps)
    per_replica_local_steps = path_utils.per_replica_local_steps(
        self.local_directory,
        step_name_format=step_lib.standard_name_format(),
        global_mesh=mesh,
        replica_axis_index=replica_axis_index,
    )
    self.assertLen(per_replica_local_steps.keys(), replica_count)
    self.assertEqual(
        per_replica_local_steps,
        {replica_id: set(steps) for replica_id in range(replica_count)},
    )

  @parameterized.product(
      steps=([0, 1, 2], [0, 5], [0]), mesh_cfg=_mesh_configs()
  )
  def test_different_steps(self, steps: list[int], mesh_cfg: MeshConfig):
    replica_count = mesh_cfg.replica_count
    replica_axis_index = mesh_cfg.replica_axis_index
    mesh = mesh_cfg.mesh
    self.save_local_steps(steps, mesh)
    name_format = step_lib.standard_name_format()

    # Delete worker 0 step.
    (self.directory / 'local_0' / name_format.build_name(0)).rmtree()

    per_replica_local_steps = path_utils.per_replica_local_steps(
        self.local_directory,
        step_name_format=name_format,
        global_mesh=mesh,
        replica_axis_index=replica_axis_index,
    )
    self.assertLen(per_replica_local_steps.keys(), replica_count)
    expected_steps = {}
    expected_steps[0] = set(steps) - {0}
    for replica_id in range(1, replica_count):
      expected_steps[replica_id] = set(steps)
    self.assertEqual(
        per_replica_local_steps,
        expected_steps,
    )

  @parameterized.product(steps=([0, 1, 2], [0, 5], [0], []))
  def test_processes_split_between_replicas(self, steps: list[int]):
    devices = jax.devices()
    mesh = jax.sharding.Mesh(
        np.asarray([
            [devices[0], devices[1], devices[2]],
            [devices[3], devices[4], devices[5]],
        ]),
        ('replica', 'data'),
    )
    replica_count = 2
    self.assertLen(mesh.devices, replica_count)
    self.assertEqual(multihost_pathways.worker_count(mesh), 3)
    self.save_local_steps(steps, mesh)
    name_format = step_lib.standard_name_format()

    per_replica_local_steps = path_utils.per_replica_local_steps(
        self.local_directory,
        step_name_format=name_format,
        global_mesh=mesh,
        replica_axis_index=0,
    )
    self.assertLen(per_replica_local_steps.keys(), replica_count)
    self.assertEqual(
        per_replica_local_steps,
        {replica_id: set(steps) for replica_id in range(replica_count)},
    )

    if not steps:
      return

    # Delete worker 2 step.
    (self.directory / 'local_2' / name_format.build_name(0)).rmtree()

    per_replica_local_steps = path_utils.per_replica_local_steps(
        self.local_directory,
        step_name_format=name_format,
        global_mesh=mesh,
        replica_axis_index=0,
    )
    self.assertLen(per_replica_local_steps.keys(), replica_count)
    expected_steps = {
        0: set(steps),
        1: set(steps) - {0},
    }
    self.assertEqual(
        per_replica_local_steps,
        expected_steps,
    )

  @parameterized.parameters(
      # 2 replicas
      ([[], [], [], []], 2, [{}, {}]),
      ([[1], [1], [0], [0]], 2, [{1}, {0}]),
      ([[], [], [0], [0]], 2, [{}, {0}]),
      ([[], [], [0], []], 2, [{}, {}]),
      ([[], [], [0], [1]], 2, [{}, {}]),
      ([[], [0], [], [0]], 2, [{}, {}]),
      ([[0], [1], [0], [1]], 2, [{}, {}]),
      ([[1, 2], [1, 2], [4], [4, 5]], 2, [{1, 2}, {4}]),
      (
          [[-1, 0], [-1, 0], [1, 2], [1, 2]],
          2,
          [{0}, {1, 2}],
      ),
      (
          [[-1, 0, 1], [-1, 0, 1], [0, 1, -1], [1, 0, -1]],
          2,
          [{0, 1}, {0, 1}],
      ),
      # 4 replicas
      ([[], [], [], []], 4, [{}, {}, {}, {}]),
      ([[1], [1], [0], [0]], 4, [{1}, {1}, {0}, {0}]),
      ([[], [], [0], [0]], 4, [{}, {}, {0}, {0}]),
      ([[], [], [0], []], 4, [{}, {}, {0}, {}]),
      ([[], [], [0], [1]], 4, [{}, {}, {0}, {1}]),
      ([[], [0], [], [0]], 4, [{}, {0}, {}, {0}]),
      ([[0], [1], [0], [1]], 4, [{0}, {1}, {0}, {1}]),
      ([[1, 2], [1, 2], [4], [4, 5]], 4, [{1, 2}, {1, 2}, {4}, {4, 5}]),
      (
          [[-1, 0], [-1, 0], [1, 2], [1, 2]],
          4,
          [{0}, {0}, {1, 2}, {1, 2}],
      ),
      (
          [[-1, 0, 1], [-1, 0, 1], [0, 1, -1], [1, 0, -1]],
          4,
          [{0, 1}, {0, 1}, {0, 1}, {0, 1}],
      ),
  )
  def test_per_replica_local_steps(
      self, worker_steps, num_replicas, expectation
  ):
    expected_dict = {i: set(steps) for i, steps in enumerate(expectation)}
    for replica_axis_index in [0]:
      with self.subTest(f'replica_axis_index={replica_axis_index}'):
        mesh_cfg = MeshConfig(
            replica_count=num_replicas,
            replica_axis_index=replica_axis_index,
        )
        for w, steps in enumerate(worker_steps):
          for step in steps:
            (self.directory / f'local_{w}' / str(step)).mkdir(
                parents=True, exist_ok=False
            )

        per_replica_local_steps = path_utils.per_replica_local_steps(
            self.local_directory,
            step_name_format=step_lib.standard_name_format(),
            global_mesh=mesh_cfg.mesh,
            replica_axis_index=mesh_cfg.replica_axis_index,
        )
        self.assertDictEqual(per_replica_local_steps, expected_dict)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  g3_multiprocessing.handle_test_main(googletest.main)
