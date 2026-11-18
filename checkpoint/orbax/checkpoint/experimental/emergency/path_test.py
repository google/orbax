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

"""Tests for emergency checkpointing path utilities."""

import json
from unittest import mock

from absl.testing import absltest
import jax
from jax.experimental import multihost_utils
import numpy as np
from orbax.checkpoint.experimental.emergency import path


class PathTest(absltest.TestCase):

  def test_sync_global_data_single_process(self):
    payload = {'process_id': 0, 'steps': [10, 20], 'tag': 'host-0'}

    result = path.sync_global_data(payload)

    self.assertSequenceEqual(result, [payload])

  def test_sync_global_data_pads_unequal_payloads_on_host(self):
    peer_payloads = [
        {},
        {'process_id': 1, 'steps': [100, 200, 300], 'label': 'longest-payload'},
        {'process_id': 2, 'note': 'café'},
    ]
    peer_bytes = [
        np.frombuffer(json.dumps(p).encode('utf-8'), dtype=np.uint8)
        for p in peer_payloads
    ]
    gathered_inputs = []

    def fake_allgather(in_tree, tiled=False):
      self.assertFalse(tiled)
      self.assertIsInstance(in_tree, np.ndarray)
      gathered_inputs.append(in_tree)
      if len(gathered_inputs) == 1:
        return np.array([[len(b)] for b in peer_bytes], dtype=np.int32)
      max_len = in_tree.shape[0]
      rows = np.zeros((len(peer_bytes), max_len), dtype=np.uint8)
      for idx, raw in enumerate(peer_bytes):
        rows[idx, : len(raw)] = raw
      return rows

    self.enter_context(
        mock.patch.object(
            jax,
            'process_count',
            autospec=True,
            spec_set=True,
            return_value=len(peer_payloads),
        )
    )
    self.enter_context(
        mock.patch.object(
            multihost_utils,
            'process_allgather',
            autospec=True,
            spec_set=True,
            side_effect=fake_allgather,
        )
    )

    result = path.sync_global_data(peer_payloads[0])

    self.assertSequenceEqual(result, peer_payloads)
    self.assertLen(gathered_inputs, 2)


if __name__ == '__main__':
  absltest.main()
