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

"""Unit tests for the standalone checkpoint conversion CLI."""

from __future__ import annotations

import io
import json
import os
from typing import Any

from absl.testing import absltest
import numpy as np
from orbax.checkpoint.experimental import v1 as ocp
from orbax.checkpoint.experimental.conversion import cli
import safetensors.numpy


class CliTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.root = self.create_tempdir().full_path

  def test_cli_round_trip_and_error_json(self) -> None:
    st_file = os.path.join(self.root, "in.safetensors")
    tensors = {"block.0.w": np.arange(12, dtype=np.float32).reshape(3, 4)}
    safetensors.numpy.save_file(tensors, st_file)

    orbax_dir = os.path.join(self.root, "orbax_out")
    buf = io.StringIO()
    code = cli.run_convert_cli([st_file, orbax_dir], stdout=buf)
    self.assertEqual(code, 0)
    payload = json.loads(buf.getvalue())
    self.assertEqual(payload["dest_format"], "orbax-ocdbt")
    self.assertEqual(payload["num_tensors"], 1)

    restored: Any = ocp.load(orbax_dir)
    np.testing.assert_array_equal(
        restored["block"]["0"]["w"], tensors["block.0.w"]
    )

    # Second run without --overwrite returns exit code 1 and structured error.
    err_buf = io.StringIO()
    err_code = cli.run_convert_cli([st_file, orbax_dir], stdout=err_buf)
    self.assertEqual(err_code, 1)
    err_payload = json.loads(err_buf.getvalue())
    self.assertIn("FileExistsError", err_payload["error"])

    # Convert back to Safetensors with --overwrite and --max-shard-size-bytes.
    st_out_dir = os.path.join(self.root, "st_out")
    buf2 = io.StringIO()
    code2 = cli.run_convert_cli(
        [
            orbax_dir,
            st_out_dir,
            "--to-format=safetensors",
            "--overwrite",
            "--max-shard-size-bytes=0",
        ],
        stdout=buf2,
    )
    self.assertEqual(code2, 0)
    payload2 = json.loads(buf2.getvalue())
    self.assertEqual(payload2["dest_format"], "safetensors")
    np.testing.assert_array_equal(
        safetensors.numpy.load_file(
            os.path.join(st_out_dir, "model.safetensors")
        )["block.0.w"],
        tensors["block.0.w"],
    )


if __name__ == "__main__":
  absltest.main()
