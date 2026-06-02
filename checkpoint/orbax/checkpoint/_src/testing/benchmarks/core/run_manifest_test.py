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

"""Tests for run_manifest: capture-once-per-suite environment snapshot."""

import subprocess
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from orbax.checkpoint._src.testing.benchmarks.core import run_manifest


class CaptureRunManifestTest(parameterized.TestCase):

  def test_captures_hostname_and_iso_timestamp(self):
    m = run_manifest.capture_run_manifest()
    self.assertIsNotNone(m.hostname)
    # ISO 8601 with 'T' separator.
    self.assertIn('T', m.captured_at)

  @mock.patch.object(subprocess, 'check_output')
  def test_captures_git_sha_and_dirty_flag(self, mock_check_output):
    def _fake(cmd, *unused_args, **unused_kwargs):
      if cmd[:2] == ['git', 'rev-parse']:
        return b'abc123def456\n'
      if cmd[:2] == ['git', 'status']:
        return b' M some_file.py\n'  # dirty tree
      raise subprocess.CalledProcessError(1, cmd)

    mock_check_output.side_effect = _fake
    m = run_manifest.capture_run_manifest()
    self.assertEqual(m.git_sha, 'abc123def456')
    self.assertTrue(m.git_dirty)

  @mock.patch.object(subprocess, 'check_output')
  def test_clean_tree_dirty_false(self, mock_check_output):
    def _fake(cmd, *unused_args, **unused_kwargs):
      if cmd[:2] == ['git', 'rev-parse']:
        return b'deadbeef\n'
      if cmd[:2] == ['git', 'status']:
        return b''  # clean
      raise subprocess.CalledProcessError(1, cmd)

    mock_check_output.side_effect = _fake
    m = run_manifest.capture_run_manifest()
    self.assertEqual(m.git_sha, 'deadbeef')
    self.assertFalse(m.git_dirty)

  @mock.patch.object(
      subprocess,
      'check_output',
      side_effect=subprocess.CalledProcessError(128, ['git']),
  )
  def test_no_git_falls_through_to_unknown(self, unused_mock_check_output):
    m = run_manifest.capture_run_manifest()
    self.assertEqual(m.git_sha, 'unknown')
    self.assertFalse(m.git_dirty)

  def test_captures_jax_and_orbax_versions(self):
    m = run_manifest.capture_run_manifest()
    # Format is a non-empty version string.
    self.assertNotEqual(m.jax_version, '')
    self.assertNotEqual(m.orbax_version, '')

  def test_xla_flags_picked_up_from_env(self):
    with mock.patch.dict('os.environ', {'XLA_FLAGS': '--xla_dump_to=/tmp/x'}):
      m = run_manifest.capture_run_manifest()
    self.assertEqual(m.xla_flags, '--xla_dump_to=/tmp/x')

  def test_xla_flags_default_empty_when_unset(self):
    with mock.patch.dict('os.environ', {}, clear=True):
      m = run_manifest.capture_run_manifest()
    self.assertEqual(m.xla_flags, '')

  def test_as_markdown_renders_sections(self):
    m = run_manifest.capture_run_manifest()
    text = m.as_markdown()
    self.assertIn('## Run manifest', text)
    self.assertIn('### Code', text)
    self.assertIn('### Environment', text)
    self.assertIn('### Topology', text)


if __name__ == '__main__':
  absltest.main()
