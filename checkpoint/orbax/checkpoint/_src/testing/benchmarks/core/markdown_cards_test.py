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

"""Tests for the markdown card renderers."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from orbax.checkpoint._src.testing.benchmarks.core import inventory as inv_lib
from orbax.checkpoint._src.testing.benchmarks.core import markdown_cards
from orbax.checkpoint._src.testing.benchmarks.core import run_manifest as rm_lib

_AGGS = {
    'save_background_4_throughput/save_total_gbps': {
        'max': 22.4,
        'min': 18.0,
        'mean': 20.1,
        'p50': 20.2,
        'p99': 22.3,
    },
    'load_4_throughput/load_total_gbps': {
        'max': 18.9,
        'min': 17.0,
        'mean': 18.0,
        'p50': 18.1,
        'p99': 18.8,
    },
    'save_background_5_inventory/save_total_gb': {
        'max': 140.2,
        'min': 140.2,
        'mean': 140.2,
    },
    'save_blocking_2_save_breakdown/blocking_async_s': {
        'max': 4.21,
        'min': 4.0,
        'mean': 4.1,
    },
}


class ScorecardTest(parameterized.TestCase):

  def test_headline_save_load_throughput_in_output(self):
    out = markdown_cards.render_scorecard('llama70b', _AGGS, None, None)
    self.assertIn('llama70b', out)
    self.assertIn('Save total throughput', out)
    self.assertIn('Load throughput', out)
    self.assertIn('22.4', out)
    self.assertIn('18.9', out)

  def test_inventory_section_appears_when_provided(self):
    inv = inv_lib.CheckpointInventory(
        total_bytes=140 * 1024**3,
        file_count=4096,
        small_file_count=128,
        small_file_pct=0.031,
        largest_file_bytes=64 * 1024**2,
        smallest_file_bytes=32,
        format={'ocdbt': 4090, 'metadata': 6},
    )
    out = markdown_cards.render_scorecard('llama70b', _AGGS, inv, None)
    self.assertIn('### Inventory', out)
    self.assertIn('4,096', out)
    self.assertIn('3.1%', out)
    self.assertIn('ocdbt', out)

  def test_inventory_section_omitted_when_none(self):
    out = markdown_cards.render_scorecard('llama70b', _AGGS, None, None)
    self.assertNotIn('### Inventory', out)

  def test_manifest_section_appears_when_provided(self):
    m = rm_lib.RunManifest(
        captured_at='2026-05-25T20:00:00+00:00',
        hostname='h',
        git_sha='abc123',
        git_dirty=False,
        jax_version='0.10.1',
        orbax_version='0.11.40',
        tensorstore_version='0.1.84',
        jax_process_count=4,
        jax_process_index=0,
        jax_device_count=8,
        jax_device_kind='cpu',
        xla_flags='',
        libtpu_init_args='',
    )
    out = markdown_cards.render_scorecard('llama70b', _AGGS, None, m)
    self.assertIn('## Run manifest', out)
    self.assertIn('abc123', out)
    self.assertIn('0.10.1', out)

  def test_no_aggregates_renders_skeleton(self):
    out = markdown_cards.render_scorecard('llama70b', {}, None, None)
    self.assertIn('llama70b', out)
    self.assertNotIn('Save throughput', out)


class ConfigurationTest(parameterized.TestCase):

  def test_options_table_renders(self):
    out = markdown_cards.render_configuration(
        'bench', {'a': 1, 'b': True}, None
    )
    self.assertIn('## bench', out)
    self.assertIn('### Benchmark options', out)
    self.assertIn('| `a` | `1` |', out)
    self.assertIn('| `b` | `True` |', out)

  def test_nested_spec_split_to_json_block(self):
    out = markdown_cards.render_configuration(
        'bench', {'a': 1}, {'spec': {'weights': {'shape': [4, 4]}}}
    )
    self.assertIn('### Checkpoint config — `spec`', out)
    self.assertIn('```json', out)

  def test_missing_checkpoint_section_omitted(self):
    out = markdown_cards.render_configuration('bench', {'a': 1}, None)
    self.assertNotIn('Checkpoint config', out)


class AggregatedMetricsTest(parameterized.TestCase):

  @dataclasses.dataclass
  class _FakeStats:
    mean: float
    std: float
    min: float
    max: float
    count: int

  def test_groups_by_section_prefix(self):
    stats = {
        '2_save_breakdown/blocking_s': self._FakeStats(1.0, 0.1, 0.9, 1.1, 3),
        '3_load_breakdown/blocking_s': self._FakeStats(2.0, 0.2, 1.8, 2.2, 3),
    }
    out = markdown_cards.render_aggregated_metrics(
        'b',
        stats,
        {
            '2_save_breakdown/blocking_s': 's',
            '3_load_breakdown/blocking_s': 's',
        },
    )
    self.assertIn('### 2_save_breakdown', out)
    self.assertIn('### 3_load_breakdown', out)

  def test_host_label_lands_in_title(self):
    stats = {
        '2_save_breakdown/blocking_s': self._FakeStats(1.0, 0.0, 1.0, 1.0, 1)
    }
    out = markdown_cards.render_aggregated_metrics(
        'b', stats, {}, host_label='host_3'
    )
    self.assertIn('host_3', out.splitlines()[0])

  def test_empty_returns_no_runs_message(self):
    out = markdown_cards.render_aggregated_metrics('b', {}, {})
    self.assertIn('No successful runs', out)


class OptionsToHparamsTest(parameterized.TestCase):

  @dataclasses.dataclass(frozen=True)
  class _Opts:
    async_enabled: bool = True
    chunk_byte_size: int | None = None
    notes: str = 'baseline'

  def test_primitive_fields_pass_through(self):
    h = markdown_cards.options_to_hparams(self._Opts(False, 256, 'x'))
    self.assertEqual(
        h, {'async_enabled': False, 'chunk_byte_size': 256, 'notes': 'x'}
    )

  def test_none_value_serialized_as_string(self):
    h = markdown_cards.options_to_hparams(self._Opts(chunk_byte_size=None))
    self.assertEqual(h['chunk_byte_size'], 'None')

  def test_non_dataclass_non_dict_returns_empty(self):
    self.assertEqual(markdown_cards.options_to_hparams(None), {})
    self.assertEqual(markdown_cards.options_to_hparams('a string'), {})


if __name__ == '__main__':
  absltest.main()
