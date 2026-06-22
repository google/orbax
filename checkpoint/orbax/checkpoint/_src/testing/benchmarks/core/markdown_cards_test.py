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

# Aggregates reach the renderer already sliced to one operation name, so keys
# are bare tags (no `load::` / `save::` prefix).
_LOAD_AGGS = {
    '3_load_breakdown/blocking_s': {'mean': 12.0, 'min': 11.5, 'max': 12.5},
    '3_load_breakdown/total_s': {'mean': 13.0, 'min': 13.0, 'max': 13.0},
    '3_load_breakdown/worker_io_s': {'mean': 9.8, 'min': 9.7, 'max': 9.9},
    '6_tensorstore/tensorstore_kvstore_file_bytes_read_value_diff': {
        'mean': 8.75 * 1024**3,
        'min': 8.7 * 1024**3,
        'max': 8.8 * 1024**3,
    },
    '5_inventory/load_total_gb': {'mean': 140.0, 'min': 140.0, 'max': 140.0},
    '6_tensorstore/tensorstore_kvstore_file_read_value_diff': {
        'mean': 6.0,
        'min': 5.0,
        'max': 7.0,
    },
    '4_throughput/load_per_host_gbps': {'mean': 0.7, 'min': 0.6, 'max': 0.8},
    '4_throughput/load_total_gbps': {'mean': 11.0, 'min': 10.0, 'max': 11.3},
}

_LOAD_PER_HOST = [
    (
        0,
        {
            '6_tensorstore/tensorstore_kvstore_file_bytes_read_value_diff': (
                8.7 * 1024**3
            ),
            '6_tensorstore/tensorstore_kvstore_file_read_value_diff': 6,
            '3_load_breakdown/blocking_s': 11.9,
            '3_load_breakdown/total_s': 12.9,
        },
    ),
    (
        1,
        {
            '6_tensorstore/tensorstore_kvstore_file_bytes_read_value_diff': (
                8.8 * 1024**3
            ),
            '6_tensorstore/tensorstore_kvstore_file_read_value_diff': 7,
            '3_load_breakdown/blocking_s': 12.1,
            '3_load_breakdown/total_s': 13.1,
        },
    ),
]

# 140 GiB on-disk checkpoint — the per-host % column divides by this.
_INV = inv_lib.CheckpointInventory(
    total_bytes=140 * 1024**3,
    file_count=4096,
    small_file_count=128,
    small_file_pct=0.031,
    largest_file_bytes=64 * 1024**2,
    smallest_file_bytes=32,
    format={'ocdbt': 4090, 'metadata': 6},
)


class GlanceCardTest(parameterized.TestCase):

  def test_load_headline_renders(self):
    out = markdown_cards.render_glance_card(
        'llama70b', _LOAD_AGGS, _LOAD_PER_HOST, _INV, None
    )
    self.assertIn('## llama70b — at a glance', out)
    self.assertIn('| Total time | 13.00 s |', out)
    self.assertIn('| Throughput | 11.30 GiB/s |', out)
    self.assertIn('Bytes / host (mean)', out)
    self.assertIn('8.75 GiB', out)
    self.assertIn('Checkpoint size (on-disk)', out)

  def test_per_host_table_with_pct_of_checkpoint(self):
    out = markdown_cards.render_glance_card(
        'b', _LOAD_AGGS, _LOAD_PER_HOST, _INV, None
    )
    self.assertIn('#### Per-host', out)
    self.assertIn('| 0 |', out)
    self.assertIn('| 1 |', out)
    self.assertRegex(out, r'\|\s*reads/writes\s*\|')
    self.assertIn('% of total', out)
    self.assertIn('6.2%', out)  # 8.7 / 140 ≈ 6.2%

  def test_inventory_and_manifest_folded_in(self):
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
    out = markdown_cards.render_glance_card(
        'b', _LOAD_AGGS, _LOAD_PER_HOST, _INV, m
    )
    self.assertIn('### Inventory', out)
    self.assertIn('4,096', out)
    self.assertIn('## Run manifest', out)
    self.assertIn('abc123', out)

  def test_save_metrics_render(self):
    save_aggs = {
        '2_save_breakdown/blocking_s': {'mean': 30.0, 'min': 29.0, 'max': 31.0},
        '2_save_breakdown/total_s': {'mean': 31.0, 'min': 31.0, 'max': 31.0},
        '6_tensorstore/tensorstore_kvstore_file_bytes_written_value_diff': {
            'mean': 8.75 * 1024**3,
            'min': 8.7 * 1024**3,
            'max': 8.8 * 1024**3,
        },
        '4_throughput/save_total_gbps': {'mean': 4.0, 'min': 3.5, 'max': 4.6},
    }
    per_host = [
        (
            0,
            {
                '6_tensorstore/tensorstore_kvstore_file_bytes_written_value_diff': (
                    8.7 * 1024**3
                ),
                '2_save_breakdown/blocking_s': 29.5,
            },
        ),
        (
            1,
            {
                '6_tensorstore/tensorstore_kvstore_file_bytes_written_value_diff': (
                    8.8 * 1024**3
                ),
                '2_save_breakdown/blocking_s': 30.5,
            },
        ),
    ]
    out = markdown_cards.render_glance_card(
        'b', save_aggs, per_host, None, None
    )
    self.assertIn('| Total time | 31.00 s |', out)
    self.assertIn('| Throughput | 4.60 GiB/s |', out)
    self.assertIn('Bytes / host (mean)', out)
    self.assertIn('8.75 GiB', out)

  def test_sub_gib_bytes_are_humanized(self):
    aggs = {
        '3_load_breakdown/blocking_s': {'mean': 1.0, 'min': 1.0, 'max': 1.0},
        # 0.5 GiB per host -> should render as MiB, not "0.00 GiB".
        '6_tensorstore/tensorstore_kvstore_file_bytes_read_value_diff': {
            'mean': 0.5 * 1024**3,
            'min': 0.5 * 1024**3,
            'max': 0.5 * 1024**3,
        },
    }
    per_host = [(
        0,
        {
            '6_tensorstore/tensorstore_kvstore_file_bytes_read_value_diff': (
                0.5 * 1024**3
            )
        },
    )]
    out = markdown_cards.render_glance_card('b', aggs, per_host, None, None)
    self.assertIn('512.00 MiB', out)
    self.assertNotIn('0.00 GiB', out)

  def test_safetensors_io_bytes_drive_load_card(self):
    # Safetensors has no kvstore counters: per-host bytes + reads arrive via
    # jax.monitoring under 6_io/, and must still populate the bytes row.
    aggs = {
        '3_load_breakdown/blocking_s': {'mean': 2.0, 'min': 2.0, 'max': 2.0},
        '6_io/file_bytes_read_per_host': {
            'mean': 8.75 * 1024**3,
            'min': 8.7 * 1024**3,
            'max': 8.8 * 1024**3,
        },
        '6_io/file_reads_per_host_count': {'mean': 3.0, 'min': 3.0, 'max': 3.0},
    }
    per_host = [
        (
            0,
            {
                '6_io/file_bytes_read_per_host': 8.7 * 1024**3,
                '6_io/file_reads_per_host_count': 3,
            },
        ),
        (
            1,
            {
                '6_io/file_bytes_read_per_host': 8.8 * 1024**3,
                '6_io/file_reads_per_host_count': 3,
            },
        ),
    ]
    out = markdown_cards.render_glance_card('st', aggs, per_host, _INV, None)
    self.assertIn('Bytes / host (mean)', out)
    self.assertIn('8.75 GiB', out)
    self.assertIn('% of total', out)


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
