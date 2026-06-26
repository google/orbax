# 📊 Orbax Checkpoint Benchmark Framework

> Measure the performance — and shape — of Orbax checkpoint operations (save,
> restore, reshard, broadcast, …) across model sizes, mesh topologies, and option
> combinations — locally, on a multi-host cluster, or on Pathways/cloud.

A benchmark has **two parts, and you provide both**:

| | |
|---|---|
| **① A benchmark class** | A small Python class whose `test_fn` runs the code you want to measure. You wrap the operations of interest in a one-line `measure(...)` block. *Reuse a built-in or write your own.* |
| **② A YAML config** | Selects which class to run, supplies its options, and describes the checkpoint(s) and mesh(es) to run against. |

Given those, the framework does everything else: expands every option
combination, runs each (optionally several times), collects a rich metric suite
**per host**, aggregates it **across hosts**, and writes results to the logs and
TensorBoard — and can capture a baseline and diff later runs against it.

> ⚠️ **The YAML alone is not enough.** It always points at a benchmark class; the
> measured code lives in that class's `test_fn`, not in the config.

### Capabilities at a glance

| Capability | What you get | Where |
|---|---|---|
| ✍️ **Write your own benchmark** | A class + `test_fn`; instrument code in one line | [§3](#3-writing-a-benchmark) |
| ♻️ **Reuse built-ins** | Save/restore, resharding, restore+broadcast, … | [§4](#4-reusing-a-built-in-benchmark) |
| ⏱️ **One-line metric capture** | `with measure("op"):` captures the whole suite; blocks **nest** (per-step breakdowns) | [§3](#3-writing-a-benchmark) |
| 📈 **Rich metric suite** | time, host & device memory, throughput, per-stage timings, TensorStore I/O, compile-cache | [§7](#7-metrics-reference) |
| 🔀 **Parameter sweeps** | list-valued options × meshes × checkpoints (Cartesian product) | [§8](#8-features) |
| 🔁 **Repeats + cross-host aggregation** | `num_repeats`; min/mean/max across all hosts | [§8](#8-features) |
| 🧪 **Synthetic or real data** | generate from a `spec`, or load from a `path` | [§5](#5-the-config-file) |
| 🗺️ **Multi-topology meshes** | list several meshes; incompatible ones are auto-skipped | [§5](#5-the-config-file) |
| 🆚 **Baselines (A/B)** | capture `<git_sha>.json`, then compare → per-metric delta | [§8](#8-features) |
| 🔬 **Profiler traces + HLO** | `enable_trace`, `--enable_hlo_dump` | [§8](#8-features) |
| 📊 **TensorBoard** | scalars, HParams, profile traces, inventory cards | [§8](#8-features) |
| ☁️ **Pathways & cloud** | auto backend init; colocated-Python load dispatcher; XPK launcher | [§6](#6-running-on-pathways-and-cloud) |

### Contents

1. [60-second tour](#1-60-second-tour)
2. [How it works](#2-how-it-works)
3. [Writing a benchmark](#3-writing-a-benchmark)
4. [Reusing a built-in benchmark](#4-reusing-a-built-in-benchmark)
5. [The config file](#5-the-config-file)
6. [Running on Pathways and cloud](#6-running-on-pathways-and-cloud)
7. [Metrics reference](#7-metrics-reference)
8. [Features](#8-features)
9. [Recipes](#9-recipes)
10. [Output layout](#10-output-layout)
11. [Cheat sheet](#11-cheat-sheet)
12. [Source map](#12-source-map)

---

## 1. 60-second tour

🚀 The fastest path: reuse the built-in save/restore benchmark.

**1 — a config** (`sweep.yaml`):

```yaml
suite_name: "ocdbt vs non-ocdbt"
num_repeats: 3
checkpoint_config:                          # synthetic data - no real model needed
  spec:
    params: {dtype: bfloat16, shape: [8192, 8192], sharding: [fsdp]}
mesh_config:
  mesh_axes: ["fsdp"]
  ici_parallelism: {"fsdp": 8}
benchmarks:
  - generator: "orbax.checkpoint._src.testing.benchmarks.v1.benchmark.Benchmark"
    options:
      use_ocdbt: [true, false]              # a list ⇒ swept
```

**2 — run it** (by file, from the repo root):

```bash
python checkpoint/orbax/checkpoint/_src/testing/benchmarks/run_benchmarks.py \
  --config_file=sweep.yaml --output_directory=/tmp/bench/
```

**3 — read the results** — per-operation timing, throughput, memory and I/O are
printed to the logs and written to TensorBoard:

```bash
tensorboard --logdir=/tmp/bench/tensorboard/
```

Two benchmarks ran (`use_ocdbt` true/false), each 3×, aggregated across hosts.
To benchmark something the built-ins don't cover, write a class — see [§3](#3-writing-a-benchmark).

---

## 2. How it works

🧠 The pieces (in `orbax/checkpoint/_src/testing/benchmarks/`):

| Object | Role |
|---|---|
| [`BenchmarkOptions`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/core/core.py) | Frozen dataclass of your knobs. **Any list field becomes a sweep axis.** |
| [`BenchmarksGenerator`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/core/core.py) | The class you write/reuse. Its **`test_fn`** runs the measured code, once per `(option × mesh × checkpoint)`. |
| [`Metrics.measure(...)`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/core/metric.py) | The one-line context manager you wrap measured code in. |
| [`TestContext`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/core/core.py) | `test_fn`'s input: the `pytree`, a working `path`, the `options`, the `mesh`, `trace_path()`. |
| [`TestResult`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/core/core.py) | `test_fn`'s output: the collected `Metrics`. |
| [`TestSuite`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/core/core.py) | Orchestrator built from your YAML: expands, repeats, aggregates, baselines. |

```
your YAML ──► TestSuite ──► your BenchmarksGenerator.generate()
                                │   options × meshes × checkpoints
                                ▼
                          [ Benchmark, … ] ──► Benchmark.run() × num_repeats
                                                   │
                    ┌──────── your test_fn(TestContext) ────────┐
                    │   with metrics.measure("save"): ...        │  ← the code you measure
                    │   return TestResult(metrics)               │
                    └────────────────────────────────────────────┘
                                                   │
        cross-host aggregation ──► logs + TensorBoard (+ optional baseline A/B)
```

Everything outside the box is handled for you. Your job: the `test_fn` and the
YAML that drives it.

---

## 3. Writing a benchmark

✍️ A benchmark class is three small pieces: an **options dataclass**, the
**`@benchmark_options`** decorator, and a **`BenchmarksGenerator`** with a
**`test_fn`**.

### Capturing metrics — the `measure()` block

Wrap any block in `metrics.measure("name")` and the framework captures the whole
default metric suite around it — time, host & device memory, compile-cache,
TensorStore I/O, and Orbax's per-stage timings/throughput:

```python
with metrics.measure("save"):       # one line ⇒ the whole metric suite
  ocp.save(path, pytree)
```

The name becomes the **operation prefix** on every metric the block emits
(`save_0_basics/time_s`, …). Use one `measure()` per operation you want broken
out.

> 💡 **`measure()` is not checkpoint-specific** — it times any block. Time, memory,
> device and compile-cache metrics populate for anything; the Orbax save/load
> breakdown tags simply stay empty unless the block does Orbax I/O.

**Blocks nest** — capture an aggregate *and* each step at once:

```python
 with metrics.measure("train_steps"):                 # whole-loop aggregate
   for step in range(options.num_steps):
     with metrics.measure(f"step_{step}"):            # per-step breakdown
       params, opt_state = train_step(params, opt_state, next(data_iter))
```
yields `train_steps_0_basics/time_s` alongside `step_0_…`, `step_1_…` (the
collectors are reentrant-safe, so nesting is fine).

### A complete example

```python
import dataclasses

import jax
from orbax.checkpoint import v1 as ocp
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib


# ① Your knobs. A list-typed field is swept (see §8).
@dataclasses.dataclass(frozen=True)
class SaveRestoreOptions(benchmarks_core.BenchmarkOptions):
  use_ocdbt: bool | list[bool] = True

  def is_valid(self) -> bool:        # optional: drop nonsensical combinations
    return True


# ② Bind the options to the generator.
@benchmarks_core.benchmark_options(SaveRestoreOptions)
class SaveRestoreBenchmark(benchmarks_core.BenchmarksGenerator):

  # ③ The measured code. Called once per (option × mesh × checkpoint).
  def test_fn(
      self, context: benchmarks_core.TestContext
  ) -> benchmarks_core.TestResult:
    metrics = metric_lib.Metrics()

    options = context.options                  # a SaveRestoreOptions instance
    pytree = context.pytree                    # generated from checkpoint_config.spec
    abstract = jax.tree.map(ocp.arrays.to_shape_dtype_struct, pytree)
    path = context.path / "ckpt"               # a fresh per-run directory

    with ocp.Context(
        array_options=ocp.options.ArrayOptions(
            saving=ocp.options.ArrayOptions.Saving(use_ocdbt=options.use_ocdbt)
        )
    ):
      with metrics.measure("save"):
        ocp.save(path, pytree)
      with metrics.measure("load"):
        _ = ocp.load(path, abstract_state=abstract)

    return benchmarks_core.TestResult(metrics=metrics)
```

That's the whole benchmark. The config (§5) decides `context.pytree` /
`context.mesh`, the repeat count, and which `use_ocdbt` values to sweep.

### `test_fn` input — `TestContext`

| Field | Meaning |
|---|---|
| `pytree` | Checkpoint data the framework generated (from `spec`) or loaded (from `path`); `None` if neither is set. |
| `path` | A fresh per-run working directory. |
| `options` | The resolved options for this sweep point. |
| `mesh` | The `jax.sharding.Mesh` from `mesh_config` (or `None`). |
| `repeat_index` | Which repeat this is (or `None`). |
| `trace_path(op)` | Profiler-trace directory for operation `op` when `enable_trace` is on, else `None`. |

### `test_fn` output — `TestResult`

Return `TestResult(metrics=metrics)`. The framework fills in the run path and a
checkpoint **inventory** (bytes/files) automatically. If `test_fn` raises, the
error is recorded and the run exits non-zero — you don't catch exceptions
yourself.

---

## 4. Reusing a built-in benchmark

♻️ For common cases, just reference a built-in under `benchmarks: - generator:`.

### [`v1.benchmark.Benchmark`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/v1/benchmark.py) — save + restore round-trip

The canonical generator: saves the pytree then restores it, measuring
`save_blocking`, `save_background`, and `load`. Options
([`v1.benchmark.BenchmarkOptions`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/v1/benchmark.py)):

| Option | Default | Meaning |
|---|---|---|
| `async_enabled` | `true` | Use `save_async` (splits blocking vs background time). |
| `use_ocdbt` | `true` | Use the OCDBT driver. |
| `use_zarr3` | `true` | Use Zarr v3. |
| `use_compression` | `true` | Compress array data. |
| `use_replica_parallel` | `false` | Parallelize writes across replicas. |
| `enable_replica_parallel_separate_folder` | `false` | Separate folder per replica (needs `use_replica_parallel` + `use_ocdbt`). |
| `use_load_and_broadcast` | `false` | Primary host loads, then broadcasts. |
| `chunk_byte_size` | `None` | Array chunk size. |
| `save_concurrent_gb` / `restore_concurrent_gb` | `None` | Concurrency budgets (GiB). |
| `metric_tracemalloc_enabled` | `false` | Add the `tracemalloc` metric. |
| `enable_trace` / `trace_every_repeat` | `false` | Profiler traces (§8). |

### Other built-ins

| Generator | Measures |
|---|---|
| [`v1.resharding_benchmark.ReshardingBenchmark`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/v1/resharding_benchmark.py) | Loading an existing checkpoint into a target sharding (`reference_checkpoint_path` + `reference_sharding_path`). |
| [`v1.restore_and_broadcast_benchmark.RestoreAndBroadcastBenchmark`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/v1/restore_and_broadcast_benchmark.py) | Restore-on-one-replica-then-broadcast. |
| [`v1.replica_parallel_multislice_benchmark.ReplicaParallelMultislice`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/v1/replica_parallel_multislice_benchmark.py) | Replica-parallel saving across slices. |

> Other generators in this directory (`checkpoint_manager_benchmark`,
> `array_handler_benchmark`, `pytree_checkpoint_benchmark`,
> `emergency_checkpoint_manager_benchmark`, `single_replica_benchmark`,
> `pytorch_checkpoint_benchmark`, …) target specific
> subsystems and follow the same pattern.

---

## 5. The config file

⚙️ A single YAML file that selects and parametrizes your benchmark class.

### Top-level keys

| Key | Required | Default | Meaning |
|---|---|---|---|
| `suite_name` | yes | — | Human-readable run name. |
| `num_repeats` | no | `1` | Times to run each generated benchmark. |
| `checkpoint_config` / `checkpoint_configs` | no | one empty config | Checkpoint(s) to save/load. Plural ⇒ swept. |
| `mesh_config` / `mesh_configs` | no | none | Device mesh(es). Plural ⇒ swept. Omitted ⇒ `context.mesh` is `None`. |
| `benchmarks` | yes | — | List of `{generator, options}` entries. |
| `baseline_capture_path` | no | none | Write captured baseline JSON here. |
| `baseline_path` | no | none | Compare against this stored baseline. |

### `benchmarks`

```yaml
benchmarks:
  - generator: "my.module.SaveRestoreBenchmark"   # import path (built-in or yours)
    options:
      use_ocdbt: [true, false]                      # any list ⇒ swept
```

### `checkpoint_config`

Describes `context.pytree` — **generated from a `spec`** or **loaded from a
`path`** (exactly one).

| Field | Default | Meaning |
|---|---|---|
| `spec` | `None` | Synthetic pytree: `name → {dtype, shape, sharding}` for arrays, or `name → "int"` / `"str"` for scalars. `sharding` is a `PartitionSpec`-style list of axis names. |
| `path` | `None` | Load an existing checkpoint. Mutually exclusive with `spec`. |
| `random_seed` | `0` | Seed for synthetic generation (deterministic). |
| `sharding_config_path` | `None` | Per-tensor target sharding JSON, used with `path` (format in `core/configs.py`). |
| `load_with_colocated_python` | `false` | On Pathways, load via the colocated-Python dispatcher (§6). |

### `mesh_config`

Translated into `context.mesh`.

| Field | Default | Meaning |
|---|---|---|
| `mesh_axes` | — | Parallelism dimension names, e.g. `["data", "fsdp", "tensor"]`. |
| `ici_parallelism` | `{}` | Per-axis degree *within* a slice, e.g. `{"fsdp": 8}`. |
| `dcn_parallelism` | `None` | Per-axis degree *across* slices (multi-slice). |
| `allow_split_physical_axes` | `false` | Allow splitting physical axes. |
| `process_is_granule` | `false` | Treat processes as the outer-network unit. |

> 💡 The product of all axis degrees must equal the device count, or the mesh is
> **skipped**. List several meshes and only the ones that fit the live hardware
> run.

### Running it

Run the `run_benchmarks.py` script **directly** (paths relative to the repo
root):

```bash
python checkpoint/orbax/checkpoint/_src/testing/benchmarks/run_benchmarks.py \
  --config_file=<config.yaml> \
  --output_directory=<dir> \
  [flags...]
```

| Flag | Required | Meaning |
|---|---|---|
| `--config_file` | yes | Path to the YAML config. |
| `--output_directory` | yes | Where results, TensorBoard logs, and traces go. |
| `--local_directory` | no | Local scratch directory (some checkpoint-manager benchmarks). |
| `--enable_hlo_dump` | no | Dump XLA HLO protos to `<output_directory>/hlo_dump/`. |
| `--remove_repeated_dir` | no | Delete the generated `repeat_*` directories after the run. |

The runner enables `jax_enable_x64`. On a single process it runs locally; on
CPU, simulate devices with
`XLA_FLAGS=--xla_force_host_platform_device_count=8`.

---

## 6. Running on Pathways and cloud

☁️ The same config and the same benchmark class run unchanged on a local host, a
multi-process cluster, or the **Pathways single-controller** backend — the
framework adapts automatically.

- **Automatic backend init.** [`run_benchmarks.py`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/run_benchmarks.py)
  detects the active backend: in a multi-process JAX cluster it calls
  `jax.distributed.initialize()` from the standard env vars
  (`JAX_COORDINATOR_ADDRESS`, `JAX_PROCESS_ID`, `JAX_NUM_PROCESSES`,
  `JAX_COORDINATOR_PORT`); if Pathways is in use
  (`pathwaysutils.is_pathways_backend_used()`) it initializes Pathways instead.
  No change to your config or class.

- **Colocated-Python load dispatcher.** On Pathways, the framework's checkpoint
  loader runs through Orbax's **colocated-Python** path: set
  [`load_with_colocated_python`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/core/configs.py)`: true`
  in `checkpoint_config` (with a `path`). The loader
  ([`checkpoint_generation.load_checkpoint`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/core/checkpoint_generation.py))
  then builds the Pathways `CheckpointingImpl` with colocated Python and
  registers the colocated type handlers, so deserialization runs *colocated with
  the TPU workers*, dispatched from the single controller — the production
  Pathways load path, measured end to end.

  ```yaml
  checkpoint_config:
    path: "gs://my-bucket/ckpt/items"
    sharding_config_path: "gs://my-bucket/sharding/abstract_state.json"
    load_with_colocated_python: true        # colocated-Python dispatcher on Pathways
  ```

- **Cloud benchmarking via XPK.** The `xpk/` launcher runs a suite on a
  GKE/Pathways cluster:
  [`xpk/launch_xpk.py`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/xpk/launch_xpk.py)
  `--enable_pathways` provisions the Pathways server / proxy / colocated-Python
  **sidecar** images and runs `run_benchmarks` on the cluster (the sidecar
  executes the colocated-Python code). For the end-to-end setup see
  **[`xpk/PathwaysColocatedPythonGuide.md`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/xpk/PathwaysColocatedPythonGuide.md)**;
  for the launchers see
  [`xpk/README.md`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/xpk/README.md)
  (GKE/XPK) and
  [`tpu_vm/README.md`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/tpu_vm/README.md)
  (single TPU VM).

---

## 7. Metrics reference

📈 Every result from a `measure("<operation>", metric_keys)` block is a
TensorBoard scalar named:

```
<operation>_<namespace>/<metric>        e.g.  load_0_basics/time_s
```

The `<operation>` prefix is the string you passed to `measure()`.

### Collectors

`measure()` with no `metric_keys` uses the **defaults**: `time`, `rss`,
`jax_monitoring`, `device_memory`, `tensorstore`. `tracemalloc` is opt-in.

| Key | Records |
|---|---|
| `time` | Wall-clock duration of the block. |
| `rss` | Host RSS memory delta. |
| `jax_monitoring` | Orbax's `jax.monitoring` emissions: per-stage timings, throughput, bytes, compile-cache. |
| `device_memory` | `jax.live_arrays()` delta (leak canary) + per-device HBM peak delta. |
| `tensorstore` | TensorStore kvstore op counts, cache hit/miss, tcmalloc deltas (whitelisted). |
| `tracemalloc` | Python allocation peak + top sites (opt-in; `metric_tracemalloc_enabled: true`). |

Scope it explicitly with a list:
`metrics.measure("load", ["time", "device_memory"])`.

### Namespaces

Results group into ordered namespaces so the dashboard reads top-to-bottom:

| Namespace | Source | Representative metrics (units) |
|---|---|---|
| `0_basics/` | time, rss | `time_s` (s), `host_rss_diff_mb` (MB) |
| `2_save_breakdown/` | jax_monitoring | `blocking_s`, `total_s`, `directory_creation_s`, `metadata_write_s`, `commit_s`, `ocdbt_merge_s` |
| `3_load_breakdown/` | jax_monitoring | `blocking_s`, `total_s`, `worker_io_s`, `primary_deserialization_s`, `broadcast_s` |
| `4_throughput/` | jax_monitoring | `save_blocking_gbps`, `save_total_gbps`, `load_total_gbps`, `load_per_host_gbps` (GiB/s) |
| `5_inventory/` | jax_monitoring | `save_total_gb`, `replicated_array_gb`, `sharded_array_gb`, `load_requested_gb` (GiB) |
| `6_tensorstore/` | tensorstore | `changed_metric_count`, per-bucket kvstore-op / cache-hit / tcmalloc `_diff` counters |
| `7_memory/` | device_memory, tracemalloc | `jax_live_arrays_count_delta` (count), `device_hbm_peak_diff_gb` (GiB), `tracemalloc_peak_diff_mb` (MB) |
| `8_jax/` | jax_monitoring | `cache_hit_rate` (derived), compilation-cache hit/miss tallies, compile-time-saved |

> `0_basics` / `6_tensorstore` / `7_memory` populate for **any** measured block;
> the `2_`–`5_` / `8_` namespaces come from Orbax's `jax.monitoring` events, so
> they populate when the block performs Orbax I/O. The collectors live in
> [`core/metric.py`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/core/metric.py);
> the event→tag map is in
> [`core/jax_monitoring_tags.py`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/core/jax_monitoring_tags.py)
> (unmapped events still surface).

### Headline metrics

🎯 The five to watch:

- **`<op>_4_throughput/*_gbps`** — effective bandwidth (the primary perf signal).
- **`<op>_0_basics/time_s`** — wall-clock per operation.
- **`<op>_2_save_breakdown/*` / `*_3_load_breakdown/*`** — where the time went.
- **`<op>_7_memory/device_hbm_peak_diff_gb`** — peak device-memory cost.
- **`<op>_8_jax/cache_hit_rate`** — compilation-cache effectiveness.

---

## 8. Features

### 🔀 Parameter sweeps

Any options field set to a **list** is a sweep axis; the benchmark runs the
Cartesian product of all axes. `checkpoint_configs` / `mesh_configs` (plural)
sweep too. `is_valid()` on your options drops invalid combinations.

```yaml
options:
  use_ocdbt: [true, false]
  use_zarr3: [true, false]   # ⇒ 4 benchmarks
```

### 🔁 Repeats & cross-host aggregation

`num_repeats` runs each benchmark N times; metrics are aggregated **across
hosts** (min/mean/max/…), so multi-process runs report cluster-wide
statistics, not just rank 0. `--remove_repeated_dir` cleans up per-repeat
checkpoint directories.

### 🆚 Baselines (capture / compare)

Set these in the config (suite level), not via flags:

```yaml
# Capture on the baseline revision (writes cross-host aggregates as <git_sha>.json):
baseline_capture_path: gs://bucket/baselines/my_suite/
# Compare a later revision (logs a per-metric delta): set this instead:
baseline_path: gs://bucket/baselines/my_suite/<git_sha>.json
```

If no git sha resolves, the baseline is written as `unknown.json`.

### 🔬 Profiler traces & HLO dumps

- `enable_trace: true` (an options field) captures a `jax.profiler` trace per
  measured operation, surfaced as its own run in the TensorBoard **Profile**
  tab. Only the first repeat is traced by default; `trace_every_repeat: true`
  traces all.
- `--enable_hlo_dump` writes XLA HLO protos to `<output_directory>/hlo_dump/`.

### 📊 TensorBoard output

Under `<output_directory>/tensorboard/`: **scalars** (every metric), **HParams**
(the option combination per benchmark, for filtering/comparison), **profile
traces** (when `enable_trace` is on), and **markdown cards** (checkpoint
inventory + run summary).

---

## 9. Recipes

### A — Sweep storage options (built-in class)

See the [60-second tour](#1-60-second-tour).

### B — Run a benchmark class you wrote

With `SaveRestoreBenchmark` from §3 importable as `my.module.SaveRestoreBenchmark`:

```yaml
suite_name: "my save/restore"
num_repeats: 3
checkpoint_config:
  spec: {params: {dtype: float32, shape: [4096, 4096], sharding: [data]}}
mesh_config:
  mesh_axes: ["data"]
  ici_parallelism: {"data": 8}
benchmarks:
  - generator: "my.module.SaveRestoreBenchmark"
    options: {use_ocdbt: [true, false]}
```

### C — Load a real checkpoint into a target sharding (built-in)

```yaml
suite_name: "resharding"
num_repeats: 10
mesh_config:
  mesh_axes: ["data", "fsdp", "tensor"]
  ici_parallelism: {"data": 1, "fsdp": 16, "tensor": 1}
benchmarks:
  - generator: "orbax.checkpoint._src.testing.benchmarks.v1.resharding_benchmark.ReshardingBenchmark"
    options:
      reference_checkpoint_path: "gs://my-bucket/ckpt/items"
      reference_sharding_path: "gs://my-bucket/sharding/abstract_state.json"
```

### D — Capture a baseline, then compare

Set the baseline path in the config (`baseline_capture_path` to capture,
`baseline_path` to compare), then run the same command on each revision:

```bash
RB=checkpoint/orbax/checkpoint/_src/testing/benchmarks/run_benchmarks.py
# baseline revision (sweep.yaml has `baseline_capture_path: gs://my-bucket/baselines/sweep/`):
python $RB --config_file=sweep.yaml --output_directory=/tmp/baseline/
# candidate revision (sweep.yaml has `baseline_path: gs://my-bucket/baselines/sweep/<git_sha>.json`):
python $RB --config_file=sweep.yaml --output_directory=/tmp/candidate/
```

---

## 10. Output layout

```none
<output_directory>/
├── tensorboard/                 # scalars, HParams, markdown cards
│   └── <benchmark>__<op>/       # per-operation profiler traces (enable_trace)
├── hlo_dump/                    # XLA HLO protos (--enable_hlo_dump)
└── <benchmark>/repeat_*/        # per-run checkpoint dirs (unless --remove_repeated_dir)
```

Metrics are also printed to the logs as a per-process report after each
benchmark, and (when `baseline_path` is set) as a per-metric delta table.

---

## 11. Cheat sheet

```none
RUN          python checkpoint/orbax/checkpoint/_src/testing/benchmarks/run_benchmarks.py \
               --config_file=cfg.yaml --output_directory=/tmp/out/

MEASURE      with metrics.measure("op"):  <code>        # captures the whole suite; blocks nest
             return TestResult(metrics=metrics)

CLASS        @benchmarks_core.benchmark_options(MyOptions)
             class MyBenchmark(benchmarks_core.BenchmarksGenerator):
               def test_fn(self, context) -> benchmarks_core.TestResult: ...

CONFIG       suite_name / num_repeats / checkpoint_config(.spec|.path) / mesh_config / benchmarks:[{generator, options}]
SWEEP        any list-valued option, or checkpoint_configs / mesh_configs
DEFAULT      metrics: time, rss, jax_monitoring, device_memory, tensorstore   (+ tracemalloc opt-in)
HEADLINE     <op>_4_throughput/*_gbps · <op>_0_basics/time_s · <op>_7_memory/device_hbm_peak_diff_gb · <op>_8_jax/cache_hit_rate
PATHWAYS     auto-init on Pathways · checkpoint_config.load_with_colocated_python: true · xpk/launch_xpk.py --enable_pathways
```

---

## 12. Source map

Every moving part, on `google/orbax` `main`:

**Core framework** — [`core/`](https://github.com/google/orbax/tree/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/core)

| File | Contains |
|---|---|
| [`core.py`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/core/core.py) | `BenchmarkOptions`, `BenchmarksGenerator`, `Benchmark`, `TestContext`, `TestResult`, `TestSuite` |
| [`metric.py`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/core/metric.py) | `Metrics.measure()` + every metric collector |
| [`jax_monitoring_tags.py`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/core/jax_monitoring_tags.py) | event → TensorBoard-tag map |
| [`configs.py`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/core/configs.py) | `CheckpointConfig` (incl. `load_with_colocated_python`), `MeshConfig` |
| [`config_parsing.py`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/core/config_parsing.py) | YAML → `TestSuite` |
| [`checkpoint_generation.py`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/core/checkpoint_generation.py) | synthetic generation + the Pathways colocated-Python load path |
| [`device_mesh.py`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/core/device_mesh.py) | `MeshConfig` → `jax.sharding.Mesh` |
| [`baseline.py`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/core/baseline.py) | baseline capture / compare |
| [`metrics_manager.py`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/core/metrics_manager.py) | cross-host aggregation + TensorBoard writing |

**Entrypoint** — [`run_benchmarks.py`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/run_benchmarks.py) (CLI flags, distributed / Pathways init)

**Built-in generators** — [`v1/`](https://github.com/google/orbax/tree/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/v1):
[`benchmark.py`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/v1/benchmark.py) ·
[`resharding_benchmark.py`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/v1/resharding_benchmark.py) ·
[`restore_and_broadcast_benchmark.py`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/v1/restore_and_broadcast_benchmark.py) ·
[`replica_parallel_multislice_benchmark.py`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/v1/replica_parallel_multislice_benchmark.py)

**Cloud / launchers** —
[`xpk/`](https://github.com/google/orbax/tree/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/xpk)
(GKE/XPK; [`launch_xpk.py`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/xpk/launch_xpk.py),
[`PathwaysColocatedPythonGuide.md`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/xpk/PathwaysColocatedPythonGuide.md)) ·
[`tpu_vm/`](https://github.com/google/orbax/tree/main/checkpoint/orbax/checkpoint/_src/testing/benchmarks/tpu_vm)
(single TPU VM)
