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

"""Core classes and functions for benchmarking Orbax."""

import abc
from collections.abc import Callable, Sequence
import dataclasses
import hashlib
import itertools
import sys
from typing import Any

from absl import logging
from etils import epath
import jax
from orbax.checkpoint._src.testing.benchmarks.core import baseline as baseline_lib
from orbax.checkpoint._src.testing.benchmarks.core import checkpoint_generation
from orbax.checkpoint._src.testing.benchmarks.core import configs
from orbax.checkpoint._src.testing.benchmarks.core import device_mesh
from orbax.checkpoint._src.testing.benchmarks.core import directory_setup
from orbax.checkpoint._src.testing.benchmarks.core import inventory as inventory_lib
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib
from orbax.checkpoint._src.testing.benchmarks.core import metrics_manager
from orbax.checkpoint._src.testing.benchmarks.core import multihost


@dataclasses.dataclass(frozen=True, kw_only=True)
class BenchmarkOptions:
  """Base class for benchmark generator options.

  The baseline_* fields are orchestration switches configured once per run
  (not swept); subclasses inherit them automatically.

  Attributes:
    enable_trace: Whether to capture a per-operation `jax.profiler` trace for
      each measured save/load. Off by default.
    trace_every_repeat: Whether to capture traces on every repeat, or only the
      first repeat (the default); traces on every repeat overflow the Trace
      Viewer.
    baseline_path: Path to a stored baseline to compare this run against, or
      None to skip comparison.
    baseline_capture_path: Path to write this run's baseline to, or None to
      skip capture.
  """

  enable_trace: bool = False
  trace_every_repeat: bool = False
  baseline_path: str | None = None
  baseline_capture_path: str | None = None

  @classmethod
  def from_dict(cls, options_dict: dict[str, Any]) -> "BenchmarkOptions":
    """Creates a BenchmarkOptions subclass from a dictionary of options."""
    return cls(**options_dict)

  def is_valid(self) -> bool:
    """Returns whether the current option combination is valid."""
    return True


def benchmark_options(options_cls):
  """Decorator to associate a BenchmarkOptions subclass with a BenchmarksGenerator."""
  if not issubclass(options_cls, BenchmarkOptions):
    raise TypeError(
        "Decorating class must be a subclass of BenchmarkOptions, got"
        f" {options_cls}"
    )

  def wrapper(generator_cls):
    if not issubclass(generator_cls, BenchmarksGenerator):
      raise TypeError(
          "benchmark_options decorator can only be used on BenchmarksGenerator"
          f" subclasses, got {generator_cls}"
      )

    generator_cls.options_class = options_cls
    return generator_cls

  return wrapper


@dataclasses.dataclass
class TestContext:
  """Input object passed to each test function, providing pre-configured components for the test run.

  Attributes:
    pytree: The generated or loaded checkpoint data. May be None.
    path: The test directory path.
    options: The specific BenchmarkOptions for this test variant.
    mesh: The mesh used for sharding the checkpoint data.
    repeat_index: The index of the repeat run, if this test is run multiple
      times.
    local_path: The local path to store the checkpoint data.
    output_dir: The suite-level output directory. Used by `trace_path` to
      compose TB Profile-plugin-discoverable trace paths.
    name: The benchmark's stable hashed name. Used by `trace_path` as the
      Profile-plugin <run> segment so traces show up alongside scalars.
  """

  pytree: Any | None
  path: epath.Path
  options: BenchmarkOptions  # The specific options for this test variant.
  mesh: jax.sharding.Mesh | None = None
  repeat_index: int | None = None
  local_path: epath.Path | None = None
  output_dir: epath.Path | str | None = None
  name: str | None = None

  def trace_path(self, operation: str) -> epath.Path | None:
    """Trace destination for an operation, or None if no trace is captured.

    Args:
      operation: The operation label (e.g. `save`, `load`) used as the
        Profile-plugin run segment so each operation traces as a distinct run.

    Returns:
      The per-operation run directory under the suite's TensorBoard logdir
      (`<output_dir>/tensorboard/<name>__<operation>`), beneath which
      `jax.profiler` writes its `plugins/profile/<timestamp>/` tree so each
      operation surfaces as a separate Profile-tab run alongside the scalar
      runs. None when `options.enable_trace` is False, when `repeat_index` > 0
      and `options.trace_every_repeat` is False (first-repeat-only is the
      default; traces on every repeat overflow the Trace Viewer), or when
      `output_dir` or `name` is missing.
    """
    if not self.options.enable_trace:
      return None
    if (
        self.repeat_index is not None
        and self.repeat_index > 0
        and not self.options.trace_every_repeat
    ):
      return None
    if self.output_dir is None or self.name is None:
      return None
    return (
        epath.Path(self.output_dir)
        / "tensorboard"
        / f"{self.name}__{operation}"
    )


@dataclasses.dataclass
class TestResult:
  """Output object returned by each test function, containing the results of the test run, including collected metrics."""

  metrics: metric_lib.Metrics
  error: Exception | None = (
      None  # The error raised during the test run, if any.
  )
  path: epath.Path | None = None
  local_path: epath.Path | None = None
  inventory: inventory_lib.CheckpointInventory | None = None

  def is_successful(self) -> bool:
    """Returns whether the test run was successful."""
    return self.error is None


class Benchmark(abc.ABC):
  """An object that encapsulates a single, runnable benchmark test case, including its configuration and metadata."""

  def __init__(
      self,
      test_fn: Callable[[TestContext], TestResult],
      checkpoint_config: configs.CheckpointConfig,
      options: BenchmarkOptions,
      name: str,
      output_dir: str | None = None,
      mesh: jax.sharding.Mesh | None = None,
      local_directory: str | None = None,
  ):
    self.test_fn = test_fn
    self.checkpoint_config = checkpoint_config
    self.options = options
    self.mesh = mesh
    self.name = name
    self.output_dir = output_dir
    self.local_directory = local_directory

  def _build_test_context_summary(self, context: TestContext) -> str:
    """Builds a string summary of the test context."""
    pytree_summary = jax.tree_util.tree_map(
        lambda x: (
            f"shape={x.shape}, dtype={x.dtype}"
            if hasattr(x, "shape") and hasattr(x, "dtype")
            else str(x)
        ),
        context.pytree,
    )
    test_context_str = f"TestContext for '{self.name}':\n"
    test_context_str += f"Path: {context.path}\n"
    test_context_str += f"Options: {context.options}\n"
    test_context_str += f"PyTree Summary:\n{pytree_summary}\n"
    return test_context_str

  def run(self, repeat_index: int | None = None) -> TestResult:
    """Executes the benchmark test case."""
    name = self.name
    if repeat_index is not None:
      name += f"_repeat_{repeat_index}"
    logging.info(
        "[process_id=%s] Setting up test: %s",
        multihost.get_process_index(),
        name,
    )

    benchmark_metrics = metric_lib.Metrics(name=f"{name} Internal")
    with benchmark_metrics.measure("sync_global_processes:benchmark:run"):
      multihost.sync_global_processes("benchmark:run")

    path = directory_setup.setup_test_directory(
        self.name, self.output_dir, repeat_index
    )
    local_path = None
    if self.local_directory is not None:
      local_path = epath.Path(self.local_directory) / name
      if repeat_index is not None:
        local_path = local_path / f"repeat_{repeat_index}"

    with benchmark_metrics.measure(
        "sync_global_processes:benchmark:setup_test_directory"
    ):
      multihost.sync_global_processes("benchmark:setup_test_directory")

    if self.checkpoint_config.path is not None:
      pytree = checkpoint_generation.load_checkpoint(self.checkpoint_config)
    elif self.checkpoint_config.spec is not None:
      pytree = checkpoint_generation.generate_checkpoint(
          self.checkpoint_config, mesh=self.mesh
      )
    else:
      pytree = None

    with benchmark_metrics.measure(
        "sync_global_processes:benchmark:setup_pytree"
    ):
      multihost.sync_global_processes("benchmark:setup_pytree")

    context = TestContext(
        pytree=pytree,
        path=path,
        options=self.options,
        mesh=self.mesh,
        repeat_index=repeat_index,
        local_path=local_path,
        output_dir=self.output_dir,
        name=self.name,
    )

    test_context_summary = self._build_test_context_summary(context)
    logging.info(test_context_summary)

    logging.info(
        "[process_id=%s] Executing test function: %s",
        multihost.get_process_index(),
        name,
    )
    try:
      result = self.test_fn(context)
      result.path = path
      result.local_path = local_path
      # Inventory captures bytes/files the benchmark wrote under context.path
      # (the per-run test dir). Only the primary host scans; the result is
      # suite-level (same across hosts) and parallel walks would race.
      if (
          multihost.get_process_index() == 0
          and result.inventory is None
          and path is not None
      ):
        result.inventory = inventory_lib.scan_checkpoint(path)
    except Exception as e:  # pylint: disable=broad-exception-caught
      # We catch all exceptions to ensure that any error during the test
      # execution is recorded in the TestResult.
      if sys.version_info >= (3, 11):
        e.add_note(
            f"[process_id={multihost.get_process_index()}],"
            f" {test_context_summary[:100]}"
        )
      logging.error(
          "[process_id=%s] Test function '%s' context: %s, raised an"
          " exception: %s",
          multihost.get_process_index(),
          name,
          test_context_summary[:100],
          e,
          exc_info=True,
      )
      result = TestResult(
          metrics=metric_lib.Metrics(),
          error=e,
          path=path,
          local_path=local_path,
      )
    result.metrics.name = name

    result.metrics.report()
    benchmark_metrics.report()

    logging.info(
        "[process_id=%s] Test finished: %s",
        multihost.get_process_index(),
        name,
    )

    return result


class BenchmarksGenerator(abc.ABC):
  """An abstract base class for generating a matrix of benchmark tests.

  Subclasses define a set of configurable options. This class then computes
  the Cartesian product of all provided option values to generate a list
  of distinct `Benchmark` objects.

  Attributes:
    options_class: The dataclass type for the benchmark options, set by the
      @benchmark_options decorator.
  """

  options_class: type[BenchmarkOptions] | None = None

  def __init__(
      self,
      checkpoint_configs: Sequence[configs.CheckpointConfig],
      options: BenchmarkOptions,
      output_dir: str | None = None,
      mesh_configs: Sequence[configs.MeshConfig] | None = None,
      local_directory: str | None = None,
  ):
    """Initializes the generator.

    Args:
        checkpoint_configs: The checkpoint configurations, shared across all
          generated benchmarks.
        options: A dataclass instance defining the parameters to sweep over.
        output_dir: The directory to store the benchmark results in.
        mesh_configs: The mesh configurations, shared across all generated
          benchmarks. If None, no mesh will be created.
        local_directory: The local directory to store the benchmark results in.
    """
    if self.options_class is None:
      raise TypeError(
          f"BenchmarksGenerator subclass {self.__class__.__name__} must be"
          " decorated with @benchmark_options to set the 'options_class'"
          " attribute."
      )
    if not isinstance(options, self.options_class):
      raise TypeError(
          f"Expected options of type {self.options_class.__name__}, but got"
          f" {type(options).__name__}"
      )

    self._checkpoint_configs = checkpoint_configs
    self._mesh_configs = mesh_configs
    self._options = options
    self._output_dir = output_dir
    self._local_directory = local_directory

  @abc.abstractmethod
  def test_fn(self, test_context: TestContext) -> TestResult:
    """A user-defined test function that will be run for every generated benchmark variant."""

  def _get_options_product(self) -> Sequence[BenchmarkOptions]:
    """Computes the Cartesian product of all options in the dataclass."""
    option_value_lists = []
    option_fields = dataclasses.fields(self._options)
    for option_field in option_fields:
      value = getattr(self._options, option_field.name)
      if not isinstance(value, list):
        value = [value]
      option_value_lists.append(value)

    product = itertools.product(*option_value_lists)

    option_instances = []
    option_names = [field.name for field in option_fields]
    for p in product:
      kwargs = dict(zip(option_names, p))
      option_instance = self._options.__class__(**kwargs)
      if option_instance.is_valid():
        option_instances.append(option_instance)
        logging.info(
            "[process_id=%s] Generating valid option combination: %s",
            multihost.get_process_index(),
            option_instance,
        )
      else:
        logging.info(
            "[process_id=%s] Skipping invalid option combination: %s",
            multihost.get_process_index(),
            option_instance,
        )
    return option_instances

  def _get_meshes(
      self, skip_incompatible_mesh_configs: bool
  ) -> Sequence[jax.sharding.Mesh]:
    """Returns a list of meshes for all mesh configs that are compatible with the runtime environment."""
    meshes = []
    for mesh_config in self._mesh_configs:
      try:
        mesh = device_mesh.create_mesh(mesh_config)
        meshes.append(mesh)
      except ValueError as e:
        if not skip_incompatible_mesh_configs:
          raise ValueError(
              f"Failed to create mesh with config {mesh_config}: {e}"
          ) from e
        else:
          logging.warning(
              "Failed to create mesh with config %s: %s", mesh_config, e
          )
    if not meshes:
      raise ValueError("No compatibile meshes found.")
    return meshes

  def generate_benchmark_name(
      self,
      option: BenchmarkOptions,
      mesh: jax.sharding.Mesh,
      checkpoint_config: configs.CheckpointConfig,
  ) -> str:
    """Generates a unique and short benchmark name."""
    class_name = self.__class__.__name__
    long_name = (
        f"{class_name}-{repr(option)}-{repr(mesh)}-{repr(checkpoint_config)}"
    )
    # The concise, one-liner version of the hashing logic
    short_hash = hashlib.md5(long_name.encode()).hexdigest()[:12]
    return f"{class_name}_{short_hash}"

  def generate(
      self, skip_incompatible_mesh_configs: bool = True
  ) -> Sequence[Benchmark]:
    """Generates a list of `Benchmark` objects for all option combinations."""
    benchmarks = []
    option_combinations = self._get_options_product()
    if self._mesh_configs is None:
      meshes = [None]
    else:
      meshes = self._get_meshes(skip_incompatible_mesh_configs)

    for test_config_options, mesh, checkpoint_config in itertools.product(
        option_combinations, meshes, self._checkpoint_configs
    ):
      benchmark_name = self.generate_benchmark_name(
          test_config_options, mesh, checkpoint_config
      )
      benchmark_obj = Benchmark(
          test_fn=self.test_fn,
          name=benchmark_name,
          checkpoint_config=checkpoint_config,
          options=test_config_options,
          output_dir=self._output_dir,
          mesh=mesh,
          local_directory=self._local_directory,
      )
      benchmarks.append(benchmark_obj)

    return benchmarks


class TestSuite:
  """A class to orchestrate running and comparing a list of benchmarks."""

  def __init__(
      self,
      name: str,
      benchmarks_generators: Sequence[BenchmarksGenerator],
      output_dir: str | None = None,
      skip_incompatible_mesh_configs: bool = True,
      num_repeats: int = 1,
      local_directory: str | None = None,
      remove_repeated_dir: bool = False,
  ):
    self._name = name
    self._benchmarks_generators = benchmarks_generators
    self._skip_incompatible_mesh_configs = skip_incompatible_mesh_configs
    self._num_repeats = num_repeats
    self._output_dir = output_dir
    self._local_directory = local_directory
    self._remove_repeated_dir = remove_repeated_dir
    tensorboard_dir = None
    if output_dir:
      tensorboard_dir = epath.Path(output_dir) / "tensorboard"

    self._suite_metrics = metrics_manager.MetricsManager(
        name=name, num_repeats=num_repeats, tensorboard_dir=tensorboard_dir
    )

  def _remove_repeat_directory(self, path: epath.Path | None):
    """Removes the repeat directory for a specific repeat."""
    if path is None:
      return
    if multihost.get_process_index() != 0:
      return

    try:
      logging.info("Removing repeat directory: %s", path)
      path.rmtree()
    except FileNotFoundError:
      logging.warning("Repeat directory %s not found. Skipping removal.", path)


  def run(self) -> Sequence[TestResult]:
    """Runs all benchmarks in the suite sequentially."""
    logging.info(
        "\n%s Running Test Suite: %s %s", "=" * 25, self._name, "=" * 25
    )

    all_results = []
    all_benchmarks: list[Benchmark] = []
    for i, generator in enumerate(self._benchmarks_generators):
      logging.info(
          "\n%s Running Generator %d: %s %s",
          "-" * 15,
          i + 1,
          generator.__class__.__name__,
          "-" * 15,
      )
      generated_benchmarks = generator.generate(
          self._skip_incompatible_mesh_configs
      )
      if not generated_benchmarks:
        logging.warning(
            "Generator %s produced no benchmarks.",
            generator.__class__.__name__,
        )
        continue

      for benchmark in generated_benchmarks:
        all_benchmarks.append(benchmark)
        for i in range(self._num_repeats):
          repeat_index = i if self._num_repeats > 1 else None
          logging.info(
              "\n--- Running test: %s (Repeat %d/%d) ---",
              benchmark.name,
              i + 1,
              self._num_repeats,
          )
          result = benchmark.run(repeat_index=repeat_index)
          all_results.append(result)
          self._suite_metrics.add_result(
              benchmark.name,
              result.metrics,
              benchmark_options=benchmark.options,
              checkpoint_config=benchmark.checkpoint_config,
              error=result.error,
              inventory=result.inventory,
          )
          if self._remove_repeated_dir:
            multihost.sync_global_processes("test_suite:repeat_cleanup")
            self._remove_repeat_directory(result.path)
            self._remove_repeat_directory(result.local_path)

    if not all_results:
      logging.warning("No benchmarks were run for this suite.")

    # Aggregate first (the cross-host gather runs here), then capture/compare
    # baselines from that aggregate.
    self._suite_metrics.generate_report()
    for benchmark in all_benchmarks:
      self._capture_or_compare_baseline(benchmark)
    multihost.sync_global_processes("test_suite:run_end")
    return all_results

  def _capture_or_compare_baseline(self, benchmark: "Benchmark") -> None:
    """Captures and/or compares a baseline for one benchmark, post-aggregation.

    Runs after generate_report, so it reads the cross-host metric aggregate the
    suite gather produced. Only the primary host writes/reads.

    Args:
      benchmark: The benchmark whose options select the capture/compare paths
        and whose name keys the baseline + its aggregated metrics.
    """
    if multihost.get_process_index() != 0:
      return
    options = benchmark.options
    capture = options.baseline_capture_path
    compare = options.baseline_path
    if not capture and not compare:
      return
    metrics = self._baseline_metrics(benchmark.name)
    if capture:
      manifest = self._suite_metrics.run_manifest
      baseline = baseline_lib.Baseline(
          benchmark_name=benchmark.name,
          captured_at=manifest.captured_at,
          captured_at_sha=manifest.git_sha,
          fixture=str(benchmark.checkpoint_config.path or "generated"),
          config=dataclasses.asdict(options),
          metrics=metrics,
          manifest=dataclasses.asdict(manifest),
      )
      out = baseline_lib.BaselineRecorder(capture).write(baseline)
      logging.info("Wrote baseline for %s to %s", benchmark.name, out)
    if compare:
      try:
        report = baseline_lib.BaselineComparer(compare).compare(metrics)
      except FileNotFoundError:
        logging.warning(
            "baseline_path=%s missing for %s; skipping compare.",
            compare,
            benchmark.name,
        )
        return
      lines = [_format_baseline_delta(d) for d in report.deltas]
      logging.info(
          "[baseline] %s vs %s:\n%s",
          benchmark.name,
          report.baseline.captured_at_sha,
          "\n".join(lines) if lines else "  (no overlapping metrics)",
      )

  def _baseline_metrics(
      self, benchmark_name: str
  ) -> dict[str, dict[str, float]]:
    """Cross-host metric aggregate for a benchmark, falling back to rank 0.

    Args:
      benchmark_name: The benchmark to aggregate.

    Returns:
      Metric key -> {stat: value}. Uses the all-host gather when available;
      otherwise wraps rank-0's per-repeat means as a `{"mean": value}` fallback.
    """
    aggregate = self._suite_metrics.cross_host_aggregates(benchmark_name)
    if aggregate is not None:
      return aggregate
    logging.warning(
        "Baseline for %s reflects rank-0 only; enable TensorBoard for the "
        "cross-host aggregate.",
        benchmark_name,
    )
    means = self._suite_metrics.mean_metrics(benchmark_name)
    return {key: {"mean": value} for key, value in means.items()}


def _format_baseline_delta(d: baseline_lib.MetricDelta) -> str:
  """Formats one baseline metric delta as a single report line.

  Args:
    d: The metric delta to render.

  Returns:
    A line with baseline/current values, plus the speedup ratio when defined.
  """
  line = f"  {d.key}: baseline={d.baseline:.4f} current={d.current:.4f}"
  if d.ratio is not None:
    line += f" ratio={d.ratio:.3f}x"
  return line
