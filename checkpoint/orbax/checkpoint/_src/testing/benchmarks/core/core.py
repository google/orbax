# Copyright 2025 The Orbax Authors.
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
from collections.abc import MutableMapping, Sequence
import contextlib
import dataclasses
import itertools
import time
from typing import Any, Callable

from absl import logging
from etils import epath
import jax
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.testing.benchmarks.core import checkpoint_generation
from orbax.checkpoint._src.testing.benchmarks.core import configs
from orbax.checkpoint._src.testing.benchmarks.core import device_mesh
from orbax.checkpoint._src.testing.benchmarks.core import directory_setup


@dataclasses.dataclass(frozen=True)
class BenchmarkOptions:
  """Base class for benchmark generator options."""


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
class Metrics:
  """A simple dataclass to store and report test metrics.

  Attributes:
    timings: A dictionary mapping metric names to their duration in seconds.
    name: The name of the metric group.
  """

  timings: MutableMapping[str, float] = dataclasses.field(default_factory=dict)
  name: str = ""

  @contextlib.contextmanager
  def time(self, name: str):
    """A context manager to time a block of code and record it."""
    logging.info(
        "[process_id=%s] Starting metric: '%s'...",
        multihost.process_index(),
        name,
    )
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    duration = end_time - start_time
    self.timings[name] = duration
    logging.info(
        "[process_id=%s] Finished metric: '%s' took %.4f s.",
        multihost.process_index(),
        name,
        duration,
    )

  def report(self):
    """Logs a formatted report of all collected metrics."""
    report_lines = []
    report_lines.append(
        f"---[process_id={multihost.process_index()}] {self.name} Metrics"
        " Report ---"
    )
    if not self.timings:
      report_lines.append(
          f"[process_id={multihost.process_index()}] No metrics recorded."
      )
    else:
      for name, duration in self.timings.items():
        report_lines.append(f"{name}: {duration:.4f} seconds")
    report_lines.append("----------------------")
    logging.info("\n".join(report_lines))


@dataclasses.dataclass
class TestContext:
  """Input object passed to each test function, providing pre-configured components for the test run.

  Attributes:
    pytree: The generated or loaded checkpoint data.
    path: The test directory path.
    options: The specific BenchmarkOptions for this test variant.
    mesh: The mesh used for sharding the checkpoint data.
  """

  pytree: Any
  path: epath.Path
  options: BenchmarkOptions  # The specific options for this test variant.
  mesh: jax.sharding.Mesh | None = None


@dataclasses.dataclass
class TestResult:
  """Output object returned by each test function, containing the results of the test run, including collected metrics."""

  metrics: Metrics


class Benchmark(abc.ABC):
  """An object that encapsulates a single, runnable benchmark test case, including its configuration and metadata."""

  def __init__(
      self,
      test_fn: Callable[[TestContext], TestResult],
      checkpoint_config: configs.CheckpointConfig,
      options: BenchmarkOptions,
      name: str,
      output_dir: str | None = None,
      mesh_config: configs.MeshConfig | None = None,
  ):
    self.test_fn = test_fn
    self.checkpoint_config = checkpoint_config
    self.options = options
    self.mesh_config = mesh_config
    self.name = name
    self.output_dir = output_dir

  def run(self) -> TestResult:
    """Executes the benchmark test case."""
    logging.info(
        "[process_id=%s] Setting up test: %s",
        multihost.process_index(),
        self.name,
    )

    benchmark_metrics = Metrics(name=f"{self.name} Internal")
    with benchmark_metrics.time("sync_global_processes:benchmark:run"):
      multihost.sync_global_processes("benchmark:run")

    path = directory_setup.setup_test_directory(self.name, self.output_dir)

    with benchmark_metrics.time(
        "sync_global_processes:benchmark:setup_test_directory"
    ):
      multihost.sync_global_processes("benchmark:setup_test_directory")

    if self.mesh_config is not None:
      mesh = device_mesh.create_mesh(self.mesh_config)
    else:
      mesh = None

    if self.checkpoint_config.path is None:
      data = checkpoint_generation.generate_checkpoint(
          self.checkpoint_config, mesh=mesh
      )
    else:
      data = checkpoint_generation.load_checkpoint(self.checkpoint_config.path)

    with benchmark_metrics.time("sync_global_processes:benchmark:setup_pytree"):
      multihost.sync_global_processes("benchmark:setup_pytree")

    context = TestContext(
        pytree=data, path=path, options=self.options, mesh=mesh
    )

    pytree_summary = jax.tree_util.tree_map(
        lambda x: (
            f"shape={x.shape}, dtype={x.dtype}"
            if hasattr(x, "shape") and hasattr(x, "dtype")
            else str(x)
        ),
        context.pytree,
    )
    logging.info(
        "--- TestContext for '%s' ---\n"
        "Path: %s\n"
        "Options: %s\n"
        "PyTree Summary:\n%s\n"
        "----------------------------------",
        self.name,
        context.path,
        context.options,
        pytree_summary,
    )

    logging.info(
        "[process_id=%s] Executing test function: %s",
        multihost.process_index(),
        self.name,
    )
    result = self.test_fn(context)
    result.metrics.name = self.name

    result.metrics.report()
    benchmark_metrics.report()

    logging.info(
        "[process_id=%s] Test finished: %s",
        multihost.process_index(),
        self.name,
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
      checkpoint_config: configs.CheckpointConfig,
      options: BenchmarkOptions,
      output_dir: str | None = None,
      mesh_config: configs.MeshConfig | None = None,
  ):
    """Initializes the generator.

    Args:
        checkpoint_config: The checkpoint configuration, shared across all
          generated benchmarks.
        options: A dataclass instance defining the parameters to sweep over.
        output_dir: The directory to store the benchmark results in.
        mesh_config: The mesh configuration, shared across all generated
          benchmarks. If None, no mesh will be created.
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

    self._checkpoint_config = checkpoint_config
    self._mesh_config = mesh_config
    self._options = options
    self._output_dir = output_dir

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
      option_instances.append(type(self._options)(**kwargs))
    return option_instances

  def generate(self) -> Sequence[Benchmark]:
    """Generates a list of `Benchmark` objects for all option combinations."""
    benchmarks = []
    option_combinations = self._get_options_product()

    for test_config_options in option_combinations:
      name_parts = [
          f"{field.name}_{getattr(test_config_options, field.name)}"
          for field in dataclasses.fields(test_config_options)
      ]
      benchmark_name = f"{self.__class__.__name__}_{'_'.join(name_parts)}"

      benchmark_obj = Benchmark(
          test_fn=self.test_fn,
          name=benchmark_name,
          checkpoint_config=self._checkpoint_config,
          options=test_config_options,
          output_dir=self._output_dir,
          mesh_config=self._mesh_config,
      )
      benchmarks.append(benchmark_obj)

    return benchmarks


class TestSuite:
  """A class to orchestrate running and comparing a list of benchmarks."""

  def __init__(
      self, name: str, benchmarks_generators: Sequence[BenchmarksGenerator]
  ):
    self._name = name
    self._benchmarks_generators = benchmarks_generators

  def run(self):
    """Runs all benchmarks in the suite sequentially."""
    logging.info(
        "\n%s Running Test Suite: %s %s", "=" * 25, self._name, "=" * 25
    )

    has_run_benchmarks = False
    for i, generator in enumerate(self._benchmarks_generators):
      logging.info(
          "\n%s Running Generator %d: %s %s",
          "-" * 15,
          i + 1,
          generator.__class__.__name__,
          "-" * 15,
      )
      generated_benchmarks = generator.generate()
      if not generated_benchmarks:
        logging.warning(
            "Generator %s produced no benchmarks.", generator.__class__.__name__
        )
        continue

      has_run_benchmarks = True
      for benchmark in generated_benchmarks:
        logging.info("\n--- Running test: %s ---", benchmark.name)
        benchmark.run()

    if not has_run_benchmarks:
      logging.warning("No benchmarks were run for this suite.")

    logging.info(
        "\n%s Finished Test Suite: %s %s", "=" * 25, self._name, "=" * 25
    )
