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
from collections.abc import Sequence
import dataclasses
import hashlib
import itertools
from typing import Any, Callable

from absl import logging
from etils import epath
import jax
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.testing.benchmarks.core import checkpoint_generation
from orbax.checkpoint._src.testing.benchmarks.core import configs
from orbax.checkpoint._src.testing.benchmarks.core import device_mesh
from orbax.checkpoint._src.testing.benchmarks.core import directory_setup
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib


@dataclasses.dataclass(frozen=True)
class BenchmarkOptions:
  """Base class for benchmark generator options."""

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
  """

  pytree: Any | None
  path: epath.Path
  options: BenchmarkOptions  # The specific options for this test variant.
  mesh: jax.sharding.Mesh | None = None
  repeat_index: int | None = None
  local_path: epath.Path | None = None


@dataclasses.dataclass
class TestResult:
  """Output object returned by each test function, containing the results of the test run, including collected metrics."""

  metrics: metric_lib.Metrics
  error: Exception | None = (
      None  # The error raised during the test run, if any.
  )

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
        multihost.process_index(),
        name,
    )

    benchmark_metrics = metric_lib.Metrics(name=f"{name} Internal")
    with benchmark_metrics.measure("sync_global_processes:benchmark:run"):
      multihost.sync_global_processes("benchmark:run")

    path = directory_setup.setup_test_directory(
        self.name, self.output_dir, repeat_index
    )

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
        local_path=self.local_directory,
    )

    test_context_summary = self._build_test_context_summary(context)
    logging.info(test_context_summary)

    logging.info(
        "[process_id=%s] Executing test function: %s",
        multihost.process_index(),
        name,
    )
    try:
      result = self.test_fn(context)
    except Exception as e:  # pylint: disable=broad-exception-caught
      # We catch all exceptions to ensure that any error during the test
      # execution is recorded in the TestResult.
      e.add_note(
          f"[process_id={multihost.process_index()}],"
          f" {test_context_summary[:100]}"
      )
      logging.error(
          "[process_id=%s] Test function '%s' context: %s, raised an"
          " exception: %s",
          multihost.process_index(),
          name,
          test_context_summary[:100],
          e,
          exc_info=True,
      )
      result = TestResult(metrics=metric_lib.Metrics(), error=e)
    result.metrics.name = name

    result.metrics.report()
    benchmark_metrics.report()

    logging.info(
        "[process_id=%s] Test finished: %s",
        multihost.process_index(),
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
            multihost.process_index(),
            option_instance,
        )
      else:
        logging.info(
            "[process_id=%s] Skipping invalid option combination: %s",
            multihost.process_index(),
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
  ):
    self._name = name
    self._benchmarks_generators = benchmarks_generators
    self._skip_incompatible_mesh_configs = skip_incompatible_mesh_configs
    self._num_repeats = num_repeats
    self._output_dir = output_dir
    self._local_directory = local_directory
    tensorboard_dir = None
    if output_dir:
      tensorboard_dir = epath.Path(output_dir) / "tensorboard"

    self._suite_metrics = metric_lib.MetricsManager(
        name=name, num_repeats=num_repeats, tensorboard_dir=tensorboard_dir
    )

  def run(self) -> Sequence[TestResult]:
    """Runs all benchmarks in the suite sequentially."""
    logging.info(
        "\n%s Running Test Suite: %s %s", "=" * 25, self._name, "=" * 25
    )

    all_results = []
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
          )

    if not all_results:
      logging.warning("No benchmarks were run for this suite.")

    self._suite_metrics.generate_report()
    multihost.sync_global_processes("test_suite:run_end")
    return all_results
