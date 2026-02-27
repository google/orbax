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

"""Benchmarks for torch.distributed.checkpoint (DCP)."""

from collections.abc import Generator
from concurrent import futures
import contextlib
import dataclasses
import gc
import io
import os
from typing import Any, Union, cast

from absl import logging
from etils import epath
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib
import safetensors
import torch
import torch.distributed as dist
from torch.distributed import device_mesh
import torch.distributed.checkpoint as dcp
import torch.distributed.tensor


safe_open = safetensors.safe_open
Replicate = torch.distributed.tensor.Replicate
Shard = torch.distributed.tensor.Shard
DTensor = torch.distributed.tensor.DTensor


def _metrics_to_measure(options: "PyTorchCheckpointOptions") -> list[str]:
  """Returns the list of metrics to measure."""
  metrics = ["time", "rss", "io", "network"]
  if options.metric_tracemalloc_enabled:
    metrics.append("tracemalloc")
  return metrics


@dataclasses.dataclass(frozen=True)
class PyTorchCheckpointOptions(benchmarks_core.BenchmarkOptions):
  """Configuration options for benchmarks targeting PyTorch DCP.

  Attributes:
    enable_async_save: Whether to use async checkpointer.
    metric_tracemalloc_enabled: Whether to enable tracemalloc metrics.
    save_thread_count: The number of threads to use for saving the checkpoint.
    save_per_thread_copy_ahead_mb: The number of MB to copy ahead per thread.
  """

  reference_checkpoint_path: str
  metric_tracemalloc_enabled: bool = False
  enable_async_save: bool = False
  save_thread_count: int = 1
  save_per_thread_copy_ahead_mb: int = 10


@benchmarks_core.benchmark_options(PyTorchCheckpointOptions)
class PyTorchCheckpointBenchmark(benchmarks_core.BenchmarksGenerator):
  """A generator for `torch.distributed.checkpoint`."""

  def _build_benchmark_state_dict(
      self,
      path: epath.Path,
      mesh: device_mesh.DeviceMesh,
      rank: int,
      shard_dim: int = 0,
      get_structure: bool = False,
  ) -> dict[str, Any]:
    """Builds the state dict structure for a single save/restore cycle."""
    state_dict = {}
    world_size = mesh.size()
    for sf_file in path.iterdir():
      if sf_file.is_file() and sf_file.name.endswith(".safetensors"):
        logging.info("[Rank %d] Found safetensors file: %s", rank, sf_file.name)
      else:
        continue
      with safe_open(str(sf_file), framework="pt") as f:
        for key in f.keys():
          global_shape = f.get_slice(key).get_shape()
          local_shape = list(global_shape)
          if local_shape[shard_dim] % world_size != 0:
            placements = [Replicate()]
            start, end = 0, global_shape[shard_dim]
          else:
            placements = [Shard(shard_dim)]
            local_shape[shard_dim] = local_shape[shard_dim] // world_size
            start = rank * local_shape[shard_dim]
            end = start + local_shape[shard_dim]
          gpu_tensor = torch.empty(
              local_shape, dtype=torch.bfloat16, device="cuda"
          )
          if not get_structure:
            local_data = f.get_slice(key)[start:end]
            gpu_tensor.copy_(local_data)
          state_dict[key] = DTensor.from_local(gpu_tensor, mesh, placements)

    return state_dict

  def test_fn(
      self, context: benchmarks_core.TestContext
  ) -> benchmarks_core.TestResult:
    """The core test logic for a single save/restore cycle."""
    options = context.options
    assert isinstance(options, PyTorchCheckpointOptions)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    logging.info(
        "[Rank %d] Attempt to set device to %d and get %s",
        rank,
        local_rank,
        device,
    )

    # Log memory at the start
    if torch.cuda.is_available():
      logging.info(
          "[Rank %d] [START OF TEST_FN] CUDA Memory Allocated: %.2f GB",
          rank,
          torch.cuda.memory_allocated() / 1e9,
      )
      logging.info(
          "[Rank %d] [START OF TEST_FN] CUDA Memory Reserved: %.2f GB",
          rank,
          torch.cuda.memory_reserved() / 1e9,
      )
      logging.info(
          "[Rank %d] [START OF TEST_FN] CUDA Memory Summary:\n%s",
          rank,
          torch.cuda.memory_summary(),
      )

    mesh = device_mesh.init_device_mesh("cuda", (world_size,))
    logging.info("[Rank %d] Initialized device mesh: %s", rank, mesh)

    safetensor_path = epath.Path(options.reference_checkpoint_path)
    state_dict = self._build_benchmark_state_dict(
        safetensor_path, mesh, rank, shard_dim=0
    )
    logging.info("[Rank %d] Built state dict from safetensors.", rank)

    metrics = metric_lib.Metrics()
    save_path = context.path / "pytorch_dtensor_ckpt"
    save_path_str = str(save_path)
    if rank == 0:
      epath.Path(save_path_str).mkdir(parents=True, exist_ok=True)
    logging.info(
        "[Rank %d] Saving sharded (DTensor) checkpoint to: %s",
        rank,
        save_path_str,
    )
    dist.barrier()

    @contextlib.contextmanager
    def create_stream(
        path: Union[str, os.PathLike[str]], mode: str
    ) -> Generator[io.IOBase, None, None]:
      new_path = epath.Path(path)
      with new_path.open(mode) as stream:
        yield cast(io.IOBase, stream)

    metrics_to_measure = _metrics_to_measure(options)
    writer = dcp.FileSystemWriter(save_path_str, single_file_per_rank=True)
    writer.thread_count = options.save_thread_count
    writer.per_thread_copy_ahead = (
        options.save_per_thread_copy_ahead_mb * 1024 * 1024
    )
    writer.fs.create_stream = create_stream
    writer.fs.mkdir = lambda path: epath.Path(path).mkdir(
        parents=True, exist_ok=True
    )
    writer.fs.rename = lambda src, dst: epath.Path(src).rename(epath.Path(dst))
    dist.barrier()

    response = None
    with metrics.measure("save", metrics_to_measure):
      logging.info("[Rank %d] Start saving checkpoint.", rank)
      if options.enable_async_save:
        future = dcp.async_save(
            state_dict,
            storage_writer=writer,
            async_checkpointer_type=dcp.state_dict_saver.AsyncCheckpointerType.THREAD,
        )
        try:
          response = cast(futures.Future[Any], future)
          logging.info("[Rank %d] Blocking part of save finished.", rank)
        except TypeError:
          logging.info("[Rank %d] The future is not a Future.", rank)
          raise
      else:
        dcp.save(state_dict, storage_writer=writer)
        logging.info("[Rank %d] Save complete.", rank)

    if options.enable_async_save and response is not None:
      with metrics.measure("wait_until_finished", metrics_to_measure):
        dist.barrier()
        response.result()
        logging.info("[Rank %d] Wait until async save finished.", rank)

    # Clear the cache to avoid OOMs when loading the checkpoint.
    del state_dict
    torch.cuda.empty_cache()
    dist.barrier()

    state_dict = self._build_benchmark_state_dict(
        safetensor_path, mesh, rank, shard_dim=0, get_structure=True
    )

    reader = dcp.FileSystemReader(save_path_str)
    reader.fs.create_stream = create_stream
    dist.barrier()
    with metrics.measure("restore", metrics_to_measure):
      dcp.load(state_dict, storage_reader=reader)
    logging.info("[Rank %d] Restore complete.", rank)

    dist.barrier()
    # Remove the checkpoint to avoid too many files in the directory.
    if rank == 0:
      epath.Path(save_path_str).rmtree()
    # Clear the cache to avoid OOMs when loading the checkpoint.
    del state_dict
    del reader
    del writer
    del response
    del mesh
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    dist.barrier()
    return benchmarks_core.TestResult(metrics=metrics)
