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

# GCS Connector for PyTorch - A thin wrapper around vanilla PyTorch DCP.
try:
  # pylint: disable=g-import-not-at-top
  from dataflux_pytorch.lightning.gcs_filesystem import GCSDistributedReader
  from dataflux_pytorch.lightning.gcs_filesystem import GCSDistributedWriter
  from dataflux_pytorch.lightning.path_utils import parse_gcs_path
  from dataflux_pytorch.multipart_upload.multipart import upload_chunks_concurrently_from_bytesio as upload
  # pylint: enable=g-import-not-at-top
except ImportError:
  GCSDistributedReader = None
  GCSDistributedWriter = None
  parse_gcs_path = None
  upload = None


# CRITICAL: Prevent TensorFlow (used by CLU/TensorBoard) from stealing GPU VRAM.
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

safe_open = safetensors.safe_open
Replicate = torch.distributed.tensor.Replicate
Shard = torch.distributed.tensor.Shard
DTensor = torch.distributed.tensor.DTensor
DefaultSavePlanner = dcp.DefaultSavePlanner
BlockingAsyncStager = dcp.staging.BlockingAsyncStager
try:
  # PyTorch 2.6.0+ usually requires explicit import of the internal module
  # pylint: disable=g-import-not-at-top
  import torch.distributed.checkpoint._fsspec_filesystem as dcp_fsspec_internal
  # pylint: enable=g-import-not-at-top
  FsspecReader = dcp_fsspec_internal.FsspecReader
  FsspecWriter = dcp_fsspec_internal.FsspecWriter
except (ImportError, AttributeError):
  try:
    # pylint: disable=g-import-not-at-top
    FsspecReadeer = dcp._fsspec_filesystem.FsspecReader  # pylint: disable=protected-access
    FsspecWriter = dcp._fsspec_filesystem.FsspecWriter  # pylint: disable=protected-access
    # pylint: enable=g-import-not-at-top
  except ImportError:
    FsspecReader = None  # pylint: disable=invalid-name
    FsspecWriter = None  # pylint: disable=invalid-name


def _metrics_to_measure(options: "PyTorchCheckpointOptions") -> list[str]:
  """Returns the list of metrics to measure."""
  metrics = ["time", "rss", "io"]
  if options.metric_tracemalloc_enabled:
    metrics.append("tracemalloc")
  return metrics


@dataclasses.dataclass(frozen=True)
class PyTorchCheckpointOptions(benchmarks_core.BenchmarkOptions):
  """Configuration options for benchmarks targeting PyTorch DCP."""

  reference_checkpoint_path: str
  metric_tracemalloc_enabled: bool = False
  enable_async_save: bool = False
  save_thread_count: int = 10
  save_per_thread_copy_ahead_mb: int = 100
  cache_staged_state_dict: bool = False
  enable_gcs_connector: bool = False
  single_file_per_rank: bool = False


class ProtectedBytesIO(io.BytesIO):
  """A BytesIO that ignores close calls until explicitly allowed."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._should_allow_close = False

  def close(self):
    if self._should_allow_close:
      super().close()

  def force_close(self):
    self._should_allow_close = True
    self.close()


@contextlib.contextmanager
def buffered_fsspec_create_stream(
    path: Union[str, os.PathLike[str]], mode: str
) -> Generator[io.IOBase, None, None]:
  """Buffered create_stream to support torch.save on non-POSIX filesystems."""
  if mode == "wb":
    stream = ProtectedBytesIO()
    try:
      yield cast(io.IOBase, stream)
      stream.seek(0)
      # Write the full buffer to GCS in one go
      with epath.Path(path).open("wb") as f:
        f.write(stream.getvalue())
    finally:
      stream.force_close()
  else:
    # For reading, we can stream directly or buffer.
    # Buffering is safer for some GCS versions.
    stream = io.BytesIO()
    with epath.Path(path).open("rb") as f:
      stream.write(f.read())
    stream.seek(0)
    yield cast(io.IOBase, stream)


if GCSDistributedWriter is not None and GCSDistributedReader:

  class StreamingGCSWriter(GCSDistributedWriter):
    """A DCP writer that also acts as a Stager for PyTorch using DataFlux."""

    def __init__(
        self,
        path: str,
        project_name: str,
        cache_staged_state_dict: bool = True,
        **kwargs,
    ):
      super().__init__(path, project_name=project_name, **kwargs)
      self._stager = BlockingAsyncStager(
          cache_staged_state_dict=cache_staged_state_dict
      )
      self.fs.create_stream = self._fixed_create_stream

    @contextlib.contextmanager
    def _fixed_create_stream(
        self, path: str, mode: str
    ) -> Generator[io.IOBase, None, None]:
      """Ensure upload is finished before buffer close."""
      bucket, gcs_path = parse_gcs_path(path)
      blob = self.fs.storage_client.bucket(bucket).blob(gcs_path)

      if mode == "wb":
        stream = ProtectedBytesIO()
        try:
          yield stream
          stream.seek(0)
          upload(stream, blob)
        finally:
          stream.force_close()
      else:
        stream = io.BytesIO()
        blob.download_to_file(stream)
        stream.seek(0)
        yield stream

    def stage(self, *args, **kwargs):
      return self._stager.stage(*args, **kwargs)

    def synchronize_staging(self):
      return self._stager.synchronize_staging()

else:
  StreamingGCSWriter = None  # pylint: disable=invalid-name

if BlockingAsyncStager is not None and StreamingGCSWriter is not None:
  BlockingAsyncStager.register(StreamingGCSWriter)


@benchmarks_core.benchmark_options(PyTorchCheckpointOptions)
class PyTorchCheckpointBenchmark(benchmarks_core.BenchmarksGenerator):
  """A generator for `torch.distributed.checkpoint`."""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._cached_mesh = None

  def _build_benchmark_state_dict(
      self,
      path: epath.Path,
      mesh: device_mesh.DeviceMesh,
      rank: int,
      device: torch.device,
      shard_dim: int = 0,
      get_structure: bool = False,
  ) -> dict[str, Any]:
    """Builds the state dict structure for a single save/restore cycle."""
    state_dict = {}
    world_size = mesh.size()
    reference_epath = epath.Path(path)

    with torch.no_grad():
      for sf_file in reference_epath.iterdir():
        if not (sf_file.is_file() and sf_file.name.endswith(".safetensors")):
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

            if get_structure:
              t = torch.empty(local_shape, dtype=torch.bfloat16, device="meta")
              state_dict[key] = DTensor.from_local(t, mesh, placements)
            else:
              t = torch.empty(local_shape, dtype=torch.bfloat16, device=device)
              local_data = f.get_slice(key)[start:end]
              t.copy_(local_data)
              del local_data
              state_dict[key] = DTensor.from_local(t, mesh, placements)

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

    # 1. Initialize Mesh (Cached)
    if self._cached_mesh is None:
      self._cached_mesh = device_mesh.init_device_mesh("cuda", (world_size,))

    safetensor_path = epath.Path(options.reference_checkpoint_path)

    # RE-LOAD state dict every time to ensure plan freshness and clear OOM risks
    logging.info("[Rank %d] Loading model into VRAM...", rank)
    state_dict = self._build_benchmark_state_dict(
        safetensor_path, self._cached_mesh, rank, device, shard_dim=0
    )

    metrics = metric_lib.Metrics()
    save_path = context.path / "pytorch_dtensor_ckpt"
    save_path_str = str(save_path)
    if rank == 0:
      logging.info("[Rank 0] Direct GCS Benchmark Path: %s", save_path_str)
      epath.Path(save_path_str).mkdir(parents=True, exist_ok=True)
    dist.barrier()

    metrics_to_measure = _metrics_to_measure(options)
    planner = DefaultSavePlanner(enable_plan_caching=True)

    # New writer every time to ensure fresh buffers
    if not options.enable_gcs_connector or StreamingGCSWriter is None:
      logging.info(
          "[Rank %d] Using standard FileSystemWriter (fsspec + shared buffer).",
          rank,
      )
      writer = FsspecWriter(
          save_path_str,
          single_file_per_rank=options.single_file_per_rank,
          thread_count=options.save_thread_count,
          per_thread_copy_ahead=options.save_per_thread_copy_ahead_mb
          * 1024
          * 1024,
      )
      writer.fs.create_stream = buffered_fsspec_create_stream
    else:
      bucket, _ = parse_gcs_path(save_path_str)
      writer = StreamingGCSWriter(
          save_path_str,
          project_name=bucket,
          cache_staged_state_dict=options.cache_staged_state_dict,
          single_file_per_rank=options.single_file_per_rank,
          thread_count=options.save_thread_count,
      )
      writer.per_thread_copy_ahead = (
          options.save_per_thread_copy_ahead_mb * 1024 * 1024
      )

    dist.barrier()

    response = None
    torch.cuda.synchronize()
    with metrics.measure("save", metrics_to_measure):
      if options.enable_async_save:
        future = dcp.async_save(
            state_dict,
            storage_writer=writer,
            planner=planner,
            async_checkpointer_type=dcp.state_dict_saver.AsyncCheckpointerType.THREAD,
        )
        response = cast(futures.Future[Any], future)
      else:
        dcp.save(state_dict, storage_writer=writer, planner=planner)

    if options.enable_async_save and response is not None:
      with metrics.measure("wait_until_finished", metrics_to_measure):
        dist.barrier()
        response.result()

    # Clear VRAM before restoration
    del state_dict
    del writer
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    # 2. Prepare structure for restoration (Fresh Meta-device)
    state_dict = self._build_benchmark_state_dict(
        safetensor_path,
        self._cached_mesh,
        rank,
        device,
        shard_dim=0,
        get_structure=True,
    )

    if not options.enable_gcs_connector or GCSDistributedReader is None:
      logging.info(
          "[Rank %d] Using standard FileSystemReader (fsspec + shared buffer).",
          rank,
      )
      reader = FsspecReader(save_path_str)
      reader.fs.create_stream = buffered_fsspec_create_stream
    else:
      reader = GCSDistributedReader(
          save_path_str, project_name="orbax-checkpoint"
      )

    dist.barrier()
    with metrics.measure("restore", metrics_to_measure):
      dcp.load(state_dict, storage_reader=reader)

    dist.barrier()
    # DO NOT delete from GCS inside the loop to avoid MetadataIndex KeyError

    # Final cleanup
    del state_dict
    del reader
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return benchmarks_core.TestResult(metrics=metrics)
