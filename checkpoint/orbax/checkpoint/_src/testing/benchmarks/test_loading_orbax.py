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

"""Tests loading Orbax checkpoints from CNS."""

from concurrent import futures
import time
from typing import Sequence

from absl import app
from etils import epath
from google.cloud import storage
import zstandard as zstd


ThreadPoolExecutor = futures.ThreadPoolExecutor
CNS_PATH = "s/ig-d/home/gemax-prod-team/llama-checkpoint/llama-3.1-70B-checkpoints/0/items/"
GCS_PATH = "gs://safetensor-kimi-central/test_model_orbax/llama-3.1-70B-checkpoints/0/items/items"

# Initialize the client globally so workers share the connection pool.
# This prevents opening a brand new TCP connection for every single file.
_GCS_CLIENT = None


def get_gcs_client():
  global _GCS_CLIENT
  if _GCS_CLIENT is None:
    _GCS_CLIENT = storage.Client()
  return _GCS_CLIENT


def read_file_gcs(p: epath.Path):
  """Reads a file from GCS using the native cloud storage client."""
  start = time.time()

  # 1. Parse the gs:// path to get the bucket and blob name
  path_str = str(p)
  if not path_str.startswith("/big" + "store/"):
    raise ValueError(f"Expected a /bistore/ path, got {path_str}")

  # Strip "gs://" and split
  path_without_scheme = path_str[10:]
  bucket_name, blob_name = path_without_scheme.split("/", 1)

  # 2. Download directly to memory using the GCS client
  client = get_gcs_client()
  bucket = client.bucket(bucket_name)
  blob = bucket.blob(blob_name)

  # download_as_bytes() releases the GIL during network transit
  data = blob.download_as_bytes()

  # 3. Decompress (Same logic as before)
  zstd_magic = b"\x28\xb5\x2f\xfd"
  start_offset = data.find(zstd_magic)

  if start_offset == -1:
    raise ValueError("Could not find a valid Zstd magic number.")

  compressed_payload = data[start_offset:]

  dctx = zstd.ZstdDecompressor()
  dctx.decompress(compressed_payload, max_output_size=int(20e9))

  end = time.time()
  mb = len(data) / 1e6
  return p.name, mb, end - start


def read_file_cns(p: epath.Path):
  """Reads and decompresses a single Orbax checkpoint file.

  Args:
    p: The epath.Path to the file to read.

  Returns:
    A tuple containing the file name, size in MB, and the time taken to read
    and decompress.

  Raises:
    ValueError: If a valid Zstd magic number is not found in the file.
  """
  start = time.time()
  data = p.read_bytes()

  # Zstandard frames always start with the magic number: 0xFD2FB528
  # We need to find where this starts in the raw OCDBT file
  zstd_magic = b"\x28\xb5\x2f\xfd"
  start_offset = data.find(zstd_magic)

  if start_offset == -1:
    raise ValueError(
        "Could not find a valid Zstd magic number in the file. "
        "It might be uncompressed or use a different algorithm."
    )

  # print(f"Found Zstd payload at offset: {start_offset}")

  # Slice the data from the magic number to the end
  compressed_payload = data[start_offset:]

  dctx = zstd.ZstdDecompressor()
  dctx.decompress(compressed_payload, max_output_size=int(20e9))

  end = time.time()
  mb = len(data) / 1e6
  return p.name, mb, end - start


def read_concurrently(workers):
  """Reads and decompresses Orbax checkpoint files concurrently.

  Args:
    workers: The number of worker threads to use for concurrent reading.
  """
  global_start = time.time()

  filepaths_cns = list(epath.Path(CNS_PATH + "/cn").glob("ocdbt.process*/d/*"))
  filepaths_gcs = list(epath.Path(GCS_PATH).glob("ocdbt.process*/d/*"))
  print(f"Total files in CNS: {len(filepaths_cns)}")
  print(f"Total files in GCS: {len(filepaths_gcs)}")
  with ThreadPoolExecutor(max_workers=workers) as pool:
    # ---------------------------------------------------------
    # 1. READ FROM CNS
    # ---------------------------------------------------------
    print("\n--- Reading from CNS ---")
    cns_start = time.time()
    cns_total_mb = 0
    results_cns = pool.map(read_file_cns, filepaths_cns)
    for r in results_cns:
      name, mb, t = r
      cns_total_mb += mb
      print(f"CNS: Opened {name} ({mb:.2f} MB) in {t:.2f} sec")
    cns_total_time = time.time() - cns_start
    print(f"CNS Finished: {cns_total_mb:.2f} MB in {cns_total_time:.2f} sec")
    print(f"CNS Throughput: {cns_total_mb / cns_total_time:.2f} MB/sec")

    # ---------------------------------------------------------
    # 2. READ FROM GCS
    # ---------------------------------------------------------
    print("\n--- Reading from GCS ---")
    gcs_start = time.time()
    gcs_total_mb = 0
    # The workers are now free and will immediately start on this:
    results_gcs = pool.map(read_file_gcs, filepaths_gcs)
    for r in results_gcs:
      name, mb, t = r
      gcs_total_mb += mb
      print(f"GCS: Opened {name} ({mb:.2f} MB) in {t:.2f} sec")
    gcs_total_time = time.time() - gcs_start
    print(f"GCS Finished: {gcs_total_mb:.2f} MB in {gcs_total_time:.2f} sec")
    print(f"GCS Throughput: {gcs_total_mb / gcs_total_time:.2f} MB/sec")

  global_end = time.time()
  print(
      f"\nTotal benchmark time for {workers} workers:"
      f" {global_end - global_start:.2f} sec"
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  read_concurrently(4)
  read_concurrently(16)
  read_concurrently(32)


if __name__ == "__main__":
  app.run(main)
