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

"""MTC Checkpoint Converter.

Converts multi-node checkpoints saved by Multi-Tier Checkpointing (MTC) on GCS
or local storage into standard, unified Orbax checkpoints loadable by
standard CheckpointManager, evaluation pipelines, or Hugging Face converters.
"""

from collections.abc import Sequence
from concurrent import futures
import dataclasses
import functools
import json
import os
import re
import shutil
from typing import Any

from absl import logging
from etils import epath
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.path import gcs_utils
from orbax.checkpoint._src.serialization import ocdbt_utils
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils

try:
  # pylint: disable=g-import-not-at-top
  from google.cloud import storage
except ImportError:
  storage = None
  # pylint: enable=g-import-not-at-top

# Regex to parse MTC metadata filenames:
# Format: <job_name>-s<step>-n<node_rank>-w<worker_rank>.meta
# e.g., maxtext-s100-n0-w0.meta
_META_FILE_PATTERN = re.compile(
    r"^(?P<job_name>.+)-s(?P<step>\d+)-n(?P<node_rank>\d+)-w(?P<worker_rank>\d+)\.meta$"
)

# Regex to detect timestamp-based backup directories:
# e.g., 2026-09-09_18-00
_TIMESTAMP_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}$")

_COMMIT_SUCCESS_FILE = "commit_success.txt"
_CHECKPOINT_METADATA_FILE = "_CHECKPOINT_METADATA"
_PROCESS_METADATA_DIR = "process_metadata"
_OCDBT_MANIFEST_FILE = "manifest.ocdbt"
_METADATA_FILE = "_METADATA"
_SHARDING_FILE = "_sharding"


@dataclasses.dataclass(frozen=True)
class MtcMetaFile:
  """Represents a parsed MTC metadata (.meta) file.

  Attributes:
    path: Path to the .meta file.
    job_name: Job name prefix.
    step: Step number.
    node_rank: Physical node index (0..N-1).
    worker_rank: Worker rank on the node (typically 0).
    data_hash: Content hash pointing to D<hash>.data directory.
  """

  path: epath.Path
  job_name: str
  step: int
  node_rank: int
  worker_rank: int
  data_hash: str


def parse_meta_filename(filename: str) -> tuple[str, int, int, int] | None:
  """Parses an MTC metadata filename into its components."""
  m = _META_FILE_PATTERN.match(filename)
  if not m:
    return None
  return (
      m.group("job_name"),
      int(m.group("step")),
      int(m.group("node_rank")),
      int(m.group("worker_rank")),
  )


def find_data_dir(backup_dir: epath.Path, data_hash: str) -> epath.Path:
  """Finds the .data directory corresponding to the given hash."""
  candidates = [
      backup_dir / f"{data_hash}.data",
      backup_dir / f"{data_hash}",
  ]
  if not data_hash.startswith("D"):
    candidates.append(backup_dir / f"D{data_hash}.data")
  else:
    candidates.append(backup_dir / f"{data_hash[1:]}.data")

  for candidate in candidates:
    if candidate.exists() and candidate.is_dir():
      return candidate

  raise FileNotFoundError(
      f"Could not find data directory for hash '{data_hash}' in '{backup_dir}'."
      f" Looked for: {[c.as_posix() for c in candidates]}"
  )


def discover_mtc_meta_files(
    backup_dir: epath.Path,
) -> dict[int, list[MtcMetaFile]]:
  """Discovers and parses all .meta files in backup_dir grouped by step."""
  steps_to_meta: dict[int, list[MtcMetaFile]] = {}

  for entry in backup_dir.iterdir():
    if not entry.name.endswith(".meta"):
      continue
    parsed = parse_meta_filename(entry.name)
    if parsed is None:
      continue
    job_name, step, node_rank, worker_rank = parsed
    data_hash = entry.read_text().strip()
    meta_obj = MtcMetaFile(
        path=entry,
        job_name=job_name,
        step=step,
        node_rank=node_rank,
        worker_rank=worker_rank,
        data_hash=data_hash,
    )
    steps_to_meta.setdefault(step, []).append(meta_obj)

  for step in steps_to_meta:
    steps_to_meta[step].sort(key=lambda m: (m.node_rank, m.worker_rank))

  return steps_to_meta


def _to_gcs_uri(path: epath.PathLike) -> str:
  """Converts epath / GCS paths to standard gs:// URI format."""
  s = os.fspath(path)
  if s.startswith("/gcs/"):
    return "gs://" + s[len("/gcs/") :]
  return s


def _parse_gcs_path(path: epath.PathLike) -> tuple[str, str]:
  """Parses a GCS path into (bucket_name, prefix)."""
  uri = _to_gcs_uri(path)
  if not uri.startswith("gs://"):
    raise ValueError(f"Expected gs:// URI, got: {uri}")
  parts = uri[5:].split("/", 1)
  bucket_name = parts[0]
  prefix = parts[1].strip("/") if len(parts) > 1 else ""
  return bucket_name, prefix


def _copy_file(
    src: epath.Path,
    dst: epath.Path,
    overwrite: bool = True,
) -> None:
  """Copies a single file, using GCS server-side copy_blob if on GCS."""
  if gcs_utils.is_gcs_path(src) and gcs_utils.is_gcs_path(dst):
    if storage is None:
      raise ImportError("google.cloud.storage is not available")
    src_bucket_name, src_blob_name = _parse_gcs_path(src)
    dst_bucket_name, dst_blob_name = _parse_gcs_path(dst)
    client = storage.Client()
    src_bucket = client.bucket(src_bucket_name)
    dst_bucket = client.bucket(dst_bucket_name)
    src_blob = src_bucket.blob(src_blob_name)
    if not overwrite and dst_bucket.blob(dst_blob_name).exists():
      return
    src_bucket.copy_blob(src_blob, dst_bucket, dst_blob_name)
  elif not gcs_utils.is_gcs_path(src) and not gcs_utils.is_gcs_path(dst):
    src.copy(dst, overwrite=overwrite)
  else:
    raise ValueError(
        f"Cross-filesystem copy between GCS and local paths is not supported:"
        f" src={src}, dst={dst}"
    )


def _copy_dir_gcs(
    src: epath.Path,
    dst: epath.Path,
    overwrite: bool = True,
    max_workers: int = 32,
) -> None:
  """Copies all blobs in GCS directory src to dst server-side in parallel."""
  if storage is None:
    raise ImportError("google.cloud.storage is not available")

  src_bucket_name, src_prefix = _parse_gcs_path(src)
  dst_bucket_name, dst_prefix = _parse_gcs_path(dst)

  client = storage.Client()
  src_bucket = client.bucket(src_bucket_name)
  dst_bucket = client.bucket(dst_bucket_name)

  src_prefix_clean = src_prefix.strip("/")
  dst_prefix_clean = dst_prefix.strip("/")

  prefix_query = f"{src_prefix_clean}/" if src_prefix_clean else ""
  blobs = list(src_bucket.list_blobs(prefix=prefix_query))

  if not blobs and src_prefix_clean:
    single_blob = src_bucket.blob(src_prefix_clean)
    if single_blob.exists():
      blobs = [single_blob]
      prefix_query = src_prefix_clean

  tasks: list[tuple[Any, str]] = []
  for blob in blobs:
    if blob.name.endswith("_$folder$"):
      continue
    if prefix_query and blob.name.startswith(prefix_query):
      rel_path = blob.name[len(prefix_query) :]
    elif src_prefix_clean and blob.name == src_prefix_clean:
      rel_path = os.path.basename(blob.name)
    else:
      rel_path = os.path.basename(blob.name)

    if not rel_path:
      continue

    dst_blob_name = (
        f"{dst_prefix_clean}/{rel_path}".strip("/")
        if dst_prefix_clean
        else rel_path
    )
    tasks.append((blob, dst_blob_name))

  def _copy_blob_task(task: tuple[Any, str]) -> None:
    src_blob, dst_name = task
    dst_blob = dst_bucket.blob(dst_name)
    if not overwrite and dst_blob.exists():
      return
    src_bucket.copy_blob(src_blob, dst_bucket, dst_name)

  with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    list(executor.map(_copy_blob_task, tasks))


def _cleanup_gcs_folder_markers(step_dir: epath.Path) -> None:
  """Cleans up legacy 0-byte _$folder$ marker objects created by epath on GCS."""
  if not gcs_utils.is_gcs_path(step_dir) or storage is None:
    return

  try:
    bucket_name, prefix = _parse_gcs_path(step_dir)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    folder_prefix = f"{prefix.strip('/')}/" if prefix else ""
    for blob in bucket.list_blobs(prefix=folder_prefix):
      if blob.name.endswith("_$folder$"):
        blob.delete()
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.debug("Python SDK folder marker cleanup failed: %s", e)


def _link_dir(
    src: epath.Path,
    dst: epath.Path,
    overwrite: bool = True,
    max_workers: int = 32,
) -> None:
  """Links directory src to dst.

  Uses filesystem symlinks for local / POSIX filesystems (instantaneous, 0-byte
  duplication). On GCS object stores (which lack native OS symlinks), mirrors
  the directory structure via server-side copy_blob with parallel threads.

  Args:
    src: Source directory.
    dst: Destination directory.
    overwrite: Whether to overwrite existing destination.
    max_workers: Number of concurrent threads for GCS blob copying.

  Raises:
    FileExistsError: If destination exists and overwrite=False.
  """
  if not gcs_utils.is_gcs_path(src) and not gcs_utils.is_gcs_path(dst):
    if dst.exists() or os.path.islink(str(dst)):
      if overwrite:
        if os.path.islink(str(dst)) or dst.is_file():
          dst.unlink()
        else:
          shutil.rmtree(str(dst))
      else:
        raise FileExistsError(f"Destination '{dst}' already exists.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.abspath(str(src)), str(dst))
  elif gcs_utils.is_gcs_path(src) and gcs_utils.is_gcs_path(dst):
    _copy_dir_gcs(src, dst, overwrite=overwrite, max_workers=max_workers)
  else:
    raise ValueError(
        "Cross-filesystem linking between GCS and local paths is not"
        f" supported: src={src}, dst={dst}"
    )


def get_default_output_dir(input_dir: epath.PathLike) -> epath.Path:
  """Derives standard checkpoint output directory in the same location as input_dir.

  Args:
    input_dir: GCS or local path to MTC backup directory.

  Returns:
    Target output directory path located alongside or inside the input location.
  """
  input_path = epath.Path(input_dir)
  if _TIMESTAMP_DIR_PATTERN.match(input_path.name):
    return input_path.parent / "standard_checkpoints"
  return input_path / "standard_checkpoints"


def convert_mtc_step(
    backup_dir: epath.PathLike,
    target_step_dir: epath.PathLike,
    meta_files: Sequence[MtcMetaFile],
    overwrite: bool = False,
    max_workers: int = 32,
) -> epath.Path:
  """Converts a single MTC checkpoint step into a standard Orbax checkpoint.

  Links node shards and merges OCDBT manifests with strict parameter validation.

  Args:
    backup_dir: Directory containing MTC backup files (.meta and D<hash>.data).
    target_step_dir: Target directory for the converted step (e.g.
      `<output_dir>/<step>`).
    meta_files: List of MtcMetaFile objects representing all nodes for this
      step.
    overwrite: If True, overwrite target directory if it already exists.
    max_workers: Number of concurrent threads for GCS blob copying.

  Returns:
    The path to the converted step directory.

  Raises:
    ValueError: If meta_files is empty or belongs to multiple steps.
    FileExistsError: If target_step_dir already exists and overwrite=False.
  """
  backup_dir = epath.Path(backup_dir)
  target_step_dir = epath.Path(target_step_dir)

  if not meta_files:
    raise ValueError("No metadata files provided for conversion.")

  steps = {m.step for m in meta_files}
  if len(steps) > 1:
    raise ValueError(
        f"All meta_files must belong to the same step. Found steps: {steps}"
    )
  step = meta_files[0].step

  logging.info(
      "Converting MTC step %d from '%s' to '%s' (%d node metadata files)...",
      step,
      backup_dir,
      target_step_dir,
      len(meta_files),
  )

  if target_step_dir.exists():
    if not overwrite:
      raise FileExistsError(
          f"Target step directory '{target_step_dir}' already exists and"
          " overwrite=False."
      )
    logging.info("Cleaning up existing target directory: %s", target_step_dir)
    target_step_dir.rmtree()

  target_step_dir.mkdir(parents=True, exist_ok=True)

  # Check that each node's data directory exists.
  node_data_dirs: dict[int, epath.Path] = {}
  for m in meta_files:
    node_data_dirs[m.node_rank] = find_data_dir(backup_dir, m.data_hash)

  first_node_data = node_data_dirs[meta_files[0].node_rank]

  # 1. Link process_metadata if present.
  src_proc_meta = first_node_data / _PROCESS_METADATA_DIR
  if src_proc_meta.exists() and src_proc_meta.is_dir():
    _link_dir(
        src_proc_meta,
        target_step_dir / _PROCESS_METADATA_DIR,
        max_workers=max_workers,
    )

  # 2. Discover checkpointable items (e.g. 'state', 'dataset').
  item_names: list[str] = []
  for entry in first_node_data.iterdir():
    if entry.is_dir() and entry.name != _PROCESS_METADATA_DIR:
      item_names.append(entry.name)

  if not item_names:
    raise ValueError(
        f"No checkpoint items found in node data directory '{first_node_data}'."
    )

  # 3. Setup _CHECKPOINT_METADATA with item_handlers mapping.
  pytree_handler_str = (
      f"{pytree_checkpoint_handler.PyTreeCheckpointHandler.__module__}."
      f"{pytree_checkpoint_handler.PyTreeCheckpointHandler.__qualname__}"
  )
  src_ckpt_meta = first_node_data / _CHECKPOINT_METADATA_FILE
  meta_json: dict[str, Any] = {}
  if src_ckpt_meta.exists():
    try:
      meta_json = json.loads(src_ckpt_meta.read_text())
    except (json.JSONDecodeError, OSError):
      meta_json = {}

  if "item_handlers" not in meta_json or not meta_json["item_handlers"]:
    meta_json["item_handlers"] = {
        item: pytree_handler_str for item in item_names
    }
  elif isinstance(meta_json["item_handlers"], dict):
    # Remove any internal process_metadata handlers that cause standard Orbax
    # restore to fail if the MTC handler class is not imported.
    keys_to_remove = [
        k
        for k, v in meta_json["item_handlers"].items()
        if _PROCESS_METADATA_DIR in k
        or "ProcessMetadataCheckpointHandler" in str(v)
    ]
    for k in keys_to_remove:
      meta_json["item_handlers"].pop(k, None)

    for item in item_names:
      if item not in meta_json["item_handlers"]:
        meta_json["item_handlers"][item] = pytree_handler_str

  if (
      "state" in meta_json["item_handlers"]
      and "items" not in meta_json["item_handlers"]
  ):
    meta_json["item_handlers"]["items"] = meta_json["item_handlers"]["state"]

  (target_step_dir / _CHECKPOINT_METADATA_FILE).write_text(
      json.dumps(meta_json, indent=2)
  )

  ts_context = ts_utils.get_ts_context(use_ocdbt=True)

  # 4. Process each item.
  for item_name in item_names:
    target_item_dir = target_step_dir / item_name
    target_item_dir.mkdir(parents=True, exist_ok=True)

    first_item_dir = first_node_data / item_name
    for file_entry in first_item_dir.iterdir():
      if file_entry.is_file():
        if file_entry.name in (
            _OCDBT_MANIFEST_FILE,
            _METADATA_FILE,
            _SHARDING_FILE,
        ):
          continue
        _copy_file(file_entry, target_item_dir / file_entry.name)

    # Merge _METADATA across all node shards.
    tree_metas: list[tree_metadata.InternalTreeMetadata] = []
    for node_rank in sorted(node_data_dirs.keys()):
      data_dir = node_data_dirs[node_rank]
      meta_file = data_dir / item_name / _METADATA_FILE
      if meta_file.exists():
        try:
          data = json.loads(meta_file.read_text())
          tree_metas.append(tree_metadata.InternalTreeMetadata.from_json(data))
        except (json.JSONDecodeError, OSError, ValueError) as e:
          logging.warning(
              "Failed to parse %s from node %d: %s",
              _METADATA_FILE,
              node_rank,
              e,
          )
    if tree_metas:
      merged_tree_meta = functools.reduce(
          lambda a, b: a.merge(b, overwrite=True), tree_metas
      )
      (target_item_dir / _METADATA_FILE).write_text(
          json.dumps(merged_tree_meta.to_json(), indent=2)
      )
    elif (first_item_dir / _METADATA_FILE).exists():
      _copy_file(
          first_item_dir / _METADATA_FILE,
          target_item_dir / _METADATA_FILE,
      )

    # Merge _sharding across all node shards if present.
    merged_sharding: dict[str, Any] = {}
    has_sharding = False
    for node_rank in sorted(node_data_dirs.keys()):
      data_dir = node_data_dirs[node_rank]
      sharding_file = data_dir / item_name / _SHARDING_FILE
      if sharding_file.exists():
        try:
          data = json.loads(sharding_file.read_text())
          merged_sharding.update(data)
          has_sharding = True
        except (json.JSONDecodeError, OSError, ValueError) as e:
          logging.warning(
              "Failed to parse %s from node %d: %s",
              _SHARDING_FILE,
              node_rank,
              e,
          )
    if has_sharding:
      (target_item_dir / _SHARDING_FILE).write_text(
          json.dumps(merged_sharding, indent=2)
      )

    is_ocdbt_item = (first_item_dir / _OCDBT_MANIFEST_FILE).exists()

    # Link each node's shard into target_item_dir in parallel.
    link_tasks: list[tuple[epath.Path, epath.Path]] = []
    linked_proc_dirs: set[str] = set()
    for node_rank, data_dir in node_data_dirs.items():
      src_node_item = data_dir / item_name
      if not src_node_item.exists():
        continue

      if is_ocdbt_item:
        existing_proc_dirs = [
            p
            for p in src_node_item.iterdir()
            if p.is_dir() and p.name.startswith("ocdbt.process_")
        ]
        if existing_proc_dirs:
          for proc_dir in existing_proc_dirs:
            if proc_dir.name not in linked_proc_dirs:
              target_proc_dir = target_item_dir / proc_dir.name
              link_tasks.append((proc_dir, target_proc_dir))
              linked_proc_dirs.add(proc_dir.name)
        else:
          target_process_dir = target_item_dir / f"ocdbt.process_{node_rank}"
          link_tasks.append((src_node_item, target_process_dir))
      else:
        link_tasks.append((src_node_item, target_item_dir))

    task_workers = max(4, max_workers // max(1, len(link_tasks)))

    def _execute_link(
        task: tuple[epath.Path, epath.Path],
        workers: int = task_workers,
    ) -> None:
      src_d, dst_d = task
      _link_dir(
          src_d,
          dst_d,
          overwrite=True,
          max_workers=workers,
      )

    with futures.ThreadPoolExecutor(
        max_workers=min(max_workers, max(1, len(link_tasks)))
    ) as executor:
      list(executor.map(_execute_link, link_tasks))

    # Merge all per-process OCDBT manifests with strict parameter validation.
    if is_ocdbt_item:
      logging.info(
          "Merging OCDBT manifests with validation for item '%s' in '%s'...",
          item_name,
          target_item_dir,
      )
      asyncio_utils.run_sync(
          ocdbt_utils.merge_ocdbt_per_process_files(
              target_item_dir,
              ts_context=ts_context,
              use_zarr3=True,
              enable_validation=True,
          )
      )

  # 5. Create 'items' alias via symlink if 'state' was converted
  if "state" in item_names and "items" not in item_names:
    target_state_dir = target_step_dir / "state"
    target_items_dir = target_step_dir / "items"
    logging.info("Creating 'items' link from 'state'...")
    if not gcs_utils.is_gcs_path(target_step_dir):
      if target_items_dir.exists() or os.path.islink(str(target_items_dir)):
        if os.path.islink(str(target_items_dir)) or target_items_dir.is_file():
          target_items_dir.unlink()
        else:
          shutil.rmtree(str(target_items_dir))
      os.symlink("state", str(target_items_dir))
    else:
      _link_dir(
          target_state_dir,
          target_items_dir,
          overwrite=True,
          max_workers=max_workers,
      )

  # 6. Finalize step with commit_success.txt
  commit_file = target_step_dir / _COMMIT_SUCCESS_FILE
  commit_file.write_text("")

  # Clean up legacy 0-byte _$folder$ marker objects on GCS
  _cleanup_gcs_folder_markers(target_step_dir)

  logging.info(
      "Successfully converted step %d into standard Orbax checkpoint: %s",
      step,
      target_step_dir,
  )
  return target_step_dir


def convert_mtc_backup(
    input_dir: epath.PathLike,
    output_dir: epath.PathLike | None = None,
    step: int | None = None,
    overwrite: bool = False,
    max_workers: int = 32,
) -> list[epath.Path]:
  """Converts an MTC backup directory into standard Orbax checkpoints.

  Args:
    input_dir: GCS or local directory containing the MTC backup.
    output_dir: Target root directory for standard Orbax checkpoints. If None,
      derived in the same location as input_dir.
    step: Specific step number to convert. If None, converts all steps found.
    overwrite: Whether to overwrite target step directories if they exist.
    max_workers: Number of concurrent threads for GCS blob copying.

  Returns:
    List of converted step directory paths.

  Raises:
    FileNotFoundError: If input_dir does not exist or contains no valid backup.
    ValueError: If a specified step is not found.
  """
  input_dir = epath.Path(input_dir)
  if output_dir is None:
    output_dir = get_default_output_dir(input_dir)
  else:
    output_dir = epath.Path(output_dir)

  if not input_dir.exists():
    raise FileNotFoundError(f"Input directory does not exist: '{input_dir}'")

  backup_dirs: list[epath.Path] = []
  direct_meta = discover_mtc_meta_files(input_dir)
  if direct_meta:
    backup_dirs.append(input_dir)
  else:
    for child in input_dir.iterdir():
      if child.is_dir() and _TIMESTAMP_DIR_PATTERN.match(child.name):
        backup_dirs.append(child)
    backup_dirs.sort(key=lambda p: p.name, reverse=True)

  if not backup_dirs:
    raise FileNotFoundError(
        "No MTC .meta files or timestamp backup directories found in"
        f" '{input_dir}'."
    )

  discovered_steps: dict[int, tuple[epath.Path, list[MtcMetaFile]]] = {}
  for b_dir in backup_dirs:
    steps_in_dir = discover_mtc_meta_files(b_dir)
    for s, meta_list in steps_in_dir.items():
      if s not in discovered_steps:
        discovered_steps[s] = (b_dir, meta_list)

  if not discovered_steps:
    raise FileNotFoundError(
        "No valid MTC step metadata found in"
        f" {[b.as_posix() for b in backup_dirs]}."
    )

  steps_to_process = (
      [step] if step is not None else sorted(discovered_steps.keys())
  )

  converted_paths: list[epath.Path] = []
  for s in steps_to_process:
    if s not in discovered_steps:
      raise ValueError(
          f"Step {s} requested for conversion, but available steps are:"
          f" {sorted(discovered_steps.keys())}"
      )
    b_dir, meta_list = discovered_steps[s]
    target_step_dir = output_dir / str(s)
    converted_path = convert_mtc_step(
        backup_dir=b_dir,
        target_step_dir=target_step_dir,
        meta_files=meta_list,
        overwrite=overwrite,
        max_workers=max_workers,
    )
    converted_paths.append(converted_path)

  return converted_paths
