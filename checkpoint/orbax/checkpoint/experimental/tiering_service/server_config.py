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

"""Server configuration."""

from collections.abc import Mapping
import datetime
from typing import Any
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2
import pytimeparse
import yaml


def _parse_timedelta(val: str) -> datetime.timedelta:
  """Parses a duration string (e.g. '1s', '30m', '1h') into a timedelta."""
  if not isinstance(val, str):
    raise ValueError(
        f"Invalid duration type for client_keep_alive_interval: {type(val)},"
        " expected str."
    )
  seconds = pytimeparse.parse(val)
  if seconds is None:
    raise ValueError(
        f"Invalid duration format for client_keep_alive_interval: {val}"
    )
  return datetime.timedelta(seconds=seconds)


def _parse_client_keep_alive(
    data: Mapping[str, Any], config: tiering_service_pb2.ServerConfig
) -> None:
  """Parses client keep-alive interval into ServerConfig."""
  if "client_keep_alive_interval_seconds" in data:
    config.client_keep_alive_interval_seconds = int(
        data["client_keep_alive_interval_seconds"]
    )
  elif "client_keep_alive_interval" in data:
    val = data["client_keep_alive_interval"]
    if isinstance(val, (int, float)):
      config.client_keep_alive_interval_seconds = int(val)
    else:
      timedelta_val = _parse_timedelta(str(val))
      config.client_keep_alive_interval_seconds = int(
          timedelta_val.total_seconds()
      )


def _parse_db_connection(
    data: Mapping[str, Any], config: tiering_service_pb2.ServerConfig
) -> None:
  """Parses database connection string into ServerConfig."""
  if "db_connection_str" in data:
    config.db_connection_str = str(data["db_connection_str"])


def _parse_storage_backend(
    b_data: Mapping[str, Any],
    backend: tiering_service_pb2.StorageBackend,
) -> None:
  """Parses a dictionary into a StorageBackend proto."""
  if "level" in b_data and b_data["level"] is not None:
    backend.level = int(b_data["level"])
  else:
    raise ValueError(
        "StorageBackend configuration missing required key: 'level'"
    )

  if "backend_type" in b_data and b_data["backend_type"]:
    b_type = b_data["backend_type"]
    if isinstance(b_type, str):
      b_type_upper = b_type.upper()
      if b_type_upper in ("LUSTRE", "BACKEND_TYPE_LUSTRE"):
        backend.backend_type = tiering_service_pb2.BACKEND_TYPE_LUSTRE
      elif b_type_upper in ("GCS", "BACKEND_TYPE_GCS"):
        backend.backend_type = tiering_service_pb2.BACKEND_TYPE_GCS
      else:
        raise ValueError(f"Unknown storage backend_type: {b_type}")
    else:
      backend.backend_type = b_type
  else:
    raise ValueError(
        "StorageBackend configuration missing required key: 'backend_type'"
    )

  if "prefix" in b_data and b_data["prefix"] is not None:
    backend.prefix = str(b_data["prefix"])
  else:
    raise ValueError(
        "StorageBackend configuration missing required key: 'prefix'"
    )

  if "zone" in b_data and b_data["zone"]:
    backend.zone = str(b_data["zone"])
  elif "region" in b_data and b_data["region"]:
    backend.region = str(b_data["region"])
  elif "multi_regions" in b_data and b_data["multi_regions"]:
    mr = b_data["multi_regions"]
    if isinstance(mr, dict) and "regions" in mr:
      backend.multi_regions.regions.extend(mr["regions"])
    elif isinstance(mr, (list, tuple)):
      backend.multi_regions.regions.extend(mr)
    else:
      raise ValueError(f"Invalid multi_regions format: {mr}")


def _parse_storage_backends(
    data: Mapping[str, Any], config: tiering_service_pb2.ServerConfig
) -> None:
  """Parses storage backends list into ServerConfig."""
  backends_data = data.get("storage_backends", [])
  for b_data in backends_data:
    backend = config.storage_backends.add()
    _parse_storage_backend(b_data, backend)


def _parse_max_active_jobs_per_backend(
    data: Mapping[str, Any], config: tiering_service_pb2.ServerConfig
) -> None:
  """Parses max active jobs per backend into ServerConfig."""
  if "max_active_jobs_per_backend" in data:
    config.max_active_jobs_per_backend = int(
        data["max_active_jobs_per_backend"]
    )


def parse_config(data: Mapping[str, Any]) -> tiering_service_pb2.ServerConfig:
  """Parses a dictionary into a ServerConfig proto.

  Args:
    data: A dictionary (usually loaded from YAML) containing server
      configuration parameters.

  Returns:
    A ServerConfig proto instance populated with the parsed data.
  """
  config = tiering_service_pb2.ServerConfig()
  _parse_client_keep_alive(data, config)
  _parse_db_connection(data, config)
  _parse_storage_backends(data, config)
  _parse_max_active_jobs_per_backend(data, config)
  return config


def load_config(yaml_path: str) -> tiering_service_pb2.ServerConfig:
  """Loads and parses a ServerConfig from a YAML file.

  Args:
    yaml_path: Path to the YAML configuration file.

  Returns:
    A ServerConfig proto instance populated with the parsed data.
  """
  with open(yaml_path, "r") as f:
    config_dict = yaml.safe_load(f)
  return parse_config(config_dict)
