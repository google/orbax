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

"""FastAPI server for the Storage Service."""

from __future__ import annotations

import json

from absl import app
from absl import flags
from absl import logging
import fastapi
from orbax.checkpoint.experimental.caching import service

FLAGS = flags.FLAGS

uvicorn_app = fastapi.applications.FastAPI()
storage_service = service.StorageService(service.DEFAULT_CONFIG)


@uvicorn_app.get("/")
async def handle_hello():
  try:
    logging.info("Received HELLO request: on node %d", service.node_id())
    return "Hello from Storage Service!"
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Failed to handle HELLO: %s", e)
    return json.dumps({"error": str(e)}), 400


@uvicorn_app.post("/resolve")
async def handle_resolve(data=fastapi.params.Body(...)):
  try:
    logging.info("Received RESOLVE request: on node %d", service.node_id())
    execution_id = data.get("execution_id")
    step = data.get("step")
    asset_id = service.AssetId(execution_id=execution_id, step=step)
    path = storage_service.resolve(asset_id)
    return {"path": path}
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Failed to resolve asset: %s", e)
    return json.dumps({"error": str(e)}), 400


@uvicorn_app.post("/exists")
async def handle_exists(data=fastapi.params.Body(...)):
  try:
    logging.info("Received EXISTS request: on node %d", service.node_id())
    execution_id = data.get("execution_id")
    step = data.get("step")
    asset_id = service.AssetId(execution_id=execution_id, step=step)
    exists = storage_service.assets.exists(asset_id)
    return {"exists": exists}
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Failed to check asset existence: %s", e)
    return json.dumps({"error": str(e)}), 400


@uvicorn_app.post("/finalize")
async def handle_finalize(data=fastapi.params.Body(...)):
  try:
    logging.info("Received FINALIZE request: on node %d", service.node_id())
    execution_id = data.get("execution_id")
    step = data.get("step")
    asset_id = service.AssetId(execution_id=execution_id, step=step)
    storage_service.finalize(asset_id)
    return {"status": "ok"}
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Failed to finalize asset: %s", e)
    return json.dumps({"error": str(e)}), 400


@uvicorn_app.post("/await_transfer")
async def handle_await_transfer(data=fastapi.params.Body(...)):
  """Handles await_transfer request."""
  try:
    logging.info(
        "Received AWAIT_TRANSFER request: on node %d", service.node_id()
    )
    execution_id = data.get("execution_id")
    step = data.get("step")
    asset_id = service.AssetId(execution_id=execution_id, step=step)
    storage_service.await_transfer(asset_id)
    return {"status": "ok"}
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Failed to await transfer: %s", e)
    return json.dumps({"error": str(e)}), 400


@uvicorn_app.post("/prefetch")
async def handle_prefetch(data=fastapi.params.Body(...)):
  try:
    logging.info("Received PREFETCH request: on node %d", service.node_id())
    execution_id = data.get("execution_id")
    step = data.get("step")
    asset_id = service.AssetId(execution_id=execution_id, step=step)
    storage_service.prefetch(asset_id)
    return {"status": "ok"}
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Failed to prefetch asset: %s", e)
    return json.dumps({"error": str(e)}), 400


@uvicorn_app.get("/inspect")
async def handle_inspect():
  try:
    logging.info("Received INSPECT request: on node %d", service.node_id())
    return {
        "storages": storage_service.storages.to_json(),
        "assets": storage_service.assets.to_json(),
    }
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Failed to inspect: %s", e)
    return json.dumps({"error": str(e)}), 400


def main(_):
  import uvicorn  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  FLAGS.alsologtostderr = True
  logging.set_verbosity(logging.INFO)
  uvicorn.run(uvicorn_app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
  app.run(main)
