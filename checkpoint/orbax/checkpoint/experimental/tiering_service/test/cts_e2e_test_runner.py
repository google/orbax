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

#!/usr/bin/env python3
"""Automated CTS E2E Integration Test Runner CLI.

This script manages end-to-end testing and cluster environment setup
for the Checkpoint Tiering Service (CTS).

Available Commands:
  create_demo_cluster:
    Creates a GKE demo cluster and optionally attaches a Lustre instance.
    Parameters:
      --project: GCP project ID (inferred from environment if omitted).
      --cluster_name: Name of the GKE cluster (default: "cts_demo_cluster").
      --zone: GCP zone (default: "us-east1-b").
      --instance_type / --machine_type: Node machine type
        (default: "e2-standard-4").
      --num_nodes: Number of nodes to allocate (default: 1).
      --network_tier: Network tier, e.g. "premium" or "standard".
      --spot: Flag to create cluster using Spot VM instances.
      --luster_instance / --lustre_instance: Optional Lustre instance name
        to attach.
      --luster_mount / --lustre_mount: Mount point for Lustre
        (default: "/lustre").

  bootstrap:
    Ensures the GKE test pod is running, checks out a GitHub PR if specified,
    parses dependencies from pyproject.toml, installs required packages inside
    the pod, copies Orbax source code and mock alerter into site-packages,
    compiles protobufs, and initializes the SQLite database.
    Parameters:
      --pod_name: Pod name (default: "cts-test-pod").
      --namespace: Kubernetes namespace (default: "default").
      --cluster: GKE cluster name.
      --github_pr: Optional GitHub PR number or branch to fetch and install.
      --repo_url: GitHub repository URL
        (default: "https://github.com/google/orbax.git").

  run_test:
    Starts the tiering service server inside the pod, runs the E2E
    integration test suite (`cts_integration_run.py`), verifies checkpoint
    save and restore operations, and cleans up server processes.

  quick_test:
    Runs a fast verification test inside the pod container by invoking
    `cts_client_cli --help` to ensure client CLI and dependencies are
    properly installed.

  all:
    Runs the complete E2E workflow by executing `bootstrap` followed by
    `run_test`.
"""

import os
import subprocess
import sys
import time

from absl import logging
import fire  # pylint: disable=g-import-not-at-top

# Auto-detect WORKSPACE_DIR
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir and os.path.basename(current_dir) != "":
  parent = os.path.dirname(current_dir)
  if parent == current_dir:
    break
  current_dir = parent
WORKSPACE_DIR = current_dir

if not WORKSPACE_DIR or not os.path.exists(WORKSPACE_DIR):
  raise RuntimeError(
      f"Could not resolve  workspace directory: {WORKSPACE_DIR}"
  )

# Add  path to sys.path to allow imports from third_party
sys.path.extend([
    os.path.dirname(WORKSPACE_DIR),
    os.path.join(WORKSPACE_DIR, "third_party/py"),
])

PACKAGE_PATH = "orbax/checkpoint/experimental/tiering_service"
POD_SITE_PACKAGES = "/usr/local/lib/python3.13/site-packages"


class CTSE2ERunner:
  """CTS E2E Integration Test Runner CLI."""

  def __init__(
      self,
      pod_name: str = "cts-test-pod",
      namespace: str = "default",
      cluster: str | None = None,
      zone: str = "us-east1-b",
      project: str | None = None,
      github_pr: int | str | None = None,
      repo_url: str = "https://github.com/google/orbax.git",
      pod_timeout: int = 300,
  ):
    self.pod_name = pod_name
    self.namespace = namespace
    self.cluster = cluster
    self.zone = zone
    self.project = project
    self.github_pr = str(github_pr) if github_pr else None
    self.repo_url = repo_url
    self.pod_timeout = pod_timeout

  def run_cmd(
      self,
      cmd: str,
      check: bool = True,
      capture_output: bool = True,
      text: bool = True,
  ) -> subprocess.CompletedProcess[str]:
    """Helper to run shell commands on host."""
    logging.info("[HOST EXEC] Running: %s", cmd)
    try:
      result = subprocess.run(
          cmd,
          shell=True,
          check=check,
          stdout=subprocess.PIPE if capture_output else None,
          stderr=subprocess.PIPE if capture_output else None,
          text=text,
      )
      if capture_output and result.stdout:
        logging.info("Stdout:\n%s", result.stdout.strip())
      if capture_output and result.stderr:
        logging.warning("Stderr:\n%s", result.stderr.strip())
      return result
    except subprocess.CalledProcessError as e:
      if capture_output and e.stdout:
        logging.error("FAILED Command Stdout:\n%s", e.stdout.strip())
      if capture_output and e.stderr:
        logging.error("FAILED Command Stderr:\n%s", e.stderr.strip())
      raise e

  def run_pod_cmd(
      self, cmd: str, check: bool = True, background: bool = False
  ) -> subprocess.Popen[bytes] | subprocess.CompletedProcess[str]:
    """Helper to run commands inside GKE test pod."""
    kube_cmd = (
        f"kubectl exec {self.pod_name} -n {self.namespace} -- {cmd}".strip()
    )
    if background:
      logging.info("[POD BG EXEC] Running: %s", kube_cmd)
      return subprocess.Popen(kube_cmd, shell=True)
    else:
      logging.info("[POD EXEC] Running: %s", kube_cmd)
      return self.run_cmd(kube_cmd, check=check)

  def _ensure_cluster_context(self):
    """Configures cluster context if cluster parameter is specified."""
    if self.cluster:
      zone_arg = f" --zone={self.zone}" if self.zone else ""
      project_arg = f" --project={self.project}" if self.project else ""
      logging.info(
          "Connecting to GKE cluster '%s' (zone: %s)...",
          self.cluster,
          self.zone,
      )
      self.run_cmd(
          "gcloud container clusters get-credentials"
          f" {self.cluster}{zone_arg}{project_arg}",
          check=False,
      )

  def _ensure_pod_exists(self):
    """Verifies test pod is running, starts it if missing."""
    self._ensure_cluster_context()

    logging.info("Checking status of pod '%s'...", self.pod_name)
    check_cmd = f"kubectl get pod {self.pod_name} -n {self.namespace}"
    res = self.run_cmd(check_cmd, check=False)
    if res.returncode != 0 or "Running" not in res.stdout:
      if res.returncode == 0 and "Pending" in res.stdout:
        logging.info(
            "Pod '%s' is stuck in Pending state. Recreating pod...",
            self.pod_name,
        )
        self.run_cmd(
            f"kubectl delete pod {self.pod_name} -n {self.namespace} --now",
            check=False,
        )

      logging.info("Pod '%s' not running. Starting pod...", self.pod_name)
      manifest_path = os.path.join(
          WORKSPACE_DIR, PACKAGE_PATH, "test/cts_test_pod.yaml"
      )
      if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
          content = f.read()
        content = content.replace(
            "name: cts-test-pod", f"name: {self.pod_name}"
        )
        pvc_check = self.run_cmd(
            f"kubectl get pvc lustre-pvc -n {self.namespace}", check=False
        )
        if pvc_check.returncode == 0 and "Bound" in pvc_check.stdout:
          content = content.replace(
              "emptyDir: {}",
              "persistentVolumeClaim:\n      claimName: lustre-pvc",
          )

        temp_manifest = f"/tmp/pod_manifest_{self.pod_name}.yaml"
        with open(temp_manifest, "w", encoding="utf-8") as f:
          f.write(content)
        self.run_cmd(f"kubectl apply -f {temp_manifest} -n {self.namespace}")
        if os.path.exists(temp_manifest):
          os.remove(temp_manifest)
      else:
        self.run_cmd(
            f"kubectl run {self.pod_name} -n {self.namespace}"
            " --image=python:3.13 -- sleep 86400"
        )
      logging.info(
          "Waiting for pod '%s' to reach Running state (timeout: %ds)...",
          self.pod_name,
          self.pod_timeout,
      )
      wait_cmd = (
          f"kubectl wait --for=condition=Ready pod/{self.pod_name} -n"
          f" {self.namespace} --timeout={self.pod_timeout}s"
      )
      wait_res = self.run_cmd(wait_cmd, check=False)
      if wait_res.returncode != 0:
        describe_res = self.run_cmd(
            f"kubectl describe pod {self.pod_name} -n {self.namespace}",
            check=False,
        )
        raise RuntimeError(
            f"Pod '{self.pod_name}' failed to reach Ready state within"
            f" {self.pod_timeout}s. Pod details:\n{describe_res.stdout}"
        )

  def _fetch_github_pr(self) -> str:
    """Fetches GitHub PR if specified, returning path to source root."""
    if not self.github_pr:
      return WORKSPACE_DIR

    pr_dir = f"/tmp/orbax_pr_{self.github_pr}"
    logging.info(
        "Fetching GitHub PR #%s from %s into %s...",
        self.github_pr,
        self.repo_url,
        pr_dir,
    )
    if os.path.exists(pr_dir):
      self.run_cmd(f"rm -rf {pr_dir}")

    self.run_cmd(f"git clone --depth 1 {self.repo_url} {pr_dir}")
    self.run_cmd(
        f"git -C {pr_dir} fetch origin"
        f" pull/{self.github_pr}/head:pr-{self.github_pr}"
    )
    self.run_cmd(f"git -C {pr_dir} checkout pr-{self.github_pr}")
    logging.info(
        "Successfully checked out PR #%s into %s", self.github_pr, pr_dir
    )
    return pr_dir

  def _setup_pod_environment(self, source_dir: str):
    """Prepares directory structure and pip dependencies in GKE pod."""
    logging.info(
        "Setting up directory structure and pip dependencies in pod..."
    )
    self.run_pod_cmd("mkdir -p /app /tmp/orbax_pkg")

    possible_pkg_paths = [
        os.path.join(source_dir, "checkpoint/pyproject.toml"),
        os.path.join(
            source_dir, "orbax/checkpoint/pyproject.toml"
        ),
        os.path.join(source_dir, "pyproject.toml"),
        os.path.join(
            WORKSPACE_DIR, "orbax/checkpoint/pyproject.toml"
        ),
    ]
    pyproject_file = next(
        (p for p in possible_pkg_paths if os.path.exists(p)), None
    )

    if pyproject_file:
      pkg_dir = os.path.dirname(pyproject_file)
      logging.info("Installing dependencies from %s via pip...", pyproject_file)
      self.run_cmd(
          f"kubectl cp {pkg_dir}/."
          f" {self.pod_name}:/tmp/orbax_pkg/ -n {self.namespace}".strip()
      )
      self.run_pod_cmd("pip install '/tmp/orbax_pkg[tiering_service]'")
    else:
      logging.warning(
          "pyproject.toml not found. Installing fallback packages..."
      )
      self.run_pod_cmd(
          "pip install httpx grpcio uvloop fire sqlalchemy aiosqlite pyyaml"
          " pytimeparse greenlet protobuf grpcio-tools absl-py jax jaxlib"
          " google-auth google-cloud-storage orbax-checkpoint"
      )

  def _copy_source_files(self, source_dir: str):
    """Copies orbax.checkpoint source files from PR/workspace into pod site-packages."""
    logging.info("Copying source files to site-packages in pod...")
    possible_orbax_paths = [
        os.path.join(source_dir, "checkpoint/orbax/checkpoint"),
        os.path.join(WORKSPACE_DIR, "orbax/checkpoint"),
        os.path.join(source_dir, "orbax/checkpoint"),
    ]
    local_orbax = next(
        (p for p in possible_orbax_paths if os.path.exists(p)), None
    )
    if not local_orbax:
      raise FileNotFoundError(
          f"Could not locate orbax/checkpoint source directory in {source_dir}"
      )

    self.run_cmd(
        f"kubectl cp {local_orbax}/."
        f" {self.pod_name}:{POD_SITE_PACKAGES}/orbax/checkpoint/"
        f" -n {self.namespace}".strip()
    )

    self.run_pod_cmd(
        f"touch {POD_SITE_PACKAGES}/orbax/checkpoint/experimental/__init__.py"
    )

  def _compile_protos(self):
    """Compiles tiering service protos inside pod."""
    logging.info("Compiling tiering service protos...")
    self.run_pod_cmd(
        "sed -i '/datapol/d'"
        f" {POD_SITE_PACKAGES}/orbax/checkpoint/experimental/tiering_service/proto/tiering_service.proto"
    )
    self.run_pod_cmd(
        "python3 -m grpc_tools.protoc"
        f" -I{POD_SITE_PACKAGES}"
        f" -I{POD_SITE_PACKAGES}/orbax/checkpoint/experimental/tiering_service/proto"
        f" --python_out={POD_SITE_PACKAGES}"
        f" --grpc_python_out={POD_SITE_PACKAGES}"
        f" {POD_SITE_PACKAGES}/orbax/checkpoint/experimental/tiering_service/proto/tiering_service.proto"
    )

  def _setup_server_config_and_db(self):
    """Copies server configuration yaml and initializes SQLite database schema."""
    logging.info("Copying server config and initializing database...")
    config_path = os.path.join(
        WORKSPACE_DIR, PACKAGE_PATH, "test/cts_server_example.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
      content = f.read()

    temp_config_path = "/tmp/server_config.yaml"
    with open(temp_config_path, "w", encoding="utf-8") as f:
      f.write(content)

    self.run_pod_cmd("mkdir -p /app")
    self.run_pod_cmd("rm -f /app/server_config.yaml")
    self.run_cmd(
        f"kubectl cp {temp_config_path}"
        f" {self.pod_name}:/app/server_config.yaml"
        f" -n {self.namespace}".strip()
    )
    if os.path.exists(temp_config_path):
      os.remove(temp_config_path)

    self.run_pod_cmd(
        "python3 -m orbax.checkpoint.experimental.tiering_service.db_cli"
        " initialize_db /app/server_config.yaml"
    )

  def bootstrap(self):
    """Installs GKE pod dependencies, compiles protos, copy configs, and initializes DB."""
    logging.info("=== Bootstrapping GKE pod '%s' ===", self.pod_name)
    self._ensure_pod_exists()
    source_dir = self._fetch_github_pr()
    self._setup_pod_environment(source_dir)
    self._copy_source_files(source_dir)
    self._compile_protos()
    self._setup_server_config_and_db()
    logging.info("Bootstrap completed successfully!")
    self.quick_test()

  def quick_test(self):
    """Executes cts_client_cli --help inside the GKE pod to verify setup."""
    logging.info("=== Running Quick Test: cts_client_cli --help ===")
    res = self.run_pod_cmd(
        "python3 -m"
        " orbax.checkpoint.experimental.tiering_service.test.cts_client_cli"
        " --help"
    )
    logging.info(
        "cts_client_cli --help output:\n%s",
        res.stdout
        if hasattr(res, "stdout") and res.stdout
        else "Executed successfully.",
    )
    logging.info("Quick test completed successfully!")

  def run_server(self):
    """Starts the CTS server in the GKE pod background and leaves it running."""
    logging.info("Stopping existing server if running...")
    self.run_pod_cmd("pkill -f 'tiering_service.server'", check=False)

    logging.info("Starting CTS Server in GKE pod background...")
    self.run_pod_cmd(
        "sh -c 'python3 -u -m"
        " orbax.checkpoint.experimental.tiering_service.server serve"
        " /app/server_config.yaml --start-tiering-service-worker=True >"
        " /app/server.log 2>&1 &'"
    )
    logging.info("Server started. Logs are at /app/server.log inside the pod.")

  def _prepare_test_environment(self):
    """Cleans up previous test state, copies integration script, and initializes DB."""
    logging.info("Preparing test environment and resetting state...")
    self.run_pod_cmd("pkill -f 'tiering_service.server'", check=False)
    self.run_pod_cmd("rm -f /app/cts.db")
    self.run_pod_cmd("rm -rf /lustre/checkpoints")
    self.run_pod_cmd(
        f"cp {POD_SITE_PACKAGES}/orbax/checkpoint/experimental/tiering_service/test/cts_integration_run.py"
        " /app/cts_integration_run.py"
    )
    self.run_pod_cmd(
        "sed -i 's|/tmp/lustre-mount/checkpoints|/lustre/checkpoints|g'"
        " /app/cts_integration_run.py"
    )
    self.run_pod_cmd(
        "python3 -m orbax.checkpoint.experimental.tiering_service.db_cli"
        " initialize_db /app/server_config.yaml"
    )

  def _wait_for_gcs_copy_job(self):
    """Polls database for GCS copy job completion."""
    logging.info("Waiting for GCS copy job to complete...")
    for i in range(120):
      res = self.run_pod_cmd(
          'python3 -c "import sqlite3; conn ='
          " sqlite3.connect('/app/cts.db'); print(conn.execute('SELECT status"
          " FROM asset_jobs LIMIT 1;').fetchone()[0])\""
      )
      status = ""
      if isinstance(res, subprocess.CompletedProcess):
        out = res.stdout
        if isinstance(out, str):
          status = out.strip()
      logging.info(
          "GCS copy job polling attempt %d/120 status: %s", i + 1, status
      )
      if status == "JOB_STATUS_COMPLETED" or status == "2":
        logging.info("GCS copy job completed successfully!")
        break
      time.sleep(5)
    else:
      logging.error("Timed out waiting for GCS copy job to complete.")
      raise TimeoutError("Timed out waiting for GCS copy job to complete.")

  def _simulate_eviction_and_seed_db(self):
    """Simulates local eviction and updates database for prefetch testing."""
    logging.info("Simulating eviction from Lustre...")
    self.run_pod_cmd("rm -rf /lustre/checkpoints/1")

    seeder_code = """import sqlite3
import sys
import uuid

db_path = "/app/cts.db"
asset_path = "/lustre/checkpoints/1"
gcs_path = "gs://orbax-benchmarks-us-east5/dnlng/tiering_service/e2e_test/checkpoints/1/"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT asset_uuid FROM assets WHERE path = ?", (asset_path,))
row = cursor.fetchone()
if not row:
  print(f"Error: Asset not found in database for path {asset_path}")
  sys.exit(1)
asset_uuid = row[0]

cursor.execute("SELECT id FROM storage_backends WHERE level = 1")
row = cursor.fetchone()
if not row:
  print("Error: GCS backend (level 1) not found in database!")
  sys.exit(1)
backend_id = row[0]

cursor.execute("SELECT id FROM storage_backends WHERE level = 0")
row = cursor.fetchone()
if not row:
  print("Error: Lustre backend (level 0) not found in database!")
  sys.exit(1)
lustre_backend_id = row[0]

cursor.execute(
    "DELETE FROM tier_paths WHERE asset_uuid = ? AND storage_backend_id = ?",
    (asset_uuid, lustre_backend_id)
)

cursor.execute(
    "SELECT id FROM tier_paths WHERE asset_uuid = ? AND storage_backend_id = ?",
    (asset_uuid, backend_id)
)
row = cursor.fetchone()
if row:
  cursor.execute(
      "UPDATE tier_paths SET ready_at = datetime('now'), state = 'READY' "
      "WHERE asset_uuid = ? AND storage_backend_id = ?",
      (asset_uuid, backend_id)
  )
else:
  tp_uuid = str(uuid.uuid4())
  cursor.execute(
      "INSERT INTO tier_paths (asset_uuid, storage_backend_id, path, ready_at, state, tier_path_uuid) "
      "VALUES (?, ?, ?, datetime('now'), 'READY', ?)",
      (asset_uuid, backend_id, gcs_path, tp_uuid)
  )

conn.commit()
conn.close()
print("Database seeded successfully.")
"""
    local_seeder_path = "/tmp/cts_seed_db.py"
    with open(local_seeder_path, "w", encoding="utf-8") as f:
      f.write(seeder_code)

    cluster_arg = f"--cluster {self.cluster}" if self.cluster else ""
    self.run_cmd(
        f"kubectl cp {local_seeder_path}"
        f" {self.pod_name}:/app/seed_db.py -n {self.namespace}"
        f" {cluster_arg}".strip()
    )
    if os.path.exists(local_seeder_path):
      os.remove(local_seeder_path)

    logging.info("Executing database seeder in pod...")
    self.run_pod_cmd("python3 -u /app/seed_db.py")

  def run_test(self):
    """Starts the server, runs E2E integration test, and kills the server on finish."""
    logging.info("=== Running E2E Integration Tests ===")
    self._prepare_test_environment()

    logging.info("Starting Server and Worker process in background...")
    server_proc = self.run_pod_cmd(
        "python3 -u -m orbax.checkpoint.experimental.tiering_service.server"
        " serve /app/server_config.yaml --start-tiering-service-worker=True",
        background=True,
    )
    time.sleep(3)

    try:
      logging.info("Executing JAX save...")
      self.run_pod_cmd(
          "python3 -u /app/cts_integration_run.py --mode=save --step=1"
      )

      self._wait_for_gcs_copy_job()
      self._simulate_eviction_and_seed_db()

      logging.info("Executing JAX restore...")
      res = self.run_pod_cmd(
          "python3 -u /app/cts_integration_run.py --mode=restore --step=1"
      )
      res_stdout = (
          res.stdout
          if isinstance(res, subprocess.CompletedProcess) and res.stdout
          else ""
      )
      if "INTEGRATION_TEST_SUCCESS" in res_stdout:
        logging.info("E2E INTEGRATION TEST SUCCESSFUL!")
      else:
        logging.error("Restore verification failed: %s", res_stdout)
        raise RuntimeError(f"Restore verification failed: {res_stdout}")
    finally:
      logging.info("Cleaning up server process...")
      self.run_pod_cmd("pkill -f 'tiering_service.server'", check=False)
      if isinstance(server_proc, subprocess.Popen):
        server_proc.terminate()

  def all(self):
    """Reinstalls everything, reinitializes DB, restarts server, runs tests, and cleans up."""
    logging.info("Running full E2E workflow (bootstrap + run_test)...")
    self.bootstrap()
    self.run_test()

  def _infer_gcp_project(self, project: str | None) -> str:
    """Infers GCP project from environment if not explicitly provided."""
    if project:
      return project
    for env_var in (
        "GCP_PROJECT",
        "GOOGLE_CLOUD_PROJECT",
        "CLOUDSDK_CORE_PROJECT",
    ):
      val = os.environ.get(env_var)
      if val:
        return val
    try:
      res = subprocess.run(
          "gcloud config get-value project",
          shell=True,
          stdout=subprocess.PIPE,
          text=True,
          check=False,
      )
      if res.returncode == 0 and res.stdout.strip():
        return res.stdout.strip()
    except Exception:  # pylint: disable=broad-except
      pass
    return "orbax-checkpoint"

  def _mount_lustre_instance(
      self,
      lustre_instance: str,
      lustre_mount: str,
      project: str,
  ):
    """Mounts a Lustre instance to GKE cluster."""
    logging.info(
        "Attempting to mount Lustre instance '%s' at '%s'...",
        lustre_instance,
        lustre_mount,
    )
    manifest_path = os.path.join(
        WORKSPACE_DIR, PACKAGE_PATH, "test/lustre_attach.yaml"
    )
    if os.path.exists(manifest_path):
      logging.info("Applying Lustre attach manifest: %s", manifest_path)
      self.run_cmd(f"kubectl apply -f {manifest_path} -n {self.namespace}")
    else:
      logging.info(
          "Configuring Lustre instance '%s' via gcloud/kubectl...",
          lustre_instance,
      )
      self.run_cmd(
          "gcloud deployment-manager deployments create"
          f" lustre-{lustre_instance} --project={project}",
          check=False,
      )

  def create_demo_cluster(
      self,
      project: str | None = None,
      cluster_name: str = "cts_demo_cluster",
      zone: str = "us-east1-b",
      instance_type: str = "e2-standard-4",
      num_nodes: int = 1,
      network_tier: str | None = None,
      spot: bool = False,
      luster_instance: str | None = None,
      luster_mount: str = "/lustre",
      machine_type: str | None = None,
      lustre_instance: str | None = None,
      lustre_mount: str | None = None,
  ):
    """Creates a GKE demo cluster and optionally mounts a Lustre instance."""
    resolved_project = self._infer_gcp_project(project or self.project)
    resolved_machine_type = machine_type or instance_type
    resolved_lustre_instance = lustre_instance or luster_instance
    resolved_lustre_mount = (
        lustre_mount if lustre_mount is not None else luster_mount
    )

    self.cluster = cluster_name
    clean_network_tier = network_tier.lower() if network_tier else None
    network_tier_arg = (
        f" --network-tier={clean_network_tier}" if clean_network_tier else ""
    )
    spot_arg = " --spot" if spot else ""
    logging.info(
        "Creating GKE cluster '%s' in project '%s', zone '%s' (instance_type:"
        " %s, num_nodes: %d%s%s)...",
        cluster_name,
        resolved_project,
        zone,
        resolved_machine_type,
        num_nodes,
        f", network_tier: {clean_network_tier}" if clean_network_tier else "",
        ", spot: True" if spot else "",
    )
    cmd = (
        f"gcloud container clusters create {cluster_name}"
        f" --project={resolved_project}"
        f" --zone={zone}"
        f" --machine-type={resolved_machine_type}"
        f" --num-nodes={num_nodes}"
        f"{network_tier_arg}"
        f"{spot_arg}"
    )
    self.run_cmd(cmd)

    logging.info("Getting credentials for cluster '%s'...", cluster_name)
    self.run_cmd(
        f"gcloud container clusters get-credentials {cluster_name}"
        f" --project={resolved_project} --zone={zone}"
    )

    if resolved_lustre_instance:
      self._mount_lustre_instance(
          lustre_instance=resolved_lustre_instance,
          lustre_mount=resolved_lustre_mount,
          project=resolved_project,
      )

    logging.info("Cluster '%s' created successfully!", cluster_name)


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  fire.Fire(CTSE2ERunner)
