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

r"""XPK Launcher for Orbax Benchmarks.

This script simplifies running Orbax benchmarks on XPK (Accelerated Processing Kit) clusters.
It acts as a high-level adapter that:
1.  **Config Management**: Uploads local YAML configs to GCS so they are accessible by remote pods.
2.  **Command Construction**: Generates complex `xpk workload create` commands, handling the differences
    between MC-JAX and Pathways (Sidecar) modes.
3.  **Lifecycle**: Can optionally auto-delete clusters upon completion.

Usage:
  python3 launch_xpk.py \
    --cluster_name=my-cluster \
    --config_file=../configs/pytree_checkpoint_benchmark.yaml \
    --tpu_type=v5litepod-8 \
    --num_slices=1
"""

from collections.abc import Sequence
import datetime
import itertools
import os
import subprocess
import sys
import uuid

from absl import app
from absl import flags
from absl import logging

# LINT.IfChange(launch_xpk_flags)

# ==============================================================================
# 1. Required Flags
# ==============================================================================
_CLUSTER_NAME = flags.DEFINE_string(
    'cluster_name', None, 'Name of the XPK cluster.', required=True
)
_CONFIG_FILE = flags.DEFINE_string(
    'config_file',
    None,
    'Path to the local benchmark YAML config file.',
    required=True,
)
flags.register_validator(
    'config_file',
    os.path.exists,
    message='--config_file must be a path to an existing file.',
)
_OUTPUT_DIRECTORY = flags.DEFINE_string(
    'output_directory',
    None,
    'GCS bucket/path for artifacts and config upload.',
    required=True,
)
flags.register_validator(
    'output_directory',
    lambda value: value.startswith('gs://'),
    message='--output_directory must start with gs://',
)

# ==============================================================================
# 2. Basic Flags (Commonly Used)
# ==============================================================================
_DOCKER_IMAGE = flags.DEFINE_string(
    'docker_image',
    'gcr.io/orbax-checkpoint/orbax-benchmarks:stable',
    'Docker image to use. Defaults to the stable Orbax benchmark image.',
)
_TPU_TYPE = flags.DEFINE_string(
    'tpu_type',
    None,
    'TPU type (e.g., v5litepod-8, v4-8). Determines topology.',
)
_NUM_SLICES = flags.DEFINE_integer('num_slices', 1, 'Number of slices.')
_PROJECT = flags.DEFINE_string('project', 'orbax-checkpoint', 'GCP Project ID.')
_ZONE = flags.DEFINE_string('zone', 'europe-west4-a', 'GCP Zone.')
_WORKLOAD_NAME = flags.DEFINE_string(
    'workload_name', None, 'Name of the workload. Defaults to generated name.'
)
_CREATE_CLUSTER = flags.DEFINE_boolean(
    'create_cluster',
    True,
    'If True, automatically create the cluster if it does not exist.',
)
_DELETE_CLUSTER_ON_COMPLETION = flags.DEFINE_boolean(
    'delete_cluster_on_completion',
    True,
    'If True, delete the cluster after the workload completes.',
)
_WAIT = flags.DEFINE_boolean(
    'wait', True, 'If True, wait for the workload to complete.'
)

# ==============================================================================
# 3. Advanced Flags (Fine-tuning)
# ==============================================================================
# --- Cluster Customization ---
_CUSTOM_CLUSTER_ARGUMENTS = flags.DEFINE_string(
    'custom_cluster_arguments',
    None,
    'Extra arguments for cluster creation (e.g. --network=...).',
)
_RESERVATION = flags.DEFINE_string(
    'reservation',
    None,
    'Reservation ID for cluster creation.',
)
_ON_DEMAND = flags.DEFINE_boolean(
    'on_demand',
    False,
    'If True, use on-demand capacity (default is reservation if not set).',
)
_SPOT = flags.DEFINE_boolean(
    'spot',
    True,
    'If True, use spot/preemptible capacity.',
)
_DEVICE_TYPE = flags.DEFINE_string(
    'device_type',
    None,
    'Device type for GPU clusters (e.g., h100-mega-80gb-8).',
)
_ENABLE_TPU_AUTOLOCK = flags.DEFINE_boolean(
    'enable_tpu_autolock',
    False,
    'If True, enable TPU autolock.',
)
_CPU_LIMIT = flags.DEFINE_string(
    'cpu_limit',
    None,
    'CPU limit for the cluster (e.g., 112).',
)
_MEMORY_LIMIT = flags.DEFINE_string(
    'memory_limit',
    None,
    'Memory limit for the cluster (e.g., 192Gi).',
)
_PRIVATE = flags.DEFINE_boolean(
    'private',
    False,
    'If True, create a private cluster.',
)
_AUTHORIZED_NETWORKS = flags.DEFINE_list(
    'authorized_networks',
    None,
    'List of authorized networks for private cluster (CIDR ranges).',
)
_CREATE_VERTEX_TENSORBOARD = flags.DEFINE_boolean(
    'create_vertex_tensorboard',
    False,
    'If True, create a Vertex AI Tensorboard instance.',
)
_TENSORBOARD_REGION = flags.DEFINE_string(
    'tensorboard_region',
    None,
    'Region for Vertex AI Tensorboard.',
)
_TENSORBOARD_NAME = flags.DEFINE_string(
    'tensorboard_name',
    None,
    'Name for Vertex AI Tensorboard.',
)

# --- Workload Customization ---
_PRIORITY = flags.DEFINE_enum(
    'priority',
    'medium',
    ['very-low', 'low', 'medium', 'high', 'very-high'],
    'Priority of the workload (very-low, low, medium, high, very-high).',
)
_MAX_RESTARTS = flags.DEFINE_integer(
    'max_restarts',
    0,
    'Maximum number of restarts for the workload.',
)
_SCHEDULER = flags.DEFINE_string(
    'scheduler',
    None,
    'Scheduler to use (e.g., kueue).',
)
_DEBUG_DUMP_GCS = flags.DEFINE_string(
    'debug_dump_gcs',
    None,
    'GCS path for debug dumps.',
)
_USE_VERTEX_TENSORBOARD = flags.DEFINE_boolean(
    'use_vertex_tensorboard',
    False,
    'If True, use Vertex AI Tensorboard for the workload.',
)
_EXPERIMENT_NAME = flags.DEFINE_string(
    'experiment_name',
    None,
    'Name of the Vertex AI experiment.',
)
_ENV = flags.DEFINE_list(
    'env',
    None,
    'List of environment variables to set (KEY=VALUE).',
)
_ENV_FILE = flags.DEFINE_string(
    'env_file',
    None,
    'Path to a file containing environment variables.',
)
_DOCKER_NAME = flags.DEFINE_string(
    'docker_name',
    None,
    'Name of the docker container.',
)
_BENCHMARK_BINARY_PATH = flags.DEFINE_string(
    'benchmark_binary_path',
    '/app/orbax_repo/checkpoint/orbax/checkpoint/_src/testing/benchmarks/run_benchmarks.py',
    'Path to the benchmark runner script within the Docker image.',
)
_SA = flags.DEFINE_string(
    'sa',
    None,
    'Service account to use for the workload.',
)
_RUN_NAME = flags.DEFINE_string(
    'run_name',
    None,
    'Run name for the workload (overrides generated name if set).',
)
_ENABLE_OPS_AGENT = flags.DEFINE_boolean(
    'enable_ops_agent',
    False,
    'If True, enable Ops Agent for the workload.',
)
_RESTART_ON_USER_CODE_FAILURE = flags.DEFINE_boolean(
    'restart_on_user_code_failure',
    False,
    'If True, restart the workload if the user code fails.',
)
_RAMDISK_DIRECTORY = flags.DEFINE_string(
    'ramdisk_directory',
    None,
    'Directory to mount a ramdisk (e.g., /tmp/ramdisk).',
)
_XPK_PATH = flags.DEFINE_string(
    'xpk_path', 'xpk', 'Path to xpk binary or command.'
)
_SKIP_PREFLIGHT_CHECKS = flags.DEFINE_boolean(
    'skip_preflight_checks', False, 'If True, skip pre-flight checks.'
)
_VERBOSE = flags.DEFINE_boolean(
    'verbose', False, 'If True, show logs from XPK commands.'
)

# --- Pathways Flags ---
# Pathways uses a "Sidecar" architecture on XPK:
# - Server Container: Runs the Pathways Worker (manages TPUs).
# - Proxy Container: Runs the IFRT Proxy (bridges Client <-> Server).
# - Sidecar Container: Runs YOUR code (the Client).
_ENABLE_PATHWAYS = flags.DEFINE_boolean(
    'enable_pathways', False, 'Enable Pathways backend (Single-Controller).'
)
_PATHWAYS_SERVER_IMAGE = flags.DEFINE_string(
    'pathways_server_image',
    'us-docker.pkg.dev/cloud-tpu-v2-images/pathways-colocated-python/server:2025-10-03',
    'Pathways server image (manages TPU chips).',
)
_PATHWAYS_PROXY_IMAGE = flags.DEFINE_string(
    'pathways_proxy_image',
    'us-docker.pkg.dev/cloud-tpu-v2-images/pathways-colocated-python/proxy_server:2025-10-03',
    'Pathways proxy image (bridges user code to server).',
)
# LINT.ThenChange(README.md:launch_xpk_flags_table)


class Console:
  """Helper for rich console output."""

  GREEN = '\033[92m'
  YELLOW = '\033[93m'
  RED = '\033[91m'
  BLUE = '\033[94m'
  BOLD = '\033[1m'
  RESET = '\033[0m'

  @staticmethod
  def print_step(step_num: int, total_steps: int, message: str):
    print(f'{Console.BOLD}[{step_num}/{total_steps}] {message}{Console.RESET}')

  @staticmethod
  def print_success(message: str):
    print(f'  {Console.GREEN}✔{Console.RESET} {message}')

  @staticmethod
  def print_warning(message: str):
    print(f'  {Console.YELLOW}⚠ {message}{Console.RESET}')

  @staticmethod
  def print_error(message: str):
    print(f'  {Console.RED}✖ {message}{Console.RESET}')

  @staticmethod
  def print_info(message: str):
    print(f'  ℹ {message}')

  @staticmethod
  def print_link(label: str, url: str):
    print(f'  🔗 {label}: {Console.BLUE}{url}{Console.RESET}')


class PreconditionError(Exception):
  """Raised when a pre-flight check fails."""


def run_command(
    cmd: Sequence[str],
    *,
    capture_output: bool = False,
    suppress_output: bool = False,
    cwd: str | None = None,
) -> str | None:
  """Runs a shell command."""
  cmd_str = ' '.join(cmd)
  if not suppress_output:
    logging.debug('Running command: %s', cmd_str)

  try:
    if capture_output:
      return (
          subprocess.check_output(cmd, stderr=subprocess.STDOUT, cwd=cwd)
          .decode('utf-8')
          .strip()
      )
    else:
      subprocess.check_call(
          cmd,
          stdout=subprocess.DEVNULL if suppress_output else None,
          stderr=subprocess.DEVNULL if suppress_output else None,
          cwd=cwd,
      )
      return None
  except subprocess.CalledProcessError as e:
    if not suppress_output:
      logging.exception('Command failed')
      if capture_output and e.output:
        logging.error('Output: %s', e.output.decode('utf-8'))
    raise


def check_preconditions() -> bool:
  """Runs pre-flight checks. Returns True if cluster exists, False otherwise.

  Raises:
    PreconditionError: If any precondition check fails.

  Returns:
    True if the cluster already exists, False otherwise.
  """
  if _SKIP_PREFLIGHT_CHECKS.value:
    Console.print_warning('Skipping pre-flight checks.')
    return True

  Console.print_step(1, 6, 'Running Pre-flight Checks')

  # 1. Check Dependencies
  dependencies = ('xpk', 'gcloud')
  for dep in dependencies:
    try:
      run_command(['which', dep], capture_output=True, suppress_output=True)
      Console.print_success(f'Found {dep}')
    except subprocess.CalledProcessError as exc:
      Console.print_error(f'{dep} not found in PATH.')
      raise PreconditionError(f'{dep} not found.') from exc

  # 2. Check GCS Access
  bucket = _OUTPUT_DIRECTORY.value.split('/')[2]
  try:
    run_command(
        ['gcloud', 'storage', 'buckets', 'describe', f'gs://{bucket}'],
        capture_output=True,
        suppress_output=True,
    )
    Console.print_success(f'GCS Bucket accessible: gs://{bucket}')
  except subprocess.CalledProcessError as exc:
    Console.print_error(f'Cannot access GCS bucket: gs://{bucket}')
    raise PreconditionError(f'Cannot access GCS bucket: gs://{bucket}') from exc

  # 3. Check Docker Image
  images_to_check = [_DOCKER_IMAGE.value]

  if _ENABLE_PATHWAYS.value:
    images_to_check.extend(
        [_PATHWAYS_SERVER_IMAGE.value, _PATHWAYS_PROXY_IMAGE.value]
    )

  for img in images_to_check:
    # Simple check: if it's a gcr/pkg.dev image, try describing it.
    # If it's local or other, we might skip or just warn.
    if img.startswith(('gcr.io/', 'pkg.dev/')):
      try:
        run_command(
            [
                'gcloud',
                'container',
                'images',
                'describe',
                img,
                f'--project={_PROJECT.value}',
            ],
            capture_output=True,
            suppress_output=True,
        )
        Console.print_success(f'Docker image found: {img}')
      except subprocess.CalledProcessError:
        Console.print_warning(
            f'Could not verify Docker image: {img} (might be private or'
            ' missing)'
        )
    else:
      Console.print_info(f'Skipping verification for non-GCP image: {img}')

  # 4. Check Cluster Existence
  try:
    clusters = run_command(
        [
            _XPK_PATH.value,
            'cluster',
            'list',
            f'--project={_PROJECT.value}',
            f'--zone={_ZONE.value}',
        ],
        capture_output=True,
        suppress_output=not _VERBOSE.value,
    )
  except subprocess.CalledProcessError:
    Console.print_warning('Could not list clusters to verify existence.')
    return False
  else:
    if _CLUSTER_NAME.value in clusters:
      Console.print_success(f'Cluster found: {_CLUSTER_NAME.value}')
      return True

    if _CREATE_CLUSTER.value:
      Console.print_warning(
          f'Cluster {_CLUSTER_NAME.value} not found. Will create it.'
      )
      return False

    Console.print_error(
        f'Cluster {_CLUSTER_NAME.value} not found in'
        f' {_PROJECT.value}/{_ZONE.value}.'
    )
    raise PreconditionError(
        f'Cluster {_CLUSTER_NAME.value} not found. Use --create_cluster to'
        ' create it automatically.'
    )


def create_cluster() -> None:
  """Creates the XPK cluster."""
  Console.print_info(f'Creating cluster {_CLUSTER_NAME.value}...')

  cmd = [
      _XPK_PATH.value,
      'cluster',
      'create-pathways' if _ENABLE_PATHWAYS.value else 'create',
      f'--cluster={_CLUSTER_NAME.value}',
      f'--project={_PROJECT.value}',
      f'--zone={_ZONE.value}',
      f'--num-slices={_NUM_SLICES.value}',
  ]

  if _TPU_TYPE.value is not None:
    cmd.append(f'--tpu-type={_TPU_TYPE.value}')
  if _DEVICE_TYPE.value is not None:
    cmd.append(f'--device-type={_DEVICE_TYPE.value}')

  # Capacity Type
  if _RESERVATION.value is not None:
    cmd.append(f'--reservation={_RESERVATION.value}')
  elif _ON_DEMAND.value:
    cmd.append('--on-demand')
  elif _SPOT.value:
    cmd.append('--spot')

  # Custom Args
  if _CUSTOM_CLUSTER_ARGUMENTS.value is not None:
    cmd.append(f'--custom-cluster-arguments={_CUSTOM_CLUSTER_ARGUMENTS.value}')
  if _ENABLE_TPU_AUTOLOCK.value:
    cmd.append('--enable-tpu-autolock')
  if _CPU_LIMIT.value is not None:
    cmd.append(f'--cpu-limit={_CPU_LIMIT.value}')
  if _MEMORY_LIMIT.value is not None:
    cmd.append(f'--memory-limit={_MEMORY_LIMIT.value}')

  # Private Cluster
  if _PRIVATE.value:
    cmd.append('--private')
  if _AUTHORIZED_NETWORKS.value:
    cmd.append('--authorized-networks')
    cmd.extend(_AUTHORIZED_NETWORKS.value)

  # Vertex AI Tensorboard
  if _CREATE_VERTEX_TENSORBOARD.value:
    cmd.append('--create-vertex-tensorboard')
  if _TENSORBOARD_REGION.value is not None:
    cmd.append(f'--tensorboard-region={_TENSORBOARD_REGION.value}')
  if _TENSORBOARD_NAME.value is not None:
    cmd.append(f'--tensorboard-name={_TENSORBOARD_NAME.value}')

  run_command(cmd, suppress_output=not _VERBOSE.value)
  Console.print_success(f'Cluster {_CLUSTER_NAME.value} created.')


def construct_workload_command(
    *,
    config_file: str,
    output_directory: str,
    run_id: str,
    enable_pathways: bool,
    benchmark_binary_path: str,
) -> str:
  """Constructs the command to run inside the workload."""
  # Environment variables
  if enable_pathways:
    env_vars = [
        'export JAX_PLATFORMS=proxy',
        'export ENABLE_PATHWAYS_PERSISTENCE=1',
        'export ENABLE_PJRT_COMPATIBILITY=true',
    ]
  else:
    env_vars = ['export JAX_PLATFORMS=tpu,cpu']

  env_cmd = ' && '.join(env_vars) + ' && ' if env_vars else ''

  python_cmd = (
      f'python3 {benchmark_binary_path} '
      f'--config_file={config_file} '
      f'--output_directory={os.path.join(output_directory, run_id)} '
      '--v=1 '
      '--alsologtostderr'
  )

  return f'{env_cmd}{python_cmd}'


def construct_xpk_command(
    workload_name: str, workload_command: str
) -> Sequence[str]:
  """Constructs the XPK CLI command."""
  base_cmd = [
      _XPK_PATH.value,
      'workload',
      'create-pathways' if _ENABLE_PATHWAYS.value else 'create',
      f'--cluster={_CLUSTER_NAME.value}',
      f'--project={_PROJECT.value}',
      f'--zone={_ZONE.value}',
      f'--workload={workload_name}',
      f'--num-slices={_NUM_SLICES.value}',
      f'--priority={_PRIORITY.value}',
      '--storage=test-service-lustre',
  ]

  if _TPU_TYPE.value is not None:
    base_cmd.append(f'--tpu-type={_TPU_TYPE.value}')
  if _DEVICE_TYPE.value is not None:
    base_cmd.append(f'--device-type={_DEVICE_TYPE.value}')

  if _MAX_RESTARTS.value > 0:
    base_cmd.append(f'--max-restarts={_MAX_RESTARTS.value}')

  # Workload Customization
  if _SCHEDULER.value is not None:
    base_cmd.append(f'--scheduler={_SCHEDULER.value}')
  if _DEBUG_DUMP_GCS.value is not None:
    base_cmd.append(f'--debug-dump-gcs={_DEBUG_DUMP_GCS.value}')
  if _USE_VERTEX_TENSORBOARD.value:
    base_cmd.append('--use-vertex-tensorboard')
  if _EXPERIMENT_NAME.value is not None:
    base_cmd.append(f'--experiment-name={_EXPERIMENT_NAME.value}')
  if _ENV.value:
    for env_var in _ENV.value:
      base_cmd.append(f'--env={env_var}')
  if _ENV_FILE.value is not None:
    base_cmd.append(f'--env-file={_ENV_FILE.value}')
  if _DOCKER_NAME.value is not None:
    base_cmd.append(f'--docker-name={_DOCKER_NAME.value}')
  if _SA.value is not None:
    base_cmd.append(f'--sa={_SA.value}')
  if _RUN_NAME.value is not None:
    base_cmd.append(f'--run-name={_RUN_NAME.value}')
  if _ENABLE_OPS_AGENT.value:
    base_cmd.append('--enable-ops-agent')

  if _ENABLE_PATHWAYS.value:
    if not _PATHWAYS_SERVER_IMAGE.value:
      raise ValueError(
          'Pathways requires --pathways_server_image to be specified.'
      )
    if not _PATHWAYS_PROXY_IMAGE.value:
      raise ValueError(
          'Pathways requires --pathways_proxy_image to be specified.'
      )

    image_args = [
        f'--server-image={_PATHWAYS_SERVER_IMAGE.value}',
        f'--proxy-server-image={_PATHWAYS_PROXY_IMAGE.value}',
        f'--colocated-python-sidecar-image={_DOCKER_IMAGE.value}',
    ]

  else:
    # Standard mode
    image_args = [f'--docker-image={_DOCKER_IMAGE.value}']

  optional_args = []
  if _RESTART_ON_USER_CODE_FAILURE.value:
    optional_args.append('--restart-on-user-code-failure')
  if _RAMDISK_DIRECTORY.value is not None:
    optional_args.append(f'--ramdisk-directory={_RAMDISK_DIRECTORY.value}')

  return list(itertools.chain(
      base_cmd, image_args, optional_args, ['--command', workload_command]
  ))


def print_summary(
    *,
    workload_name: str,
    run_id: str,
    project: str,
    zone: str,
    cluster: str,
    output_directory: str,
):
  """Prints a clean summary of the launched workload."""
  gcs_bucket = output_directory.replace('gs://', '')

  summary = [
      '',
      '=' * 80,
      f'{Console.BOLD}🚀 Workload Launched Successfully{Console.RESET}',
      '=' * 80,
      f'  🆔 {Console.BOLD}Workload Name:{Console.RESET} {workload_name}',
      f'  🆔 {Console.BOLD}Run ID:{Console.RESET}        {run_id}',
      '',
      (
          f'  📄 {Console.BOLD}Logs:{Console.RESET}      '
          f' https://console.cloud.google.com/logs/query;query=resource.labels.pod_name:"{workload_name}"?project={project}'
      ),
      (
          f'  📦 {Console.BOLD}Artifacts:{Console.RESET} '
          f' https://console.cloud.google.com/storage/browser/{gcs_bucket}/{run_id}?project={project}'
      ),
      (
          f'  🔍 {Console.BOLD}Status:{Console.RESET}     xpk workload list'
          f' --cluster={cluster} --workload={workload_name} --project={project}'
          f' --zone={zone}'
      ),
      '=' * 80,
      '',
  ]
  print('\n'.join(summary))


def upload_config_to_gcs(local_path: str, gcs_root: str, run_id: str) -> str:
  """Uploads the local config file to GCS and returns the GCS path."""
  filename = os.path.basename(local_path)
  gcs_path = os.path.join(gcs_root, run_id, filename)

  Console.print_info(f'Uploading config to {gcs_path}')
  run_command(
      ['gcloud', 'storage', 'cp', local_path, gcs_path], suppress_output=True
  )
  return gcs_path


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # 1. Validation & Pre-flight
  cluster_exists = check_preconditions()

  # 2. Cluster Creation (if needed)
  if not cluster_exists:
    create_cluster()

  # 3. Preparation
  Console.print_step(2, 6, 'Preparing Workload')
  timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  run_id = (
      f"{os.environ.get('USER', 'user')}-{timestamp}-{uuid.uuid4().hex[:6]}"
  )

  if _WORKLOAD_NAME.value is not None:
    workload_name = _WORKLOAD_NAME.value
  else:
    base_name, _ = os.path.splitext(os.path.basename(_CONFIG_FILE.value))
    # XPK requires workload name < 40 chars.
    # Format: orbax-{base_name}-{timestamp}
    # timestamp (15) + orbax- (6) + separators (1) = 22 chars.
    # Max base_name = 40 - 22 = 18 chars. We use 15 to be safe.
    if len(base_name) > 15:
      base_name = base_name[:15]
    workload_name = f'orbax-{base_name}-{timestamp}'.replace('_', '-').lower()

  Console.print_info(f'Workload: {workload_name}')
  Console.print_info(f'Run ID:   {run_id}')

  # 4. Upload Config
  Console.print_step(3, 6, 'Uploading Configuration')
  remote_config_path = upload_config_to_gcs(
      _CONFIG_FILE.value, _OUTPUT_DIRECTORY.value, run_id
  )
  Console.print_success('Config uploaded.')

  # 5. Construct Commands
  Console.print_step(4, 6, 'Constructing Commands')
  workload_cmd = construct_workload_command(
      config_file=remote_config_path,
      output_directory=_OUTPUT_DIRECTORY.value,
      run_id=run_id,
      enable_pathways=_ENABLE_PATHWAYS.value,
      benchmark_binary_path=_BENCHMARK_BINARY_PATH.value,
  )
  xpk_cmd = construct_xpk_command(workload_name, workload_cmd)

  # 6. Launch
  Console.print_step(5, 6, 'Launching Workload')
  run_command(xpk_cmd, suppress_output=not _VERBOSE.value)

  print_summary(
      workload_name=workload_name,
      run_id=run_id,
      project=_PROJECT.value,
      zone=_ZONE.value,
      cluster=_CLUSTER_NAME.value,
      output_directory=_OUTPUT_DIRECTORY.value,
  )

  # 7. Post-Launch (Wait / Delete)
  should_wait = _WAIT.value or _DELETE_CLUSTER_ON_COMPLETION.value
  if not should_wait:
    return

  Console.print_step(6, 6, 'Post-Launch Actions')
  Console.print_info(f'Waiting for workload {workload_name}...')

  if _DELETE_CLUSTER_ON_COMPLETION.value:
    Console.print_warning(
        'Cluster auto-deletion is ENABLED. Do not interrupt if you want'
        ' auto-deletion.'
    )
  else:
    Console.print_info(
        'You can Ctrl+C to stop waiting (workload will continue).'
    )

  try:
    run_command(
        [
            _XPK_PATH.value,
            'workload',
            'list',
            f'--cluster={_CLUSTER_NAME.value}',
            f'--project={_PROJECT.value}',
            f'--zone={_ZONE.value}',
            f'--wait-for-job-completion={workload_name}',
        ],
        suppress_output=not _VERBOSE.value,
    )
  except subprocess.CalledProcessError:
    Console.print_error('Workload FAILED or was preempted.')
    Console.print_info(
        'Check logs: https://console.cloud.google.com/logs/query;'
        f'query=resource.labels.pod_name:"{workload_name}"?project={_PROJECT.value}'
    )
    sys.exit(1)
  except KeyboardInterrupt:
    Console.print_warning('\nWait interrupted by user.')
    if _DELETE_CLUSTER_ON_COMPLETION.value:
      Console.print_error('Skipping cluster deletion due to interruption.')
      sys.exit(1)
  else:
    Console.print_success('Workload completed successfully.')
    if _DELETE_CLUSTER_ON_COMPLETION.value:
      Console.print_info(f'Deleting cluster {_CLUSTER_NAME.value}...')
      run_command(
          [
              _XPK_PATH.value,
              'cluster',
              'delete',
              f'--cluster={_CLUSTER_NAME.value}',
              f'--project={_PROJECT.value}',
              f'--zone={_ZONE.value}',
              '--force',
          ],
          suppress_output=not _VERBOSE.value,
          cwd='/tmp',
      )
      Console.print_success('Cluster deleted.')


if __name__ == '__main__':
  app.run(main)
