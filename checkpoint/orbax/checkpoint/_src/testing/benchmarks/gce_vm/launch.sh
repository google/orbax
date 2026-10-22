#!/bin/bash
# Runs an Orbax benchmark on SimDevice across GCE CPU VMs created by
# manage_vms.sh. VM i runs the benchmark image as launcher task i of N, which
# starts its share of the config's simulated hosts (see core/launch_env.py and
# core/simdevice_backend.py). VM 0 hosts the JAX coordinator and the SimDevice
# barrier.

set -e
set -o pipefail

DEFAULT_ZONE="europe-west4-a"
DEFAULT_PROJECT="orbax-checkpoint"
COORDINATOR_PORT=1234
BARRIER_PORT=50051
CONTAINER_NAME="orbax-benchmark"

usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Options:"
  echo "  --config FILE          Local benchmark YAML with a simdevice: block (REQUIRED)"
  echo "  --image IMAGE          gcr.io/orbax-checkpoint/orbax-benchmarks:simdevice-<tag> (REQUIRED)"
  echo "  --output-dir GCS_URI   GCS directory for results (default: gs://orbax-benchmarks/benchmark-results/\${USER})"
  echo "  --name NAME            VM name prefix used with manage_vms.sh (default: \${USER}-simdevice)"
  echo "  --zone ZONE            GCP zone (default: $DEFAULT_ZONE)"
  echo "  --project PROJECT      GCP project (default: $DEFAULT_PROJECT)"
  echo "  --ssh-flag FLAG        Extra flag for 'gcloud compute ssh', e.g. --tunnel-through-iap (repeatable)"
  echo "  --help                 Show this help"
  exit 1
}

log() {
  echo ">>> $1" >&2
}

CONFIG_FILE=""
IMAGE=""
OUTPUT_DIR="gs://orbax-benchmarks/benchmark-results/${USER}"
NAME="${USER}-simdevice"
ZONE="$DEFAULT_ZONE"
PROJECT_ID="$DEFAULT_PROJECT"
declare -a SSH_FLAGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --project) PROJECT_ID="$2"; shift 2 ;;
    --ssh-flag) SSH_FLAGS+=("$2"); shift 2 ;;
    --help) usage ;;
    *) echo "Unknown argument: $1"; usage ;;
  esac
done

if [[ -z "$CONFIG_FILE" || -z "$IMAGE" ]]; then
  echo "Error: --config and --image are required."
  usage
fi
if [[ "$CONFIG_FILE" != /* && -n "$BUILD_WORKING_DIRECTORY" ]]; then
  # Run by `bazel run`, from the runfiles directory.
  CONFIG_FILE="$BUILD_WORKING_DIRECTORY/$CONFIG_FILE"
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Error: config file not found: $CONFIG_FILE"
  exit 1
fi
if [[ "$OUTPUT_DIR" != gs://* ]]; then
  echo "Error: --output-dir must be a gs:// path, shared by all VMs."
  exit 1
fi

vm_ssh() {
  local vm=$1
  local cmd=$2
  gcloud compute ssh "$vm" --zone="$ZONE" --project="$PROJECT_ID" \
    "${SSH_FLAGS[@]}" --command="$cmd"
}

# --- Discover the VMs: NAME-0 ... NAME-{N-1} ---
declare -a VMS=()
declare -a IPS=()
while IFS=, read -r vm ip; do
  VMS[${vm##*-}]="$vm"
  IPS[${vm##*-}]="$ip"
done < <(gcloud compute instances list --project="$PROJECT_ID" \
  --filter="name ~ ^${NAME}-[0-9]+$ AND zone:${ZONE} AND status=RUNNING" \
  --format="csv[no-heading](name,networkInterfaces[0].networkIP)")

NUM_VMS=${#VMS[@]}
if [[ "$NUM_VMS" -eq 0 ]]; then
  echo "Error: no running VMs named ${NAME}-<i> in $ZONE. Create them with manage_vms.sh."
  exit 1
fi
for ((i = 0; i < NUM_VMS; i++)); do
  if [[ -z "${VMS[$i]}" ]]; then
    echo "Error: found $NUM_VMS VMs but ${NAME}-${i} is missing."
    exit 1
  fi
done
TASK0_IP="${IPS[0]}"
log "VMs: ${VMS[*]} (task 0 at $TASK0_IP)"

# --- Upload the config ---
RUN_ID="${USER}-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${OUTPUT_DIR%/}/${RUN_ID}"
REMOTE_CONFIG="${RUN_DIR}/config/$(basename "$CONFIG_FILE")"
gcloud storage cp "$CONFIG_FILE" "$REMOTE_CONFIG"
log "Run directory: $RUN_DIR"

# Pushes the SSH key once, so the parallel connections below don't race.
vm_ssh "${VMS[0]}" "true"

REGISTRY="${IMAGE%%/*}"

# Container-Optimized OS drops incoming traffic except SSH. Accept TCP from the
# run's VMs: the JAX coordinator and SimDevice barrier on VM 0, and any
# process-to-process connections.
ALLOW_PEERS="true"
for ip in "${IPS[@]}"; do
  rule="INPUT -p tcp -s ${ip} -j ACCEPT"
  ALLOW_PEERS+=" && (sudo iptables -C ${rule} 2>/dev/null || sudo iptables -I ${rule})"
done

# Remote shell code that sets $DOCKER: plain docker if this user may use it,
# else sudo with this user's registry credentials (OS Login users are not in
# the docker group).
docker_cmd() {
  echo 'if docker info >/dev/null 2>&1; then DOCKER=docker;' \
    'else DOCKER="sudo docker --config $HOME/.docker"; fi'
}

run_on_vm() {
  local i=$1
  local cmd
  cmd="${ALLOW_PEERS}"
  cmd+=" && docker-credential-gcr configure-docker --registries=${REGISTRY} >/dev/null"
  cmd+=" && $(docker_cmd)"
  cmd+=" && (\$DOCKER rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true)"
  cmd+=" && \$DOCKER run --rm --name=${CONTAINER_NAME} --net=host"
  cmd+=" -e SIMDEVICE_TASK_ID=${i} -e SIMDEVICE_NUM_TASKS=${NUM_VMS}"
  cmd+=" -e SIMDEVICE_BARRIER_ADDRESS=${TASK0_IP}:${BARRIER_PORT}"
  cmd+=" -e SIMDEVICE_BARRIER_PORT=${BARRIER_PORT}"
  cmd+=" -e JAX_COORDINATOR_ADDRESS=${TASK0_IP}:${COORDINATOR_PORT}"
  cmd+=" ${IMAGE} --config_file=${REMOTE_CONFIG} --output_directory=${RUN_DIR}"
  cmd+=" --alsologtostderr"
  vm_ssh "${VMS[$i]}" "$cmd" 2>&1 | sed -u "s/^/[${VMS[$i]}] /"
}

stop_all() {
  for vm in "${VMS[@]}"; do
    vm_ssh "$vm" "$(docker_cmd) && \$DOCKER rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true" &
  done
  wait
}

declare -a PIDS=()
for ((i = 0; i < NUM_VMS; i++)); do
  run_on_vm "$i" &
  PIDS+=($!)
done

# Stop every VM's run as soon as one fails; the others would otherwise wait
# at a barrier until it times out.
failed=0
for ((n = 0; n < NUM_VMS; n++)); do
  if ! wait -n "${PIDS[@]}"; then
    failed=1
    log "A VM failed; stopping the others."
    stop_all
    break
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "Error: the benchmark failed. Results so far: $RUN_DIR" >&2
  exit 1
fi
log "Benchmark finished on $NUM_VMS VMs. Results: $RUN_DIR"
